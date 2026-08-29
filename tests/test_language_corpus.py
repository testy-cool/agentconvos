import json
from pathlib import Path

import pytest

from agentconvos.parser import ConversationMeta


def _language():
    try:
        from agentconvos import language_corpus
    except ImportError:
        pytest.fail("agentconvos.language_corpus is not implemented")
    return language_corpus


def _meta(path: Path, session: str, project: str, date: str = "2026-08-20"):
    return ConversationMeta(
        path=path,
        uuid=session,
        slug="",
        timestamp=f"{date}T12:00:00Z",
        cwd=project,
        preview="fixture prompt",
        source="claude",
    )


def _observation(
    session: str,
    project: str,
    text: str,
    user: str = "A different request.",
    turn_index: int = 1,
    source: str | None = None,
):
    language = _language()
    return language.ReplyObservation(
        session_id=session,
        project=project,
        date=None,
        model=None,
        turn_index=turn_index,
        text=text,
        preceding_user=user,
        source=source,
    )


def test_corpus_pairs_each_assistant_reply_with_normalized_preceding_user(tmp_path):
    language = _language()
    path = tmp_path / "session.jsonl"
    records = [
        {"type": "system", "message": {"content": "bootstrap-only"}},
        {"type": "user", "message": {"content": "Please verify the load-bearing path."}},
        {
            "type": "assistant",
            "message": {
                "model": "claude-fixture",
                "content": [
                    {"type": "thinking", "thinking": "reasoning-only"},
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "Read",
                        "input": {"file_path": "/private/path.py"},
                    },
                    {"type": "text", "text": "I will verify the load-bearing path."},
                ],
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "tool-1", "content": "result-only"}
                ]
            },
        },
        {"type": "user", "message": {"content": "Now report the result."}},
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "The result is verified."}]},
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    corpus = language.build_reply_corpus(
        [_meta(path, "session-1", "/work/project-a")]
    )

    assert [reply.text for reply in corpus.replies] == [
        "I will verify the load-bearing path.",
        "The result is verified.",
    ]
    assert [reply.preceding_user for reply in corpus.replies] == [
        "Please verify the load-bearing path.",
        "Now report the result.",
    ]
    assert [reply.turn_index for reply in corpus.replies] == [1, 3]
    assert corpus.replies[0].date == "2026-08-20"
    assert corpus.replies[0].model == "claude-fixture"
    assert corpus.replies[0].source == "claude"
    serialized = corpus.replies[0].public_dict()
    assert list(serialized) == [
        "session_id",
        "project",
        "source",
        "date",
        "model",
        "turn_index",
        "text",
        "preceding_user",
    ]
    assert "bootstrap-only" not in json.dumps(corpus.public_dict())
    assert "reasoning-only" not in json.dumps(corpus.public_dict())
    assert "result-only" not in json.dumps(corpus.public_dict())
    assert "/private/path.py" not in json.dumps(corpus.public_dict())


def test_source_identity_is_serialized_and_changes_the_manifest():
    language = _language()
    claude = _observation(
        "session-1", "project-1", "The same normalized reply.", source="claude"
    )
    codex = _observation(
        "session-1", "project-1", "The same normalized reply.", source="codex"
    )

    claude_corpus = language.ReplyCorpus.from_replies((claude,))
    codex_corpus = language.ReplyCorpus.from_replies((codex,))

    assert claude_corpus.replies[0].public_dict()["source"] == "claude"
    assert claude_corpus.manifest.sha256 != codex_corpus.manifest.sha256


def test_manifest_and_project_split_are_repeatable_and_have_no_leakage():
    language = _language()
    replies = tuple(
        _observation(
            f"session-{index}",
            f"project-{index}",
            f"Stable recurring phrase number {index}.",
        )
        for index in range(10)
    )
    first = language.ReplyCorpus.from_replies(replies)
    second = language.ReplyCorpus.from_replies(tuple(reversed(replies)))

    assert first.manifest == second.manifest
    assert first.manifest.reply_count == 10
    assert len(first.manifest.sha256) == 64
    assert first.manifest.parser_version
    assert first.manifest.analysis_version
    changed = language.ReplyCorpus.from_replies(
        replies[:-1]
        + (_observation("session-9", "project-9", "Changed recurring phrase."),)
    )
    assert changed.manifest.sha256 != first.manifest.sha256

    split_one = language.split_corpus(first)
    split_two = language.split_corpus(second)
    discovery_projects = {reply.project for reply in split_one.discovery.replies}
    validation_projects = {reply.project for reply in split_one.validation.replies}
    assert len(discovery_projects) == 7
    assert len(validation_projects) == 3
    assert discovery_projects.isdisjoint(validation_projects)
    assert split_one.public_dict() == split_two.public_dict()


def test_candidate_extraction_strips_noise_but_preserves_hyphens_and_boundaries():
    language = _language()

    features = language.extract_reply_features(
        "If you want, use the load-bearing boundary at `/tmp/private.py`. "
        "See https://example.test user_id helperName abc123.\n"
        "```python\ndef hidden_identifier(): pass\n```\nThe checks pass."
    )
    by_phrase = {feature.phrase: feature for feature in features}

    assert "load-bearing" in by_phrase
    assert "load-bearing boundary" in by_phrase
    assert "if you want" in by_phrase
    assert "sentence_opener" in by_phrase["if you want"].contexts
    assert "the checks pass" in by_phrase
    assert "sentence_ender" in by_phrase["the checks pass"].contexts
    serialized = json.dumps([feature.public_dict() for feature in features])
    for excluded in (
        "tmp",
        "private",
        "example",
        "user_id",
        "helperName",
        "abc123",
        "hidden_identifier",
    ):
        assert excluded not in serialized


def test_echo_is_measured_exactly_and_downranks_prompted_phrases():
    language = _language()
    replies = []
    for index in range(3):
        replies.append(
            _observation(
                f"session-{index}",
                f"project-{index}",
                "Echoed phrase. Original phrase.",
                "Please use echoed phraseology and echoed phrase.",
            )
        )
    corpus = language.ReplyCorpus.from_replies(tuple(replies))

    candidates = language.discover_recurring_candidates(
        corpus,
        _minimum_sessions=2,
        _minimum_projects=2,
        collapse_overlaps=False,
    )
    by_phrase = {candidate.phrase: candidate for candidate in candidates}

    assert by_phrase["echoed phrase"].echo_replies == 3
    assert by_phrase["echoed phrase"].echo_rate == 1.0
    assert by_phrase["echoed phrase"].ranking_score < by_phrase["original phrase"].ranking_score
    assert by_phrase["original phrase"].echo_replies == 0


def test_project_vocabulary_is_ineligible_and_overlap_collapse_is_deterministic():
    language = _language()
    replies = [
        _observation(
            f"single-{index}",
            "one-project",
            "Private project vocabulary. Let me verify this boundary.",
        )
        for index in range(5)
    ]
    replies.extend(
        _observation(
            f"cross-{index}",
            f"cross-project-{index}",
            "Cross project phrase. Let me verify this boundary.",
        )
        for index in range(3)
    )
    corpus = language.ReplyCorpus.from_replies(tuple(replies))

    first = language.discover_recurring_candidates(
        corpus,
        _minimum_sessions=2,
        _minimum_projects=2,
    )
    second = language.discover_recurring_candidates(
        language.ReplyCorpus.from_replies(tuple(reversed(replies))),
        _minimum_sessions=2,
        _minimum_projects=2,
    )
    phrases = [candidate.phrase for candidate in first]

    assert "private project vocabulary" not in phrases
    assert "cross project phrase" in phrases
    assert not ({"let me verify", "let me verify this", "let me verify this boundary"} <= set(phrases))
    assert [candidate.public_dict() for candidate in first] == [
        candidate.public_dict() for candidate in second
    ]
    assert all(candidate.claim_types == ("recurring",) for candidate in first)
    assert all("distinctiveness" not in candidate.public_dict() for candidate in first)


def test_default_eligibility_scales_by_session_and_project_counts():
    language = _language()

    assert language.EligibilityThresholds.for_totals(20, 20).public_dict() == {
        "minimum_sessions": 10,
        "minimum_projects": 5,
    }
    assert language.EligibilityThresholds.for_totals(600, 120).public_dict() == {
        "minimum_sessions": 12,
        "minimum_projects": 6,
    }


def test_analysis_reports_synthetic_held_out_recurrence_in_stable_order():
    language = _language()
    replies = tuple(
        _observation(
            f"session-{index}",
            f"project-{index}",
            "A load-bearing boundary. It recurs. If you want, continue carefully.",
        )
        for index in range(10)
    )
    corpus = language.ReplyCorpus.from_replies(replies)

    first = language.analyze_recurring_candidates(
        corpus,
        _minimum_sessions=2,
        _minimum_projects=2,
    )
    second = language.analyze_recurring_candidates(
        language.ReplyCorpus.from_replies(tuple(reversed(replies))),
        _minimum_sessions=2,
        _minimum_projects=2,
    )
    load_bearing = next(
        candidate for candidate in first.candidates if candidate.phrase == "load-bearing boundary"
    )

    assert load_bearing.validation_replies == 3
    assert load_bearing.validation_sessions == 3
    assert load_bearing.validation_projects == 3
    assert first.public_dict() == second.public_dict()
    assert json.loads(json.dumps(first.public_dict())) == first.public_dict()
