import json
import math

import pytest

from agentconvos.language_corpus import (
    CandidateClaimType,
    CandidateSummary,
    ReplyCorpus,
    ReplyObservation,
    discover_recurring_candidates,
)


def _baseline_module():
    try:
        from agentconvos import language_baseline
    except ImportError:
        pytest.fail("agentconvos.language_baseline is not implemented")
    return language_baseline


def _reply(
    session: str,
    project: str,
    source: str,
    text: str,
    *,
    date: str = "2026-08-20",
    model: str | None = None,
) -> ReplyObservation:
    return ReplyObservation(
        session_id=session,
        project=project,
        source=source,
        date=date,
        model=model,
        turn_index=1,
        text=text,
        preceding_user="A synthetic prompt.",
    )


def _candidate(phrase: str = "signature phrase") -> CandidateSummary:
    return CandidateSummary(
        phrase=phrase,
        contexts=("content",),
        claim_types=(CandidateClaimType.RECURRING,),
        occurrences=8,
        replies=8,
        sessions=8,
        projects=8,
        reply_prevalence=1.0,
        session_prevalence=1.0,
        project_prevalence=1.0,
        project_concentration=0.125,
        echo_replies=0,
        echo_rate=0.0,
        ranking_score=10.0,
    )


def _words(count: int, marker: str) -> str:
    return " ".join([marker] + ["word"] * (count - 1))


def test_word_count_bins_have_frozen_boundaries():
    baseline = _baseline_module()

    assert baseline.cleaned_word_count_bin(_words(49, "a")) == "0-49"
    assert baseline.cleaned_word_count_bin(_words(50, "a")) == "50-149"
    assert baseline.cleaned_word_count_bin(_words(149, "a")) == "50-149"
    assert baseline.cleaned_word_count_bin(_words(150, "a")) == "150-399"
    assert baseline.cleaned_word_count_bin(_words(399, "a")) == "150-399"
    assert baseline.cleaned_word_count_bin(_words(400, "a")) == "400+"


def test_matching_is_deterministic_without_reuse_and_counts_unmatched_replies():
    baseline = _baseline_module()
    targets = (
        _reply("target-1", "project-a", "claude", _words(60, "target")),
        _reply("target-2", "project-a", "claude", _words(60, "target")),
        _reply("target-3", "project-c", "claude", _words(60, "target")),
    )
    candidates = (
        _reply("base-1", "project-a", "codex", _words(60, "baseline")),
        _reply("base-2", "project-a", "pi", _words(60, "baseline")),
        _reply("same-source", "project-a", "claude", _words(60, "baseline")),
        _reply(
            "wrong-month",
            "project-a",
            "codex",
            _words(60, "baseline"),
            date="2026-07-20",
        ),
        _reply("wrong-bin", "project-a", "codex", _words(160, "baseline")),
    )

    first = baseline.select_matched_baseline(targets, candidates)
    second = baseline.select_matched_baseline(
        tuple(reversed(targets)), tuple(reversed(candidates))
    )

    assert first.public_dict() == second.public_dict()
    assert first.target_count == 3
    assert first.baseline_count == 5
    assert first.matched_count == 2
    assert first.unmatched_target_count == 1
    assert first.unused_baseline_count == 3
    assert len({pair.baseline.session_id for pair in first.pairs}) == 2
    assert all(pair.target.project == pair.baseline.project for pair in first.pairs)
    assert all(pair.target.source != pair.baseline.source for pair in first.pairs)


def _matches(*, projects: int, target_has_phrase: bool, baseline_has_phrase: bool):
    baseline = _baseline_module()
    targets = []
    candidates = []
    for index in range(projects):
        project = f"project-{index}"
        target_text = "signature phrase appears" if target_has_phrase else "ordinary target reply"
        baseline_text = (
            "signature phrase appears" if baseline_has_phrase else "ordinary baseline reply"
        )
        targets.append(_reply(f"target-{index}", project, "claude", target_text))
        candidates.append(_reply(f"baseline-{index}", project, "codex", baseline_text))
    return baseline.select_matched_baseline(tuple(targets), tuple(candidates))


def test_matched_log_odds_and_project_bootstrap_are_positive_and_stable():
    baseline = _baseline_module()
    discovery = _matches(projects=8, target_has_phrase=True, baseline_has_phrase=False)
    validation = _matches(projects=3, target_has_phrase=True, baseline_has_phrase=False)

    first = baseline.analyze_matched_candidates(
        (_candidate(),), discovery, validation, _bootstrap_samples=200
    )[0]
    second = baseline.analyze_matched_candidates(
        (_candidate(),),
        baseline.MatchResult.from_pairs(tuple(reversed(discovery.pairs))),
        baseline.MatchResult.from_pairs(tuple(reversed(validation.pairs))),
        _bootstrap_samples=200,
    )[0]

    assert first.matched_target_reply_prevalence == 1.0
    assert first.baseline_reply_prevalence == 0.0
    assert first.distinctiveness == pytest.approx(2 * math.log(17))
    assert first.distinctiveness_ci_low > 0
    assert first.distinctiveness_ci_high > 0
    assert first.heldout_direction == "positive"
    assert CandidateClaimType.DISTINCTIVE in first.claim_types
    assert first.public_dict() == second.public_dict()


def test_distinctive_claim_requires_positive_held_out_confirmation():
    baseline = _baseline_module()
    discovery = _matches(projects=8, target_has_phrase=True, baseline_has_phrase=False)
    validation = _matches(projects=3, target_has_phrase=False, baseline_has_phrase=True)

    result = baseline.analyze_matched_candidates(
        (_candidate(),), discovery, validation, _bootstrap_samples=100
    )[0]

    assert result.distinctiveness_ci_low > 0
    assert result.heldout_direction == "negative"
    assert result.claim_types == (CandidateClaimType.RECURRING,)


def test_candidate_dispersion_is_deterministic_across_project_time_and_model():
    replies = (
        _reply(
            "session-1",
            "project-a",
            "claude",
            "A signature phrase.",
            date="2026-08-01",
            model="model-a",
        ),
        _reply(
            "session-2",
            "project-a",
            "claude",
            "The signature phrase.",
            date="2026-08-15",
            model="model-b",
        ),
        _reply(
            "session-3",
            "project-b",
            "claude",
            "Another signature phrase.",
            date="2026-09-01",
            model="model-a",
        ),
    )

    first = discover_recurring_candidates(
        ReplyCorpus.from_replies(replies),
        _minimum_sessions=2,
        _minimum_projects=2,
        collapse_overlaps=False,
    )
    second = discover_recurring_candidates(
        ReplyCorpus.from_replies(tuple(reversed(replies))),
        _minimum_sessions=2,
        _minimum_projects=2,
        collapse_overlaps=False,
    )
    candidate = next(item for item in first if item.phrase == "signature phrase")

    assert candidate.project_dispersion == (("project-a", 2), ("project-b", 1))
    assert candidate.time_dispersion == (("2026-08", 2), ("2026-09", 1))
    assert candidate.model_dispersion == (("model-a", 2), ("model-b", 1))
    assert [item.public_dict() for item in first] == [item.public_dict() for item in second]
    assert json.loads(json.dumps(candidate.public_dict())) == candidate.public_dict()
