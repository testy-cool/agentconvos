import json
from pathlib import Path

from agentconvos.language_corpus import ReplyCorpus, ReplyObservation
from agentconvos.language_nlp import (
    DescriptorBatch,
    DescriptorRecord,
    PipelineFingerprint,
)


def _reply(index: int, *, source: str = "claude", phrase: bool = True):
    project = f"/private/client/project-{index % 10}"
    marker = "The load-bearing boundary." if phrase else "A separate response."
    return ReplyObservation(
        session_id=f"private-session-{source}-{index}",
        project=project,
        source=source,
        date=f"2026-0{7 + index % 2}-15",
        model=f"private-model-{index % 2}",
        turn_index=1,
        text=f"{marker} Never expose /private/client/secret-{index}.txt.",
        preceding_user=f"private prompt {index} /private/user/path",
    )


class _FakeAdapter:
    fingerprint = PipelineFingerprint(
        analysis_version="reply-language-v2",
        python_version="3.12.0",
        package_versions=(("spacy", "3.8.16"), ("textdescriptives", "2.8.4")),
        model_name="en_core_web_sm",
        model_version="3.8.0",
        components=("descriptive_stats",),
    )

    def describe_replies(self, replies, *, cache_path: Path | None):
        ordered = tuple(replies)
        return DescriptorBatch(
            fingerprint=self.fingerprint,
            records=tuple(
                DescriptorRecord(
                    reply_hash=f"hash-{index}",
                    session_id=reply.session_id,
                    project=reply.project,
                    descriptors={"sentence_length_mean": float(index + 1)},
                )
                for index, reply in enumerate(ordered)
            ),
            cache_hits=0,
            cache_misses=len(ordered),
        )

    def lemmatize_phrases(self, phrases):
        return {phrase: phrase.replace("remains", "remain") for phrase in phrases}


def test_deterministic_report_schema_html_evidence_and_privacy(tmp_path):
    from agentconvos.language_report import build_language_report, render_html

    target = ReplyCorpus.from_replies(_reply(index) for index in range(20))
    first = build_language_report(
        target,
        ReplyCorpus.from_replies(()),
        adapter=_FakeAdapter(),
        source="claude",
        after="2026-07-01",
        before="2026-08-31",
        seed=42,
        baseline_mode="none",
        cache_path=tmp_path / "cache.sqlite3",
        limit=20,
    )
    second = build_language_report(
        ReplyCorpus.from_replies(reversed(target.replies)),
        ReplyCorpus.from_replies(()),
        adapter=_FakeAdapter(),
        source="claude",
        after="2026-07-01",
        before="2026-08-31",
        seed=42,
        baseline_mode="none",
        cache_path=tmp_path / "cache.sqlite3",
        limit=20,
    )

    payload = first.public_dict()
    assert payload == second.public_dict()
    assert list(payload) == [
        "analysis_version",
        "manifest",
        "pipeline",
        "split",
        "filters",
        "corpus",
        "matching",
        "descriptors",
        "patterns",
    ]
    assert payload["manifest"]["sha256"] == target.manifest.sha256
    assert payload["split"]["seed"] == 42
    assert payload["filters"] == {
        "source": "claude",
        "after": "2026-07-01",
        "before": "2026-08-31",
        "models": ["private-model-0", "private-model-1"],
    }
    assert payload["corpus"]["replies"] == 20
    assert payload["split"]["discovery"]["projects"] == 7
    assert payload["split"]["validation"]["projects"] == 3
    assert payload["matching"]["mode"] == "none"
    assert payload["descriptors"]["metrics"][0]["corpus_median"] is not None
    assert payload["patterns"]
    pattern = next(item for item in payload["patterns"] if item["surface_phrase"] == "load-bearing boundary")
    assert pattern["lemma_phrase"] == "load-bearing boundary"
    assert pattern["exact_variants"] == ["load-bearing boundary"]
    assert pattern["validation"]["replies"] > 0
    assert pattern["claim_status"] == "recurring-held-out"
    assert len(pattern["examples"]) <= 3
    assert len({example["project"] for example in pattern["examples"]}) == len(pattern["examples"])

    serialized = json.dumps(payload, sort_keys=True)
    rendered = render_html(first)
    for private in (
        "private prompt",
        "private-session",
        "/private/client",
        "/private/user",
    ):
        assert private not in serialized
        assert private not in rendered
    assert "Canonical manifest" in rendered
    assert "Discovery / validation" in rendered
    assert "Project-equal median" in rendered
    assert "Exact variants" in rendered
    assert "Project / time / model dispersion" in rendered
    assert "load-bearing boundary" in rendered


def test_matched_report_requires_heldout_confirmation_for_distinctive_claim(tmp_path):
    from agentconvos.language_report import build_language_report

    targets = [_reply(index) for index in range(20)]
    baselines = [
        _reply(index, source="codex", phrase=index < 14)
        for index in range(20)
    ]
    report = build_language_report(
        ReplyCorpus.from_replies(targets),
        ReplyCorpus.from_replies(baselines),
        adapter=_FakeAdapter(),
        source="claude",
        after=None,
        before=None,
        seed=42,
        baseline_mode="matched",
        cache_path=tmp_path / "cache.sqlite3",
        limit=20,
        _bootstrap_samples=100,
    )

    payload = report.public_dict()
    assert payload["matching"]["matched"] + payload["matching"]["unmatched"] == 20
    assert 0 <= payload["matching"]["coverage"] <= 1
    pattern = next(item for item in payload["patterns"] if item["surface_phrase"] == "load-bearing boundary")
    assert "matched_log_odds" in pattern
    assert "bootstrap_ci_95" in pattern
    if pattern["heldout_direction"] != "positive":
        assert pattern["claim_status"] != "distinctive"


def test_matching_uses_the_target_project_partition_even_with_extra_baseline_projects(
    tmp_path,
):
    from agentconvos.language_report import build_language_report

    targets = [_reply(index) for index in range(20)]
    baselines = [_reply(index, source="codex", phrase=False) for index in range(20)]
    baselines.extend(
        ReplyObservation(
            session_id=f"extra-{index}",
            project=f"/unmatched/extra-{index}",
            source="codex",
            date="2026-07-15",
            model="baseline-model",
            turn_index=1,
            text="An unrelated baseline reply.",
            preceding_user=None,
        )
        for index in range(100)
    )

    report = build_language_report(
        ReplyCorpus.from_replies(targets),
        ReplyCorpus.from_replies(baselines),
        adapter=_FakeAdapter(),
        source="claude",
        after=None,
        before=None,
        seed=42,
        baseline_mode="matched",
        cache_path=tmp_path / "cache.sqlite3",
        limit=5,
        _bootstrap_samples=10,
    )

    assert report.matching["matched"] == 20
    assert report.matching["unmatched"] == 0


def test_write_report_emits_stable_json_and_html(tmp_path):
    from agentconvos.language_report import build_language_report, write_language_report

    report = build_language_report(
        ReplyCorpus.from_replies(_reply(index) for index in range(20)),
        ReplyCorpus.from_replies(()),
        adapter=_FakeAdapter(),
        source="claude",
        after=None,
        before=None,
        seed=42,
        baseline_mode="none",
        cache_path=tmp_path / "cache.sqlite3",
        limit=5,
    )
    output = tmp_path / "report.html"
    write_language_report(report, output)
    first_json = output.with_suffix(".json").read_bytes()
    write_language_report(report, output)

    assert output.is_file()
    assert output.with_suffix(".json").read_bytes() == first_json
    assert json.loads(first_json) == report.public_dict()


def test_cache_state_does_not_change_deterministic_report_artifacts(tmp_path):
    from agentconvos.language_report import build_language_report

    class StatefulAdapter(_FakeAdapter):
        calls = 0

        def describe_replies(self, replies, *, cache_path: Path | None):
            batch = super().describe_replies(replies, cache_path=cache_path)
            self.calls += 1
            if self.calls == 1:
                return batch
            return DescriptorBatch(
                fingerprint=batch.fingerprint,
                records=batch.records,
                cache_hits=len(batch.records),
                cache_misses=0,
            )

    adapter = StatefulAdapter()
    corpus = ReplyCorpus.from_replies(_reply(index) for index in range(20))
    kwargs = {
        "adapter": adapter,
        "source": "claude",
        "after": None,
        "before": None,
        "seed": 42,
        "baseline_mode": "none",
        "cache_path": tmp_path / "cache.sqlite3",
        "limit": 5,
    }

    first = build_language_report(corpus, ReplyCorpus.from_replies(()), **kwargs)
    second = build_language_report(corpus, ReplyCorpus.from_replies(()), **kwargs)

    assert first.cache_misses == 20 and first.cache_hits == 0
    assert second.cache_hits == 20 and second.cache_misses == 0
    assert first.public_dict() == second.public_dict()
    assert "cache_hits" not in first.public_dict()["descriptors"]
