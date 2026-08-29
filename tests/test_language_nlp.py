import json
import sqlite3
import tomllib
from pathlib import Path

import pytest

from agentconvos.language_corpus import ReplyObservation


def _nlp_module():
    try:
        from agentconvos import language_nlp
    except ImportError:
        pytest.fail("agentconvos.language_nlp is not implemented")
    return language_nlp


def _reply(session: str, project: str, text: str) -> ReplyObservation:
    return ReplyObservation(
        session_id=session,
        project=project,
        source="claude",
        date="2026-08-20",
        model="claude-fixture",
        turn_index=1,
        text=text,
        preceding_user="A synthetic prompt.",
    )


class _FakeDoc:
    def __init__(self, text: str):
        self.text = text
        self.tokens = [
            type("Token", (), {"lemma_": token.rstrip("s"), "is_space": False})()
            for token in text.split()
        ]

    def __iter__(self):
        return iter(self.tokens)


class _FakeNLP:
    meta = {"name": "core_web_sm", "version": "3.8.0", "lang": "en"}

    def __init__(self):
        self.added = []
        self.pipe_calls = []

    def add_pipe(self, name: str):
        self.added.append(name)

    def pipe(self, texts, *, n_process: int):
        materialized = list(texts)
        self.pipe_calls.append((materialized, n_process))
        return (_FakeDoc(text) for text in materialized)


class _FakeSpacy:
    def __init__(self, pipeline=None, error=None):
        self.pipeline = pipeline
        self.error = error
        self.loaded = []

    def load(self, model_name: str):
        self.loaded.append(model_name)
        if self.error:
            raise self.error
        return self.pipeline


class _FakeTextDescriptives:
    def __init__(self):
        self.extract_calls = []

    def extract_dict(self, doc, *, metrics, include_text):
        self.extract_calls.append((doc.text, tuple(metrics), include_text))
        return {
            "sentence_length_mean": float(len(doc.text.split())),
            "entropy": float("nan"),
            "quality_label": "not a numeric style descriptor",
        }


def _adapter(*, pipeline=None, package_versions=None):
    language_nlp = _nlp_module()
    fake_nlp = pipeline or _FakeNLP()
    fake_td = _FakeTextDescriptives()
    adapter = language_nlp.NLPAdapter.load(
        _spacy=_FakeSpacy(fake_nlp),
        _textdescriptives=fake_td,
        _package_versions=package_versions
        or {"spacy": "3.8.7", "textdescriptives": "2.8.4"},
    )
    return adapter, fake_nlp, fake_td


def test_language_extra_keeps_optional_dependencies_out_of_the_base_set():
    project = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]

    assert project["dependencies"] == ["httpx>=0.28.1", "textual>=3.0"]
    assert project["optional-dependencies"]["language"] == [
        "spacy>=3.8,<3.9",
        "textdescriptives>=2,<3",
    ]


def test_missing_language_extra_raises_one_domain_error_with_install_hint():
    language_nlp = _nlp_module()

    def missing_import(name):
        raise ImportError(name)

    with pytest.raises(language_nlp.LanguageNLPError) as raised:
        language_nlp.NLPAdapter.load(_import_module=missing_import)

    message = str(raised.value)
    assert "agentconvos[language]" in message
    assert "en_core_web_sm" in message
    assert "does not download" in message


def test_missing_explicit_model_raises_the_same_domain_error():
    language_nlp = _nlp_module()
    fake_td = _FakeTextDescriptives()

    with pytest.raises(language_nlp.LanguageNLPError) as raised:
        language_nlp.NLPAdapter.load(
            model_name="missing_trained_model",
            _spacy=_FakeSpacy(error=OSError("E050 model not found")),
            _textdescriptives=fake_td,
            _package_versions={"spacy": "3.8.7", "textdescriptives": "2.8.4"},
        )

    message = str(raised.value)
    assert "missing_trained_model" in message
    assert "agentconvos[language]" in message
    assert "does not download" in message


def test_adapter_registers_only_selected_components_and_pipes_once():
    language_nlp = _nlp_module()
    adapter, fake_nlp, fake_td = _adapter()
    replies = (
        _reply("session-1", "project-a", "One short synthetic reply."),
        _reply("session-2", "project-b", "Another synthetic reply with more words."),
    )

    batch = adapter.describe_replies(replies, cache_path=None)

    expected = tuple(f"textdescriptives/{name}" for name in language_nlp.SELECTED_COMPONENTS)
    assert tuple(fake_nlp.added) == expected
    assert language_nlp.SELECTED_COMPONENTS == (
        "descriptive_stats",
        "pos_proportions",
        "dependency_distance",
        "readability",
        "information_theory",
    )
    assert "coherence" not in expected
    assert "quality" not in expected
    assert fake_nlp.pipe_calls == [([reply.text for reply in replies], 1)]
    assert all(call[1] == language_nlp.SELECTED_COMPONENTS for call in fake_td.extract_calls)
    assert all(call[2] is False for call in fake_td.extract_calls)
    assert batch.cache_hits == 0
    assert batch.cache_misses == 2
    assert batch.records[0].descriptors["entropy"] is None
    assert "quality_label" not in batch.records[0].descriptors


def test_adapter_lemmatizes_phrases_with_single_process_batching():
    adapter, fake_nlp, _ = _adapter()

    lemmas = adapter.lemmatize_phrases(("checks remain", "tests pass"))

    assert lemmas == {"checks remain": "check remain", "tests pass": "test pa"}
    assert fake_nlp.pipe_calls == [(["checks remain", "tests pass"], 1)]


def test_adapter_accepts_textdescriptives_single_row_extract_dict_shape():
    class ListTextDescriptives(_FakeTextDescriptives):
        def extract_dict(self, doc, *, metrics, include_text):
            return [{"sentence_length_mean": 7.0}]

    language_nlp = _nlp_module()
    fake_nlp = _FakeNLP()
    adapter = language_nlp.NLPAdapter.load(
        _spacy=_FakeSpacy(fake_nlp),
        _textdescriptives=ListTextDescriptives(),
        _package_versions={"spacy": "3.8.16", "textdescriptives": "2.8.4"},
    )

    batch = adapter.describe_replies(
        (_reply("session-1", "project-a", "A synthetic reply."),),
        cache_path=None,
    )

    assert batch.records[0].descriptors == {"sentence_length_mean": 7.0}


def test_sqlite_cache_hits_and_pipeline_fingerprint_invalidation(tmp_path):
    cache_path = tmp_path / "language.sqlite3"
    replies = (
        _reply("session-1", "project-a", "One cached synthetic reply."),
        _reply("session-2", "project-b", "Another cached synthetic reply."),
    )
    first_adapter, first_nlp, _ = _adapter()

    first = first_adapter.describe_replies(replies, cache_path=cache_path)
    second = first_adapter.describe_replies(tuple(reversed(replies)), cache_path=cache_path)

    assert first.cache_hits == 0
    assert first.cache_misses == 2
    assert second.cache_hits == 2
    assert second.cache_misses == 0
    assert len(first_nlp.pipe_calls) == 1
    assert [record.public_dict() for record in first.records] == [
        record.public_dict() for record in second.records
    ]
    with sqlite3.connect(cache_path) as connection:
        assert connection.execute("SELECT count(*) FROM reply_descriptors").fetchone()[0] == 2
        stored = connection.execute(
            "SELECT descriptors_json FROM reply_descriptors ORDER BY reply_hash LIMIT 1"
        ).fetchone()[0]
    assert stored == json.dumps(json.loads(stored), sort_keys=True, separators=(",", ":"))

    changed_adapter, changed_nlp, _ = _adapter(
        package_versions={"spacy": "3.8.8", "textdescriptives": "2.8.4"}
    )
    changed = changed_adapter.describe_replies(replies, cache_path=cache_path)

    assert changed.cache_hits == 0
    assert changed.cache_misses == 2
    assert len(changed_nlp.pipe_calls) == 1
    assert changed.fingerprint.sha256 != first.fingerprint.sha256


def test_descriptor_aggregation_weights_projects_equally_and_reports_exclusions():
    language_nlp = _nlp_module()
    records = (
        language_nlp.DescriptorRecord("r1", "s1", "project-a", {"metric": 0.0}),
        language_nlp.DescriptorRecord("r2", "s2", "project-a", {"metric": 100.0}),
        language_nlp.DescriptorRecord("r3", "s3", "project-b", {"metric": 10.0}),
        language_nlp.DescriptorRecord("r4", "s4", "project-c", {"metric": None}),
    )

    aggregate = language_nlp.aggregate_descriptors(records)
    metric = next(item for item in aggregate if item.metric == "metric")

    assert metric.project_medians == (("project-a", 50.0), ("project-b", 10.0))
    assert metric.corpus_median == 30.0
    assert metric.iqr == 20.0
    assert metric.replies_covered == 3
    assert metric.replies_excluded == 1
    assert metric.projects_covered == 2
    assert metric.projects_excluded == 1
    assert json.loads(json.dumps(metric.public_dict())) == metric.public_dict()
