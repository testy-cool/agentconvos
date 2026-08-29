"""Optional spaCy/TextDescriptives adapter and deterministic descriptor cache."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import platform
import sqlite3
import statistics
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from importlib import metadata
from numbers import Real
from pathlib import Path
from typing import Any

from .language_corpus import ANALYSIS_VERSION, ReplyObservation

DEFAULT_MODEL = "en_core_web_sm"
SELECTED_COMPONENTS = (
    "descriptive_stats",
    "pos_proportions",
    "dependency_distance",
    "readability",
    "information_theory",
)


class LanguageNLPError(RuntimeError):
    """The optional language pipeline or its explicitly named model is unavailable."""


def _unavailable_message(model_name: str) -> str:
    return (
        f"Language analysis for model '{model_name}' is unavailable. "
        "Install agentconvos[language] and install that trained spaCy model explicitly; "
        "this adapter does not download models at runtime."
    )


@dataclass(frozen=True)
class PipelineFingerprint:
    analysis_version: str
    python_version: str
    package_versions: tuple[tuple[str, str], ...]
    model_name: str
    model_version: str
    components: tuple[str, ...]

    def _identity_dict(self) -> dict:
        return {
            "analysis_version": self.analysis_version,
            "python_version": self.python_version,
            "package_versions": dict(self.package_versions),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "components": list(self.components),
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self._identity_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def public_dict(self) -> dict:
        return {**self._identity_dict(), "sha256": self.sha256}


@dataclass(frozen=True)
class DescriptorRecord:
    reply_hash: str
    session_id: str
    project: str
    descriptors: dict[str, float | None]

    def public_dict(self) -> dict:
        return {
            "reply_hash": self.reply_hash,
            "session_id": self.session_id,
            "project": self.project,
            "descriptors": dict(sorted(self.descriptors.items())),
        }


@dataclass(frozen=True)
class DescriptorBatch:
    fingerprint: PipelineFingerprint
    records: tuple[DescriptorRecord, ...]
    cache_hits: int
    cache_misses: int

    def public_dict(self) -> dict:
        return {
            "fingerprint": self.fingerprint.public_dict(),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "records": [record.public_dict() for record in self.records],
        }


def normalized_reply_hash(reply: ReplyObservation) -> str:
    return hashlib.sha256(reply.text.encode()).hexdigest()


def _descriptor_values(payload: Mapping[str, Any], prefix: str = "") -> dict[str, float | None]:
    flattened: dict[str, float | None] = {}
    for raw_key, value in payload.items():
        key = f"{prefix}.{raw_key}" if prefix else str(raw_key)
        if isinstance(value, Mapping):
            flattened.update(_descriptor_values(value, key))
        elif isinstance(value, Real) and not isinstance(value, bool):
            numeric = float(value)
            flattened[key] = numeric if math.isfinite(numeric) else None
    return flattened


class _DescriptorCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        with self.connection:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reply_descriptors (
                    reply_hash TEXT NOT NULL,
                    pipeline_fingerprint TEXT NOT NULL,
                    descriptors_json TEXT NOT NULL,
                    PRIMARY KEY (reply_hash, pipeline_fingerprint)
                )
                """
            )

    def close(self) -> None:
        self.connection.close()

    def get(self, reply_hash: str, fingerprint: str) -> dict[str, float | None] | None:
        row = self.connection.execute(
            """
            SELECT descriptors_json
            FROM reply_descriptors
            WHERE reply_hash = ? AND pipeline_fingerprint = ?
            """,
            (reply_hash, fingerprint),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put_many(
        self,
        fingerprint: str,
        descriptors: Iterable[tuple[str, dict[str, float | None]]],
    ) -> None:
        rows = [
            (
                reply_hash,
                fingerprint,
                json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False),
            )
            for reply_hash, values in descriptors
        ]
        with self.connection:
            self.connection.executemany(
                """
                INSERT OR REPLACE INTO reply_descriptors (
                    reply_hash, pipeline_fingerprint, descriptors_json
                ) VALUES (?, ?, ?)
                """,
                rows,
            )


def _reply_order(reply: ReplyObservation) -> tuple:
    return (reply.project, reply.session_id, reply.turn_index, reply.source or "", reply.text)


class NLPAdapter:
    def __init__(self, nlp, textdescriptives, fingerprint: PipelineFingerprint):
        self._nlp = nlp
        self._textdescriptives = textdescriptives
        self.fingerprint = fingerprint

    @classmethod
    def load(
        cls,
        model_name: str = DEFAULT_MODEL,
        *,
        _spacy=None,
        _textdescriptives=None,
        _package_versions: Mapping[str, str] | None = None,
        _import_module: Callable[[str], Any] = importlib.import_module,
    ) -> NLPAdapter:
        try:
            spacy = _spacy or _import_module("spacy")
            textdescriptives = _textdescriptives or _import_module("textdescriptives")
            package_versions = dict(
                _package_versions
                or {
                    "spacy": metadata.version("spacy"),
                    "textdescriptives": metadata.version("textdescriptives"),
                }
            )
        except (ImportError, metadata.PackageNotFoundError) as error:
            raise LanguageNLPError(_unavailable_message(model_name)) from error

        try:
            nlp = spacy.load(model_name)
        except (ImportError, OSError) as error:
            raise LanguageNLPError(_unavailable_message(model_name)) from error

        for component in SELECTED_COMPONENTS:
            nlp.add_pipe(f"textdescriptives/{component}")
        fingerprint = PipelineFingerprint(
            analysis_version=ANALYSIS_VERSION,
            python_version=platform.python_version(),
            package_versions=tuple(sorted(package_versions.items())),
            model_name=model_name,
            model_version=str(nlp.meta.get("version", "")),
            components=SELECTED_COMPONENTS,
        )
        return cls(nlp, textdescriptives, fingerprint)

    def describe_replies(
        self,
        replies: Iterable[ReplyObservation],
        *,
        cache_path: Path | None,
    ) -> DescriptorBatch:
        ordered = tuple(sorted(replies, key=_reply_order))
        cache = _DescriptorCache(Path(cache_path)) if cache_path is not None else None
        cached: dict[str, dict[str, float | None]] = {}
        missing: list[ReplyObservation] = []
        try:
            for reply in ordered:
                reply_hash = normalized_reply_hash(reply)
                values = cache.get(reply_hash, self.fingerprint.sha256) if cache else None
                if values is None:
                    missing.append(reply)
                else:
                    cached[reply_hash] = values

            generated: dict[str, dict[str, float | None]] = {}
            if missing:
                docs = self._nlp.pipe((reply.text for reply in missing), n_process=1)
                for reply, doc in zip(missing, docs, strict=True):
                    raw = self._textdescriptives.extract_dict(
                        doc,
                        metrics=SELECTED_COMPONENTS,
                        include_text=False,
                    )
                    generated[normalized_reply_hash(reply)] = _descriptor_values(raw)
                if cache:
                    cache.put_many(self.fingerprint.sha256, generated.items())

            values_by_hash = {**cached, **generated}
            records = tuple(
                DescriptorRecord(
                    reply_hash=normalized_reply_hash(reply),
                    session_id=reply.session_id,
                    project=reply.project,
                    descriptors=values_by_hash[normalized_reply_hash(reply)],
                )
                for reply in ordered
            )
            return DescriptorBatch(
                fingerprint=self.fingerprint,
                records=records,
                cache_hits=len(cached),
                cache_misses=len(missing),
            )
        finally:
            if cache:
                cache.close()


def _percentile(values: list[float], quantile: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


@dataclass(frozen=True)
class DescriptorAggregate:
    metric: str
    project_medians: tuple[tuple[str, float], ...]
    corpus_median: float | None
    iqr: float | None
    replies_covered: int
    replies_excluded: int
    projects_covered: int
    projects_excluded: int

    def public_dict(self) -> dict:
        return {
            "metric": self.metric,
            "project_medians": [
                {"project": project, "median": median}
                for project, median in self.project_medians
            ],
            "corpus_median": self.corpus_median,
            "iqr": self.iqr,
            "replies_covered": self.replies_covered,
            "replies_excluded": self.replies_excluded,
            "projects_covered": self.projects_covered,
            "projects_excluded": self.projects_excluded,
        }


def aggregate_descriptors(
    records: Iterable[DescriptorRecord],
) -> tuple[DescriptorAggregate, ...]:
    ordered = tuple(records)
    projects = {record.project for record in ordered}
    metrics = sorted(
        {metric for record in ordered for metric in record.descriptors}
    )
    aggregates = []
    for metric in metrics:
        by_project: dict[str, list[float]] = {}
        replies_covered = 0
        for record in ordered:
            value = record.descriptors.get(metric)
            if value is None or not math.isfinite(value):
                continue
            replies_covered += 1
            by_project.setdefault(record.project, []).append(value)
        project_medians = tuple(
            (project, float(statistics.median(values)))
            for project, values in sorted(by_project.items())
        )
        medians = sorted(median for _, median in project_medians)
        corpus_median = float(statistics.median(medians)) if medians else None
        iqr = (
            _percentile(medians, 0.75) - _percentile(medians, 0.25)
            if medians
            else None
        )
        aggregates.append(
            DescriptorAggregate(
                metric=metric,
                project_medians=project_medians,
                corpus_median=corpus_median,
                iqr=iqr,
                replies_covered=replies_covered,
                replies_excluded=len(ordered) - replies_covered,
                projects_covered=len(project_medians),
                projects_excluded=len(projects) - len(project_medians),
            )
        )
    return tuple(aggregates)
