"""Opt-in reproducible reply-language analysis and local reports."""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .language_baseline import analyze_matched_candidates, select_matched_baseline
from .language_corpus import (
    ANALYSIS_VERSION,
    CandidateClaimType,
    CandidateSummary,
    CorpusManifest,
    ReplyCorpus,
    ReplyObservation,
    _clean_prose,
    analyze_recurring_candidates,
    build_reply_corpus,
    extract_reply_features,
    split_corpus,
)
from .language_nlp import NLPAdapter, PipelineFingerprint, aggregate_descriptors
from .parser import ConversationMeta

MAX_DISPLAYED_PATTERNS = 50
MAX_EXAMPLES = 3


def default_cache_path() -> Path:
    configured = os.environ.get("AGENTCONVOS_LANGUAGE_CACHE")
    if configured:
        return Path(configured)
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_root / "agentconvos" / "language-descriptors.sqlite3"


def _counts(corpus: ReplyCorpus) -> dict[str, int]:
    return {
        "replies": len(corpus.replies),
        "sessions": len({reply.session_id for reply in corpus.replies}),
        "projects": len({reply.project for reply in corpus.replies}),
    }


def _project_label(project: str) -> str:
    digest = hashlib.sha256(project.encode()).hexdigest()[:10]
    return f"project-{digest}"


def _matching_phrase(reply: ReplyObservation, phrase: str) -> bool:
    return any(feature.phrase == phrase for feature in extract_reply_features(reply.text))


def _safe_excerpt(text: str, phrase: str) -> str:
    cleaned = " ".join(_clean_prose(text).split())
    match = re.search(re.escape(phrase), cleaned, flags=re.IGNORECASE)
    if not match:
        return cleaned[:280]
    start = max(0, match.start() - 100)
    end = min(len(cleaned), match.end() + 140)
    excerpt = cleaned[start:end].strip()
    if start:
        excerpt = "…" + excerpt
    if end < len(cleaned):
        excerpt += "…"
    return excerpt


def _examples(replies: tuple[ReplyObservation, ...], phrase: str) -> list[dict[str, str]]:
    examples = []
    projects = set()
    for reply in replies:
        if reply.project in projects or not _matching_phrase(reply, phrase):
            continue
        projects.add(reply.project)
        examples.append(
            {
                "project": _project_label(reply.project),
                "date": reply.date or "[unknown]",
                "model": reply.model or "[unknown]",
                "text": _safe_excerpt(reply.text, phrase),
            }
        )
        if len(examples) == MAX_EXAMPLES:
            break
    return examples


def _exact_variants(replies: tuple[ReplyObservation, ...], phrase: str) -> list[str]:
    variants = set()
    matcher = re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", re.IGNORECASE)
    for reply in replies:
        variants.update(match.group(0) for match in matcher.finditer(_clean_prose(reply.text)))
    return sorted(variants, key=lambda value: (value.casefold(), value)) or [phrase]


def _dispersion(values: tuple[tuple[str, int], ...], *, projects: bool = False):
    return [
        {
            "value": _project_label(value) if projects else value,
            "replies": replies,
        }
        for value, replies in values
    ]


def _descriptor_dict(metric) -> dict[str, Any]:
    payload = metric.public_dict()
    payload["project_medians"] = [
        {"project": _project_label(item["project"]), "median": item["median"]}
        for item in payload["project_medians"]
    ]
    return payload


def _claim_status(candidate: CandidateSummary) -> str:
    if CandidateClaimType.DISTINCTIVE in candidate.claim_types:
        return "distinctive"
    if candidate.validation_replies > 0:
        return "recurring-held-out"
    return "descriptive-candidate"


def _pattern_dict(
    candidate: CandidateSummary,
    *,
    lemma: str,
    target_replies: tuple[ReplyObservation, ...],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "surface_phrase": candidate.phrase,
        "lemma_phrase": lemma,
        "exact_variants": _exact_variants(target_replies, candidate.phrase),
        "contexts": list(candidate.contexts),
        "claim_types": [claim.value for claim in candidate.claim_types],
        "claim_status": _claim_status(candidate),
        "occurrences": candidate.occurrences,
        "prevalence": {
            "replies": candidate.replies,
            "reply_rate": candidate.reply_prevalence,
            "sessions": candidate.sessions,
            "session_rate": candidate.session_prevalence,
            "projects": candidate.projects,
            "project_rate": candidate.project_prevalence,
        },
        "echo": {"replies": candidate.echo_replies, "rate": candidate.echo_rate},
        "project_concentration": candidate.project_concentration,
        "ranking_score": candidate.ranking_score,
        "dispersion": {
            "projects": _dispersion(candidate.project_dispersion, projects=True),
            "time": _dispersion(candidate.time_dispersion),
            "models": _dispersion(candidate.model_dispersion),
        },
        "validation": {
            "replies": candidate.validation_replies,
            "sessions": candidate.validation_sessions,
            "projects": candidate.validation_projects,
        },
        "examples": _examples(target_replies, candidate.phrase),
    }
    if candidate.distinctiveness is not None:
        result.update(
            {
                "matched_target_reply_prevalence": candidate.matched_target_reply_prevalence,
                "baseline_reply_prevalence": candidate.baseline_reply_prevalence,
                "matched_log_odds": candidate.distinctiveness,
                "bootstrap_ci_95": [
                    candidate.distinctiveness_ci_low,
                    candidate.distinctiveness_ci_high,
                ],
                "heldout_log_odds": candidate.heldout_distinctiveness,
                "heldout_direction": candidate.heldout_direction,
            }
        )
    return result


@dataclass(frozen=True)
class LanguageReport:
    analysis_version: str
    manifest: CorpusManifest
    pipeline: PipelineFingerprint
    split: dict[str, Any]
    filters: dict[str, Any]
    corpus: dict[str, int]
    matching: dict[str, Any]
    descriptors: dict[str, Any]
    patterns: tuple[dict[str, Any], ...]
    cache_hits: int
    cache_misses: int

    def public_dict(self) -> dict[str, Any]:
        return {
            "analysis_version": self.analysis_version,
            "manifest": self.manifest.public_dict(),
            "pipeline": self.pipeline.public_dict(),
            "split": self.split,
            "filters": self.filters,
            "corpus": self.corpus,
            "matching": self.matching,
            "descriptors": self.descriptors,
            "patterns": list(self.patterns),
        }


def build_language_report(
    target: ReplyCorpus,
    baseline: ReplyCorpus,
    *,
    adapter: NLPAdapter,
    source: str,
    after: str | None,
    before: str | None,
    seed: int,
    baseline_mode: str,
    cache_path: Path,
    limit: int,
    _bootstrap_samples: int = 1000,
) -> LanguageReport:
    seed_text = str(seed)
    split = split_corpus(target, seed=seed_text)
    analysis = analyze_recurring_candidates(target, split_seed=seed_text)
    candidates = analysis.candidates
    matching: dict[str, Any] = {
        "mode": baseline_mode,
        "target": 0,
        "baseline": 0,
        "matched": 0,
        "unmatched": 0,
        "unused_baseline": 0,
        "coverage": 0.0,
    }
    if baseline_mode == "matched":
        baseline_split = split_corpus(baseline, seed=seed_text)
        discovery_matches = select_matched_baseline(
            split.discovery.replies, baseline_split.discovery.replies
        )
        validation_matches = select_matched_baseline(
            split.validation.replies, baseline_split.validation.replies
        )
        candidates = analyze_matched_candidates(
            candidates,
            discovery_matches,
            validation_matches,
            _bootstrap_samples=_bootstrap_samples,
        )
        target_count = discovery_matches.target_count + validation_matches.target_count
        matched_count = discovery_matches.matched_count + validation_matches.matched_count
        matching = {
            "mode": baseline_mode,
            "target": target_count,
            "baseline": discovery_matches.baseline_count + validation_matches.baseline_count,
            "matched": matched_count,
            "unmatched": discovery_matches.unmatched_target_count
            + validation_matches.unmatched_target_count,
            "unused_baseline": discovery_matches.unused_baseline_count
            + validation_matches.unused_baseline_count,
            "coverage": round(matched_count / target_count, 6) if target_count else 0.0,
        }

    descriptor_batch = adapter.describe_replies(target.replies, cache_path=cache_path)
    descriptor_metrics = aggregate_descriptors(descriptor_batch.records)
    displayed = tuple(candidates[: min(max(1, limit), MAX_DISPLAYED_PATTERNS)])
    lemmas = adapter.lemmatize_phrases(candidate.phrase for candidate in displayed)
    patterns = tuple(
        _pattern_dict(
            candidate,
            lemma=lemmas.get(candidate.phrase, candidate.phrase),
            target_replies=target.replies,
        )
        for candidate in displayed
    )
    return LanguageReport(
        analysis_version=ANALYSIS_VERSION,
        manifest=target.manifest,
        pipeline=descriptor_batch.fingerprint,
        split={
            "seed": seed,
            "discovery": _counts(split.discovery),
            "validation": _counts(split.validation),
        },
        filters={
            "source": source,
            "after": after,
            "before": before,
            "models": sorted({reply.model or "[unknown]" for reply in target.replies}),
        },
        corpus=_counts(target),
        matching=matching,
        descriptors={
            "metrics": [_descriptor_dict(metric) for metric in descriptor_metrics],
        },
        patterns=patterns,
        cache_hits=descriptor_batch.cache_hits,
        cache_misses=descriptor_batch.cache_misses,
    )


def _format_number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_html(report: LanguageReport) -> str:
    payload = report.public_dict()
    pattern_rows = []
    for pattern in payload["patterns"]:
        dispersion = pattern["dispersion"]
        dispersion_text = " · ".join(
            f"{label} "
            + ", ".join(
                f"{item['value']} ({item['replies']})" for item in dispersion[key]
            )
            for label, key in (("projects", "projects"), ("time", "time"), ("models", "models"))
        )
        examples = "".join(
            f"<li><code>{html.escape(example['project'])}</code> "
            f"{html.escape(example['text'])}</li>"
            for example in pattern["examples"]
        )
        effect = "—"
        if "matched_log_odds" in pattern:
            low, high = pattern["bootstrap_ci_95"]
            effect = (
                f"{_format_number(pattern['matched_log_odds'])} "
                f"[{_format_number(low)}, {_format_number(high)}]"
            )
        pattern_rows.append(
            "<tr>"
            f"<td><strong>{html.escape(pattern['surface_phrase'])}</strong><br>"
            f"lemma: {html.escape(pattern['lemma_phrase'])}<br>"
            f"{html.escape(', '.join(pattern['contexts']))}<br>"
            f"Exact variants: {html.escape(', '.join(pattern['exact_variants']))}<br>"
            "Project / time / model dispersion: "
            f"{html.escape(dispersion_text)}</td>"
            f"<td>{html.escape(pattern['claim_status'])}</td>"
            f"<td>{pattern['prevalence']['replies']} replies / "
            f"{pattern['prevalence']['sessions']} sessions / "
            f"{pattern['prevalence']['projects']} projects<br>"
            f"echo {_format_number(pattern['echo']['rate'])}; concentration "
            f"{_format_number(pattern['project_concentration'])}</td>"
            f"<td>{pattern['validation']['replies']} replies / "
            f"{pattern['validation']['projects']} projects</td>"
            f"<td>{effect}</td><td><ul>{examples}</ul></td></tr>"
        )
    descriptor_rows = "".join(
        "<tr>"
        f"<td>{html.escape(metric['metric'])}</td>"
        f"<td>{_format_number(metric['corpus_median'])}</td>"
        f"<td>{_format_number(metric['iqr'])}</td>"
        f"<td>{metric['replies_covered']} / "
        f"{metric['replies_covered'] + metric['replies_excluded']}</td></tr>"
        for metric in payload["descriptors"]["metrics"]
    )
    discovery = payload["split"]["discovery"]
    validation = payload["split"]["validation"]
    matching = payload["matching"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Reply-language evidence</title>
<style>body{{font:15px/1.45 system-ui;margin:2rem;max-width:1500px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:.55rem;vertical-align:top}}th{{text-align:left;background:#f3f3f3}}code{{font-size:.9em}}ul{{margin:.2rem 0;padding-left:1.2rem}}.meta{{display:grid;grid-template-columns:repeat(3,minmax(12rem,1fr));gap:.7rem}}.card{{border:1px solid #ddd;padding:.8rem}}</style></head><body>
<h1>Reply-language evidence</h1>
<p>Descriptive candidates from normalized assistant replies. Recurrence does not prove intent, authorship, training data, or inherent style.</p>
<div class="meta">
<div class="card"><strong>Canonical manifest</strong><br>analysis {html.escape(payload['analysis_version'])}<br><code>{payload['manifest']['sha256']}</code><br>{payload['corpus']['replies']} replies · {payload['corpus']['sessions']} sessions · {payload['corpus']['projects']} projects</div>
<div class="card"><strong>Discovery / validation</strong><br>seed {payload['split']['seed']}<br>{discovery['replies']} / {validation['replies']} replies<br>{discovery['projects']} / {validation['projects']} projects</div>
<div class="card"><strong>Pipeline</strong><br>{html.escape(payload['pipeline']['model_name'])} {html.escape(payload['pipeline']['model_version'])}<br>{html.escape(', '.join(f'{name} {version}' for name, version in payload['pipeline']['package_versions'].items()))}<br><code>{payload['pipeline']['sha256']}</code><br>descriptors cached by reply and fingerprint</div>
<div class="card"><strong>Filters</strong><br>source {html.escape(payload['filters']['source'])}<br>dates {html.escape(str(payload['filters']['after']))} to {html.escape(str(payload['filters']['before']))}<br>models {html.escape(', '.join(payload['filters']['models']))}</div>
<div class="card"><strong>Matched comparison</strong><br>{matching['mode']}<br>{matching['matched']} matched · {matching['unmatched']} unmatched<br>coverage {_format_number(matching['coverage'])}</div>
</div>
<h2>Patterns</h2><p>Surface and lemma phrases include content, function/rhetorical, sentence-opening, and sentence-closing candidates. Distinctive requires a positive held-out direction and a project-bootstrap 95% interval excluding zero.</p>
<table><thead><tr><th>Phrase and exact contexts</th><th>Cautious status</th><th>Discovery prevalence</th><th>Held-out evidence</th><th>Matched log odds [95% CI]</th><th>Assistant-only examples (max 3 distinct projects)</th></tr></thead><tbody>{''.join(pattern_rows)}</tbody></table>
<h2>TextDescriptives</h2><p>Project-equal median and IQR; each project's reply median receives equal weight. Coverage excludes missing or non-finite values. Coherence and quality labels are not analyzed as style.</p>
<table><thead><tr><th>Descriptor</th><th>Project-equal median</th><th>IQR</th><th>Reply coverage</th></tr></thead><tbody>{descriptor_rows}</tbody></table>
</body></html>"""


def write_language_report(report: LanguageReport, output: Path) -> tuple[Path, Path]:
    output = Path(output)
    json_path = output.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    html_tmp = output.with_name(output.name + ".tmp")
    json_tmp = json_path.with_name(json_path.name + ".tmp")
    html_tmp.write_text(render_html(report), encoding="utf-8")
    json_tmp.write_text(
        json.dumps(report.public_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    html_tmp.replace(output)
    json_tmp.replace(json_path)
    return output, json_path


def run_language_report(
    target_conversations: list[ConversationMeta],
    baseline_conversations: list[ConversationMeta],
    *,
    source: str,
    after: str | None,
    before: str | None,
    seed: int,
    baseline_mode: str,
    spacy_model: str,
    output: Path,
    limit: int,
    cache_path: Path | None = None,
) -> LanguageReport:
    adapter = NLPAdapter.load(spacy_model)
    target = build_reply_corpus(target_conversations)
    baseline = build_reply_corpus(baseline_conversations)
    report = build_language_report(
        target,
        baseline,
        adapter=adapter,
        source=source,
        after=after,
        before=before,
        seed=seed,
        baseline_mode=baseline_mode,
        cache_path=cache_path or default_cache_path(),
        limit=limit,
    )
    write_language_report(report, output)
    return report
