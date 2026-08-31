"""Deterministic matched-baseline analysis for frozen language candidates."""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace

from .language_corpus import (
    CandidateClaimType,
    CandidateSummary,
    ReplyObservation,
    cleaned_word_count,
    extract_reply_features,
)

MATCH_SEED = "agentconvos-matched-baseline-v1"
BOOTSTRAP_SEED = "agentconvos-project-bootstrap-v1"


def cleaned_word_count_bin(text: str) -> str:
    count = cleaned_word_count(text)
    if count < 50:
        return "0-49"
    if count < 150:
        return "50-149"
    if count < 400:
        return "150-399"
    return "400+"


def _month(reply: ReplyObservation) -> str | None:
    return reply.date[:7] if reply.date and len(reply.date) >= 7 else None


def _stratum(reply: ReplyObservation) -> tuple[str, str, str] | None:
    month = _month(reply)
    if not reply.project or month is None:
        return None
    return reply.project, month, cleaned_word_count_bin(reply.text)


def _reply_identity(reply: ReplyObservation) -> tuple:
    return (
        reply.project,
        reply.session_id,
        reply.turn_index,
        reply.source or "",
        reply.date or "",
        reply.model or "",
        reply.text,
    )


def _seeded_reply_key(reply: ReplyObservation, seed: str = MATCH_SEED) -> tuple[str, tuple]:
    identity = "\0".join(str(item) for item in _reply_identity(reply))
    return hashlib.sha256(f"{seed}\0{identity}".encode()).hexdigest(), _reply_identity(reply)


@dataclass(frozen=True)
class MatchedPair:
    target: ReplyObservation
    baseline: ReplyObservation

    def public_dict(self) -> dict:
        return {
            "target": self.target.public_dict(),
            "baseline": self.baseline.public_dict(),
        }


@dataclass(frozen=True)
class MatchResult:
    pairs: tuple[MatchedPair, ...]
    target_count: int
    baseline_count: int
    matched_count: int
    unmatched_target_count: int
    unused_baseline_count: int

    @classmethod
    def from_pairs(cls, pairs: Iterable[MatchedPair]) -> MatchResult:
        ordered = tuple(sorted(pairs, key=lambda pair: _reply_identity(pair.target)))
        return cls(
            pairs=ordered,
            target_count=len(ordered),
            baseline_count=len(ordered),
            matched_count=len(ordered),
            unmatched_target_count=0,
            unused_baseline_count=0,
        )

    def public_dict(self) -> dict:
        return {
            "target_count": self.target_count,
            "baseline_count": self.baseline_count,
            "matched_count": self.matched_count,
            "unmatched_target_count": self.unmatched_target_count,
            "unused_baseline_count": self.unused_baseline_count,
            "pairs": [pair.public_dict() for pair in self.pairs],
        }


def select_matched_baseline(
    targets: Iterable[ReplyObservation],
    baselines: Iterable[ReplyObservation],
) -> MatchResult:
    ordered_targets = tuple(sorted(targets, key=_seeded_reply_key))
    ordered_baselines = tuple(sorted(baselines, key=_seeded_reply_key))
    buckets: dict[tuple[str, str, str], list[ReplyObservation]] = defaultdict(list)
    for baseline in ordered_baselines:
        stratum = _stratum(baseline)
        if stratum is not None:
            buckets[stratum].append(baseline)

    pairs = []
    used_baselines: set[tuple] = set()
    for target in ordered_targets:
        stratum = _stratum(target)
        if stratum is None:
            continue
        available = buckets[stratum]
        selected_index = next(
            (
                index
                for index, baseline in enumerate(available)
                if baseline.source != target.source
            ),
            None,
        )
        if selected_index is None:
            continue
        baseline = available.pop(selected_index)
        used_baselines.add(_reply_identity(baseline))
        pairs.append(MatchedPair(target=target, baseline=baseline))

    ordered_pairs = tuple(sorted(pairs, key=lambda pair: _reply_identity(pair.target)))
    return MatchResult(
        pairs=ordered_pairs,
        target_count=len(ordered_targets),
        baseline_count=len(ordered_baselines),
        matched_count=len(ordered_pairs),
        unmatched_target_count=len(ordered_targets) - len(ordered_pairs),
        unused_baseline_count=len(ordered_baselines) - len(used_baselines),
    )


@dataclass(frozen=True)
class _PairPhrases:
    """One matched pair reduced to the candidate phrases each side contains.

    Feature extraction runs once per reply here; the bootstrap below resamples
    pairs a thousand times per candidate, so extracting inside the resample
    loop is unusable on real archives.
    """

    project: str
    target_phrases: frozenset[str]
    baseline_phrases: frozenset[str]


def _pair_phrases(
    pairs: Iterable[MatchedPair], phrases: Iterable[str]
) -> tuple[_PairPhrases, ...]:
    wanted = frozenset(phrases)
    return tuple(
        _PairPhrases(
            project=pair.target.project,
            target_phrases=wanted.intersection(
                feature.phrase for feature in extract_reply_features(pair.target.text)
            ),
            baseline_phrases=wanted.intersection(
                feature.phrase for feature in extract_reply_features(pair.baseline.text)
            ),
        )
        for pair in pairs
    )


def _smoothed_log_odds(
    target_hits: int,
    target_total: int,
    baseline_hits: int,
    baseline_total: int,
) -> float:
    target_odds = (target_hits + 0.5) / (target_total - target_hits + 0.5)
    baseline_odds = (baseline_hits + 0.5) / (baseline_total - baseline_hits + 0.5)
    return math.log(target_odds) - math.log(baseline_odds)


def _effect(pairs: Iterable[_PairPhrases], phrase: str) -> tuple[int, int, int, float]:
    materialized = tuple(pairs)
    target_hits = sum(phrase in pair.target_phrases for pair in materialized)
    baseline_hits = sum(phrase in pair.baseline_phrases for pair in materialized)
    total = len(materialized)
    return (
        target_hits,
        baseline_hits,
        total,
        _smoothed_log_odds(target_hits, total, baseline_hits, total),
    )


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


def _bootstrap_interval(
    pairs: tuple[_PairPhrases, ...],
    phrase: str,
    *,
    samples: int,
) -> tuple[float, float]:
    by_project: dict[str, list[_PairPhrases]] = defaultdict(list)
    for pair in pairs:
        by_project[pair.project].append(pair)
    projects = sorted(by_project)
    if not projects or samples <= 0:
        return 0.0, 0.0
    seed_bytes = hashlib.sha256(f"{BOOTSTRAP_SEED}\0{phrase}".encode()).digest()[:8]
    generator = random.Random(int.from_bytes(seed_bytes))
    effects = []
    for _ in range(samples):
        sampled_pairs = []
        for _ in projects:
            sampled_pairs.extend(by_project[generator.choice(projects)])
        effects.append(_effect(sampled_pairs, phrase)[3])
    effects.sort()
    return _percentile(effects, 0.025), _percentile(effects, 0.975)


def _direction(effect: float) -> str:
    if effect > 0:
        return "positive"
    if effect < 0:
        return "negative"
    return "neutral"


def analyze_matched_candidates(
    candidates: Iterable[CandidateSummary],
    discovery_matches: MatchResult,
    validation_matches: MatchResult,
    *,
    _bootstrap_samples: int = 1000,
) -> tuple[CandidateSummary, ...]:
    analyzed = []
    materialized = tuple(candidates)
    phrases = tuple(candidate.phrase for candidate in materialized)
    discovery_pairs = _pair_phrases(discovery_matches.pairs, phrases)
    validation_pairs = _pair_phrases(validation_matches.pairs, phrases)
    for candidate in materialized:
        target_hits, baseline_hits, total, effect = _effect(
            discovery_pairs, candidate.phrase
        )
        interval_low, interval_high = _bootstrap_interval(
            discovery_pairs,
            candidate.phrase,
            samples=_bootstrap_samples,
        )
        heldout_effect = _effect(validation_pairs, candidate.phrase)[3]
        heldout_direction = _direction(heldout_effect)
        claim_types = tuple(
            claim
            for claim in candidate.claim_types
            if claim != CandidateClaimType.DISTINCTIVE
        )
        if interval_low > 0 and heldout_effect > 0:
            claim_types += (CandidateClaimType.DISTINCTIVE,)
        analyzed.append(
            replace(
                candidate,
                claim_types=claim_types,
                matched_target_reply_prevalence=(target_hits / total if total else 0.0),
                baseline_reply_prevalence=(baseline_hits / total if total else 0.0),
                distinctiveness=effect,
                distinctiveness_ci_low=interval_low,
                distinctiveness_ci_high=interval_high,
                heldout_distinctiveness=heldout_effect,
                heldout_direction=heldout_direction,
            )
        )
    return tuple(analyzed)
