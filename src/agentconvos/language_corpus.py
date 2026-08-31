"""Deterministic reply-level corpus and recurring-language candidate analysis."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from enum import StrEnum

from .parser import DETAIL_TEXT, ConversationMeta, get_stats, parse_jsonl

PARSER_VERSION = "normalized-detail-text-v1"
ANALYSIS_VERSION = "reply-language-v2"
SPLIT_SEED = "agentconvos-reply-language-v1"

_FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]+`")
_URL = re.compile(r"(?:https?://|www\.)[^\s)\]}>]+", re.IGNORECASE)
_PATH = re.compile(r"(?<!\w)(?:~|\.{0,2})?/(?:[^\s,;:!?()\[\]{}]+)")
_DOTTED_IDENTIFIER = re.compile(
    r"\b[A-Za-z][A-Za-z0-9-]*\.(?:py|pyi|js|jsx|ts|tsx|go|rs|java|rb|php|sh|toml|yaml|yml|json|md)\b",
    re.IGNORECASE,
)
_SNAKE_IDENTIFIER = re.compile(r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b")
_CAMEL_IDENTIFIER = re.compile(r"\b(?:[a-z]+[A-Z][A-Za-z0-9]*|[A-Z][a-z]+[A-Z][A-Za-z0-9]*)\b")
_ALPHANUMERIC_IDENTIFIER = re.compile(
    r"\b(?=[A-Za-z0-9-]*[A-Za-z])(?=[A-Za-z0-9-]*\d)[A-Za-z0-9-]{4,}\b"
)
_SENTENCE_BREAK = re.compile(r"(?:[.!?]+|[\r\n]+)")
_TOKEN = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", re.UNICODE)

_FUNCTION_WORDS = frozenset(
    """
    a about after all also am an and any are as at be because been before being both but by
    can could did do does doing each for from had has have having he her here him his how i if
    in into is it its just let may me might more most my no nor not now of on once only or other
    our out over same she should so some than that the their them then there these they this those
    through to too under up us very was we were what when where which while who why will with would
    you your
    """.split()
)
_ARTICLES = frozenset({"a", "an", "the"})
_CONTEXT_ORDER = {
    "content": 0,
    "function": 1,
    "sentence_opener": 2,
    "sentence_ender": 3,
}


class CandidateClaimType(StrEnum):
    RECURRING = "recurring"
    DISTINCTIVE = "distinctive"


@dataclass(frozen=True)
class ReplyObservation:
    session_id: str
    project: str
    date: str | None
    model: str | None
    turn_index: int
    text: str
    preceding_user: str | None
    source: str | None = None

    def public_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "project": self.project,
            "source": self.source,
            "date": self.date,
            "model": self.model,
            "turn_index": self.turn_index,
            "text": self.text,
            "preceding_user": self.preceding_user,
        }


@dataclass(frozen=True)
class CorpusManifest:
    parser_version: str
    analysis_version: str
    reply_count: int
    sha256: str

    def public_dict(self) -> dict:
        return {
            "parser_version": self.parser_version,
            "analysis_version": self.analysis_version,
            "reply_count": self.reply_count,
            "sha256": self.sha256,
        }


def _reply_sort_key(reply: ReplyObservation) -> tuple:
    return (
        reply.project,
        reply.session_id,
        reply.source or "",
        reply.turn_index,
        reply.text,
        reply.preceding_user or "",
        reply.date or "",
        reply.model or "",
    )


def _manifest(replies: tuple[ReplyObservation, ...]) -> CorpusManifest:
    payload = json.dumps(
        [reply.public_dict() for reply in replies],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return CorpusManifest(
        parser_version=PARSER_VERSION,
        analysis_version=ANALYSIS_VERSION,
        reply_count=len(replies),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


@dataclass(frozen=True)
class ReplyCorpus:
    replies: tuple[ReplyObservation, ...]
    manifest: CorpusManifest

    @classmethod
    def from_replies(cls, replies: Iterable[ReplyObservation]) -> ReplyCorpus:
        ordered = tuple(sorted(replies, key=_reply_sort_key))
        return cls(replies=ordered, manifest=_manifest(ordered))

    def public_dict(self) -> dict:
        return {
            "manifest": self.manifest.public_dict(),
            "replies": [reply.public_dict() for reply in self.replies],
        }


def build_reply_corpus(conversations: Iterable[ConversationMeta]) -> ReplyCorpus:
    """Build one observation per normalized assistant reply using text detail."""
    replies: list[ReplyObservation] = []
    ordered = sorted(
        conversations,
        key=lambda conversation: (
            conversation.cwd,
            conversation.uuid,
            str(conversation.path),
        ),
    )
    for conversation in ordered:
        turns = parse_jsonl(conversation.path, detail=DETAIL_TEXT)
        model = get_stats(conversation.path).model or None
        preceding_user: str | None = None
        for turn_index, turn in enumerate(turns):
            text = turn.text.strip()
            if turn.role == "user":
                preceding_user = text or None
            elif turn.role == "assistant" and text:
                replies.append(
                    ReplyObservation(
                        session_id=conversation.uuid,
                        project=conversation.cwd or str(conversation.path.parent),
                        source=conversation.source or None,
                        date=(conversation.timestamp or "")[:10] or None,
                        model=model,
                        turn_index=turn_index,
                        text=text,
                        preceding_user=preceding_user,
                    )
                )
    return ReplyCorpus.from_replies(replies)


@dataclass(frozen=True)
class CorpusSplit:
    seed: str
    discovery: ReplyCorpus
    validation: ReplyCorpus

    def public_dict(self) -> dict:
        return {
            "seed": self.seed,
            "discovery": self.discovery.public_dict(),
            "validation": self.validation.public_dict(),
        }


def split_corpus(corpus: ReplyCorpus, *, seed: str = SPLIT_SEED) -> CorpusSplit:
    """Split whole projects into a repeatable 70% discovery and 30% validation set."""
    by_project: dict[str, list[ReplyObservation]] = defaultdict(list)
    for reply in corpus.replies:
        by_project[reply.project].append(reply)
    projects = sorted(
        by_project,
        key=lambda project: (
            hashlib.sha256(f"{seed}\0{project}".encode()).hexdigest(),
            project,
        ),
    )
    if len(projects) <= 1:
        discovery_count = len(projects)
    else:
        discovery_count = min(len(projects) - 1, max(1, (7 * len(projects) + 5) // 10))
    discovery_projects = set(projects[:discovery_count])
    discovery = [
        reply for reply in corpus.replies if reply.project in discovery_projects
    ]
    validation = [
        reply for reply in corpus.replies if reply.project not in discovery_projects
    ]
    return CorpusSplit(
        seed=seed,
        discovery=ReplyCorpus.from_replies(discovery),
        validation=ReplyCorpus.from_replies(validation),
    )


def _preserve_terminal_punctuation(match: re.Match[str]) -> str:
    terminal = match.group(0)[-1:]
    return f" {terminal} " if terminal in ".!?" else " "


def _clean_prose(text: str) -> str:
    cleaned = _FENCED_CODE.sub(" ", text or "")
    for pattern in (
        _INLINE_CODE,
        _URL,
        _PATH,
        _DOTTED_IDENTIFIER,
        _SNAKE_IDENTIFIER,
        _CAMEL_IDENTIFIER,
        _ALPHANUMERIC_IDENTIFIER,
    ):
        cleaned = pattern.sub(_preserve_terminal_punctuation, cleaned)
    return cleaned


def _sentence_tokens(text: str) -> list[list[str]]:
    sentences: list[list[str]] = []
    for sentence in _SENTENCE_BREAK.split(_clean_prose(text)):
        tokens = [token.casefold().replace("’", "'") for token in _TOKEN.findall(sentence)]
        if tokens:
            sentences.append(tokens)
    return sentences


def _grams(tokens: list[str], minimum: int, maximum: int):
    for size in range(minimum, min(maximum, len(tokens)) + 1):
        for start in range(len(tokens) - size + 1):
            yield tokens[start : start + size]


@dataclass(frozen=True)
class ExtractedFeature:
    phrase: str
    contexts: tuple[str, ...]
    occurrences: int

    def public_dict(self) -> dict:
        return {
            "phrase": self.phrase,
            "contexts": list(self.contexts),
            "occurrences": self.occurrences,
        }


def extract_reply_features(text: str) -> tuple[ExtractedFeature, ...]:
    """Extract content, function/rhetorical, opener, and ender phrase candidates."""
    sentences = _sentence_tokens(text)
    contexts: dict[str, set[str]] = defaultdict(set)
    # Occurrences are counted in the same sweep that generates the grams;
    # rescanning the reply once per distinct phrase is quadratic in reply
    # length and unusable on real archives.
    occurrence_counts: Counter[str] = Counter()
    for tokens in sentences:
        for size in range(1, min(5, len(tokens)) + 1):
            for start in range(len(tokens) - size + 1):
                words = tokens[start : start + size]
                phrase = " ".join(words)
                occurrence_counts[phrase] += 1
                if (
                    size <= 3
                    and words[0] not in _FUNCTION_WORDS
                    and words[-1] not in _FUNCTION_WORDS
                ):
                    contexts[phrase].add("content")
                if size >= 2:
                    function_words = {word for word in words if word in _FUNCTION_WORDS}
                    if function_words and function_words - _ARTICLES:
                        contexts[phrase].add("function")
        if tokens[0] not in _ARTICLES:
            for size in range(2, min(5, len(tokens)) + 1):
                contexts[" ".join(tokens[:size])].add("sentence_opener")
        for size in range(2, min(5, len(tokens)) + 1):
            contexts[" ".join(tokens[-size:])].add("sentence_ender")

    features = [
        ExtractedFeature(
            phrase=phrase,
            contexts=tuple(sorted(feature_contexts, key=_CONTEXT_ORDER.get)),
            occurrences=occurrence_counts[phrase],
        )
        for phrase, feature_contexts in contexts.items()
    ]
    return tuple(sorted(features, key=lambda feature: feature.phrase))


def _phrase_gram_set(text: str | None) -> set[str]:
    """Every one-to-five-token run in the text, for O(1) phrase membership."""
    grams: set[str] = set()
    for tokens in _sentence_tokens(text or ""):
        for words in _grams(tokens, 1, 5):
            grams.add(" ".join(words))
    return grams


@dataclass(frozen=True)
class EligibilityThresholds:
    minimum_sessions: int
    minimum_projects: int

    @classmethod
    def for_totals(cls, sessions: int, projects: int) -> EligibilityThresholds:
        return cls(
            minimum_sessions=max(10, math.ceil(sessions * 0.02)),
            minimum_projects=max(5, math.ceil(projects * 0.05)),
        )

    def public_dict(self) -> dict:
        return {
            "minimum_sessions": self.minimum_sessions,
            "minimum_projects": self.minimum_projects,
        }


@dataclass(frozen=True)
class CandidateSummary:
    phrase: str
    contexts: tuple[str, ...]
    claim_types: tuple[CandidateClaimType, ...]
    occurrences: int
    replies: int
    sessions: int
    projects: int
    reply_prevalence: float
    session_prevalence: float
    project_prevalence: float
    project_concentration: float
    echo_replies: int
    echo_rate: float
    ranking_score: float
    validation_replies: int = 0
    validation_sessions: int = 0
    validation_projects: int = 0
    project_dispersion: tuple[tuple[str, int], ...] = ()
    time_dispersion: tuple[tuple[str, int], ...] = ()
    model_dispersion: tuple[tuple[str, int], ...] = ()
    distinctiveness: float | None = None
    matched_target_reply_prevalence: float | None = None
    baseline_reply_prevalence: float | None = None
    distinctiveness_ci_low: float | None = None
    distinctiveness_ci_high: float | None = None
    heldout_distinctiveness: float | None = None
    heldout_direction: str | None = None

    def public_dict(self) -> dict:
        result = {
            "phrase": self.phrase,
            "contexts": list(self.contexts),
            "claim_types": [claim.value for claim in self.claim_types],
            "occurrences": self.occurrences,
            "replies": self.replies,
            "sessions": self.sessions,
            "projects": self.projects,
            "reply_prevalence": self.reply_prevalence,
            "session_prevalence": self.session_prevalence,
            "project_prevalence": self.project_prevalence,
            "project_concentration": self.project_concentration,
            "echo_replies": self.echo_replies,
            "echo_rate": self.echo_rate,
            "ranking_score": self.ranking_score,
            "validation_replies": self.validation_replies,
            "validation_sessions": self.validation_sessions,
            "validation_projects": self.validation_projects,
            "project_dispersion": [
                {"value": value, "replies": replies}
                for value, replies in self.project_dispersion
            ],
            "time_dispersion": [
                {"value": value, "replies": replies}
                for value, replies in self.time_dispersion
            ],
            "model_dispersion": [
                {"value": value, "replies": replies}
                for value, replies in self.model_dispersion
            ],
        }
        if self.distinctiveness is not None:
            result["distinctiveness"] = self.distinctiveness
        if self.matched_target_reply_prevalence is not None:
            result["matched_target_reply_prevalence"] = self.matched_target_reply_prevalence
        if self.baseline_reply_prevalence is not None:
            result["baseline_reply_prevalence"] = self.baseline_reply_prevalence
        if self.distinctiveness_ci_low is not None:
            result["distinctiveness_ci_low"] = self.distinctiveness_ci_low
        if self.distinctiveness_ci_high is not None:
            result["distinctiveness_ci_high"] = self.distinctiveness_ci_high
        if self.heldout_distinctiveness is not None:
            result["heldout_distinctiveness"] = self.heldout_distinctiveness
        if self.heldout_direction is not None:
            result["heldout_direction"] = self.heldout_direction
        return result


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


@dataclass
class _CandidateCounts:
    contexts: set[str] = field(default_factory=set)
    occurrences: int = 0
    replies: int = 0
    sessions: set[str] = field(default_factory=set)
    projects: set[str] = field(default_factory=set)
    project_replies: Counter[str] = field(default_factory=Counter)
    time_replies: Counter[str] = field(default_factory=Counter)
    model_replies: Counter[str] = field(default_factory=Counter)
    echo_replies: int = 0


def _candidate_summaries(
    corpus: ReplyCorpus,
    wanted: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, CandidateSummary]:
    counts = {
        phrase: _CandidateCounts(contexts=set(contexts))
        for phrase, contexts in (wanted or {}).items()
    }
    for reply in corpus.replies:
        # Tokenized once per reply; the old per-feature containment scan
        # retokenized the preceding user message thousands of times.
        preceding_grams = _phrase_gram_set(reply.preceding_user)
        for feature in extract_reply_features(reply.text):
            if wanted is not None and feature.phrase not in wanted:
                continue
            candidate = counts.setdefault(feature.phrase, _CandidateCounts())
            candidate.contexts.update(feature.contexts)
            candidate.occurrences += feature.occurrences
            candidate.replies += 1
            candidate.sessions.add(reply.session_id)
            candidate.projects.add(reply.project)
            candidate.project_replies[reply.project] += 1
            month = reply.date[:7] if reply.date and len(reply.date) >= 7 else "[unknown]"
            candidate.time_replies[month] += 1
            candidate.model_replies[reply.model or "[unknown]"] += 1
            if feature.phrase in preceding_grams:
                candidate.echo_replies += 1

    total_sessions = len({reply.session_id for reply in corpus.replies})
    total_projects = len({reply.project for reply in corpus.replies})
    summaries = {}
    for phrase, candidate in counts.items():
        session_count = len(candidate.sessions)
        project_count = len(candidate.projects)
        concentration = _ratio(
            max(candidate.project_replies.values(), default=0), candidate.replies
        )
        echo_rate = _ratio(candidate.echo_replies, candidate.replies)
        ranking_score = round(
            math.log2(session_count + 1)
            * math.log2(project_count + 1)
            * (1 - concentration)
            * (1 - 0.75 * echo_rate),
            8,
        )
        summaries[phrase] = CandidateSummary(
            phrase=phrase,
            contexts=tuple(sorted(candidate.contexts, key=_CONTEXT_ORDER.get)),
            claim_types=(CandidateClaimType.RECURRING,),
            occurrences=candidate.occurrences,
            replies=candidate.replies,
            sessions=session_count,
            projects=project_count,
            reply_prevalence=_ratio(candidate.replies, len(corpus.replies)),
            session_prevalence=_ratio(session_count, total_sessions),
            project_prevalence=_ratio(project_count, total_projects),
            project_concentration=concentration,
            echo_replies=candidate.echo_replies,
            echo_rate=echo_rate,
            ranking_score=ranking_score,
            project_dispersion=tuple(sorted(candidate.project_replies.items())),
            time_dispersion=tuple(sorted(candidate.time_replies.items())),
            model_dispersion=tuple(sorted(candidate.model_replies.items())),
        )
    return summaries


def _candidate_sort_key(candidate: CandidateSummary) -> tuple:
    return (
        -candidate.ranking_score,
        -candidate.sessions,
        -candidate.projects,
        -candidate.occurrences,
        candidate.phrase.split()[0] in _ARTICLES,
        -len(candidate.phrase.split()),
        candidate.phrase,
    )


def _phrases_overlap(left: str, right: str) -> bool:
    left_tokens = left.split()
    right_tokens = right.split()
    short, long = (
        (left_tokens, right_tokens)
        if len(left_tokens) <= len(right_tokens)
        else (right_tokens, left_tokens)
    )
    return any(long[start : start + len(short)] == short for start in range(len(long) - len(short) + 1))


def collapse_overlapping_candidates(
    candidates: Iterable[CandidateSummary],
) -> tuple[CandidateSummary, ...]:
    """Keep the highest-ranked representative of each containment cluster."""
    selected: list[CandidateSummary] = []
    for candidate in sorted(candidates, key=_candidate_sort_key):
        if any(
            candidate.replies == existing.replies
            and candidate.sessions == existing.sessions
            and candidate.projects == existing.projects
            and _phrases_overlap(candidate.phrase, existing.phrase)
            for existing in selected
        ):
            continue
        selected.append(candidate)
    return tuple(selected)


def discover_recurring_candidates(
    corpus: ReplyCorpus,
    *,
    _minimum_sessions: int | None = None,
    _minimum_projects: int | None = None,
    collapse_overlaps: bool = True,
) -> tuple[CandidateSummary, ...]:
    """Discover recurring phrases; underscored thresholds are synthetic-test hooks."""
    sessions = len({reply.session_id for reply in corpus.replies})
    projects = len({reply.project for reply in corpus.replies})
    defaults = EligibilityThresholds.for_totals(sessions, projects)
    minimum_sessions = (
        defaults.minimum_sessions if _minimum_sessions is None else _minimum_sessions
    )
    minimum_projects = (
        defaults.minimum_projects if _minimum_projects is None else _minimum_projects
    )
    # Prune on distinct-session count first: holding full counting state for
    # every distinct n-gram in the archive costs gigabytes, while the phrases
    # that can pass eligibility are a tiny fraction of them.
    session_counts: Counter[str] = Counter()
    for _, session_replies in itertools.groupby(
        sorted(corpus.replies, key=lambda reply: reply.session_id),
        key=lambda reply: reply.session_id,
    ):
        session_phrases: set[str] = set()
        for reply in session_replies:
            session_phrases.update(
                feature.phrase for feature in extract_reply_features(reply.text)
            )
        session_counts.update(session_phrases)
    wanted: dict[str, tuple[str, ...]] = {
        phrase: ()
        for phrase, count in session_counts.items()
        if count >= minimum_sessions
    }
    del session_counts
    candidates = [
        summary
        for summary in _candidate_summaries(corpus, wanted).values()
        if summary.sessions >= minimum_sessions and summary.projects >= minimum_projects
    ]
    if collapse_overlaps:
        return collapse_overlapping_candidates(candidates)
    return tuple(sorted(candidates, key=_candidate_sort_key))


@dataclass(frozen=True)
class CandidateAnalysis:
    manifest: CorpusManifest
    split_seed: str
    eligibility: EligibilityThresholds
    candidates: tuple[CandidateSummary, ...]

    def public_dict(self) -> dict:
        return {
            "manifest": self.manifest.public_dict(),
            "split_seed": self.split_seed,
            "eligibility": self.eligibility.public_dict(),
            "candidates": [candidate.public_dict() for candidate in self.candidates],
        }


def analyze_recurring_candidates(
    corpus: ReplyCorpus,
    *,
    split_seed: str = SPLIT_SEED,
    _minimum_sessions: int | None = None,
    _minimum_projects: int | None = None,
) -> CandidateAnalysis:
    split = split_corpus(corpus, seed=split_seed)
    discovery_sessions = len({reply.session_id for reply in split.discovery.replies})
    discovery_projects = len({reply.project for reply in split.discovery.replies})
    defaults = EligibilityThresholds.for_totals(discovery_sessions, discovery_projects)
    eligibility = EligibilityThresholds(
        minimum_sessions=(
            defaults.minimum_sessions if _minimum_sessions is None else _minimum_sessions
        ),
        minimum_projects=(
            defaults.minimum_projects if _minimum_projects is None else _minimum_projects
        ),
    )
    discovered = discover_recurring_candidates(
        split.discovery,
        _minimum_sessions=eligibility.minimum_sessions,
        _minimum_projects=eligibility.minimum_projects,
    )
    validation_summaries = _candidate_summaries(
        split.validation,
        {candidate.phrase: candidate.contexts for candidate in discovered},
    )
    with_validation = []
    for candidate in discovered:
        validation = validation_summaries[candidate.phrase]
        with_validation.append(
            replace(
                candidate,
                validation_replies=validation.replies,
                validation_sessions=validation.sessions,
                validation_projects=validation.projects,
            )
        )
    return CandidateAnalysis(
        manifest=corpus.manifest,
        split_seed=split.seed,
        eligibility=eligibility,
        candidates=tuple(sorted(with_validation, key=_candidate_sort_key)),
    )


def cleaned_word_count(text: str) -> int:
    """Count normalized prose tokens after code, URL, path, and identifier removal."""
    return sum(len(tokens) for tokens in _sentence_tokens(text))
