"""Persistent incremental full-text index for conversation search."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path

from .parser import (
    ConversationMeta,
    SearchHit,
    Turn,
    _search_snippet,
    conversation_signature,
    parse_jsonl,
    parse_search_terms,
)

DEFAULT_INDEX_PATH = Path.home() / ".claude" / "convo-explorer" / "search-index.sqlite3"


@dataclass(frozen=True)
class IndexSyncStats:
    total: int
    checked: int = 0
    indexed: int = 0
    unchanged: int = 0
    removed: int = 0
    failed: int = 0


class ConversationSearchIndex:
    """SQLite FTS5 index keyed by transcript path, size, and mtime."""

    def __init__(self, path: Path | None = None) -> None:
        # Resolved here rather than bound at import, so the location can be
        # redirected. Tests that sync a handful of fixtures against the real
        # index delete every conversation missing from their fixture list.
        self.path = Path(path) if path is not None else Path(DEFAULT_INDEX_PATH)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS search_documents (
                path TEXT PRIMARY KEY,
                uuid TEXT NOT NULL,
                size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                metadata_key TEXT NOT NULL,
                turns_indexed INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(search_documents)")
        }
        if "turns_indexed" not in columns:
            connection.execute(
                "ALTER TABLE search_documents "
                "ADD COLUMN turns_indexed INTEGER NOT NULL DEFAULT 0"
            )
        connection.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS search_documents_fts USING fts5(
                uuid UNINDEXED,
                path UNINDEXED,
                metadata,
                content,
                tokenize='unicode61 remove_diacritics 2'
            )
            """
        )
        connection.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS search_turns_fts USING fts5(
                uuid UNINDEXED,
                path UNINDEXED,
                turn_index UNINDEXED,
                role UNINDEXED,
                timestamp UNINDEXED,
                metadata,
                content,
                tokenize='unicode61 remove_diacritics 2'
            )
            """
        )
        connection.commit()
        return connection

    @staticmethod
    def _metadata(meta: ConversationMeta) -> str:
        return "\n".join(
            value
            for value in (
                meta.uuid,
                meta.slug,
                meta.source,
                meta.cwd,
                meta.git_branch,
                meta.preview,
            )
            if value
        )

    def sync(
        self,
        conversations: Iterable[ConversationMeta],
        *,
        parse_conversation: Callable[[Path], list[Turn]] = parse_jsonl,
        on_progress: Callable[[IndexSyncStats], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> IndexSyncStats:
        metas = list(conversations)
        stats = IndexSyncStats(total=len(metas))
        connection = self._connect()
        try:
            existing = {
                row[0]: row[1:]
                for row in connection.execute(
                    """
                    SELECT path, uuid, size, mtime_ns, metadata_key, turns_indexed
                    FROM search_documents
                    """
                )
            }
            current_paths = {str(meta.path) for meta in metas}
            stale_paths = set(existing) - current_paths
            if stale_paths:
                with connection:
                    for stale_path in stale_paths:
                        connection.execute(
                            "DELETE FROM search_documents_fts WHERE path = ?",
                            (stale_path,),
                        )
                        connection.execute(
                            "DELETE FROM search_turns_fts WHERE path = ?",
                            (stale_path,),
                        )
                        connection.execute(
                            "DELETE FROM search_documents WHERE path = ?",
                            (stale_path,),
                        )
                stats = replace(stats, removed=len(stale_paths))

            for meta in metas:
                if should_cancel and should_cancel():
                    break

                path_key = str(meta.path)
                metadata = self._metadata(meta)
                try:
                    size_bytes, mtime_ns = conversation_signature(meta.path)
                    signature = (
                        meta.uuid,
                        size_bytes,
                        mtime_ns,
                        metadata,
                    )
                    existing_row = existing.get(path_key)
                    if (
                        existing_row
                        and existing_row[:4] == signature
                        and existing_row[4]
                    ):
                        stats = replace(
                            stats,
                            checked=stats.checked + 1,
                            unchanged=stats.unchanged + 1,
                        )
                    else:
                        turns = parse_conversation(meta.path)
                        content = "\n".join(turn.text for turn in turns if turn.text)
                        with connection:
                            connection.execute(
                                "DELETE FROM search_documents_fts WHERE path = ?",
                                (path_key,),
                            )
                            connection.execute(
                                "DELETE FROM search_turns_fts WHERE path = ?",
                                (path_key,),
                            )
                            connection.execute(
                                "INSERT INTO search_documents_fts "
                                "(uuid, path, metadata, content) VALUES (?, ?, ?, ?)",
                                (meta.uuid, path_key, metadata, content),
                            )
                            connection.executemany(
                                """
                                INSERT INTO search_turns_fts
                                    (uuid, path, turn_index, role, timestamp, metadata, content)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                                """,
                                [
                                    (
                                        meta.uuid,
                                        path_key,
                                        turn_index,
                                        turn.role,
                                        meta.timestamp,
                                        metadata,
                                        turn.text,
                                    )
                                    for turn_index, turn in enumerate(turns)
                                    if turn.text
                                ],
                            )
                            connection.execute(
                                """
                                INSERT INTO search_documents
                                    (path, uuid, size, mtime_ns, metadata_key, turns_indexed)
                                VALUES (?, ?, ?, ?, ?, 1)
                                ON CONFLICT(path) DO UPDATE SET
                                    uuid = excluded.uuid,
                                    size = excluded.size,
                                    mtime_ns = excluded.mtime_ns,
                                    metadata_key = excluded.metadata_key,
                                    turns_indexed = 1
                                """,
                                (
                                    path_key,
                                    meta.uuid,
                                    size_bytes,
                                    mtime_ns,
                                    metadata,
                                ),
                            )
                        stats = replace(
                            stats,
                            checked=stats.checked + 1,
                            indexed=stats.indexed + 1,
                        )
                except (OSError, sqlite3.Error, ValueError):
                    stats = replace(
                        stats,
                        checked=stats.checked + 1,
                        failed=stats.failed + 1,
                    )

                if on_progress:
                    on_progress(stats)
            return stats
        finally:
            connection.close()

    @staticmethod
    def _term_expression(term: str) -> str:
        tokens = re.findall(r"\w+", term, flags=re.UNICODE)
        if not tokens:
            return ""
        escaped = [token.replace('"', '""') for token in tokens]
        if len(escaped) == 1:
            return f'"{escaped[0]}"*'
        return f'"{" ".join(escaped)}"'

    @classmethod
    def _term_expressions(cls, terms: list[str]) -> list[str]:
        """Let each single word also match when joined to its neighbour.

        People write product names both ways, so "touch designer" has to find
        a transcript that only ever wrote "touchdesigner".
        """
        words = [
            tokens[0] if len(tokens := re.findall(r"\w+", term, flags=re.UNICODE)) == 1 else None
            for term in terms
        ]
        expressions = []
        for i, term in enumerate(terms):
            expression = cls._term_expression(term)
            if not expression:
                continue
            word = words[i]
            if word is not None:
                joined = [
                    f'"{left + right}"*'
                    for left, right in (
                        (words[i - 1] if i > 0 else None, word),
                        (word, words[i + 1] if i + 1 < len(words) else None),
                    )
                    if left is not None and right is not None
                ]
                if joined:
                    expression = "(" + " OR ".join([expression, *joined]) + ")"
            expressions.append(expression)
        return expressions

    @classmethod
    def _match_expression(cls, query: str) -> str:
        return " AND ".join(cls._term_expressions(parse_search_terms(query)))

    def search(
        self,
        query: str,
        limit: int = 5000,
        snippet_limit: int = 20,
    ) -> dict[str, str]:
        expression = self._match_expression(query)
        if not expression or not self.path.exists():
            return {}

        connection = sqlite3.connect(
            f"file:{self.path}?mode=ro",
            uri=True,
            timeout=0.1,
        )
        try:
            matches = {
                uuid: ""
                for (uuid,) in connection.execute(
                    """
                    SELECT uuid
                    FROM search_documents_fts
                    WHERE search_documents_fts MATCH ?
                    LIMIT ?
                    """,
                    (expression, limit),
                )
            }
            if not matches or snippet_limit <= 0:
                return matches

            snippet_rows = connection.execute(
                """
                SELECT uuid,
                       snippet(search_documents_fts, 3, '', '', ' … ', 18)
                FROM search_documents_fts
                WHERE content MATCH ?
                LIMIT ?
                """,
                (expression, min(limit, snippet_limit)),
            )
            for uuid, snippet in snippet_rows:
                matches[uuid] = snippet or ""
            return matches
        except sqlite3.Error:
            return {}
        finally:
            connection.close()

    def search_hits(
        self,
        query: str,
        conversations: Iterable[ConversationMeta],
        limit: int = 50,
    ) -> list[SearchHit]:
        """Return ranked exact-turn hits without reparsing transcripts.

        AND terms may be split across turns in one conversation. Candidate
        conversations are selected from the aggregate index, then matching
        turns are ranked by exact phrase, term coverage, frequency, and FTS
        relevance.
        """
        terms = parse_search_terms(query)
        all_expression = self._match_expression(query)
        any_expression = " OR ".join(self._term_expressions(terms))
        if not terms or not all_expression or not any_expression or not self.path.exists():
            return []

        metas_by_path = {str(meta.path): meta for meta in conversations}
        if not metas_by_path:
            return []
        candidate_limit = min(max(limit * 100, 5000), 20_000)
        connection = sqlite3.connect(
            f"file:{self.path}?mode=ro",
            uri=True,
            timeout=1,
        )
        try:
            rows = connection.execute(
                """
                SELECT uuid,
                       path,
                       turn_index,
                       role,
                       timestamp,
                       content,
                       bm25(search_turns_fts)
                FROM search_turns_fts
                WHERE search_turns_fts MATCH ?
                  AND path IN (
                      SELECT path
                      FROM search_documents_fts
                      WHERE content MATCH ?
                  )
                ORDER BY bm25(search_turns_fts)
                LIMIT ?
                """,
                (any_expression, all_expression, candidate_limit),
            ).fetchall()
        except sqlite3.Error:
            return []
        finally:
            connection.close()

        phrase = " ".join(terms)
        joined_phrase = "".join(terms) if len(terms) > 1 else ""
        scored: list[tuple[tuple[int, int, int, float, str], SearchHit]] = []
        for _uuid, path_key, turn_index, role, timestamp, content, fts_rank in rows:
            meta = metas_by_path.get(path_key)
            if meta is None:
                continue
            content = str(content or "")
            folded = content.casefold()
            matched_terms = [term for term in terms if term in folded]
            if not matched_terms:
                continue
            exact_phrase = bool(phrase and phrase in folded) or bool(
                joined_phrase and joined_phrase in folded
            )
            occurrences = sum(folded.count(term) for term in matched_terms)
            score = (
                int(exact_phrase),
                len(matched_terms),
                occurrences,
                -float(fts_rank or 0),
                str(timestamp or meta.timestamp or ""),
            )
            scored.append(
                (
                    score,
                    SearchHit(
                        meta=meta,
                        turn_index=int(turn_index),
                        role=str(role or ""),
                        snippet=_search_snippet(content, matched_terms),
                    ),
                )
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        return [hit for _, hit in scored[:limit]]
