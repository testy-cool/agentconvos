import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import agentconvos.search_index as search_index_module
from agentconvos.parser import ConversationMeta, Turn, conversation_signature


def _meta(path: Path, uuid: str = "session-one") -> ConversationMeta:
    return ConversationMeta(
        path=path,
        uuid=uuid,
        slug="fix-auth-middleware",
        timestamp="2026-07-18T10:30:00",
        cwd=str(path.parent),
        preview="Investigate request IDs",
        source="codex",
        git_branch="fix/request-id",
    )


class DefaultIndexPathTests(unittest.TestCase):
    def test_default_path_is_resolved_when_the_index_is_built(self):
        """Binding the default at import time makes the real index impossible
        to redirect, so anything constructing an index with no argument writes
        to the developer's own archive."""
        with tempfile.TemporaryDirectory() as tmp:
            redirected = Path(tmp) / "search-index.sqlite3"
            with patch.object(search_index_module, "DEFAULT_INDEX_PATH", redirected):
                index = search_index_module.ConversationSearchIndex()

            self.assertEqual(index.path, redirected)


class ConversationSearchIndexTests(unittest.TestCase):
    def test_search_hits_rank_exact_turns_and_keep_cross_turn_and_semantics(self):
        from agentconvos.search_index import ConversationSearchIndex

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_path = root / "split.jsonl"
            exact_path = root / "exact.jsonl"
            split_path.write_text("split", encoding="utf-8")
            exact_path.write_text("exact", encoding="utf-8")
            split = _meta(split_path, "split-session")
            exact = _meta(exact_path, "exact-session")
            turns = {
                split_path: [
                    Turn("user", "Investigate the auth middleware"),
                    Turn("assistant", "The request IDs disappear in the retry path"),
                ],
                exact_path: [
                    Turn("user", "The auth request IDs fail together"),
                    Turn("assistant", "I reproduced the issue"),
                ],
            }
            parse_calls = []

            def parse_conversation(path: Path):
                parse_calls.append(path)
                return turns[path]

            index = ConversationSearchIndex(root / "search.sqlite3")
            index.sync([split, exact], parse_conversation=parse_conversation)
            hits = index.search_hits("auth request", [split, exact])

        self.assertEqual(parse_calls, [split_path, exact_path])
        self.assertEqual(hits[0].meta.uuid, "exact-session")
        self.assertEqual(hits[0].turn_index, 0)
        self.assertEqual(hits[0].role, "user")
        self.assertIn("auth request", hits[0].snippet.casefold())
        self.assertIn("split-session", {hit.meta.uuid for hit in hits})

    def test_two_words_find_a_conversation_that_wrote_them_as_one(self):
        """People type product names both ways. Searching "touch designer"
        must find a session that only ever wrote "touchdesigner"."""
        from agentconvos.search_index import ConversationSearchIndex

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            joined_path = root / "joined.jsonl"
            stray_path = root / "stray.jsonl"
            joined_path.write_text("joined", encoding="utf-8")
            stray_path.write_text("stray", encoding="utf-8")
            joined = _meta(joined_path, "joined-session")
            stray = _meta(stray_path, "stray-session")
            turns = {
                joined_path: [
                    Turn("user", "can you check if i have touchdesigner installed?"),
                    Turn("assistant", "I found the installer in Downloads."),
                ],
                stray_path: [
                    Turn("assistant", "Do not touch the designer handoff, touching it twice breaks it."),
                ],
            }

            index = ConversationSearchIndex(root / "search.sqlite3")
            index.sync([joined, stray], parse_conversation=turns.__getitem__)
            documents = index.search("touch designer")
            hits = index.search_hits("touch designer", [joined, stray])

        self.assertIn("joined-session", documents)
        self.assertEqual((hits[0].meta.uuid, hits[0].turn_index), ("joined-session", 0))

    def test_sync_backfills_turn_rows_for_an_existing_conversation_index(self):
        from agentconvos.search_index import ConversationSearchIndex

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcript = root / "session.jsonl"
            transcript.write_text("existing", encoding="utf-8")
            meta = _meta(transcript)
            index_path = root / "search.sqlite3"
            size_bytes, mtime_ns = conversation_signature(transcript)
            metadata = ConversationSearchIndex._metadata(meta)

            connection = sqlite3.connect(index_path)
            try:
                connection.executescript(
                    """
                    CREATE TABLE search_documents (
                        path TEXT PRIMARY KEY,
                        uuid TEXT NOT NULL,
                        size INTEGER NOT NULL,
                        mtime_ns INTEGER NOT NULL,
                        metadata_key TEXT NOT NULL
                    );
                    CREATE VIRTUAL TABLE search_documents_fts USING fts5(
                        uuid UNINDEXED,
                        path UNINDEXED,
                        metadata,
                        content,
                        tokenize='unicode61 remove_diacritics 2'
                    );
                    """
                )
                connection.execute(
                    "INSERT INTO search_documents VALUES (?, ?, ?, ?, ?)",
                    (str(transcript), meta.uuid, size_bytes, mtime_ns, metadata),
                )
                connection.execute(
                    "INSERT INTO search_documents_fts VALUES (?, ?, ?, ?)",
                    (meta.uuid, str(transcript), metadata, "legacy searchable text"),
                )
                connection.commit()
            finally:
                connection.close()

            parse_calls = []

            def parse_conversation(path: Path):
                parse_calls.append(path)
                return [Turn("assistant", "Backfilled exact turn text")]

            index = ConversationSearchIndex(index_path)
            result = index.sync([meta], parse_conversation=parse_conversation)
            hits = index.search_hits("backfilled exact", [meta])

        self.assertEqual(result.indexed, 1)
        self.assertEqual(parse_calls, [transcript])
        self.assertEqual(hits[0].turn_index, 0)
        self.assertEqual(hits[0].role, "assistant")

    def test_sync_reuses_unchanged_rows_and_returns_prefix_matches(self):
        from agentconvos.search_index import ConversationSearchIndex

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcript = root / "session.jsonl"
            transcript.write_text("first version", encoding="utf-8")
            meta = _meta(transcript)
            parse_calls = []

            def parse_conversation(path: Path):
                parse_calls.append(path)
                return [Turn("user", "The auth middleware drops request IDs")]

            index = ConversationSearchIndex(root / "search.sqlite3")
            first = index.sync([meta], parse_conversation=parse_conversation)
            second = index.sync([meta], parse_conversation=parse_conversation)
            matches = index.search("auth midd")

        self.assertEqual(first.indexed, 1)
        self.assertEqual(second.unchanged, 1)
        self.assertEqual(parse_calls, [transcript])
        self.assertIn(meta.uuid, matches)
        self.assertIn("auth middleware", matches[meta.uuid].casefold())

    def test_sync_removes_conversations_missing_from_the_latest_scan(self):
        from agentconvos.search_index import ConversationSearchIndex

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_path = root / "first.jsonl"
            second_path = root / "second.jsonl"
            first_path.write_text("first", encoding="utf-8")
            second_path.write_text("second", encoding="utf-8")
            first = _meta(first_path, "first-session")
            second = _meta(second_path, "second-session")
            index = ConversationSearchIndex(root / "search.sqlite3")

            def parse_conversation(path: Path):
                return [Turn("user", f"unique text from {path.stem}")]

            index.sync(
                [first, second],
                parse_conversation=parse_conversation,
            )
            result = index.sync(
                [second],
                parse_conversation=parse_conversation,
            )

            first_matches = index.search("unique first")
            second_matches = index.search("unique second")

        self.assertEqual(result.removed, 1)
        self.assertNotIn(first.uuid, first_matches)
        self.assertIn(second.uuid, second_matches)

    def test_search_keeps_all_matches_without_rendering_every_snippet(self):
        from agentconvos.search_index import ConversationSearchIndex

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_path = root / "first.jsonl"
            second_path = root / "second.jsonl"
            first_path.write_text("first", encoding="utf-8")
            second_path.write_text("second", encoding="utf-8")
            first = _meta(first_path, "first-session")
            second = _meta(second_path, "second-session")
            index = ConversationSearchIndex(root / "search.sqlite3")
            index.sync(
                [first, second],
                parse_conversation=lambda path: [
                    Turn("user", f"shared searchable text from {path.stem}")
                ],
            )

            matches = index.search("shared searchable", snippet_limit=1)

        self.assertEqual(set(matches), {first.uuid, second.uuid})
        self.assertEqual(sum(bool(snippet) for snippet in matches.values()), 1)


if __name__ == "__main__":
    unittest.main()
