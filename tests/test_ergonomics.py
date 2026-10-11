import contextlib
import io
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from textual.widgets import Input, Static, Tree

import agentconvos.app as app_module
import agentconvos.parser as parser_module
from agentconvos.parser import ConversationMeta, Turn
from agentconvos.scanner import Project


def _meta(path: Path, uuid: str, timestamp: str, cwd: Path) -> ConversationMeta:
    return ConversationMeta(
        path=path,
        uuid=uuid,
        slug=f"session-{uuid[:4]}",
        timestamp=timestamp,
        cwd=str(cwd),
        preview="search preview",
        source="codex",
    )


class SearchErgonomicsTests(unittest.TestCase):
    def test_navigation_timestamps_stay_compact(self):
        formatter = getattr(app_module, "_fmt_nav_ts", None)
        self.assertIsNotNone(formatter, "compact navigation timestamps are missing")
        self.assertEqual(formatter("2026-07-18T12:23:45"), "07-18 12:23")

    def test_query_parser_supports_words_and_quoted_phrases(self):
        parse_terms = getattr(parser_module, "parse_search_terms", None)
        self.assertIsNotNone(parse_terms, "search query parser is missing")

        self.assertEqual(
            parse_terms('auth "request id" AUTH'),
            ["auth", "request id"],
        )

    def test_search_matches_terms_across_turns_and_ranks_exact_phrase_first(self):
        root = Path("/tmp/search-ranking")
        split_path = root / "split.jsonl"
        exact_path = root / "exact.jsonl"
        metas = {
            split_path: _meta(split_path, "split-session", "2026-07-15T12:00:00", root),
            exact_path: _meta(exact_path, "exact-session", "2026-07-14T12:00:00", root),
        }
        turns = {
            split_path: [
                Turn("user", "The authentication flow is failing."),
                Turn("assistant", "I will inspect the middleware next."),
            ],
            exact_path: [
                Turn("user", "Please inspect the auth middleware implementation."),
            ],
        }

        with (
            patch("agentconvos.parser.get_meta", side_effect=lambda path: metas[path]),
            patch("agentconvos.parser.parse_jsonl", side_effect=lambda path: turns[path]),
        ):
            hits = parser_module.search_conversations(
                [split_path, exact_path],
                "auth middleware",
            )

        self.assertGreaterEqual(len(hits), 3)
        self.assertEqual(hits[0].meta.uuid, "exact-session")
        self.assertIn("auth middleware", hits[0].snippet.lower())
        self.assertEqual(
            {hit.meta.uuid for hit in hits},
            {"split-session", "exact-session"},
        )

    def test_rich_highlight_marks_every_query_term_case_insensitively(self):
        highlight = getattr(app_module, "_highlight_matches", None)
        self.assertIsNotNone(highlight, "Rich search highlighting is missing")

        rendered = highlight("Auth middleware and AUTH logs", ["auth", "middleware"])

        self.assertEqual(rendered.plain, "Auth middleware and AUTH logs")
        highlighted_text = [rendered.plain[span.start:span.end] for span in rendered.spans]
        self.assertEqual(highlighted_text, ["Auth", "middleware", "AUTH"])

    def test_match_excerpt_centers_the_visible_text_on_the_hit(self):
        excerpt = getattr(app_module, "_matching_excerpt", None)
        self.assertIsNotNone(excerpt, "match-centered excerpts are missing")

        text = "prefix " * 30 + "AUTH MIDDLEWARE failed here" + " suffix" * 30
        result = excerpt(text, ["auth", "middleware"], width=70)

        self.assertLessEqual(len(result), 76)
        self.assertTrue(result.startswith("…"))
        self.assertTrue(result.endswith("…"))
        self.assertIn("AUTH MIDDLEWARE", result)

    def test_elide_measures_terminal_cells_not_characters(self):
        """A CJK character occupies two cells, so counting characters lets a
        row overflow the sidebar it was measured against."""
        elide = getattr(app_module, "_elide", None)
        self.assertIsNotNone(elide, "row eliding helper is missing")
        cells = getattr(app_module, "_cell_len", None)
        self.assertIsNotNone(cells, "terminal width helper is missing")

        self.assertEqual(cells("abc"), 3)
        self.assertEqual(cells("界界"), 4)
        self.assertLessEqual(cells(elide("界" * 20, 16)), 16)
        self.assertLessEqual(cells(elide("a" * 40, 16)), 16)
        self.assertEqual(elide("short", 16), "short")

    def test_markdown_highlight_joins_touching_matches(self):
        """Two matches that touch would emit **** between them, which markdown
        renders as literal asterisks instead of bold text."""
        highlight = getattr(app_module, "_highlight_markdown", None)

        self.assertEqual(
            highlight("X-RateLimit-Reset", ["rate", "limit"]),
            "X-**RateLimit**-Reset",
        )

    def test_markdown_highlight_makes_matches_visible_in_preview(self):
        highlight = getattr(app_module, "_highlight_markdown", None)
        self.assertIsNotNone(highlight, "preview search highlighting is missing")

        self.assertEqual(
            highlight("Auth middleware failed", ["auth", "middleware"]),
            "**Auth** **middleware** failed",
        )


class FuzzyFinderTests(unittest.TestCase):
    def test_large_transcript_preview_uses_cached_context_without_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "huge.jsonl"
            path.write_text("large transcript", encoding="utf-8")
            meta = ConversationMeta(
                path=path,
                uuid="huge-session",
                slug="huge-auth-session",
                timestamp="2026-07-18T10:30:00",
                cwd=str(root),
                preview="First prompt about auth middleware",
                source="codex",
            )
            stream = io.StringIO()

            with (
                patch("agentconvos.parser.get_meta", return_value=meta),
                patch(
                    "agentconvos.summarize.load_summaries",
                    return_value={meta.uuid: "Cached authentication summary"},
                ),
                patch(
                    "agentconvos.app.parse_jsonl",
                    side_effect=AssertionError("large preview must stay lazy"),
                ),
                contextlib.redirect_stdout(stream),
            ):
                app_module._print_fuzzy_preview(path, max_parse_bytes=1)

        output = stream.getvalue()
        self.assertIn("Cached authentication summary", output)
        self.assertIn("First prompt about auth middleware", output)
        self.assertIn("Large transcript", output)

    def test_fuzzy_picker_searches_cached_metadata_and_previews_lazily(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "session.jsonl"
            path.write_text("", encoding="utf-8")
            meta = ConversationMeta(
                path=path,
                uuid="019f1234-abcd-7000-8000-123456789abc",
                slug="fix-auth-middleware",
                timestamp="2026-07-18T10:30:00",
                cwd=str(root / "checkout"),
                preview="Investigate dropped request IDs",
                source="codex",
                git_branch="fix/request-id",
            )
            captured = {}

            def fake_runner(command, **kwargs):
                captured["command"] = command
                captured["input"] = kwargs["input"]
                selected_row = kwargs["input"].splitlines()[0]
                return SimpleNamespace(returncode=0, stdout=selected_row + "\n")

            with (
                patch(
                    "agentconvos.summarize.load_summaries",
                    return_value={meta.uuid: "Authentication debugging notes"},
                ),
                patch(
                    "agentconvos.app.parse_jsonl",
                    side_effect=AssertionError("picker must not preload transcripts"),
                ),
            ):
                selected = app_module._pick_conversation_fuzzy(
                    [meta],
                    initial_query="auth reqid",
                    fzf_path="/usr/bin/fzf",
                    runner=fake_runner,
                )

        self.assertEqual(selected, meta)
        self.assertIn("fix-auth-middleware", captured["input"])
        self.assertIn("Investigate dropped request IDs", captured["input"])
        self.assertIn("Authentication debugging notes", captured["input"])
        self.assertIn("fix/request-id", captured["input"])
        self.assertIn("--nth=2..7", captured["command"])
        self.assertIn("--with-nth=2..6", captured["command"])
        self.assertIn("--query=auth reqid", captured["command"])
        preview_arg = next(
            arg for arg in captured["command"] if arg.startswith("--preview=")
        )
        self.assertIn("agentconvos.app --peek {8}", preview_arg)

    def test_find_flag_selects_without_starting_the_textual_tui(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "session.jsonl"
            path.write_text("", encoding="utf-8")
            meta = _meta(path, "selected-session", "2026-07-18T10:30:00", root)
            project = Project("tmp", str(root), [meta])
            stream = io.StringIO()
            old_argv = sys.argv
            sys.argv = ["agentconvos", "--find", "auth reqid"]
            try:
                with (
                    patch("agentconvos.scanner.scan_projects", return_value=[project]),
                    patch(
                        "agentconvos.app._pick_conversation_fuzzy",
                        return_value=meta,
                    ) as picker,
                    patch(
                        "agentconvos.app.ConvoExplorer.run",
                        side_effect=AssertionError("--find must not start Textual"),
                    ),
                    contextlib.redirect_stdout(stream),
                    contextlib.redirect_stderr(stream),
                ):
                    app_module.main()
            finally:
                sys.argv = old_argv

        picker.assert_called_once_with([meta], initial_query="auth reqid")
        output = stream.getvalue()
        self.assertIn(meta.uuid, output)
        self.assertIn(f"agentconvos --show {meta.uuid}", output)
        self.assertIn(f"agentconvos --resume {meta.uuid}", output)

    def test_direct_preview_path_accepts_non_jsonl_conversation_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agy-conversation.db"
            path.write_bytes(b"")

            with patch(
                "agentconvos.scanner.resolve_ids",
                side_effect=AssertionError("direct paths must not trigger a full rescan"),
            ):
                resolved = app_module._resolve_args([str(path)])

        self.assertEqual(resolved, [path])


class ResumeErgonomicsTests(unittest.TestCase):
    def test_bare_resume_accepts_latest_pi_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            session_path = cwd / "pi-session.jsonl"
            session_path.write_text("", encoding="utf-8")
            meta = ConversationMeta(
                path=session_path,
                uuid="pi-session",
                slug="session-pi",
                timestamp="2026-07-15T10:00:00",
                cwd=str(cwd),
                preview="Pi session",
                source="pi",
            )
            project = Project("pi:tmp", f"[pi] {cwd}", [meta])

            old_argv = sys.argv
            old_cwd = os.getcwd()
            sys.argv = ["agentconvos", "--resume", "--dry-run"]
            stream = io.StringIO()
            try:
                os.chdir(cwd)
                with (
                    patch("agentconvos.scanner.scan_projects", return_value=[project]),
                    contextlib.redirect_stdout(stream),
                    contextlib.redirect_stderr(stream),
                ):
                    app_module.main()
            finally:
                sys.argv = old_argv
                os.chdir(old_cwd)

        output = stream.getvalue()
        self.assertIn("pi --session pi-session", output)

    def test_bare_resume_selects_latest_session_for_current_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            older_path = cwd / "older.jsonl"
            newer_path = cwd / "newer.jsonl"
            older_path.write_text("", encoding="utf-8")
            newer_path.write_text("", encoding="utf-8")
            older = _meta(older_path, "older-session", "2026-07-14T10:00:00", cwd)
            newer = _meta(newer_path, "newer-session", "2026-07-15T10:00:00", cwd)
            project = Project("tmp", str(cwd), [older, newer])

            old_argv = sys.argv
            old_cwd = os.getcwd()
            sys.argv = ["agentconvos", "--resume", "--dry-run"]
            stream = io.StringIO()
            try:
                os.chdir(cwd)
                with (
                    patch("agentconvos.scanner.scan_projects", return_value=[project]),
                    contextlib.redirect_stdout(stream),
                    contextlib.redirect_stderr(stream),
                ):
                    try:
                        app_module.main()
                    except SystemExit as exc:
                        self.fail(f"bare --resume should be accepted, exited with {exc.code}")
            finally:
                sys.argv = old_argv
                os.chdir(old_cwd)

        output = stream.getvalue()
        self.assertIn("Resuming: session-newe", output)
        self.assertIn("newer-session", output)
        self.assertNotIn("older-session", output)

    def test_resume_confirmation_includes_agent_cwd_full_id_and_command(self):
        describe = getattr(app_module, "_resume_description", None)
        self.assertIsNotNone(describe, "resume confirmation details are missing")

        meta = ConversationMeta(
            path=Path("/tmp/project/session.jsonl"),
            uuid="019f1234-abcd-7000-8000-123456789abc",
            slug="useful-session",
            timestamp="2026-07-15T14:30:00",
            cwd="/tmp/project",
            preview="",
            source="codex",
        )
        description = describe(meta)

        self.assertIn("Codex", description)
        self.assertIn("/tmp/project", description)
        self.assertIn(meta.uuid, description)
        self.assertIn("codex resume", description)


class TuiErgonomicsTests(unittest.IsolatedAsyncioTestCase):
    async def test_status_rail_sits_above_the_footer_on_small_terminals(self):
        class ReadyIndex:
            def search(self, _query):
                return {}

            def sync(self, conversations, **_kwargs):
                conversations = list(conversations)
                return SimpleNamespace(
                    total=len(conversations),
                    checked=len(conversations),
                    indexed=0,
                    unchanged=len(conversations),
                    removed=0,
                    failed=0,
                )

        with patch("agentconvos.app.scan_projects", return_value=[]):
            tui = app_module.ConvoExplorer(search_index=ReadyIndex())
            async with tui.run_test(size=(80, 24)) as pilot:
                await pilot.pause()

                status_rail = tui.query_one("#status-rail")
                footer = tui.query_one("Footer")
                left_title = tui.query_one("#left-title")
                search = tui.query_one("#filter-input", Input)
                self.assertLess(status_rail.region.bottom, footer.region.bottom)
                self.assertEqual(status_rail.region.bottom, footer.region.y)
                self.assertLessEqual(left_title.region.bottom, search.region.y)

    async def test_projects_render_before_background_index_completes(self):
        class BlockingIndex:
            def __init__(self):
                self.started = threading.Event()
                self.release = threading.Event()

            def search(self, _query):
                return {}

            def sync(self, conversations, **_kwargs):
                conversations = list(conversations)
                self.started.set()
                self.release.wait(timeout=3)
                return SimpleNamespace(
                    total=len(conversations),
                    checked=len(conversations),
                    indexed=len(conversations),
                    unchanged=0,
                    removed=0,
                    failed=0,
                )

        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            path = cwd / "session.jsonl"
            path.write_text("", encoding="utf-8")
            meta = _meta(path, "instant-session", "2026-07-18T10:30:00", cwd)
            project = Project("tmp", str(cwd), [meta])
            index = BlockingIndex()

            with (
                patch("agentconvos.app.scan_projects", return_value=[project]),
                patch("agentconvos.app.parse_jsonl") as parse_transcript,
            ):
                tui = app_module.ConvoExplorer(search_index=index)
                async with tui.run_test(size=(110, 36)) as pilot:
                    try:
                        for _ in range(20):
                            await pilot.pause()
                            result_nodes = [
                                node
                                for node in tui._walk_tree_nodes()
                                if node.data and node.data.kind == "convo"
                            ]
                            if result_nodes and index.started.is_set():
                                break

                        self.assertEqual(len(result_nodes), 1)
                        self.assertTrue(index.started.is_set())
                        self.assertFalse(index.release.is_set())
                        parse_transcript.assert_not_called()
                        index_label = str(
                            tui.query_one("#index-status", Static).render()
                        )
                        self.assertIn("INDEXING", index_label)
                    finally:
                        index.release.set()

    async def test_search_uses_persisted_snippets_without_preloading_transcripts(self):
        class ReadyIndex:
            def __init__(self, uuid):
                self.uuid = uuid

            def search(self, query):
                if query == "auth middleware":
                    return {self.uuid: "The auth middleware drops request IDs"}
                return {}

            def sync(self, conversations, **_kwargs):
                conversations = list(conversations)
                return SimpleNamespace(
                    total=len(conversations),
                    checked=len(conversations),
                    indexed=0,
                    unchanged=len(conversations),
                    removed=0,
                    failed=0,
                )

        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            path = cwd / "session.jsonl"
            path.write_text("", encoding="utf-8")
            meta = _meta(path, "indexed-session", "2026-07-18T10:30:00", cwd)
            meta.preview = "Unrelated first prompt"
            project = Project("tmp", str(cwd), [meta])
            index = ReadyIndex(meta.uuid)

            with (
                patch("agentconvos.app.scan_projects", return_value=[project]),
                patch("agentconvos.app.parse_jsonl") as parse_transcript,
            ):
                tui = app_module.ConvoExplorer(search_index=index)
                async with tui.run_test(size=(180, 36)) as pilot:
                    await pilot.pause()
                    search = tui.query_one("#filter-input", Input)
                    search.value = "auth middleware"
                    for _ in range(30):
                        await pilot.pause(0.03)
                        result_nodes = [
                            node
                            for node in tui._walk_tree_nodes()
                            if node.data and node.data.kind == "convo"
                        ]
                        if result_nodes and "drops request IDs" in result_nodes[0].label.plain:
                            break

                    result_nodes = [
                        node
                        for node in tui._walk_tree_nodes()
                        if node.data and node.data.kind == "convo"
                    ]
                    self.assertEqual(len(result_nodes), 1)
                    self.assertIn("drops request IDs", result_nodes[0].label.plain)
                    parse_transcript.assert_not_called()

    async def test_search_lists_the_best_match_first_not_the_newest(self):
        """A newer session that only says "touch" in one turn and "designer"
        in another must not bury an older one about TouchDesigner."""
        from agentconvos.search_index import ConversationSearchIndex

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weak_dir = root / "busy-project"
            strong_dir = root / "quiet-project"
            weak_dir.mkdir()
            strong_dir.mkdir()
            weak_path = weak_dir / "weak.jsonl"
            strong_path = strong_dir / "strong.jsonl"
            weak_path.write_text("weak", encoding="utf-8")
            strong_path.write_text("strong", encoding="utf-8")
            weak = _meta(weak_path, "weak-session", "2026-10-10T22:23:00", weak_dir)
            strong = _meta(strong_path, "strong-session", "2026-10-09T14:04:00", strong_dir)
            turns = {
                weak_path: [
                    Turn("user", "Do not touch the pipeline."),
                    Turn("assistant", "The designer handoff is ready."),
                ],
                strong_path: [
                    Turn("user", "can you check if i have touchdesigner installed?"),
                ],
            }
            index = ConversationSearchIndex(root / "search.sqlite3")
            index.sync([weak, strong], parse_conversation=turns.__getitem__)
            projects = [
                Project("busy", str(weak_dir), [weak]),
                Project("quiet", str(strong_dir), [strong]),
            ]

            with patch("agentconvos.app.scan_projects", return_value=projects):
                tui = app_module.ConvoExplorer(search_index=index)
                async with tui.run_test(size=(180, 36)) as pilot:
                    await pilot.pause()
                    tui.query_one("#filter-input", Input).value = "touch designer"
                    for _ in range(30):
                        await pilot.pause(0.03)
                        result_nodes = [
                            node
                            for node in tui._walk_tree_nodes()
                            if node.data and node.data.kind == "convo"
                        ]
                        # History already lists both; wait for the search snippet.
                        if any("touchdesigner" in node.label.plain for node in result_nodes):
                            break

                    self.assertEqual(
                        [node.data.meta.uuid for node in result_nodes],
                        ["strong-session", "weak-session"],
                    )
                    self.assertIn("touchdesigner", result_nodes[0].label.plain)

    async def test_filter_input_debounces_intermediate_queries(self):
        class RecordingIndex:
            def __init__(self):
                self.queries = []

            def search(self, query):
                self.queries.append(query)
                return {}

            def sync(self, conversations, **_kwargs):
                conversations = list(conversations)
                return SimpleNamespace(
                    total=len(conversations),
                    checked=len(conversations),
                    indexed=0,
                    unchanged=len(conversations),
                    removed=0,
                    failed=0,
                )

        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            path = cwd / "session.jsonl"
            path.write_text("", encoding="utf-8")
            meta = _meta(path, "debounce-session", "2026-07-18T10:30:00", cwd)
            project = Project("tmp", str(cwd), [meta])
            index = RecordingIndex()

            with patch("agentconvos.app.scan_projects", return_value=[project]):
                tui = app_module.ConvoExplorer(search_index=index)
                async with tui.run_test(size=(110, 36)) as pilot:
                    for _ in range(20):
                        await pilot.pause()
                        if "INDEX READY" in str(
                            tui.query_one("#index-status", Static).render()
                        ):
                            break

                    index.queries.clear()
                    search = tui.query_one("#filter-input", Input)
                    search.value = "a"
                    await pilot.pause(0.03)
                    search.value = "ag"
                    await pilot.pause(0.03)
                    search.value = "agentconvos"
                    await pilot.pause(0.03)

                    self.assertEqual(index.queries, [])

                    await pilot.pause(0.25)
                    for _ in range(20):
                        await pilot.pause()
                        if index.queries:
                            break

                    self.assertEqual(index.queries, ["agentconvos"])

    async def test_stale_filter_result_is_discarded_before_tree_mutation(self):
        with patch("agentconvos.app.scan_projects", return_value=[]):
            tui = app_module.ConvoExplorer()
            async with tui.run_test(size=(100, 32)):
                tui._filter_query = "agentconvos"
                tui._filter_generation = 2
                stale_result = SimpleNamespace(
                    projects=[],
                    indexed_matches={},
                    filtered_count=0,
                )

                with (
                    patch.object(tui, "_populate_tree") as populate_tree,
                    patch.object(tui, "_show_search_summary") as show_summary,
                ):
                    tui._filter_finished("a", 1, stale_result)

                populate_tree.assert_not_called()
                show_summary.assert_not_called()

    async def test_preview_header_shortens_the_home_directory(self):
        """The tree groups paths under ~, so the preview should agree instead
        of printing the whole absolute path."""
        shorten = getattr(app_module, "_short_path", None)
        self.assertIsNotNone(shorten, "path shortening helper is missing")
        self.assertEqual(shorten("/srv/elsewhere"), "/srv/elsewhere")

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            cwd = home / "work" / "billing-api"
            cwd.mkdir(parents=True)
            meta = _meta(cwd / "s.jsonl", "preview-home", "2026-07-18T10:30:00", cwd)
            turns = [Turn("user", "hello"), Turn("assistant", "hi")]
            captured: list[str] = []

            with patch("agentconvos.app.scan_projects", return_value=[]):
                tui = app_module.ConvoExplorer()
                async with tui.run_test(size=(100, 32)) as pilot:
                    with (
                        patch("agentconvos.app.parse_jsonl", return_value=turns),
                        patch("pathlib.Path.home", return_value=home),
                        patch.object(
                            tui, "_set_preview",
                            side_effect=lambda md, *a, **k: captured.append(md),
                        ),
                    ):
                        tui.request_preview(meta)
                        for _ in range(60):
                            await pilot.pause(0.05)
                            if captured:
                                break

            self.assertTrue(captured, "preview was never rendered")
            self.assertIn("**CWD:** ~/work/billing-api", captured[0])
            self.assertNotIn(str(cwd), captured[0])

    async def test_late_index_callbacks_after_shutdown_are_ignored(self):
        """A slow index sync can finish after the app exits; its UI callbacks
        must not crash on the already-removed status widgets."""
        with patch("agentconvos.app.scan_projects", return_value=[]):
            tui = app_module.ConvoExplorer()
            async with tui.run_test(size=(100, 32)):
                pass

            stats = SimpleNamespace(total=5, checked=5, failed=0)
            tui._set_index_progress(stats)
            tui._index_finished(stats)
            tui._index_failed("boom")

    async def test_full_transcript_is_readable_without_leaving_the_browser(self):
        """The preview shows a tail; v must open the whole conversation, and
        the header must say how much is on screen either way."""
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            meta = _meta(cwd / "long.jsonl", "long-session", "2026-08-10T09:00:00", cwd)
            turns = [
                Turn("user" if index % 2 == 0 else "assistant", f"turn number {index}")
                for index in range(40)
            ]
            captured: list[str] = []

            with patch("agentconvos.app.scan_projects", return_value=[]):
                tui = app_module.ConvoExplorer()
                async with tui.run_test(size=(110, 36)) as pilot:
                    with (
                        patch("agentconvos.app.parse_jsonl", return_value=turns),
                        patch.object(
                            tui, "_set_preview",
                            side_effect=lambda md, *a, **k: captured.append(md),
                        ),
                    ):
                        tui.current_meta = meta
                        tui.request_preview(meta)
                        for _ in range(60):
                            await pilot.pause(0.05)
                            if captured:
                                break

                        self.assertTrue(captured, "preview was never rendered")
                        self.assertIn("showing last 10 of 40 · press v for all", captured[0])
                        self.assertNotIn("turn number 0", captured[0])

                        tui.action_view_full()
                        for _ in range(60):
                            await pilot.pause(0.05)
                            if len(captured) > 1:
                                break

            self.assertGreater(len(captured), 1, "full transcript was never rendered")
            self.assertIn("turn number 0", captured[-1])
            self.assertIn("turn number 39", captured[-1])

    async def test_tree_never_scrolls_sideways_whatever_the_names_are(self):
        """Rows wider than the tree only produce a horizontal scrollbar, so
        every row kind has to fit: long paths and long first messages alike."""
        for width in (100, 124, 160):
            with tempfile.TemporaryDirectory() as tmp:
                cwd = Path(tmp) / ("deep/" * 8 + "a-very-long-project-name")
                cwd.mkdir(parents=True)
                metas = [
                    _meta(cwd / "a.jsonl", "wide-session", "2026-08-10T09:00:00", cwd),
                    _meta(cwd / "b.jsonl", "other-session", "2026-08-09T09:00:00", cwd),
                ]
                metas[0].preview = "w" * 300
                project = Project("tmp", str(cwd), metas)

                with patch("agentconvos.app.scan_projects", return_value=[project]):
                    tui = app_module.ConvoExplorer()
                    async with tui.run_test(size=(width, 36)) as pilot:
                        tree = tui.query_one("#nav-tree", Tree)
                        for _ in range(60):
                            await pilot.pause(0.05)
                            if tree.root.children:
                                break
                        tree.root.expand_all()
                        await pilot.pause(0.2)

                        rows = [
                            str(node.label)
                            for node in tui._walk_tree_nodes()
                            if node.data and node.data.kind == "convo"
                        ]
                        virtual = tree.virtual_size.width
                        viewport = tree.container_size.width

            self.assertTrue(rows, f"no rows rendered at {width} columns")
            self.assertLessEqual(
                virtual, viewport,
                f"tree scrolls sideways at {width} columns ({virtual} > {viewport})",
            )
            self.assertTrue(
                any(row.rstrip().endswith("…") for row in rows),
                f"no row was elided at {width} columns: {rows}",
            )

    async def test_preview_header_shows_branch_model_and_size(self):
        """Deciding whether to resume a session needs more than a date; the
        branch, model and token cost belong in the header."""
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            meta = _meta(cwd / "s.jsonl", "stats-session", "2026-08-10T09:05:00", cwd)
            meta.git_branch = "fix/jwt-refresh"
            stats = app_module.ConversationStats(
                model="claude-opus-4-6",
                input_tokens=41200,
                output_tokens=1850,
            )
            captured: list[str] = []

            with patch("agentconvos.app.scan_projects", return_value=[]):
                tui = app_module.ConvoExplorer()
                async with tui.run_test(size=(120, 36)) as pilot:
                    with (
                        patch("agentconvos.app.parse_jsonl", return_value=[Turn("user", "hi")]),
                        patch("agentconvos.app.get_stats", return_value=stats),
                        patch.object(
                            tui, "_set_preview",
                            side_effect=lambda md, *a, **k: captured.append(md),
                        ),
                    ):
                        tui.request_preview(meta)
                        for _ in range(60):
                            await pilot.pause(0.05)
                            if captured:
                                break

        self.assertTrue(captured, "preview was never rendered")
        header = captured[0]
        self.assertIn("2026-08-10 09:05", header)
        self.assertIn("fix/jwt-refresh", header)
        self.assertIn("claude-opus-4-6", header)
        self.assertIn("43,050", header)

    async def test_session_id_can_be_copied_with_one_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            meta = _meta(cwd / "s.jsonl", "copy-session-uuid", "2026-08-10T09:00:00", cwd)

            with patch("agentconvos.app.scan_projects", return_value=[]):
                tui = app_module.ConvoExplorer()
                async with tui.run_test(size=(110, 36)) as pilot:
                    tui.current_meta = meta
                    with patch.object(tui, "copy_to_clipboard") as copy:
                        tui.action_copy_id()
                        await pilot.pause(0.05)

        copy.assert_called_once_with("copy-session-uuid")

    async def test_project_row_previews_the_project_instead_of_going_stale(self):
        """Landing on a project should say what is in it, not leave whatever
        conversation was previewed last on screen."""
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            metas = [
                _meta(cwd / "a.jsonl", "recent-session", "2026-08-10T09:00:00", cwd),
                _meta(cwd / "b.jsonl", "older-session", "2026-06-02T09:00:00", cwd),
            ]
            project = Project("tmp", str(cwd), metas)
            captured: list[str] = []

            with patch("agentconvos.app.scan_projects", return_value=[project]):
                tui = app_module.ConvoExplorer()
                async with tui.run_test(size=(120, 36)) as pilot:
                    tree = tui.query_one("#nav-tree", Tree)
                    for _ in range(60):
                        await pilot.pause(0.05)
                        if tree.root.children:
                            break
                    tree.root.expand_all()
                    await pilot.pause(0.1)

                    node = next(
                        node
                        for node in tui._walk_tree_nodes()
                        if node.data and node.data.kind == "project"
                    )
                    with patch.object(
                        tui, "_set_preview",
                        side_effect=lambda md, *a, **k: captured.append(md),
                    ):
                        tree.select_node(node)
                        tree.action_select_cursor()
                        for _ in range(40):
                            await pilot.pause(0.05)
                            if captured:
                                break

        self.assertTrue(captured, "project row rendered no preview")
        digest = captured[-1]
        self.assertIn("2 conversations", digest)
        self.assertIn("2026-08-10", digest)

    def test_footer_leads_with_the_flagship_actions(self):
        """Resume, handoff and search must be visible in the footer instead of
        being crowded out by bulk-action keys; the rest lives behind ?."""
        shown = [
            binding.key
            for binding in app_module.ConvoExplorer.BINDINGS
            if getattr(binding, "show", True)
        ]
        self.assertEqual(shown[:6], ["slash", "v", "r", "h", "e", "a"])
        self.assertIn("question_mark", shown)

    async def test_help_overlay_opens_from_the_tree_but_not_the_search_box(self):
        with patch("agentconvos.app.scan_projects", return_value=[]):
            tui = app_module.ConvoExplorer()
            async with tui.run_test(size=(110, 36)) as pilot:
                await pilot.pause(0.1)

                # In the search box, "?" is a character, not a shortcut.
                await pilot.press("question_mark")
                self.assertEqual(tui.query_one("#filter-input", Input).value, "?")

                tui.query_one("#filter-input", Input).value = ""
                await pilot.press("tab")
                await pilot.press("question_mark")
                await pilot.pause(0.05)
                self.assertIsInstance(tui.screen, app_module.HelpScreen)

                await pilot.press("escape")
                await pilot.pause(0.05)
                self.assertNotIsInstance(tui.screen, app_module.HelpScreen)

    async def test_scanning_past_rows_does_not_parse_every_transcript(self):
        """Arrowing through results should preview where the cursor rests, not
        parse every conversation it passes over on the way."""
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            metas = [
                _meta(cwd / f"s{index}.jsonl", f"scan-{index}", f"2026-08-{10 - index:02d}T09:00:00", cwd)
                for index in range(6)
            ]
            project = Project("tmp", str(cwd), metas)
            parsed: list[str] = []

            with patch("agentconvos.app.scan_projects", return_value=[project]):
                tui = app_module.ConvoExplorer()
                async with tui.run_test(size=(110, 36)) as pilot:
                    tree = tui.query_one("#nav-tree", Tree)
                    for _ in range(60):
                        await pilot.pause(0.05)
                        if tree.root.children:
                            break
                    tree.root.expand_all()
                    await pilot.pause(0.1)

                    convos = [
                        node
                        for node in tui._walk_tree_nodes()
                        if node.data and node.data.kind == "convo"
                    ]
                    self.assertGreaterEqual(len(convos), 6)

                    with patch(
                        "agentconvos.app.parse_jsonl",
                        side_effect=lambda path: parsed.append(path.name) or [Turn("user", "x")],
                    ):
                        # Sweep the cursor across every row without resting.
                        # move_cursor is what an arrow key does; select_node
                        # would be an explicit open, which loads immediately.
                        for node in convos:
                            tree.move_cursor(node)
                            await pilot.pause(0.01)
                        # Then rest, the way a reader does.
                        for _ in range(40):
                            await pilot.pause(0.05)
                            if parsed:
                                break
                        await pilot.pause(0.3)

        self.assertTrue(parsed, "the conversation the cursor rested on was never previewed")
        self.assertLessEqual(
            len(parsed), 2,
            f"scanning parsed {len(parsed)} transcripts: {parsed}",
        )
        self.assertEqual(parsed[-1], convos[-1].data.meta.path.name)

    async def test_returning_to_a_conversation_after_a_project_row_previews_it(self):
        """The project digest replaces the transcript, so coming back to the
        same conversation has to render it again rather than assume it is up."""
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            metas = [
                _meta(cwd / "a.jsonl", "back-a", "2026-08-10T09:00:00", cwd),
                _meta(cwd / "b.jsonl", "back-b", "2026-08-09T09:00:00", cwd),
            ]
            project = Project("tmp", str(cwd), metas)
            shown: list[str] = []

            with patch("agentconvos.app.scan_projects", return_value=[project]):
                tui = app_module.ConvoExplorer()
                async with tui.run_test(size=(120, 36)) as pilot:
                    tree = tui.query_one("#nav-tree", Tree)
                    for _ in range(60):
                        await pilot.pause(0.05)
                        if tree.root.children:
                            break
                    tree.root.expand_all()
                    await pilot.pause(0.2)

                    convo = next(
                        n for n in tui._walk_tree_nodes()
                        if n.data and n.data.kind == "convo"
                    )
                    project_row = next(
                        n for n in tui._walk_tree_nodes()
                        if n.data and n.data.kind == "project"
                    )

                    with (
                        patch("agentconvos.app.parse_jsonl", return_value=[Turn("user", "hello")]),
                        patch.object(
                            tui, "_set_preview",
                            side_effect=lambda md, *a, **k: shown.append(md),
                        ),
                    ):
                        tree.move_cursor(convo)
                        await pilot.pause(0.4)
                        tree.move_cursor(project_row)
                        await pilot.pause(0.4)
                        tree.move_cursor(convo)
                        await pilot.pause(0.4)

        self.assertGreaterEqual(len(shown), 3, f"only rendered {len(shown)} previews")
        self.assertIn("Session ID:", shown[0])          # the conversation
        self.assertIn("2 conversations", shown[1])      # the project digest
        self.assertIn("Session ID:", shown[-1])         # the conversation again

    async def test_opening_the_highlighted_row_does_not_parse_it_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            meta = _meta(cwd / "a.jsonl", "once", "2026-08-10T09:00:00", cwd)
            project = Project("tmp", str(cwd), [meta])
            parsed: list[str] = []

            with patch("agentconvos.app.scan_projects", return_value=[project]):
                tui = app_module.ConvoExplorer()
                async with tui.run_test(size=(120, 36)) as pilot:
                    tree = tui.query_one("#nav-tree", Tree)
                    for _ in range(60):
                        await pilot.pause(0.05)
                        if tree.root.children:
                            break
                    tree.root.expand_all()
                    await pilot.pause(0.2)
                    convo = next(
                        n for n in tui._walk_tree_nodes()
                        if n.data and n.data.kind == "convo"
                    )

                    with patch(
                        "agentconvos.app.parse_jsonl",
                        side_effect=lambda p: parsed.append(p.name) or [Turn("user", "x")],
                    ):
                        tree.move_cursor(convo)
                        await pilot.pause(0.4)
                        tree.select_node(convo)   # what Enter does
                        await pilot.pause(0.4)

        self.assertEqual(len(parsed), 1, f"transcript parsed {len(parsed)} times: {parsed}")

    async def test_arrow_keys_walk_results_from_the_search_box(self):
        """↑/↓ must move the result cursor and preview it while the search box
        keeps focus, and Enter must open the highlighted conversation."""
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            metas = [
                _meta(cwd / "a.jsonl", "session-one", "2026-08-10T09:00:00", cwd),
                _meta(cwd / "b.jsonl", "session-two", "2026-08-09T09:00:00", cwd),
            ]
            project = Project("tmp", str(cwd), metas)
            previewed: list[str] = []

            with patch("agentconvos.app.scan_projects", return_value=[project]):
                tui = app_module.ConvoExplorer()
                async with tui.run_test(size=(110, 36)) as pilot:
                    tree = tui.query_one("#nav-tree", Tree)
                    for _ in range(60):
                        await pilot.pause(0.05)
                        if tree.root.children:
                            break
                    tree.root.expand_all()
                    await pilot.pause(0.1)

                    search = tui.query_one("#filter-input", Input)
                    self.assertTrue(search.has_focus)

                    with patch.object(
                        tui,
                        "request_preview",
                        side_effect=lambda meta, query="": previewed.append(meta.uuid),
                    ):
                        node = None
                        for _ in range(8):
                            await pilot.press("down")
                            await pilot.pause(0.02)
                            node = tree.cursor_node
                            if node and node.data and node.data.kind == "convo":
                                break

                        self.assertTrue(search.has_focus, "arrows must not steal focus")
                        self.assertIsNotNone(node.data, "cursor never reached a conversation")
                        self.assertEqual(node.data.kind, "convo")
                        # The highlight preview waits for the cursor to settle.
                        for _ in range(40):
                            await pilot.pause(0.05)
                            if node.data.meta.uuid in previewed:
                                break
                        self.assertIn(node.data.meta.uuid, previewed)

                        await pilot.press("enter")
                        await pilot.pause(0.05)

                    self.assertEqual(tui.current_meta.uuid, node.data.meta.uuid)
                    self.assertTrue(tree.has_focus, "Enter should hand focus to the tree")

    async def test_projects_order_by_recent_activity_not_alphabet(self):
        """With hundreds of projects, the active ones must sit on top; the
        alphabet buries them under long-dormant names."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dormant_dir = root / "alpha-archive"
            active_dir = root / "zeta-current"
            dormant = Project(
                "alpha-archive",
                str(dormant_dir),
                [_meta(dormant_dir / "old.jsonl", "old-session", "2026-01-05T09:00:00", dormant_dir)],
            )
            active = Project(
                "zeta-current",
                str(active_dir),
                [_meta(active_dir / "new.jsonl", "new-session", "2026-08-10T09:00:00", active_dir)],
            )

            with patch("agentconvos.app.scan_projects", return_value=[active, dormant]):
                tui = app_module.ConvoExplorer()
                async with tui.run_test(size=(110, 36)) as pilot:
                    for _ in range(60):
                        await pilot.pause(0.05)
                        rows = [
                            node.data.project.folder_name
                            for node in tui._walk_tree_nodes()
                            if node.data and node.data.kind == "project"
                        ]
                        if len(rows) == 2:
                            break

        self.assertEqual(rows, ["zeta-current", "alpha-archive"])

    async def test_broad_filter_is_bounded_and_applied_in_one_batch(self):
        class BroadIndex:
            def search(self, query):
                if query != "a":
                    return {}
                return {meta.uuid: "a broad match" for meta in metas}

            def sync(self, conversations, **_kwargs):
                conversations = list(conversations)
                return SimpleNamespace(
                    total=len(conversations),
                    checked=len(conversations),
                    indexed=0,
                    unchanged=len(conversations),
                    removed=0,
                    failed=0,
                )

        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            metas = [
                _meta(
                    cwd / f"session-{index}.jsonl",
                    f"broad-session-{index}",
                    "2026-07-18T10:30:00",
                    cwd,
                )
                for index in range(225)
            ]
            project = Project("tmp", str(cwd), metas)

            with patch("agentconvos.app.scan_projects", return_value=[project]):
                tui = app_module.ConvoExplorer(search_index=BroadIndex())
                with patch.object(tui, "batch_update", wraps=tui.batch_update) as batch_update:
                    async with tui.run_test(size=(110, 36)) as pilot:
                        await pilot.pause()
                        search = tui.query_one("#filter-input", Input)
                        search.value = "a"
                        for _ in range(40):
                            await pilot.pause(0.03)
                            result_nodes = [
                                node
                                for node in tui._walk_tree_nodes()
                                if node.data and node.data.kind == "convo"
                            ]
                            if result_nodes and "RESULTS" in str(
                                tui.query_one("#left-title", Static).render()
                            ):
                                break

                        self.assertEqual(len(result_nodes), 200)

                    self.assertGreaterEqual(batch_update.call_count, 2)

    async def test_preview_shows_loading_state_while_transcript_parses(self):
        class ReadyIndex:
            def search(self, _query):
                return {}

            def sync(self, conversations, **_kwargs):
                conversations = list(conversations)
                return SimpleNamespace(
                    total=len(conversations),
                    checked=len(conversations),
                    indexed=0,
                    unchanged=len(conversations),
                    removed=0,
                    failed=0,
                )

        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            path = cwd / "session.jsonl"
            path.write_text("", encoding="utf-8")
            meta = _meta(path, "preview-session", "2026-07-18T10:30:00", cwd)
            project = Project("tmp", str(cwd), [meta])
            parse_started = threading.Event()
            release_parse = threading.Event()

            def slow_parse(_path):
                parse_started.set()
                release_parse.wait(timeout=3)
                return [Turn("assistant", "Loaded preview")]

            with (
                patch("agentconvos.app.scan_projects", return_value=[project]),
                patch("agentconvos.app.parse_jsonl", side_effect=slow_parse),
            ):
                tui = app_module.ConvoExplorer(search_index=ReadyIndex())
                async with tui.run_test(size=(110, 36)) as pilot:
                    try:
                        await pilot.pause()
                        tui.request_preview(meta)
                        for _ in range(20):
                            await pilot.pause()
                            if parse_started.is_set():
                                break

                        preview_scroll = tui.query_one("#preview-scroll")
                        self.assertTrue(parse_started.is_set())
                        self.assertTrue(preview_scroll.loading)

                        release_parse.set()
                        for _ in range(20):
                            await pilot.pause()
                            if not preview_scroll.loading:
                                break
                        self.assertFalse(preview_scroll.loading)
                    finally:
                        release_parse.set()

    async def test_cancelled_preview_cannot_overwrite_the_latest_conversation(self):
        class ReadyIndex:
            def search(self, _query):
                return {}

            def sync(self, conversations, **_kwargs):
                conversations = list(conversations)
                return SimpleNamespace(
                    total=len(conversations),
                    checked=len(conversations),
                    indexed=0,
                    unchanged=len(conversations),
                    removed=0,
                    failed=0,
                )

        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            first_path = cwd / "first.jsonl"
            second_path = cwd / "second.jsonl"
            first_path.write_text("", encoding="utf-8")
            second_path.write_text("", encoding="utf-8")
            first = _meta(first_path, "first-preview", "2026-07-18T10:30:00", cwd)
            second = _meta(second_path, "second-preview", "2026-07-18T10:31:00", cwd)
            project = Project("tmp", str(cwd), [second, first])
            first_started = threading.Event()
            release_first = threading.Event()

            def parse_preview(path):
                if path == first_path:
                    first_started.set()
                    release_first.wait(timeout=3)
                    return [Turn("user", "First"), Turn("assistant", "First result")]
                return [Turn("assistant", "Second result")]

            with (
                patch("agentconvos.app.scan_projects", return_value=[project]),
                patch("agentconvos.app.parse_jsonl", side_effect=parse_preview),
            ):
                tui = app_module.ConvoExplorer(search_index=ReadyIndex())
                async with tui.run_test(size=(110, 36)) as pilot:
                    try:
                        await pilot.pause()
                        tui.request_preview(first)
                        for _ in range(20):
                            await pilot.pause()
                            if first_started.is_set():
                                break

                        tui.request_preview(second)
                        for _ in range(20):
                            await pilot.pause()
                            if "1 turns" in str(
                                tui.query_one("#right-title", Static).render()
                            ):
                                break

                        release_first.set()
                        for _ in range(20):
                            await pilot.pause()

                        title = str(tui.query_one("#right-title", Static).render())
                        self.assertEqual(title, "CONVERSATION · 1 turn")
                    finally:
                        release_first.set()

    async def test_active_search_refreshes_as_background_index_makes_progress(self):
        class ProgressiveIndex:
            def __init__(self, uuid):
                self.uuid = uuid
                self.indexed = False
                self.started = threading.Event()
                self.advance = threading.Event()
                self.progress_sent = threading.Event()
                self.finish = threading.Event()

            def search(self, query):
                if self.indexed and query == "auth middleware":
                    return {self.uuid: "auth middleware indexed in the background"}
                return {}

            def sync(self, conversations, on_progress, **_kwargs):
                conversations = list(conversations)
                self.started.set()
                self.advance.wait(timeout=3)
                self.indexed = True
                progress = SimpleNamespace(
                    total=2,
                    checked=1,
                    indexed=1,
                    unchanged=0,
                    removed=0,
                    failed=0,
                )
                on_progress(progress)
                self.progress_sent.set()
                self.finish.wait(timeout=3)
                return SimpleNamespace(
                    total=2,
                    checked=2,
                    indexed=2,
                    unchanged=0,
                    removed=0,
                    failed=0,
                )

        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            path = cwd / "session.jsonl"
            path.write_text("", encoding="utf-8")
            meta = _meta(path, "progress-session", "2026-07-18T10:30:00", cwd)
            meta.preview = "Unrelated prompt"
            project = Project("tmp", str(cwd), [meta])
            index = ProgressiveIndex(meta.uuid)

            with patch("agentconvos.app.scan_projects", return_value=[project]):
                tui = app_module.ConvoExplorer(search_index=index)
                async with tui.run_test(size=(110, 36)) as pilot:
                    try:
                        for _ in range(20):
                            await pilot.pause()
                            if index.started.is_set():
                                break
                        search = tui.query_one("#filter-input", Input)
                        search.value = "auth middleware"
                        for _ in range(30):
                            await pilot.pause(0.03)
                            result_nodes = [
                                node
                                for node in tui._walk_tree_nodes()
                                if node.data and node.data.kind == "convo"
                            ]
                            if not any(
                                node.data and node.data.kind == "convo"
                                for node in tui._walk_tree_nodes()
                            ):
                                break
                        self.assertFalse(
                            any(
                                node.data and node.data.kind == "convo"
                                for node in tui._walk_tree_nodes()
                            )
                        )

                        index.advance.set()
                        for _ in range(30):
                            await pilot.pause()
                            result_nodes = [
                                node
                                for node in tui._walk_tree_nodes()
                                if node.data and node.data.kind == "convo"
                            ]
                            if index.progress_sent.is_set() and result_nodes:
                                break

                        self.assertTrue(index.progress_sent.is_set())
                        self.assertEqual(len(result_nodes), 1)
                        self.assertFalse(index.finish.is_set())
                        search_title = str(
                            tui.query_one("#right-title", Static).render()
                        )
                        self.assertEqual(search_title, "SEARCH · 1")
                    finally:
                        index.advance.set()
                        index.finish.set()

    async def test_index_progress_does_not_replace_an_open_search_preview(self):
        class ReadyIndex:
            def __init__(self, uuid):
                self.uuid = uuid

            def search(self, query):
                return (
                    {self.uuid: "auth middleware indexed in the background"}
                    if query == "auth middleware"
                    else {}
                )

            def sync(self, conversations, **_kwargs):
                conversations = list(conversations)
                return SimpleNamespace(
                    total=len(conversations),
                    checked=len(conversations),
                    indexed=0,
                    unchanged=len(conversations),
                    removed=0,
                    failed=0,
                )

        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            path = cwd / "session.jsonl"
            path.write_text("", encoding="utf-8")
            meta = _meta(path, "stable-preview", "2026-07-18T10:30:00", cwd)
            project = Project("tmp", str(cwd), [meta])
            turns = [
                Turn("user", "Inspect the auth middleware"),
                Turn("assistant", "The auth middleware is ready"),
            ]

            with (
                patch("agentconvos.app.scan_projects", return_value=[project]),
                patch("agentconvos.app.parse_jsonl", return_value=turns),
            ):
                tui = app_module.ConvoExplorer(search_index=ReadyIndex(meta.uuid))
                async with tui.run_test(size=(110, 36)) as pilot:
                    await pilot.pause()
                    search = tui.query_one("#filter-input", Input)
                    search.value = "auth middleware"
                    await pilot.press("enter")
                    for _ in range(20):
                        await pilot.pause()
                        if "MATCHES" in str(
                            tui.query_one("#right-title", Static).render()
                        ):
                            break

                    tui._indexing = True
                    tui._set_index_progress(
                        SimpleNamespace(
                            total=2,
                            checked=1,
                            indexed=1,
                            unchanged=0,
                            removed=0,
                            failed=0,
                        )
                    )

                    title = str(tui.query_one("#right-title", Static).render())
                    self.assertEqual(title, "MATCHES (2 turns) · R RESUME")

    async def test_search_enter_opens_first_match_with_highlighted_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            path = cwd / "session.jsonl"
            path.write_text("", encoding="utf-8")
            meta = _meta(path, "search-session", "2026-07-15T14:30:00", cwd)
            project = Project("tmp", str(cwd), [meta])
            turns = [
                Turn("user", "Please inspect the auth middleware failure."),
                Turn("assistant", "The auth middleware drops the request id."),
            ]
            from agentconvos.search_index import ConversationSearchIndex

            index = ConversationSearchIndex(cwd / "search.sqlite3")
            index.sync([meta], parse_conversation=lambda _path: turns)

            with (
                patch("agentconvos.app.scan_projects", return_value=[project]),
                patch("agentconvos.app.parse_jsonl", return_value=turns),
            ):
                tui = app_module.ConvoExplorer(search_index=index)
                async with tui.run_test(size=(160, 36)) as pilot:
                    await pilot.pause()
                    search = tui.query_one("#filter-input", Input)
                    search.focus()
                    search.value = "auth middleware"
                    for _ in range(30):
                        await pilot.pause(0.03)
                        result_nodes = [
                            node
                            for node in tui._walk_tree_nodes()
                            if node.data and node.data.kind == "convo"
                        ]
                        if result_nodes and "RESULTS" in str(
                            tui.query_one("#left-title", Static).render()
                        ):
                            break

                    result_nodes = [
                        node
                        for node in tui._walk_tree_nodes()
                        if node.data and node.data.kind == "convo"
                    ]
                    self.assertEqual(len(result_nodes), 1)
                    self.assertGreaterEqual(len(result_nodes[0].label.spans), 2)

                    await pilot.press("enter")
                    await pilot.pause()

                    self.assertEqual(tui.current_meta, meta)
                    title = str(tui.query_one("#right-title", Static).render())
                    self.assertEqual(title, "MATCHES (2 turns) · R RESUME")

    async def test_escape_clears_search_from_the_search_box(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            path = cwd / "session.jsonl"
            path.write_text("", encoding="utf-8")
            meta = _meta(path, "search-session", "2026-07-15T14:30:00", cwd)
            project = Project("tmp", str(cwd), [meta])

            with (
                patch("agentconvos.app.scan_projects", return_value=[project]),
                patch("agentconvos.app.parse_jsonl", return_value=[Turn("user", "auth")]),
            ):
                from agentconvos.search_index import ConversationSearchIndex

                tui = app_module.ConvoExplorer(
                    search_index=ConversationSearchIndex(cwd / "search.sqlite3")
                )
                async with tui.run_test(size=(100, 32)) as pilot:
                    await pilot.pause()
                    search = tui.query_one("#filter-input", Input)
                    search.focus()
                    search.value = "auth"
                    await pilot.pause()

                    await pilot.press("escape")
                    await pilot.pause()

                    self.assertEqual(search.value, "")

    async def test_escape_dismisses_resume_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            path = cwd / "session.jsonl"
            path.write_text("", encoding="utf-8")
            meta = _meta(path, "resume-session", "2026-07-15T14:30:00", cwd)
            project = Project("tmp", str(cwd), [meta])

            with (
                patch("agentconvos.app.scan_projects", return_value=[project]),
                patch(
                    "agentconvos.app.parse_jsonl",
                    return_value=[Turn("user", "searchable conversation")],
                ),
            ):
                from agentconvos.search_index import ConversationSearchIndex

                tui = app_module.ConvoExplorer(
                    search_index=ConversationSearchIndex(cwd / "search.sqlite3")
                )
                async with tui.run_test(size=(100, 32)) as pilot:
                    await pilot.pause()
                    tui.current_meta = meta
                    tui.action_resume()
                    await pilot.pause()
                    self.assertIsInstance(tui.screen, app_module.ResumeScreen)

                    await pilot.press("escape")
                    await pilot.pause()

                    self.assertNotIsInstance(tui.screen, app_module.ResumeScreen)


if __name__ == "__main__":
    unittest.main()
