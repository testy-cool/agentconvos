import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentconvos.app import _handoff_cmd, main
from agentconvos.parser import ConversationMeta, ConversationStats
from agentconvos.scanner import Project


class HandoffCommandTests(unittest.TestCase):
    def test_codex_handoff_can_use_yolo(self):
        self.assertEqual(
            _handoff_cmd("codex", "handoff message", codex_yolo=True),
            ["codex", "--yolo", "handoff message"],
        )

    def test_handoff_agent_targets_codex_independent_of_conversation_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            convo_path = cwd / "conversation.jsonl"
            convo_path.write_text("", encoding="utf-8")
            meta = ConversationMeta(
                path=convo_path,
                uuid="abc123",
                slug="claude-source",
                timestamp="2026-06-03T12:00:00",
                cwd=str(cwd),
                preview="Previous work",
                source="claude",
            )
            project = Project("tmp", str(cwd), [meta])

            old_argv = sys.argv
            old_cwd = os.getcwd()
            sys.argv = [
                "agentconvos",
                "--handoff",
                "--handoff-agent",
                "codex",
                "--yolo",
                "--dry-run",
            ]
            stream = io.StringIO()
            try:
                os.chdir(cwd)
                with (
                    patch("agentconvos.scanner.scan_projects", return_value=[project]),
                    patch("agentconvos.app.parse_jsonl", return_value=[]),
                    patch("agentconvos.app.get_stats", return_value=ConversationStats()),
                    patch("agentconvos.app.to_markdown", return_value="exported"),
                    contextlib.redirect_stdout(stream),
                ):
                    main()
            finally:
                sys.argv = old_argv
                os.chdir(old_cwd)

        output = stream.getvalue()
        self.assertIn("codex --yolo", output)
        self.assertNotIn("claude --dangerously-skip-permissions", output)
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", output)


if __name__ == "__main__":
    unittest.main()
