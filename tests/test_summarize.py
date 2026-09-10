import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentconvos.llm_config import LLMConfig
from agentconvos.parser import DETAIL_TEXT, ConversationMeta, Turn
from agentconvos.summarize import _needs_summary, summarize_session


def _meta(path: Path) -> ConversationMeta:
    return ConversationMeta(
        path=path,
        uuid="summary-session",
        slug="summary-session",
        timestamp="2026-08-03T12:00:00Z",
        cwd=str(path.parent),
        preview="Summarize the session",
        source="codex",
    )


class _Response:
    status_code = 200

    def __init__(self, text: str):
        self._text = text

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self._text}}]}


class SummarizeSessionTests(unittest.TestCase):
    def test_legacy_one_pass_cache_is_regenerated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "session.jsonl"
            path.write_text("conversation", encoding="utf-8")
            summaries = root / "summaries"
            summaries.mkdir()
            (summaries / "summary-session.json").write_text(
                json.dumps({"summary": "Legacy summary", "model": "old-model"}),
                encoding="utf-8",
            )

            with patch("agentconvos.summarize.SUMMARIES_DIR", summaries):
                needs_summary = _needs_summary(_meta(path))

        self.assertTrue(needs_summary)

    def test_summarize_session_reads_every_normalized_turn(self):
        turns = [
            Turn("user", "The first turn must not be dropped."),
            Turn("assistant", "The middle turn carries the decision."),
            Turn("user", "The final turn confirms the outcome."),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.jsonl"
            with (
                patch("agentconvos.summarize.SUMMARIES_DIR", Path(tmp) / "summaries"),
                patch("agentconvos.summarize.parse_jsonl", return_value=turns) as parse,
                patch("agentconvos.summarize._call_llm", return_value="Session summary"),
            ):
                summarize_session(_meta(path), "test-key")

        parse.assert_called_once_with(path, detail=DETAIL_TEXT)

    def test_second_pass_extends_the_first_request_for_prompt_caching(self):
        turns = [
            Turn("user", "Implement cache-aware summaries."),
            Turn("assistant", "The implementation was completed and verified."),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.jsonl"
            with (
                patch("agentconvos.summarize.SUMMARIES_DIR", Path(tmp) / "summaries"),
                patch("agentconvos.summarize.parse_jsonl", return_value=turns),
                patch(
                    "httpx.post",
                    side_effect=[
                        _Response("Draft recap from the complete transcript."),
                        _Response("Implement cache-aware full-session summaries."),
                    ],
                ) as post,
            ):
                summary = summarize_session(_meta(path), "test-key")

        self.assertEqual(summary, "Implement cache-aware full-session summaries.")
        self.assertEqual(post.call_count, 2)
        first_payload = post.call_args_list[0].kwargs["json"]
        second_payload = post.call_args_list[1].kwargs["json"]
        first_messages = first_payload["messages"]
        second_messages = second_payload["messages"]
        self.assertEqual(second_messages[: len(first_messages)], first_messages)
        self.assertEqual(
            second_messages[len(first_messages)],
            {"role": "assistant", "content": "Draft recap from the complete transcript."},
        )
        self.assertEqual(second_messages[-1]["role"], "user")

    def test_summaries_use_the_configured_endpoint_and_model(self):
        turns = [Turn("user", "Hi."), Turn("assistant", "Hello.")]
        cfg = LLMConfig("http://gw/v1", "k", "custom/model", "custom/pro", "test")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.jsonl"
            with (
                patch("agentconvos.summarize.SUMMARIES_DIR", Path(tmp) / "summaries"),
                patch("agentconvos.summarize.parse_jsonl", return_value=turns),
                patch("agentconvos.summarize.load_config", return_value=cfg),
                patch("httpx.post", side_effect=[_Response("draft"), _Response("Done.")]) as post,
            ):
                summarize_session(_meta(path))
            cached = json.loads((Path(tmp) / "summaries" / "summary-session.json").read_text())

        self.assertEqual(post.call_args.args[0], "http://gw/v1/chat/completions")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "custom/model")
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer k")
        self.assertEqual(cached["model"], "custom/model")


if __name__ == "__main__":
    unittest.main()
