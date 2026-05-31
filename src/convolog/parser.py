"""Parse Claude Code, Codex, and Pi .jsonl conversation logs into structured data."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class Turn:
    role: str  # "user" or "assistant"
    text: str


@dataclass
class ToolCall:
    name: str
    summary: str  # human-readable one-liner


@dataclass
class ConversationStats:
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_create_tokens: int = 0
    duration_ms: int = 0
    tool_calls: int = 0
    api_errors: int = 0

    @property
    def cost_estimate(self) -> float:
        """Rough cost estimate in USD. Uses Sonnet-tier pricing as default."""
        # $3/M input, $15/M output (Sonnet-ish), cache read ~$0.30/M
        return (
            self.input_tokens * 3 / 1_000_000
            + self.output_tokens * 15 / 1_000_000
            + self.cache_read_tokens * 0.3 / 1_000_000
        )


@dataclass
class ConversationMeta:
    path: Path
    uuid: str
    slug: str  # human-readable name, may be empty
    timestamp: str  # ISO 8601
    cwd: str
    preview: str  # first user message, truncated
    turn_count: int = 0
    source: str = "claude"  # "claude", "codex", or "pi"


def _detect_format(path: Path) -> str:
    """Detect whether a .jsonl file is Claude Code, Codex, or Pi format."""
    try:
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
            if isinstance(rec, dict) and isinstance(rec.get("items"), list):
                return "codex_json"

        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline()
            if not first_line.strip():
                first_line = f.readline()
            rec = json.loads(first_line)
            if rec.get("type") == "session" and "version" in rec:
                return "pi"
            if rec.get("type") == "session_meta" or (
                rec.get("type") in ("event_msg", "response_item", "turn_context")
                and "payload" in rec
            ):
                return "codex"
    except (json.JSONDecodeError, OSError, KeyError):
        pass
    return "claude"


def get_meta(path: Path) -> ConversationMeta | None:
    """Quick-scan a .jsonl to extract metadata from the first user record."""
    fmt = _detect_format(path)
    if fmt == "codex":
        return _get_meta_codex(path)
    if fmt == "codex_json":
        return _get_meta_codex_json(path)
    if fmt == "pi":
        return _get_meta_pi(path)
    return _get_meta_claude(path)


def _get_meta_claude(path: Path) -> ConversationMeta | None:
    """Extract metadata from a Claude Code .jsonl."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("type") != "user":
                    continue
                content = rec.get("message", {}).get("content", "")
                if isinstance(content, list) or not content or len(content.strip()) < 3:
                    continue
                preview = content.strip().replace("\n", " ")[:120]
                return ConversationMeta(
                    path=path,
                    uuid=path.stem,
                    slug=rec.get("slug", ""),
                    timestamp=rec.get("timestamp", ""),
                    cwd=rec.get("cwd", ""),
                    preview=preview,
                )
    except (json.JSONDecodeError, OSError):
        pass
    return None


_CODEX_THREAD_NAMES: dict[str, str] | None = None


def _codex_home() -> Path:
    home = os.environ.get("CODEX_HOME")
    return Path(home).expanduser() if home else Path.home() / ".codex"


def _codex_thread_names() -> dict[str, str]:
    """Return Codex session id -> thread name from the lightweight local index."""
    global _CODEX_THREAD_NAMES
    if _CODEX_THREAD_NAMES is not None:
        return _CODEX_THREAD_NAMES

    names: dict[str, str] = {}
    index_path = _codex_home() / "session_index.jsonl"
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = rec.get("id")
                thread_name = rec.get("thread_name")
                if sid and thread_name:
                    names[sid] = thread_name
    except OSError:
        pass
    _CODEX_THREAD_NAMES = names
    return names


def _unix_to_iso(value) -> str:
    if not isinstance(value, (int, float)):
        return ""
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _codex_timestamp_from_rollout_stem(stem: str) -> str:
    raw = stem[8:27]
    if "T" not in raw:
        return ""
    date_part, time_part = raw.split("T", 1)
    return f"{date_part}T{time_part.replace('-', ':')}Z"


def _is_codex_context_block(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith((
        "<environment_context>",
        "<permissions instructions>",
        "<collaboration_mode>",
        "<apps_instructions>",
        "<skills_instructions>",
        "<plugins_instructions>",
    ))


def _extract_codex_message_text(content, role: str = "") -> str:
    """Extract human-readable text from Codex message content blocks."""
    if isinstance(content, str):
        text = content.strip()
        return "" if _is_codex_context_block(text) else text
    if not isinstance(content, list):
        return ""

    parts = []
    saw_image = False
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")
        if btype in ("input_image", "image"):
            saw_image = True
            continue
        if btype not in ("input_text", "output_text", "text"):
            continue
        text = (block.get("text") or "").strip()
        if not text or _is_codex_context_block(text):
            continue
        text = re.sub(r"<image name=([^>]+)>", r"\1", text).replace("</image>", "")
        text = re.sub(r"(\[Image[^\]]+\])\s+\1", r"\1", text)
        parts.append(text)

    if saw_image and not any(part.lower().startswith("image") or "[image" in part.lower() for part in parts):
        parts.append("[image]")
    return "\n".join(parts).strip()


def _codex_user_dedupe_key(text: str) -> str:
    cleaned = re.sub(r"\[Image[^\]]+\]|</?image[^>]*>", " ", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned[:200] or text[:200]


def _get_meta_codex(path: Path) -> ConversationMeta | None:
    """Extract metadata from a Codex .jsonl."""
    try:
        session_id = ""
        timestamp = ""
        cwd = ""
        preview = ""
        preview_source = ""
        with open(path, "r", encoding="utf-8") as f:
            line_count = 0
            for line in f:
                line_count += 1
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rtype = rec.get("type", "")
                payload = rec.get("payload", {})

                if rtype == "session_meta":
                    session_id = payload.get("id", path.stem)
                    timestamp = payload.get("timestamp", rec.get("timestamp", ""))
                    cwd = payload.get("cwd", "")

                if rtype == "event_msg" and payload.get("type") == "user_message" and preview_source != "event":
                    msg = payload.get("message", "")
                    if isinstance(msg, str) and msg.strip() and len(msg.strip()) >= 3:
                        preview = msg.strip().replace("\n", " ")[:120]
                        preview_source = "event"

                if rtype == "response_item" and payload.get("type") == "message" and not preview:
                    if payload.get("role") == "user":
                        msg = _extract_codex_message_text(payload.get("content", []), role="user")
                        if msg:
                            preview = msg.replace("\n", " ")[:120]
                            preview_source = "response"

                if session_id and preview and (preview_source == "event" or line_count > 50):
                    break

        if not session_id:
            session_id = path.stem
        if not timestamp:
            # Fall back to filename timestamp: rollout-2026-04-30T08-05-54-{uuid}.jsonl
            stem = path.stem
            if stem.startswith("rollout-") and len(stem) > 28:
                timestamp = _codex_timestamp_from_rollout_stem(stem)

        return ConversationMeta(
            path=path,
            uuid=session_id,
            slug=_codex_thread_names().get(session_id, ""),
            timestamp=timestamp,
            cwd=cwd,
            preview=preview,
            source="codex",
        )
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _get_meta_codex_json(path: Path) -> ConversationMeta | None:
    """Extract metadata from legacy Codex JSON conversation files."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            rec = json.load(f)
        session_id = rec.get("id", path.stem)
        timestamp = _unix_to_iso(rec.get("created_at")) or _unix_to_iso(rec.get("updated_at")) or ""
        preview = ""
        for item in rec.get("items", []):
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message" and item.get("role") == "user":
                preview = _extract_codex_message_text(item.get("content", []), role="user")
                if preview:
                    preview = preview.replace("\n", " ")[:120]
                    break

        return ConversationMeta(
            path=path,
            uuid=session_id,
            slug=rec.get("title") or _codex_thread_names().get(session_id, ""),
            timestamp=timestamp,
            cwd=rec.get("cwd", ""),
            preview=preview,
            source="codex",
        )
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _get_meta_pi(path: Path) -> ConversationMeta | None:
    """Extract metadata from a Pi .jsonl."""
    try:
        session_id = ""
        timestamp = ""
        cwd = ""
        preview = ""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rtype = rec.get("type", "")

                if rtype == "session":
                    session_id = rec.get("id", path.stem)
                    timestamp = rec.get("timestamp", "")
                    cwd = rec.get("cwd", "")

                if rtype == "message" and not preview:
                    msg = rec.get("message", {})
                    if msg.get("role") == "user":
                        content = msg.get("content", [])
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text = block.get("text", "").strip()
                                if text and len(text) >= 3:
                                    preview = text.replace("\n", " ")[:120]
                                    break

                if session_id and preview:
                    break

        if not session_id:
            session_id = path.stem

        return ConversationMeta(
            path=path,
            uuid=session_id,
            slug="",
            timestamp=timestamp,
            cwd=cwd,
            preview=preview,
            source="pi",
        )
    except (json.JSONDecodeError, OSError):
        pass
    return None


# ---------------------------------------------------------------------------
# Detail levels for parse_jsonl
# ---------------------------------------------------------------------------
DETAIL_TEXT = "text"        # user/assistant text only (default, backward compat)
DETAIL_TOOLS = "tools"      # + tool call summaries
DETAIL_RESULTS = "results"  # + truncated tool results
DETAIL_FULL = "full"        # + untruncated tool results

RESULT_TRUNCATE = 500  # chars to keep per tool result in "results" mode


def _summarize_tool(name: str, inp: dict) -> str:
    """One-line human-readable summary of a tool call."""
    if name == "Bash":
        cmd = inp.get("command", "")
        return f"`{cmd[:120]}`" if cmd else "(empty)"
    if name == "Read":
        return inp.get("file_path", "?")
    if name == "Write":
        return inp.get("file_path", "?")
    if name == "Edit":
        fp = inp.get("file_path", "?")
        old = (inp.get("old_string", "") or "")[:60].replace("\n", " ")
        return f"{fp} — replace `{old}...`"
    if name == "Grep":
        pat = inp.get("pattern", "?")
        path = inp.get("path", "")
        return f'"{pat}"' + (f" in {path}" if path else "")
    if name == "Glob":
        return inp.get("pattern", "?")
    if name == "Agent":
        desc = inp.get("description", "")
        prompt = inp.get("prompt", "")[:100].replace("\n", " ")
        return desc or prompt or "?"
    if name == "Skill":
        return inp.get("skill", "?")
    # Fallback: dump keys
    keys = ", ".join(f"{k}={str(v)[:40]}" for k, v in list(inp.items())[:4])
    return keys or "(no input)"


def _extract_tool_result_text(content) -> str:
    """Extract text from a tool_result content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts)
    return ""


def parse_jsonl(path: Path, detail: str = DETAIL_TEXT, last_n: int = 0) -> list[Turn]:
    """Extract turns from a .jsonl conversation log.

    detail levels:
      "text"    — user/assistant text only (default)
      "tools"   — also include tool call summaries
      "results" — also include tool results (truncated)
      "full"    — also include tool results (untruncated)

    last_n: if > 0, return only the last N turns (default 0 = all turns)
    """
    fmt = _detect_format(path)
    if fmt == "codex":
        turns = _parse_jsonl_codex(path, detail)
    elif fmt == "codex_json":
        turns = _parse_jsonl_codex_json(path, detail)
    elif fmt == "pi":
        turns = _parse_jsonl_pi(path, detail)
    else:
        turns = _parse_jsonl_claude(path, detail)
    if last_n > 0:
        return turns[-last_n:]
    return turns


def _parse_jsonl_claude(path: Path, detail: str = DETAIL_TEXT) -> list[Turn]:
    """Parse Claude Code format .jsonl."""
    include_tools = detail in (DETAIL_TOOLS, DETAIL_RESULTS, DETAIL_FULL)
    include_results = detail in (DETAIL_RESULTS, DETAIL_FULL)
    truncate_results = detail == DETAIL_RESULTS

    # Two-pass: tool results arrive in user records AFTER the assistant that called them.
    # Pass 1: collect all tool results keyed by tool_use_id.
    # Pass 2: build turns, attaching results to their tool calls.
    tool_results: dict[str, str] = {}
    if include_results:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") != "user":
                    continue
                content = rec.get("message", {}).get("content", "")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tool_id = block.get("tool_use_id", "")
                        if tool_id:
                            tool_results[tool_id] = _extract_tool_result_text(block.get("content", ""))

    # Pass 2: build turns
    turns: list[Turn] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg_type = rec.get("type")

            if msg_type == "user":
                content = rec.get("message", {}).get("content", "")
                if isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                    text = "\n".join(text_parts).strip()
                    if text and len(text) >= 3:
                        turns.append(Turn(role="user", text=text))
                elif isinstance(content, str) and content.strip() and len(content.strip()) >= 3:
                    turns.append(Turn(role="user", text=content.strip()))

            elif msg_type == "assistant":
                msg = rec.get("message", {})
                blocks = msg.get("content", [])
                parts = []
                for block in blocks:
                    if isinstance(block, str):
                        parts.append(block)
                        continue
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type", "")
                    if btype == "text":
                        parts.append(block.get("text", ""))
                    elif btype == "tool_use" and include_tools:
                        name = block.get("name", "?")
                        inp = block.get("input", {})
                        tool_id = block.get("id", "")
                        summary = _summarize_tool(name, inp)
                        tool_line = f"> **{name}**: {summary}"
                        if include_results and tool_id in tool_results:
                            result = tool_results[tool_id]
                            if result:
                                if truncate_results and len(result) > RESULT_TRUNCATE * 2:
                                    head = result[:RESULT_TRUNCATE]
                                    tail = result[-RESULT_TRUNCATE:]
                                    result = f"{head}\n... ({len(result):,} chars total) ...\n{tail}"
                                tool_line += f"\n> ```\n> {result}\n> ```"
                        parts.append(tool_line)

                text = "\n\n".join(p for p in parts if p.strip())
                if text and len(text) > 10:
                    turns.append(Turn(role="assistant", text=text))

    return turns


def _summarize_codex_tool(name: str, arguments: str) -> str:
    """One-line summary for a Codex function_call."""
    try:
        args = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        return arguments[:80] if arguments else "(no args)"
    if name in ("exec_command", "shell"):
        cmd = args.get("cmd", args.get("command", ""))
        if isinstance(cmd, list):
            cmd = " ".join(str(part) for part in cmd)
        return f"`{cmd[:120]}`" if cmd else "(empty)"
    if name == "write_stdin":
        chars = args.get("chars", "")
        session_id = args.get("session_id", "")
        preview = str(chars).replace("\n", "\\n")[:80]
        return f"session {session_id}: {preview}" if session_id else preview or "(empty)"
    if name == "update_plan":
        steps = args.get("plan", [])
        return f"{len(steps)} steps" if steps else "?"
    if name == "request_user_input":
        return args.get("prompt", args.get("question", "?"))[:80]
    if name == "spawn_agent":
        return args.get("task", args.get("prompt", "?"))[:80]
    if name.startswith("mcp__"):
        parts = name.split("__")
        short = parts[-1] if len(parts) > 1 else name
        return short
    keys = ", ".join(f"{k}={str(v)[:40]}" for k, v in list(args.items())[:3])
    return keys or "(no args)"


def _parse_jsonl_codex(path: Path, detail: str = DETAIL_TEXT) -> list[Turn]:
    """Parse Codex format .jsonl into turns."""
    include_tools = detail in (DETAIL_TOOLS, DETAIL_RESULTS, DETAIL_FULL)
    include_results = detail in (DETAIL_RESULTS, DETAIL_FULL)
    truncate_results = detail == DETAIL_RESULTS

    # Collect all events
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Build tool results map: call_id -> output text
    tool_results: dict[str, str] = {}
    if include_results:
        for rec in events:
            if rec.get("type") != "response_item":
                continue
            payload = rec.get("payload", {})
            if payload.get("type") == "function_call_output":
                call_id = payload.get("call_id", "")
                output = payload.get("output", "")
                # Strip the Codex header (Chunk ID, Wall time, etc.)
                if "\nOutput:\n" in output:
                    output = output.split("\nOutput:\n", 1)[1]
                if call_id:
                    tool_results[call_id] = output
        for rec in events:
            if rec.get("type") != "event_msg":
                continue
            payload = rec.get("payload", {})
            if payload.get("type") == "exec_command_end":
                call_id = payload.get("call_id", "")
                output = payload.get("aggregated_output", "") or payload.get("stdout", "")
                if call_id and call_id not in tool_results:
                    tool_results[call_id] = output

    turns: list[Turn] = []
    # Track seen agent messages to avoid duplicates (event_msg/agent_message vs response_item/message)
    seen_agent_texts: set[str] = set()
    seen_user_texts: dict[str, int] = {}

    for rec in events:
        rtype = rec.get("type", "")
        payload = rec.get("payload", {})

        # User messages from event_msg
        if rtype == "event_msg" and payload.get("type") == "user_message":
            msg = payload.get("message", "")
            if isinstance(msg, str) and msg.strip() and len(msg.strip()) >= 3:
                text = msg.strip()
                text_key = _codex_user_dedupe_key(text)
                if text_key in seen_user_texts:
                    turns[seen_user_texts[text_key]] = Turn(role="user", text=text)
                else:
                    seen_user_texts[text_key] = len(turns)
                    turns.append(Turn(role="user", text=text))

        # Agent messages from event_msg (prefer these over response_item duplicates)
        elif rtype == "event_msg" and payload.get("type") == "agent_message":
            msg = payload.get("message", "")
            if isinstance(msg, str) and msg.strip() and len(msg.strip()) > 10:
                text_key = msg.strip()[:200]
                if text_key not in seen_agent_texts:
                    seen_agent_texts.add(text_key)
                    turns.append(Turn(role="assistant", text=msg.strip()))

        # User text from response_item/message (newer Codex records).
        elif rtype == "response_item" and payload.get("type") == "message" and payload.get("role") == "user":
            text = _extract_codex_message_text(payload.get("content", []), role="user")
            if text and len(text) >= 3:
                text_key = _codex_user_dedupe_key(text)
                if text_key not in seen_user_texts:
                    seen_user_texts[text_key] = len(turns)
                    turns.append(Turn(role="user", text=text))

        # Tool calls from response_item/function_call
        elif rtype == "response_item" and payload.get("type") == "function_call" and include_tools:
            name = payload.get("name", "?")
            arguments = payload.get("arguments", "")
            call_id = payload.get("call_id", "")
            summary = _summarize_codex_tool(name, arguments)
            tool_line = f"> **{name}**: {summary}"
            if include_results and call_id in tool_results:
                result = tool_results[call_id]
                if result:
                    if truncate_results and len(result) > RESULT_TRUNCATE * 2:
                        head = result[:RESULT_TRUNCATE]
                        tail = result[-RESULT_TRUNCATE:]
                        result = f"{head}\n... ({len(result):,} chars total) ...\n{tail}"
                    tool_line += f"\n> ```\n> {result}\n> ```"
            # Append to last assistant turn or create new one
            if turns and turns[-1].role == "assistant":
                turns[-1] = Turn(role="assistant", text=turns[-1].text + "\n\n" + tool_line)
            else:
                turns.append(Turn(role="assistant", text=tool_line))

        # Assistant text from response_item/message (fallback if not seen via agent_message)
        elif rtype == "response_item" and payload.get("type") == "message" and payload.get("role") == "assistant":
            text = _extract_codex_message_text(payload.get("content", []), role="assistant")
            if text and len(text) > 10:
                text_key = text[:200]
                if text_key not in seen_agent_texts:
                    seen_agent_texts.add(text_key)
                    turns.append(Turn(role="assistant", text=text))

    return turns


def _parse_jsonl_codex_json(path: Path, detail: str = DETAIL_TEXT) -> list[Turn]:
    """Parse legacy Codex single-file JSON conversations into turns."""
    include_tools = detail in (DETAIL_TOOLS, DETAIL_RESULTS, DETAIL_FULL)
    include_results = detail in (DETAIL_RESULTS, DETAIL_FULL)
    truncate_results = detail == DETAIL_RESULTS

    with open(path, "r", encoding="utf-8") as f:
        rec = json.load(f)
    items = rec.get("items", [])
    if not isinstance(items, list):
        return []

    tool_results: dict[str, str] = {}
    if include_results:
        for item in items:
            if not isinstance(item, dict) or item.get("type") != "function_call_output":
                continue
            call_id = item.get("call_id", "")
            output = item.get("output", "")
            if call_id and isinstance(output, str):
                tool_results[call_id] = output

    turns: list[Turn] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "message":
            role = item.get("role")
            if role not in ("user", "assistant"):
                continue
            text = _extract_codex_message_text(item.get("content", []), role=role)
            if text and (role == "user" or len(text) > 10):
                turns.append(Turn(role=role, text=text))
        elif itype == "function_call" and include_tools:
            name = item.get("name", "?")
            arguments = item.get("arguments", "")
            call_id = item.get("call_id", "")
            tool_line = f"> **{name}**: {_summarize_codex_tool(name, arguments)}"
            if include_results and call_id in tool_results:
                result = tool_results[call_id]
                if truncate_results and len(result) > RESULT_TRUNCATE * 2:
                    head = result[:RESULT_TRUNCATE]
                    tail = result[-RESULT_TRUNCATE:]
                    result = f"{head}\n... ({len(result):,} chars total) ...\n{tail}"
                tool_line += f"\n> ```\n> {result}\n> ```"
            if turns and turns[-1].role == "assistant":
                turns[-1] = Turn(role="assistant", text=turns[-1].text + "\n\n" + tool_line)
            else:
                turns.append(Turn(role="assistant", text=tool_line))

    return turns


def _summarize_pi_tool(name: str, arguments: dict) -> str:
    """One-line summary for a Pi toolCall."""
    if name == "Bash":
        cmd = arguments.get("command", "")
        return f"`{cmd[:120]}`" if cmd else "(empty)"
    if name == "Read":
        return arguments.get("file_path", "?")
    if name == "Write":
        return arguments.get("file_path", "?")
    if name == "Edit":
        fp = arguments.get("file_path", "?")
        old = (arguments.get("old_string", "") or "")[:60].replace("\n", " ")
        return f"{fp} — replace `{old}...`"
    if name == "Grep":
        pat = arguments.get("pattern", "?")
        path = arguments.get("path", "")
        return f'"{pat}"' + (f" in {path}" if path else "")
    if name == "Glob":
        return arguments.get("pattern", "?")
    if name == "Agent":
        desc = arguments.get("description", "")
        prompt = arguments.get("prompt", "")[:100].replace("\n", " ")
        return desc or prompt or "?"
    keys = ", ".join(f"{k}={str(v)[:40]}" for k, v in list(arguments.items())[:4])
    return keys or "(no args)"


def _parse_jsonl_pi(path: Path, detail: str = DETAIL_TEXT) -> list[Turn]:
    """Parse Pi format .jsonl into turns."""
    include_tools = detail in (DETAIL_TOOLS, DETAIL_RESULTS, DETAIL_FULL)
    include_results = detail in (DETAIL_RESULTS, DETAIL_FULL)
    truncate_results = detail == DETAIL_RESULTS

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Build tool results map: tool_call_id -> result text
    tool_results: dict[str, str] = {}
    if include_results:
        for rec in records:
            if rec.get("type") != "message":
                continue
            msg = rec.get("message", {})
            if msg.get("role") != "toolResult":
                continue
            call_id = msg.get("toolCallId", "")
            content = msg.get("content", [])
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            if call_id:
                tool_results[call_id] = "\n".join(text_parts)

    turns: list[Turn] = []
    for rec in records:
        if rec.get("type") != "message":
            continue
        msg = rec.get("message", {})
        role = msg.get("role", "")
        content = msg.get("content", [])

        if role == "user":
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            text = "\n".join(text_parts).strip()
            if text and len(text) >= 3:
                turns.append(Turn(role="user", text=text))

        elif role == "assistant":
            parts = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype == "text":
                    parts.append(block.get("text", ""))
                elif btype == "toolCall" and include_tools:
                    name = block.get("name", "?")
                    arguments = block.get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                    call_id = block.get("id", "")
                    summary = _summarize_pi_tool(name, arguments)
                    tool_line = f"> **{name}**: {summary}"
                    if include_results and call_id in tool_results:
                        result = tool_results[call_id]
                        if result:
                            if truncate_results and len(result) > RESULT_TRUNCATE * 2:
                                head = result[:RESULT_TRUNCATE]
                                tail = result[-RESULT_TRUNCATE:]
                                result = f"{head}\n... ({len(result):,} chars total) ...\n{tail}"
                            tool_line += f"\n> ```\n> {result}\n> ```"
                    parts.append(tool_line)
            text = "\n\n".join(p for p in parts if p.strip())
            if text and len(text) > 10:
                turns.append(Turn(role="assistant", text=text))

    return turns


@dataclass
class SearchHit:
    meta: ConversationMeta
    turn_index: int
    role: str
    snippet: str  # context around match


def search_conversations(paths: list[Path], query: str, max_hits: int = 50) -> list[SearchHit]:
    """Search across conversation files for a string (case-insensitive).

    Returns matches with surrounding context.
    """
    query_lower = query.lower()
    hits: list[SearchHit] = []
    for path in paths:
        meta = get_meta(path)
        if not meta:
            continue
        try:
            turns = parse_jsonl(path)
        except Exception:
            continue
        for i, turn in enumerate(turns):
            text_lower = turn.text.lower()
            pos = text_lower.find(query_lower)
            if pos == -1:
                continue
            # Extract snippet with context
            start = max(0, pos - 60)
            end = min(len(turn.text), pos + len(query) + 60)
            snippet = turn.text[start:end].replace("\n", " ")
            if start > 0:
                snippet = "..." + snippet
            if end < len(turn.text):
                snippet += "..."
            hits.append(SearchHit(meta=meta, turn_index=i, role=turn.role, snippet=snippet))
            if len(hits) >= max_hits:
                return hits
    return hits


def get_stats(path: Path) -> ConversationStats:
    """Extract usage stats from a conversation without parsing full content."""
    fmt = _detect_format(path)
    if fmt == "codex":
        return _get_stats_codex(path)
    if fmt == "codex_json":
        return _get_stats_codex_json(path)
    if fmt == "pi":
        return _get_stats_pi(path)
    return _get_stats_claude(path)


def _get_stats_claude(path: Path) -> ConversationStats:
    """Extract stats from Claude Code format."""
    stats = ConversationStats()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            rtype = rec.get("type", "")

            if rtype == "assistant":
                msg = rec.get("message", {})
                if not stats.model and isinstance(msg, dict):
                    stats.model = msg.get("model", "")
                usage = msg.get("usage", {}) if isinstance(msg, dict) else {}
                stats.input_tokens += usage.get("input_tokens", 0)
                stats.output_tokens += usage.get("output_tokens", 0)
                stats.cache_read_tokens += usage.get("cache_read_input_tokens", 0)
                stats.cache_create_tokens += usage.get("cache_creation_input_tokens", 0)
                content = msg.get("content", []) if isinstance(msg, dict) else []
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            stats.tool_calls += 1

            elif rtype == "system":
                sub = rec.get("subtype", "")
                if sub == "turn_duration":
                    stats.duration_ms += rec.get("durationMs", 0)
                elif sub == "api_error":
                    stats.api_errors += 1

    return stats


def _get_stats_codex(path: Path) -> ConversationStats:
    """Extract stats from Codex format."""
    stats = ConversationStats()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            rtype = rec.get("type", "")
            payload = rec.get("payload", {})

            if rtype == "turn_context":
                if not stats.model:
                    stats.model = payload.get("model", "")

            elif rtype == "event_msg":
                ptype = payload.get("type", "")
                if ptype == "token_count":
                    info = payload.get("info")
                    if isinstance(info, dict):
                        # total_token_usage is cumulative — take the latest
                        usage = info.get("total_token_usage", {})
                        stats.input_tokens = usage.get("input_tokens", 0)
                        stats.output_tokens = usage.get("output_tokens", 0)
                        stats.cache_read_tokens = usage.get("cached_input_tokens", 0)
                elif ptype == "task_complete":
                    stats.duration_ms += payload.get("duration_ms", 0)

            elif rtype == "response_item":
                if payload.get("type") == "function_call":
                    stats.tool_calls += 1

    return stats


def _get_stats_codex_json(path: Path) -> ConversationStats:
    """Extract stats from legacy Codex JSON format."""
    stats = ConversationStats()
    with open(path, "r", encoding="utf-8") as f:
        rec = json.load(f)
    stats.model = rec.get("model", "")
    for item in rec.get("items", []):
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call":
            stats.tool_calls += 1
        usage = item.get("usage")
        if isinstance(usage, dict):
            stats.input_tokens += usage.get("input_tokens", usage.get("input", 0))
            stats.output_tokens += usage.get("output_tokens", usage.get("output", 0))
            stats.cache_read_tokens += usage.get("cached_input_tokens", usage.get("cache_read", 0))
    return stats


def _get_stats_pi(path: Path) -> ConversationStats:
    """Extract stats from Pi format."""
    stats = ConversationStats()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            rtype = rec.get("type", "")

            if rtype == "message":
                msg = rec.get("message", {})
                role = msg.get("role", "")
                if role == "assistant":
                    if not stats.model:
                        stats.model = msg.get("model", "")
                    usage = msg.get("usage", {})
                    if usage:
                        stats.input_tokens += usage.get("input", usage.get("inputTokens", 0))
                        stats.output_tokens += usage.get("output", usage.get("outputTokens", 0))
                        stats.cache_read_tokens += usage.get("cacheRead", usage.get("cacheReadTokens", 0))
                        stats.cache_create_tokens += usage.get("cacheWrite", usage.get("cacheWriteTokens", 0))
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "toolCall":
                                stats.tool_calls += 1

    return stats


def to_markdown(turns: list[Turn], stats: ConversationStats | None = None) -> str:
    """Format turns as markdown, optionally with a stats header."""
    sections = []

    if stats and stats.model:
        header_lines = [
            f"**Model:** {stats.model}",
            f"**Tokens:** {stats.input_tokens + stats.output_tokens:,} total ({stats.input_tokens:,} in / {stats.output_tokens:,} out)",
            f"**Duration:** {stats.duration_ms / 1000:.0f}s",
            f"**Tool calls:** {stats.tool_calls}",
        ]
        if stats.cost_estimate > 0.001:
            header_lines.append(f"**Est. cost:** ${stats.cost_estimate:.3f}")
        if stats.api_errors:
            header_lines.append(f"**API errors:** {stats.api_errors}")
        sections.append(" | ".join(header_lines))
        sections.append("---")

    for t in turns:
        header = "## User" if t.role == "user" else "## Assistant"
        sections.append(f"{header}\n{t.text}\n")
    return "\n".join(sections)
