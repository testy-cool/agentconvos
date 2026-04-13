"""Parse Claude Code .jsonl conversation logs into structured data."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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


def get_meta(path: Path) -> ConversationMeta | None:
    """Quick-scan a .jsonl to extract metadata from the first user record."""
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


def parse_jsonl(path: Path, detail: str = DETAIL_TEXT) -> list[Turn]:
    """Extract turns from a .jsonl conversation log.

    detail levels:
      "text"    — user/assistant text only (default)
      "tools"   — also include tool call summaries
      "results" — also include tool results (truncated)
      "full"    — also include tool results (untruncated)
    """
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
                # Count tool_use blocks
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
