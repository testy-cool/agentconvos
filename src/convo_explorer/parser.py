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


def parse_jsonl(path: Path) -> list[Turn]:
    """Extract user/assistant text turns from a .jsonl conversation log."""
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
                if isinstance(content, list) or not content or len(content.strip()) < 3:
                    continue
                turns.append(Turn(role="user", text=content.strip()))

            elif msg_type == "assistant":
                blocks = rec.get("message", {}).get("content", [])
                parts = []
                for block in blocks:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        parts.append(block)
                text = "".join(parts).strip()
                if text and len(text) > 10:
                    turns.append(Turn(role="assistant", text=text))
    return turns


def to_markdown(turns: list[Turn]) -> str:
    """Format turns as markdown."""
    sections = []
    for t in turns:
        header = "## User" if t.role == "user" else "## Assistant"
        sections.append(f"{header}\n{t.text}\n")
    return "\n".join(sections)
