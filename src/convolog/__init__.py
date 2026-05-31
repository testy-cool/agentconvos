"""convolog: discover, parse, and query AI coding agent conversations.

Library API — import these directly:

    from convolog import scan_projects, parse_jsonl, get_meta, search

Discovers sessions from Claude Code (~/.claude/projects/),
Codex (~/.codex/sessions/), and Pi (~/.pi/agent/sessions/).
"""

from .parser import (
    ConversationMeta,
    ConversationStats,
    SearchHit,
    Turn,
    get_meta,
    get_stats,
    parse_jsonl,
    search_conversations as search,
    to_markdown,
)
from .scanner import Project, scan_projects

__all__ = [
    "ConversationMeta",
    "ConversationStats",
    "Project",
    "SearchHit",
    "Turn",
    "get_meta",
    "get_stats",
    "parse_jsonl",
    "scan_projects",
    "search",
    "to_markdown",
]
