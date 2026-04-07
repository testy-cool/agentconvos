"""Scan ~/.claude/projects/ for Claude Code conversation logs."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .parser import ConversationMeta, get_meta


def _claude_projects_dir() -> Path:
    home = Path(os.environ.get("USERPROFILE", Path.home()))
    return home / ".claude" / "projects"


def _folder_to_path(folder_name: str) -> str:
    """Best-effort decode of folder name back to a path.

    Claude Code encodes paths by replacing every non-alphanumeric char with '-'.
    This is lossy, but we can recover the drive prefix and use \\ for separators.
    E.g. 'F--code-ailookup' -> 'F:\\code\\ailookup'
    """
    import re
    # Match drive prefix: single letter followed by --
    m = re.match(r"^([A-Za-z])--(.*)$", folder_name)
    if m:
        drive = m.group(1).upper()
        rest = m.group(2)
        # Replace remaining - with \ for display
        path = rest.replace("-", "\\")
        return f"{drive}:\\{path}"
    # No drive letter — just replace dashes with backslashes
    return folder_name.replace("-", "\\")


@dataclass
class Project:
    folder_name: str
    display_path: str
    conversations: list[ConversationMeta] = field(default_factory=list)


def scan_projects() -> list[Project]:
    """Find all projects and their conversations, sorted by most recent first."""
    base = _claude_projects_dir()
    if not base.is_dir():
        return []

    projects: list[Project] = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        jsonl_files = list(entry.glob("*.jsonl"))
        if not jsonl_files:
            continue

        convos = []
        for jf in jsonl_files:
            meta = get_meta(jf)
            if meta:
                convos.append(meta)

        convos.sort(key=lambda c: c.timestamp, reverse=True)

        # Use cwd from the most recent conversation as the real path (if available)
        display = convos[0].cwd if convos and convos[0].cwd else _folder_to_path(entry.name)
        projects.append(Project(
            folder_name=entry.name,
            display_path=display,
            conversations=convos,
        ))

    # Sort projects by most recent conversation
    projects.sort(
        key=lambda p: p.conversations[0].timestamp if p.conversations else "",
        reverse=True,
    )
    return projects
