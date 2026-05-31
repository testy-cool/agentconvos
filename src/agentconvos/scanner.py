"""Scan ~/.claude/projects/, ~/.codex/sessions/, and ~/.pi/agent/sessions/ for conversation logs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .parser import ConversationMeta, get_meta

_CACHE_PATH = Path(os.environ.get("USERPROFILE", Path.home())) / ".claude" / "convo-explorer" / "meta-cache.json"


def _load_cache() -> dict:
    try:
        return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")


def _get_meta_cached(path: Path, cache: dict) -> ConversationMeta | None:
    """Get metadata, using cache when file size+mtime haven't changed."""
    key = str(path)
    try:
        stat = path.stat()
        size = stat.st_size
        mtime = stat.st_mtime
    except OSError:
        return None

    cached = cache.get(key)
    if cached and cached.get("size") == size and cached.get("mtime") == mtime:
        return ConversationMeta(
            path=path,
            uuid=cached["uuid"],
            slug=cached.get("slug", ""),
            timestamp=cached.get("timestamp", ""),
            cwd=cached.get("cwd", ""),
            preview=cached.get("preview", ""),
            source=cached.get("source", "claude"),
            git_branch=cached.get("git_branch", ""),
        )

    meta = get_meta(path)
    if meta:
        cache[key] = {
            "size": size,
            "mtime": mtime,
            "uuid": meta.uuid,
            "slug": meta.slug,
            "timestamp": meta.timestamp,
            "cwd": meta.cwd,
            "preview": meta.preview,
            "source": meta.source,
            "git_branch": meta.git_branch,
        }
    return meta


def _claude_projects_dir() -> Path:
    home = Path(os.environ.get("USERPROFILE", Path.home()))
    return home / ".claude" / "projects"


def _codex_sessions_dir() -> Path:
    codex_home = os.environ.get("CODEX_HOME")
    home = Path(codex_home).expanduser() if codex_home else Path(os.environ.get("USERPROFILE", Path.home())) / ".codex"
    return home / "sessions"


def _codex_conversations_dir() -> Path:
    codex_home = os.environ.get("CODEX_HOME")
    home = Path(codex_home).expanduser() if codex_home else Path(os.environ.get("USERPROFILE", Path.home())) / ".codex"
    return home / "conversations"


def _pi_sessions_dir() -> Path:
    home = Path(os.environ.get("USERPROFILE", Path.home()))
    return home / ".pi" / "agent" / "sessions"


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


def scan_projects(
    extra_dirs: list[Path] | None = None,
    source: str | None = None,
    after: str | None = None,
    before: str | None = None,
) -> list[Project]:
    """Find all projects and their conversations, sorted by most recent first.

    Args:
        extra_dirs: Additional project directories to scan.
        source: Filter by agent — "claude", "codex", or "pi".
        after: Only include conversations after this ISO date (e.g. "2026-05-01").
        before: Only include conversations before this ISO date.
    """
    projects: list[Project] = []
    cache = _load_cache()

    # Scan Claude Code projects
    bases = [_claude_projects_dir()]
    if extra_dirs:
        bases.extend(extra_dirs)
    for base in bases:
        if not base.is_dir():
            continue
        for entry in sorted(base.iterdir()):
            if not entry.is_dir():
                continue
            jsonl_files = list(entry.glob("*.jsonl"))
            if not jsonl_files:
                continue

            convos = []
            for jf in jsonl_files:
                meta = _get_meta_cached(jf, cache)
                if meta:
                    convos.append(meta)

            convos.sort(key=lambda c: c.timestamp, reverse=True)

            display = convos[0].cwd if convos and convos[0].cwd else _folder_to_path(entry.name)
            projects.append(Project(
                folder_name=entry.name,
                display_path=display,
                conversations=convos,
            ))

    # Scan Codex sessions — group by cwd into virtual "projects"
    codex_bases = [(_codex_sessions_dir(), "*.jsonl"), (_codex_conversations_dir(), "*.json")]
    codex_convos: list[ConversationMeta] = []
    for codex_base, pattern in codex_bases:
        if not codex_base.is_dir():
            continue
        for jf in codex_base.rglob(pattern):
            meta = _get_meta_cached(jf, cache)
            if meta:
                codex_convos.append(meta)

    if codex_convos:
        # Group by cwd
        by_cwd: dict[str, list[ConversationMeta]] = {}
        for c in codex_convos:
            key = c.cwd or "(no project)"
            by_cwd.setdefault(key, []).append(c)

        for cwd, convos in by_cwd.items():
            convos.sort(key=lambda c: c.timestamp, reverse=True)
            folder = "codex:" + (Path(cwd).name if cwd and cwd != "(no project)" else "misc")
            projects.append(Project(
                folder_name=folder,
                display_path=f"[codex] {cwd}",
                conversations=convos,
            ))

    # Scan Pi sessions — group by cwd-slug subdirectory
    pi_base = _pi_sessions_dir()
    if pi_base.is_dir():
        pi_convos: list[ConversationMeta] = []
        for jf in pi_base.rglob("*.jsonl"):
            meta = _get_meta_cached(jf, cache)
            if meta:
                pi_convos.append(meta)

        by_cwd_pi: dict[str, list[ConversationMeta]] = {}
        for c in pi_convos:
            key = c.cwd or "(no project)"
            by_cwd_pi.setdefault(key, []).append(c)

        for cwd, convos in by_cwd_pi.items():
            convos.sort(key=lambda c: c.timestamp, reverse=True)
            folder = "pi:" + (Path(cwd).name if cwd and cwd != "(no project)" else "misc")
            projects.append(Project(
                folder_name=folder,
                display_path=f"[pi] {cwd}",
                conversations=convos,
            ))

    _save_cache(cache)

    # Apply source filter
    if source:
        projects = [p for p in projects if p.conversations and p.conversations[0].source == source]

    # Apply date filters to conversations within each project
    if after or before:
        filtered = []
        for p in projects:
            convos = p.conversations
            if after:
                convos = [c for c in convos if (c.timestamp or "") >= after]
            if before:
                convos = [c for c in convos if (c.timestamp or "") <= before]
            if convos:
                filtered.append(Project(
                    folder_name=p.folder_name,
                    display_path=p.display_path,
                    conversations=convos,
                ))
        projects = filtered

    # Sort projects by most recent conversation
    projects.sort(
        key=lambda p: p.conversations[0].timestamp if p.conversations else "",
        reverse=True,
    )
    return projects


def resolve_ids(ids: list[str], extra_dirs: list[Path] | None = None) -> list[Path]:
    """Resolve conversation IDs (UUID prefix or slug) to JSONL file paths.

    Matches against uuid (prefix match) and slug (exact or substring).
    Returns list of resolved paths. Prints warnings for unresolved IDs.
    """
    projects = scan_projects(extra_dirs=extra_dirs)
    all_convos = [c for p in projects for c in p.conversations]

    resolved = []
    for query in ids:
        q = query.lower().strip()
        # Try exact UUID prefix match first
        matches = [c for c in all_convos if c.uuid.lower().startswith(q)]
        if not matches:
            # Try slug match (exact or substring)
            matches = [c for c in all_convos if c.slug and q in c.slug.lower()]
        if not matches:
            print(f"Warning: no conversation found for '{query}'")
        elif len(matches) > 1:
            print(f"Warning: '{query}' matches {len(matches)} conversations, using all:")
            for m in matches:
                name = m.slug or m.uuid[:8]
                print(f"  {m.timestamp[:10]}  {name}  {m.cwd}")
            resolved.extend(m.path for m in matches)
        else:
            resolved.append(matches[0].path)
    return resolved
