"""Textual TUI for browsing conversations across local coding agents."""

from __future__ import annotations

import os
import re
import shutil
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from unicodedata import east_asian_width

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Markdown,
    Static,
    TextArea,
    Tree,
)
from textual.widgets.tree import TreeNode
from textual.worker import get_current_worker

from .analyzer import DEFAULT_MODEL, MODELS, MULTI_PROMPT, SINGLE_PROMPT
from .parser import (
    DETAIL_TEXT,
    ConversationMeta,
    ConversationStats,
    conversation_signature,
    get_stats,
    parse_jsonl,
    parse_search_terms,
    to_markdown,
)
from .scanner import Project, scan_projects
from .search_index import ConversationSearchIndex, IndexSyncStats

_SOURCE_STYLE = {
    "claude": ("Claude Code", "bold #cc5500"),
    "codex": ("Codex", "bold #00cc66"),
    "pi": ("Pi", "bold #7c6fff"),
    "agy": ("Agy", "bold #00a3ff"),
    "opencode": ("OpenCode", "bold #ffaa00"),
    "clihow": ("Clihow", "bold #a78bfa"),
}
_SOURCE_ORDER = ["claude", "codex", "pi", "agy", "opencode", "clihow"]
_RESUMABLE_SOURCES = frozenset(_SOURCE_ORDER)
_FILTER_DEBOUNCE_SECONDS = 0.18
_HIGHLIGHT_DEBOUNCE_SECONDS = 0.12
_FILTER_RESULT_LIMIT = 200


def _group_key(display_path: str, source: str) -> tuple[str, str]:
    """Return (group_name, relative_label) for a project path."""
    source_prefix = f"[{source}] "
    if display_path.startswith(source_prefix):
        display_path = display_path[len(source_prefix):]

    home = str(Path.home())
    if display_path.startswith(home):
        display_path = "~" + display_path[len(home):]

    if display_path.startswith("~/Work"):
        rest = display_path[len("~/Work/"):]
        return "~/Work", rest or display_path
    if display_path.startswith("~/."):
        parts = display_path.split("/")
        group = "/".join(parts[:2])  # e.g. "~/.claude"
        rest = "/".join(parts[2:])
        return group, rest or display_path
    if display_path == "~":
        return "~", "~"
    if display_path.startswith("~/"):
        return "~", display_path[len("~/"):]

    return "Other", display_path


def _project_real_path(project: Project, convos: list[ConversationMeta] | None = None) -> str:
    """Return the real cwd for source-prefixed virtual projects."""
    if convos:
        for convo in convos:
            if convo.cwd:
                return convo.cwd
    if project.conversations:
        for convo in project.conversations:
            if convo.cwd:
                return convo.cwd
    display_path = project.display_path
    for prefix in ("[codex] ", "[pi] ", "[agy] ", "[opencode] ", "[clihow] "):
        if display_path.startswith(prefix):
            return display_path[len(prefix):]
    return display_path


def _is_current_project(cwd: str, project_path: str) -> bool:
    if not project_path or project_path == "(no project)":
        return False
    project_path = os.path.realpath(os.path.expanduser(project_path))
    cwd = os.path.realpath(cwd)
    return cwd == project_path


def _short_path(path: str) -> str:
    """Write a path under the home directory as ~/rest, the way the tree does."""
    home = str(Path.home())
    if path == home:
        return "~"
    if path.startswith(home + "/"):
        return "~" + path[len(home):]
    return path


def _file_size_label(size_bytes: int) -> str:
    """Transcript size on disk. Deliberately not a token count: the raw file
    holds tool traffic that a normal read strips, so tokens read one to two
    orders of magnitude high and talk people out of readable conversations."""
    if size_bytes >= 1_000_000_000:
        return f"{size_bytes / 1_000_000_000:.1f}GB"
    if size_bytes >= 1_000_000:
        return f"{size_bytes / 1_000_000:.1f}MB"
    if size_bytes >= 1000:
        return f"{size_bytes // 1000}KB"
    return f"{size_bytes}B"


def _fmt_ts(ts: str, date_only: bool = False) -> str:
    """Format ISO timestamp: '2026-05-17 14:30' or '2026-05-17'."""
    if not ts:
        return ""
    if date_only or len(ts) < 16:
        return ts[:10]
    return ts[:16].replace("T", " ")


def _turns(count: int) -> str:
    return "1 turn" if count == 1 else f"{count} turns"


def _stats_line(path: Path) -> str:
    """Model and token cost for the preview header, empty when unavailable."""
    try:
        stats: ConversationStats = get_stats(path)
    except Exception:
        # The header is a nicety; a transcript it cannot summarize must not
        # take the transcript itself off the screen.
        return ""
    parts = []
    if stats.model:
        parts.append(f"**Model:** {stats.model}")
    total_tokens = stats.input_tokens + stats.output_tokens
    if total_tokens:
        parts.append(f"**Tokens:** {total_tokens:,}")
    if stats.tool_calls:
        parts.append(f"**Tool calls:** {stats.tool_calls:,}")
    return "  \n" + " · ".join(parts) if parts else ""


def _cell_len(text: str) -> int:
    """Terminal columns the text occupies; CJK and emoji take two."""
    return sum(2 if east_asian_width(char) in "WF" else 1 for char in text)


def _elide(text: str, width: int) -> str:
    """Trim to width, marking the cut so a clipped row is obvious."""
    text = text.rstrip()
    if _cell_len(text) <= width:
        return text
    budget = max(1, width - 1)  # the ellipsis needs a column
    kept: list[str] = []
    used = 0
    for char in text:
        size = 2 if east_asian_width(char) in "WF" else 1
        if used + size > budget:
            break
        kept.append(char)
        used += size
    return "".join(kept).rstrip() + "…"


def _fmt_nav_ts(ts: str) -> str:
    """Format an ISO timestamp for a narrow navigation row."""
    formatted = _fmt_ts(ts)
    return formatted[5:] if len(formatted) >= 16 else formatted


def _export_stem(meta: ConversationMeta) -> str:
    """Build a human-readable filename stem from conversation metadata.

    Priority: summary > slug > preview-derived > project+uuid fragment.
    """
    try:
        from .summarize import load_summaries
        summary = load_summaries().get(meta.uuid)
        if summary:
            words = summary.strip("- ").split()[:6]
            return " ".join(words)
    except Exception:
        pass

    if meta.slug:
        return meta.slug

    # Derive from first user message
    if meta.preview:
        # Take first ~50 chars, strip non-alphanum, collapse whitespace
        raw = meta.preview[:50].lower()
        raw = re.sub(r"[^a-z0-9\s-]", "", raw)
        raw = re.sub(r"\s+", "-", raw.strip())
        raw = raw.strip("-")
        if len(raw) >= 4:
            return raw

    # Fallback: project name + short uuid
    proj = Path(meta.cwd).name if meta.cwd else ""
    short_id = meta.uuid[:8]
    return f"{proj}-{short_id}" if proj else short_id


def _conversation_size(path: Path) -> int:
    try:
        return conversation_signature(path)[0]
    except OSError:
        return 0


def _match_ranges(text: str, terms: list[str]) -> list[tuple[int, int]]:
    """Return non-overlapping ranges for every case-insensitive term match."""
    candidates: list[tuple[int, int]] = []
    for term in sorted(set(terms), key=len, reverse=True):
        if not term:
            continue
        candidates.extend(
            (match.start(), match.end())
            for match in re.finditer(re.escape(term), text, flags=re.IGNORECASE)
        )

    ranges: list[tuple[int, int]] = []
    for start, end in sorted(candidates, key=lambda item: (item[0], -(item[1] - item[0]))):
        if ranges and start < ranges[-1][1]:
            continue
        if ranges and start == ranges[-1][1]:
            # Touching matches, as in "rate" + "limit" inside RateLimit. Two
            # separate bold spans would emit **** and render as asterisks.
            ranges[-1] = (ranges[-1][0], end)
            continue
        ranges.append((start, end))
    return ranges


def _highlight_matches(text: str, terms: list[str], style: str = "") -> Text:
    """Render text with high-contrast search matches for the tree."""
    rendered = Text(text, style=style)
    for start, end in _match_ranges(text, terms):
        rendered.stylize("bold #111111 on #ffd75f", start, end)
    return rendered


def _matching_excerpt(text: str, terms: list[str], width: int = 90) -> str:
    """Return compact context around the earliest matching search term."""
    flattened = re.sub(r"\s+", " ", text).strip()
    if not flattened:
        return ""
    ranges = _match_ranges(flattened, terms)
    if not ranges:
        return flattened if len(flattened) <= width else flattened[: max(1, width - 1)].rstrip() + "…"

    if len(flattened) <= width:
        return flattened

    # Only spend width on the ellipses that actually get rendered.
    match_start, _ = ranges[0]
    start = max(0, match_start - width // 3)
    content_width = width - (1 if start else 0)
    end = start + content_width
    if end >= len(flattened):
        end = len(flattened)
        start = max(0, end - content_width)
        start = max(0, end - (width - (1 if start else 0)))
    else:
        end = start + content_width - 1
    prefix = "…" if start else ""
    suffix = "…" if end < len(flattened) else ""
    return prefix + flattened[start:end].strip() + suffix


def _escape_markdown_inline(text: str) -> str:
    return re.sub(r"([\\`*_\[\]])", r"\\\1", text)


def _highlight_markdown(text: str, terms: list[str]) -> str:
    """Make query matches visibly bold while safely embedding preview text."""
    ranges = _match_ranges(text, terms)
    if not ranges:
        return _escape_markdown_inline(text)

    parts: list[str] = []
    cursor = 0
    for start, end in ranges:
        parts.append(_escape_markdown_inline(text[cursor:start]))
        parts.append(f"**{_escape_markdown_inline(text[start:end])}**")
        cursor = end
    parts.append(_escape_markdown_inline(text[cursor:]))
    return "".join(parts)


def _export_date(meta: ConversationMeta) -> str:
    """Return MM-DD-YYYY date string from conversation timestamp."""
    if meta.timestamp and len(meta.timestamp) >= 10:
        try:
            dt = datetime.fromisoformat(meta.timestamp[:10])
            return dt.strftime("%m-%d-%Y")
        except ValueError:
            pass
    return datetime.now().strftime("%m-%d-%Y")


def _export_filename(meta: ConversationMeta, custom_name: str = "") -> str:
    """Build export filename: MM-DD-YYYY-{name}.md"""
    date = _export_date(meta)
    name = custom_name.strip() if custom_name else _export_stem(meta)
    name = re.sub(r"[^a-zA-Z0-9_\s-]", "", name)
    name = re.sub(r"\s+", "-", name.strip()).strip("-")
    if not name:
        name = _export_stem(meta)
    return f"{date}-{name}.md"


class ExportNameScreen(ModalScreen[str]):
    """Prompt for an optional export name."""

    CSS = """
    ExportNameScreen { align: center middle; }
    #export-dialog { width: 60; height: auto; max-height: 10; border: thick $accent; background: $surface; padding: 1 2; }
    #export-name-input { width: 100%; }
    #export-hint { color: $text-muted; margin-bottom: 1; }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="export-dialog"):
            yield Static("Export name (Enter to skip):", id="export-hint")
            yield Input(placeholder=self._default_name, id="export-name-input")

    def __init__(self, default_name: str = "") -> None:
        super().__init__()
        self._default_name = default_name

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def key_escape(self) -> None:
        self.dismiss("")


class ResumeScreen(ModalScreen[bool]):
    """Confirm exactly which native session and directory will be resumed."""

    CSS = """
    ResumeScreen { align: center middle; }
    #resume-dialog { width: 76; height: auto; border: thick $accent; background: $surface; padding: 1 2; }
    #resume-title { text-style: bold; margin-bottom: 1; }
    #resume-details { color: $text-muted; margin-bottom: 1; }
    #resume-actions { height: 3; align-horizontal: right; }
    #resume-actions Button { min-width: 16; margin-left: 1; }
    """

    BINDINGS = [
        Binding("escape", "dismiss_resume", "Cancel", priority=True),
        Binding("enter", "confirm_resume", "Resume", priority=True),
    ]

    def __init__(self, meta: ConversationMeta) -> None:
        super().__init__()
        self.meta = meta

    def compose(self) -> ComposeResult:
        agent_name = _SOURCE_STYLE.get(self.meta.source, (self.meta.source.title(), ""))[0]
        with Vertical(id="resume-dialog"):
            yield Static("Resume this session?", id="resume-title")
            yield Static(_resume_description(self.meta), id="resume-details", markup=False)
            with Horizontal(id="resume-actions"):
                yield Button("Cancel", id="resume-cancel")
                yield Button(f"Resume in {agent_name}", id="resume-confirm", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#resume-confirm", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "resume-confirm")

    def action_confirm_resume(self) -> None:
        self.dismiss(True)

    def action_dismiss_resume(self) -> None:
        self.dismiss(False)


class HelpScreen(ModalScreen[None]):
    """Every key in one place; the footer only shows the headliners."""

    CSS = """
    HelpScreen { align: center middle; }
    #help-dialog { width: 64; height: auto; max-height: 90%; border: thick $accent; background: $surface; padding: 1 2; }
    #help-title { text-style: bold; margin-bottom: 1; }
    #help-hint { color: $text-muted; margin-top: 1; }
    """

    BINDINGS = [
        Binding("escape", "dismiss_help", "Close", priority=True),
        Binding("q", "dismiss_help", "Close"),
        Binding("question_mark", "dismiss_help", "Close"),
    ]

    KEYS: list[tuple[str, str]] = [
        ("/", "Search; ↑ ↓ walk the results while typing"),
        ("Enter", "Open the highlighted conversation"),
        ("v", "Read the whole transcript, not just the tail"),
        ("Tab", "Switch panel"),
        ("r", "Resume the session in its agent"),
        ("h", "Handoff into a new session"),
        ("e", "Export markdown"),
        ("c", "Export selected as one file"),
        ("a", "Analyze with Gemini"),
        ("m", "Cycle the Gemini model"),
        ("p", "Edit the analysis prompt"),
        ("s", "Select / deselect for bulk actions"),
        ("Ctrl+A", "Select all"),
        ("Ctrl+D", "Deselect all"),
        ("y", "Copy the session ID"),
        ("o", "Open the conversation's folder"),
        ("Esc", "Clear the search, cancel"),
        ("q", "Quit"),
        ("?", "This help"),
    ]

    def compose(self) -> ComposeResult:
        rows = "\n".join(f"  {key:<8} {what}" for key, what in self.KEYS)
        with Vertical(id="help-dialog"):
            yield Static("Keys", id="help-title")
            yield Static(rows, id="help-body", markup=False)
            yield Static("Esc closes", id="help-hint")

    def action_dismiss_help(self) -> None:
        self.dismiss(None)


ANALYSES_DIR = Path(os.environ.get("USERPROFILE", Path.home())) / ".claude" / "convo-explorer" / "analyses"


def _analysis_filename(project: str, count: int) -> str:
    """Generate human-readable analysis filename."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Clean project name for filename
    proj = project.replace("\\", "-").replace("/", "-").replace(":", "").strip("-")
    if len(proj) > 40:
        proj = proj[-40:]
    return f"{ts}-{proj}-{count}-convos.md"


@dataclass
class NodeData:
    """Attached to each tree node to identify what it represents."""
    kind: str  # "project" | "convo"
    project: Project | None = None
    meta: ConversationMeta | None = None
    selected: bool = False  # for multi-select
    is_cwd: bool = False


@dataclass(frozen=True)
class _FilterResult:
    projects: list[Project]
    indexed_matches: dict[str, str]
    filtered_count: int


class SearchInput(Input):
    """The search box; arrow keys walk the result tree without leaving it."""

    def _tree(self) -> Tree:
        return self.app.query_one("#nav-tree", Tree)

    def key_down(self) -> None:
        self._tree().action_cursor_down()

    def key_up(self) -> None:
        self._tree().action_cursor_up()

    def key_pagedown(self) -> None:
        self._tree().action_page_down()

    def key_pageup(self) -> None:
        self._tree().action_page_up()


class ConvoExplorer(App):
    CSS = """
    Screen {
        background: #091017;
        color: #d6e1ea;
    }
    Header {
        background: #0f1821;
        color: #e8f0f6;
    }
    Footer {
        background: #0d151d;
        color: #8da0b1;
    }
    #main { height: 1fr; background: #091017; }
    #sidebar {
        width: 42%;
        min-width: 34;
        max-width: 80;
        background: #0c141c;
        border-right: solid #22313f;
    }
    #resize-handle {
        width: 1;
        height: 1fr;
        background: #16232e;
        color: #426275;
    }
    #resize-handle:hover { background: #2dd4bf; color: #081014; }
    #content { width: 1fr; background: #091017; }
    #filter-input {
        height: 3;
        margin: 1 1 0 1;
        padding: 0 1;
        border: tall #2a3a48;
        background: #111b25;
        color: #e7eef4;
    }
    #filter-input:hover { border: tall #456276; }
    #filter-input:focus {
        border: tall #2dd4bf;
        background: #10232a;
    }
    #filter-input > .input--placeholder { color: #71889a; }
    #nav-tree {
        height: 1fr;
        padding: 1 1;
        background: #0c141c;
        overflow-x: hidden;
        scrollbar-color: #385467;
        scrollbar-color-hover: #2dd4bf;
        scrollbar-color-active: #5eead4;
    }
    #nav-tree:focus { background: #0e1822; }
    #preview-scroll {
        height: 1fr;
        background: #091017;
        scrollbar-color: #385467;
        scrollbar-color-hover: #2dd4bf;
        scrollbar-color-active: #5eead4;
    }
    #preview { padding: 1 3 3 3; color: #d6e1ea; }
    #status-rail {
        height: 1;
        background: #0d151d;
    }
    #status-bar { width: 1fr; height: 1; color: #91a3b2; padding: 0 1; }
    #index-status {
        width: auto;
        height: 1;
        min-width: 22;
        content-align: right middle;
        background: #101c25;
        padding: 0 1;
    }
    .panel-title {
        height: 2;
        content-align: left middle;
        background: #0f1923;
        color: #91a7b8;
        border-bottom: solid #1d2b37;
        padding: 0 2;
        text-style: bold;
    }
    Tree { scrollbar-size: 1 1; }
    #prompt-editor { height: 1fr; }
    #prompt-panel { height: 1fr; }
    #prompt-bar { dock: bottom; height: 3; background: #101923; }
    #prompt-bar Button { width: 1fr; margin: 0 1; }
    """

    # Footer space is scarce; it shows these in order until it runs out.
    # The flagship actions come first, the bulk keys live behind "?".
    BINDINGS = [
        Binding("slash", "search", "Search", priority=False),
        Binding("v", "view_full", "Full text", priority=False),
        Binding("r", "resume", "Resume", priority=False),
        Binding("h", "handoff", "Handoff", priority=False),
        Binding("e", "export", "Export", priority=False),
        Binding("a", "analyze", "Analyze", priority=False),
        Binding("s", "toggle_select", "Select", priority=False),
        Binding("tab", "toggle_focus", "Switch panel", priority=True),
        Binding("q", "quit", "Quit", priority=False),
        Binding("question_mark", "help", "Help", priority=False),
        Binding("y", "copy_id", "Copy session ID", show=False),
        Binding("c", "export_concat", "Export combined", show=False),
        Binding("ctrl+a", "select_all", "Select all", show=False),
        Binding("ctrl+d", "deselect_all", "Deselect all", show=False),
        Binding("m", "cycle_model", "Model", show=False),
        Binding("p", "edit_prompt", "Edit prompt", show=False),
        Binding("o", "open_folder", "Open folder", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    TITLE = "agentconvos"
    SUB_TITLE = "conversation search cockpit"

    def __init__(
        self,
        extra_dirs: list[Path] | None = None,
        search_index: ConversationSearchIndex | None = None,
        source: str | None = None,
        after: str | None = None,
        before: str | None = None,
    ) -> None:
        super().__init__()
        self.projects: list[Project] = []
        self._extra_dirs = extra_dirs
        self._source = source
        self._after = after
        self._before = before
        self.current_meta: ConversationMeta | None = None
        self._dragging_sidebar = False
        self._model_index = 0
        self.gemini_model = MODELS[0]
        self.custom_single_prompt: str = SINGLE_PROMPT
        self.custom_multi_prompt: str = MULTI_PROMPT
        self._editing_prompt: str = "single"  # which prompt is being edited
        self._analyzing = False
        self._last_action: str = ""  # "analysis" or "export"
        self._resume_meta: ConversationMeta | None = None  # set when user wants to resume
        self._handoff_meta: ConversationMeta | None = None  # set when user wants to handoff
        self._conversation_index = search_index or ConversationSearchIndex()
        self._indexing = False
        self._index_progress = IndexSyncStats(total=0)
        self._summaries: dict[str, str] = {}
        self._analyzed_set: set[str] = set()
        self._filter_timer = None
        self._filter_query = ""
        self._filter_generation = 0
        self._tree_width = 0
        self._width_timer = None
        self._highlight_timer = None
        self._previewed_meta: ConversationMeta | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with Vertical(id="sidebar"):
                yield Static("HISTORY", classes="panel-title", id="left-title")
                yield SearchInput(
                    placeholder="Search conversations…",
                    id="filter-input",
                )
                yield Tree("Conversations", id="nav-tree")
            yield Static("│", id="resize-handle")
            with Vertical(id="content"):
                yield Static("CONVERSATION", classes="panel-title", id="right-title")
                with VerticalScroll(id="preview-scroll"):
                    yield Markdown(
                        """# Pick up the thread

Type to find a session by title, prompt, path, branch, or transcript text.

Enter opens the first result. R resumes it. S marks sessions for bulk actions.

History appears immediately. Full-text results arrive live while indexing runs in the background.
""",
                        id="preview",
                    )
                with Vertical(id="prompt-panel"):
                    yield TextArea(id="prompt-editor", language="markdown")
                    with Horizontal(id="prompt-bar"):
                        yield Button("Save & Close", id="prompt-save", variant="primary")
                        yield Button("Switch Single/Multi", id="prompt-switch", variant="default")
                        yield Button("Reset Default", id="prompt-reset", variant="warning")
        with Horizontal(id="status-rail"):
            yield Static("Discovering conversation history…", id="status-bar")
            yield Static("INDEX · STARTING", id="index-status")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#prompt-panel").display = False
        tree = self.query_one("#nav-tree", Tree)
        tree.show_root = False
        tree.guide_depth = 3
        tree.loading = True
        self.query_one("#status-bar", Static).update(" Discovering conversation history…")
        self.query_one("#left-title", Static).update(" HISTORY · loading")
        self.load_projects()
        self.query_one("#filter-input", Input).focus()

    def on_resize(self, event) -> None:
        """Row labels are sized to the tree, so a resize has to redraw them."""
        self._rerender_for_width()

    def _rerender_for_width(self) -> None:
        try:
            tree = self.query_one("#nav-tree", Tree)
        except NoMatches:
            return
        width = tree.container_size.width
        if not width or width == self._tree_width or not self.projects:
            return
        self._tree_width = width
        if self._width_timer is not None:
            self._width_timer.stop()
        self._width_timer = self.set_timer(
            0.1, lambda: self._populate_tree(self.projects, self._filter_query)
        )

    # --- Resize handle ---

    def on_mouse_down(self, event) -> None:
        handle = self.query_one("#resize-handle")
        if handle.region.contains(event.screen_x, event.screen_y):
            self._dragging_sidebar = True

    def on_mouse_move(self, event) -> None:
        if self._dragging_sidebar:
            sidebar = self.query_one("#sidebar")
            new_width = max(20, min(int(event.screen_x), self.size.width - 20))
            sidebar.styles.width = new_width

    def on_mouse_up(self, event) -> None:
        if self._dragging_sidebar:
            self._dragging_sidebar = False
            self.call_after_refresh(self._rerender_for_width)

    # --- Data loading ---

    @work(thread=True)
    def load_projects(self) -> None:
        projects = scan_projects(
            extra_dirs=self._extra_dirs,
            source=self._source,
            after=self._after,
            before=self._before,
        )
        self.call_from_thread(self._projects_loaded, projects)

    def _projects_loaded(self, projects: list[Project]) -> None:
        self.query_one("#nav-tree", Tree).loading = False
        try:
            from .summarize import load_summaries

            self._summaries = load_summaries()
        except Exception:
            self._summaries = {}
        self._analyzed_set = self._get_analyzed_set()
        self._populate_tree(projects)
        conversations = [
            conversation
            for project in projects
            for conversation in project.conversations
        ]
        index_conversations = conversations
        if self._source or self._after or self._before:
            complete_projects = scan_projects(extra_dirs=self._extra_dirs)
            index_conversations = [
                conversation
                for project in complete_projects
                for conversation in project.conversations
            ]
        self._indexing = True
        self._set_index_progress(IndexSyncStats(total=len(index_conversations)))
        self.build_search_index(index_conversations)
        if self._filter_query:
            self._schedule_filter()

    @work(thread=True, exclusive=True, group="search-index")
    def build_search_index(self, conversations: list[ConversationMeta]) -> None:
        worker = get_current_worker()
        last_reported = -10

        def report(progress: IndexSyncStats) -> None:
            nonlocal last_reported
            if worker.is_cancelled:
                return
            if progress.checked == progress.total or progress.checked - last_reported >= 10:
                last_reported = progress.checked
                self.call_from_thread(self._set_index_progress, progress)

        try:
            result = self._conversation_index.sync(
                conversations,
                on_progress=report,
                should_cancel=lambda: worker.is_cancelled,
            )
        except Exception as error:
            if not worker.is_cancelled:
                self.call_from_thread(self._index_failed, str(error))
            return
        if not worker.is_cancelled:
            self.call_from_thread(self._index_finished, result)

    def _set_index_progress(self, progress: IndexSyncStats) -> None:
        self._index_progress = progress
        label = Text("INDEXING ", style="bold #f0b35a")
        label.append(f"{progress.checked}/{progress.total}", style="#d8c39b")
        try:
            status = self.query_one("#index-status", Static)
            search_input = self.query_one("#filter-input", Input)
        except NoMatches:  # index sync outlived the app; nothing to update
            return
        status.update(label)
        query = search_input.value.strip()
        if query and progress.checked:
            self._filter_query = query
            self._schedule_filter()

    def _index_finished(self, progress: IndexSyncStats) -> None:
        self._indexing = False
        self._index_progress = progress
        label = Text("INDEX READY ", style="bold #55d6a9")
        label.append(f"{progress.total}", style="#a7e8d2")
        if progress.failed:
            label.append(f" · {progress.failed} skipped", style="#f0b35a")
        try:
            status = self.query_one("#index-status", Static)
            search_input = self.query_one("#filter-input", Input)
        except NoMatches:  # index sync outlived the app; nothing to update
            return
        status.update(label)
        query = search_input.value.strip()
        if query:
            self._filter_query = query
            self._schedule_filter()

    def _index_failed(self, message: str) -> None:
        self._indexing = False
        try:
            status = self.query_one("#index-status", Static)
        except NoMatches:  # index sync outlived the app; nothing to update
            return
        status.update(Text("INDEX UNAVAILABLE", style="bold #ff7b72"))
        self.notify(f"Search index failed: {message}", severity="warning")

    def _get_analyzed_set(self) -> set[str]:
        """Scan analyses dir for previously analyzed project/convo names."""
        analyzed = set()
        if ANALYSES_DIR.is_dir():
            for f in ANALYSES_DIR.iterdir():
                if f.suffix == ".md":
                    # filename: 2026-04-07_01-32-24-projectname-1-convos.md
                    analyzed.add(f.stem.lower())
        return analyzed

    def _is_analyzed(self, name: str) -> bool:
        """Check if any analysis file contains this name."""
        name_lower = name.lower().replace("\\", " ").replace("/", " ").replace("-", " ")
        return any(name_lower in a.replace("-", " ") for a in self._analyzed_set)

    def _prepare_filter(
        self,
        projects: list[Project],
        filter_text: str,
        summaries: dict[str, str] | None = None,
    ) -> _FilterResult:
        """Search and select the conversations that should be rendered."""
        terms = parse_search_terms(filter_text)
        if not terms:
            return _FilterResult(
                projects=list(projects),
                indexed_matches={},
                filtered_count=sum(len(project.conversations) for project in projects),
            )

        try:
            indexed_matches = self._conversation_index.search(filter_text)
        except Exception:
            indexed_matches = {}

        summaries = self._summaries if summaries is None else summaries
        filtered_projects: list[Project] = []
        filtered_count = 0
        for project in projects:
            matches = []
            for conversation in project.conversations:
                metadata = "\n".join(
                    (
                        conversation.slug or "",
                        conversation.uuid,
                        conversation.preview or "",
                        summaries.get(conversation.uuid, ""),
                        project.display_path,
                    )
                ).casefold()
                if conversation.uuid in indexed_matches or all(
                    term in metadata for term in terms
                ):
                    matches.append(conversation)

            if not matches:
                continue
            remaining = _FILTER_RESULT_LIMIT - filtered_count
            if remaining <= 0:
                break
            visible = matches[:remaining]
            filtered_projects.append(
                Project(project.folder_name, project.display_path, visible)
            )
            filtered_count += len(visible)

        return _FilterResult(
            projects=filtered_projects,
            indexed_matches=indexed_matches,
            filtered_count=filtered_count,
        )

    def _schedule_filter(self) -> None:
        """Coalesce filter changes before starting the background search."""
        self._filter_generation += 1
        if self._filter_timer is not None:
            self._filter_timer.stop()
        self._filter_timer = self.set_timer(
            _FILTER_DEBOUNCE_SECONDS,
            self._run_latest_filter,
        )

    def _run_latest_filter(self) -> None:
        self._filter_timer = None
        self._run_filter_in_background(
            list(self.projects),
            self._filter_query,
            self._filter_generation,
        )

    @work(thread=True, exclusive=True, group="filter")
    def _run_filter_in_background(
        self,
        projects: list[Project],
        query: str,
        generation: int,
    ) -> None:
        worker = get_current_worker()
        result = self._prepare_filter(projects, query, dict(self._summaries))
        if not worker.is_cancelled:
            self.call_from_thread(self._filter_finished, query, generation, result)

    def _filter_finished(
        self,
        query: str,
        generation: int,
        result: _FilterResult,
    ) -> None:
        """Apply a worker result only if it still matches the current input."""
        if generation != self._filter_generation or query != self._filter_query:
            return

        self._populate_tree(
            self.projects,
            filter_text=query,
            filter_result=result,
        )
        search_input = self.query_one("#filter-input", Input)
        if query:
            if search_input.has_focus or self.current_meta is None:
                self._show_search_summary(query, result.filtered_count)
        elif self.current_meta:
            self.request_preview(self.current_meta)

    def _populate_tree(
        self,
        projects: list[Project],
        filter_text: str = "",
        *,
        filter_result: _FilterResult | None = None,
    ) -> None:
        if filter_result is None:
            filter_result = self._prepare_filter(projects, filter_text)
        with self.batch_update():
            self._render_tree(projects, filter_text, filter_result)

    def _render_tree(
        self,
        projects: list[Project],
        filter_text: str,
        filter_result: _FilterResult,
    ) -> None:
        self.projects = projects
        tree = self.query_one(Tree)
        tree.clear()
        tree.root.data = None
        terms = parse_search_terms(filter_text)
        indexed_matches = filter_result.indexed_matches

        summaries = self._summaries

        cwd = os.getcwd()

        # Every row costs guide_depth columns per level of nesting before its
        # label starts, plus the tree's own horizontal padding. Anything wider
        # than that is invisible and only makes the tree scroll sideways.
        viewport = tree.container_size.width or (self.query_one("#sidebar").size.width - 4)
        def budget(depth: int) -> int:
            return max(16, viewport - 2 - tree.guide_depth * depth)

        # Group: source → path_group → [(rel_label, proj, convos)]
        by_source: dict[str, dict[str, list[tuple[str, Project, list]]]] = {}
        filtered_count = filter_result.filtered_count

        for proj in filter_result.projects:
            convos = proj.conversations
            if not convos:
                continue

            source = convos[0].source if convos else "claude"
            gkey, rel_label = _group_key(proj.display_path, source)
            by_source.setdefault(source, {}).setdefault(gkey, []).append(
                (rel_label, proj, convos)
            )

        self.query_one("#left-title", Static).update(
            f" {'RESULTS' if terms else 'HISTORY'} · {filtered_count}"
        )

        # Find which source/group contains the cwd project
        cwd_source = None
        cwd_path_group = None
        for source, path_groups in by_source.items():
            for gkey, items in path_groups.items():
                for _, p, cv in items:
                    if _is_current_project(cwd, _project_real_path(p, cv)):
                        cwd_source = source
                        cwd_path_group = gkey
                        break
                if cwd_source:
                    break
            if cwd_source:
                break

        # Order sources: cwd source first, then standard order
        source_order = []
        if cwd_source:
            source_order.append(cwd_source)
        for s in _SOURCE_ORDER:
            if s not in source_order and s in by_source:
                source_order.append(s)

        total_projects = 0
        for source in source_order:
            name, style = _SOURCE_STYLE.get(source, (source, "bold"))
            path_groups = by_source[source]
            src_count = sum(
                len(cv) for items in path_groups.values() for _, _, cv in items
            )

            src_label = Text()
            src_label.append(name, style)
            src_label.append(f"  ({src_count})", "dim")

            is_cwd_source = source == cwd_source
            source_node = tree.root.add(
                src_label,
                data=NodeData(kind="group"),
                expand=is_cwd_source or bool(terms),
            )

            # Order path groups: cwd group first, ~/Work, then sorted, Other last
            pg_order: list[str] = []
            if is_cwd_source and cwd_path_group and cwd_path_group in path_groups:
                pg_order.append(cwd_path_group)
            if "~/Work" in path_groups and "~/Work" not in pg_order:
                pg_order.append("~/Work")
            for k in sorted(path_groups.keys()):
                if k not in pg_order and k != "Other":
                    pg_order.append(k)
            if "Other" in path_groups and "Other" not in pg_order:
                pg_order.append("Other")

            for gkey in pg_order:
                items = path_groups[gkey]
                has_cwd = any(
                    _is_current_project(cwd, _project_real_path(p, cv))
                    for _, p, cv in items
                )

                pg_node = source_node.add(
                    _highlight_matches(_elide(gkey, budget(1)), terms),
                    data=NodeData(kind="group"),
                    expand=has_cwd or bool(terms),
                )

                # Newest activity first, with the current directory's project
                # pinned on top. Alphabetical order buries active projects.
                ordered = sorted(
                    items,
                    key=lambda item: item[2][0].timestamp if item[2] else "",
                    reverse=True,
                )
                ordered.sort(
                    key=lambda item: not _is_current_project(
                        cwd, _project_real_path(item[1], item[2])
                    )
                )

                for rel_label, proj, convos in ordered:
                    project_path = _project_real_path(proj, convos)
                    is_cwd = _is_current_project(cwd, project_path)
                    date_str = _fmt_nav_ts(convos[0].timestamp) if convos else ""
                    count = len(convos)
                    total_projects += 1

                    analyzed = self._is_analyzed(proj.folder_name)
                    # The count and date matter more than a long path, so the
                    # path is what gives way when the row does not fit. The
                    # markers occupy the row too, so they come out of its budget.
                    tail = f"  ({count})  {date_str}"
                    room = budget(2) - (2 if is_cwd else 0) - (2 if analyzed else 0)
                    body = _elide(
                        _elide(rel_label, max(8, room - len(tail))) + tail, room
                    )

                    plabel = Text()
                    if is_cwd:
                        plabel.append("● ", "bold cyan")
                    plabel.append_text(_highlight_matches(body, terms))
                    if analyzed:
                        plabel.append(" ★", "yellow")

                    pnode = pg_node.add(
                        plabel,
                        data=NodeData(kind="project", project=proj, is_cwd=is_cwd),
                        expand=is_cwd or bool(terms),
                    )

                    for c in convos:
                        d = _fmt_nav_ts(c.timestamp)
                        prefix = f"  {d}  "
                        summary = summaries.get(c.uuid, "")
                        indexed_snippet = indexed_matches.get(c.uuid, "")
                        if terms and indexed_snippet:
                            preview = _matching_excerpt(
                                indexed_snippet, terms, width=max(16, budget(3) - len(prefix))
                            )
                        else:
                            preview = summary or c.preview or c.slug or c.uuid[:8]
                        label = _elide(prefix + preview, budget(3))
                        pnode.add_leaf(
                            _highlight_matches(label, terms),
                            data=NodeData(kind="convo", meta=c, project=proj),
                        )

        if terms:
            status = f" {filtered_count} matches · ↑↓ scan · Enter open · R resume · Esc clear"
        else:
            status = (
                f" {filtered_count} convos · {total_projects} projects · "
                "type to search · Tab browse"
            )
        self.query_one("#status-bar", Static).update(status)

    def _refresh_analyzed_markers(self) -> None:
        """Re-scan analyses dir and update ★ markers on project nodes."""
        self._analyzed_set = self._get_analyzed_set()
        for pnode in self._walk_tree_nodes():
            data: NodeData = pnode.data
            if not data or data.kind != "project" or not data.project:
                continue
            proj_name = Path(data.project.display_path).name if data.project.display_path else ""
            label = str(pnode.label)
            label = label.replace(" ★", "")
            if data.is_cwd and label.startswith("● "):
                label = label[2:]
            if data.is_cwd and label.startswith("✓ ● "):
                label = label[4:]
            if self._is_analyzed(proj_name):
                label += " ★"
            if data.is_cwd:
                prefix = label[:2] if label.startswith("✓ ") else ""
                body = label[2:] if prefix else label
                styled = Text(f"{prefix}● ", style="bold cyan")
                styled.append(body, style="bold cyan")
                pnode.set_label(styled)
            else:
                pnode.set_label(label)

    # --- Tree interaction ---

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        node = event.node
        data: NodeData | None = node.data
        if not data:
            return

        if data.kind == "convo" and data.meta:
            self._cancel_highlight_preview()
            self.current_meta = data.meta
            if self._previewed_meta is data.meta:
                return  # already on screen; opening it again would reparse it
            query = self.query_one("#filter-input", Input).value.strip()
            self.request_preview(data.meta, query)
        elif data.kind == "project" and data.project:
            self._cancel_highlight_preview()
            self._show_project_digest(data.project)

    def _show_project_digest(self, project: Project) -> None:
        """Summarize a project so its row does not leave a stale transcript up."""
        self._previewed_meta = None
        convos = project.conversations
        lines = [
            f"## {_short_path(project.display_path)}",
            f"**{len(convos)} conversations**",
            "",
            "---",
            "",
        ]
        summaries = self._summaries
        for meta in convos[:12]:
            title = summaries.get(meta.uuid) or meta.preview or meta.slug or meta.uuid[:8]
            lines.append(f"- `{_fmt_ts(meta.timestamp)}` {_escape_markdown_inline(title[:110])}")
        if len(convos) > 12:
            lines.append(f"\n*and {len(convos) - 12} more.*")
        self._set_preview("\n".join(lines), 0, f"PROJECT · {len(convos)} conversations")

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        """Preview the conversation the cursor rests on.

        Scanning through results passes over many rows; parsing each one would
        read transcripts nobody is going to look at, so the load waits until
        the cursor stops moving.
        """
        data: NodeData | None = event.node.data
        if not data:
            return
        if data.kind == "project" and data.project:
            # Cheap: the digest reads cached metadata, no transcripts.
            self._cancel_highlight_preview()
            self._show_project_digest(data.project)
            return
        if data.kind != "convo" or not data.meta:
            return
        self.current_meta = data.meta
        # Compare against what is on screen, not what is selected: a project
        # row replaces the transcript, so coming back needs a fresh render.
        if self._previewed_meta is data.meta:
            return
        self._cancel_highlight_preview()
        self._highlight_timer = self.set_timer(
            _HIGHLIGHT_DEBOUNCE_SECONDS,
            lambda meta=data.meta: self._preview_highlighted(meta),
        )

    def _cancel_highlight_preview(self) -> None:
        if self._highlight_timer is not None:
            self._highlight_timer.stop()
            self._highlight_timer = None

    def _preview_highlighted(self, meta: ConversationMeta) -> None:
        self._highlight_timer = None
        if self.current_meta is not meta:  # the cursor moved on
            return
        query = self.query_one("#filter-input", Input).value.strip()
        self.request_preview(meta, query)

    def request_preview(
        self, meta: ConversationMeta, query: str = "", *, full: bool = False
    ) -> None:
        preview_scroll = self.query_one("#preview-scroll", VerticalScroll)
        preview_scroll.loading = True
        self._previewed_meta = meta
        name = meta.slug or meta.uuid[:8]
        self.query_one("#right-title", Static).update(f"LOADING · {name}")
        self.load_preview(meta, query, full=full)

    @work(thread=True, exclusive=True, group="preview")
    def load_preview(
        self, meta: ConversationMeta, query: str = "", *, full: bool = False
    ) -> None:
        worker = get_current_worker()
        try:
            turns = parse_jsonl(meta.path)
        except Exception as error:
            if not worker.is_cancelled:
                self.call_from_thread(
                    self._set_preview,
                    f"## Preview unavailable\n\n`{_escape_markdown_inline(str(error))}`",
                    0,
                    "PREVIEW ERROR",
                )
            return
        if worker.is_cancelled:
            return
        meta.turn_count = len(turns)
        terms = parse_search_terms(query) if not full else []

        if terms:
            phrase = " ".join(terms)
            matches: list[tuple[int, int, object]] = []
            for index, turn in enumerate(turns):
                folded = turn.text.casefold()
                matched_terms = [term for term in terms if term in folded]
                if not matched_terms:
                    continue
                score = (
                    (1000 if phrase and phrase in folded else 0)
                    + 100 * len(matched_terms)
                    + sum(folded.count(term) for term in matched_terms)
                )
                matches.append((score, index, turn))

            if matches:
                matches.sort(key=lambda item: (item[0], -item[1]), reverse=True)
                visible = matches[:20]
                title = _highlight_markdown(meta.slug or meta.uuid, terms)
                cwd = _highlight_markdown(_short_path(meta.cwd) if meta.cwd else "(unknown)", terms)
                lines = [
                    f"## {title}",
                    f"**Search:** {_escape_markdown_inline(query)}  ",
                    f"**Matches:** {_turns(len(matches))}  ",
                    f"**Date:** {_fmt_ts(meta.timestamp)}  ",
                    f"**CWD:** {cwd}  ",
                    f"**Session ID:** `{meta.uuid}`",
                    "",
                    "---",
                    "",
                ]
                for _score, index, turn in visible:
                    excerpt = _matching_excerpt(turn.text, terms, width=320)
                    highlighted = _highlight_markdown(excerpt, terms)
                    lines.append(f"### Turn {index + 1} · {turn.role.title()}")
                    lines.append(f"> {highlighted}\n")
                if len(matches) > len(visible):
                    lines.append(f"*Showing the best {len(visible)} of {len(matches)} matching turns.*")
                if not worker.is_cancelled:
                    self.call_from_thread(
                        self._set_preview,
                        "\n".join(lines),
                        len(matches),
                        f"MATCHES ({_turns(len(matches))}) · R RESUME",
                    )
                return

        # A tail keeps browsing fast; v renders the whole transcript.
        tail = turns if full or len(turns) <= 10 else turns[-10:]
        md = to_markdown(tail)
        skipped = len(turns) - len(tail)
        title = _highlight_markdown(meta.slug or meta.uuid, terms)
        cwd = _highlight_markdown(_short_path(meta.cwd) if meta.cwd else "(unknown)", terms)
        header = f"## {title}\n**Date:** {_fmt_ts(meta.timestamp)}  \n**CWD:** {cwd}  "
        if meta.git_branch:
            header += f"\n**Branch:** {_highlight_markdown(meta.git_branch, terms)}  "
        header += f"\n**Session ID:** `{meta.uuid}`  \n**Turns:** {len(turns)} total"
        if skipped:
            header += f" (showing last {len(tail)} of {len(turns)} · press v for all)"
        header += _stats_line(meta.path)
        header += "\n\n---\n\n"
        if not worker.is_cancelled:
            self.call_from_thread(
                self._set_preview,
                header + md,
                len(turns),
                f"FULL TRANSCRIPT · {_turns(len(turns))}" if full else None,
            )

    def _set_preview(self, md: str, turn_count: int, title: str | None = None) -> None:
        self.query_one("#preview", Markdown).update(md)
        label = title or (
            f"CONVERSATION · {_turns(turn_count)}" if turn_count else "CONVERSATION"
        )
        self.query_one("#right-title", Static).update(label)
        preview_scroll = self.query_one("#preview-scroll", VerticalScroll)
        preview_scroll.loading = False
        preview_scroll.scroll_home()

    def _show_search_summary(self, query: str, count: int | None = None) -> None:
        if count is None:
            result_nodes = [
                node
                for node in self._walk_tree_nodes()
                if node.data and node.data.kind == "convo"
            ]
            count = len(result_nodes)
        escaped_query = _escape_markdown_inline(query)
        if self._indexing:
            progress = self._index_progress
            index_note = (
                f"\n\n*Full-text index: {progress.checked}/{progress.total}. "
                "Results update live.*"
            )
        else:
            index_note = ""

        if count:
            noun = "conversation" if count == 1 else "conversations"
            body = (
                f'## {count} {noun}\n\n`{escaped_query}`\n\n'
                "Use **↑/↓** to scan, **Enter** to open, then **R** to resume."
                f"{index_note}"
            )
        elif self._indexing:
            body = (
                f'## No matches yet\n\n`{escaped_query}`\n\n'
                "Metadata is searchable now. Transcript results will appear as "
                f"the background index advances.{index_note}"
            )
        else:
            body = (
                f'## No matches\n\n`{escaped_query}`\n\n'
                "Try fewer words, a path fragment, or an exact phrase in quotes."
            )
        self._set_preview(body, 0, f"SEARCH · {count}")

    # --- Filter ---

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "filter-input":
            self._filter_query = event.value.strip()
            self._schedule_filter()

    # --- Multi-select ---

    def _walk_tree_nodes(self, node: TreeNode | None = None):
        """Yield every visible tree node below node."""
        if node is None:
            node = self.query_one("#nav-tree", Tree).root
        for child in node.children:
            yield child
            yield from self._walk_tree_nodes(child)

    def _get_selected_nodes(self) -> list[NodeData]:
        """Collect all nodes marked as selected."""
        selected = []
        seen_paths = set()
        for node in self._walk_tree_nodes():
            data: NodeData = node.data
            if not data:
                continue
            if data.kind == "project" and data.selected:
                for child in self._walk_tree_nodes(node):
                    cd: NodeData = child.data
                    if cd and cd.kind == "convo" and cd.meta and cd.meta.path not in seen_paths:
                        selected.append(cd)
                        seen_paths.add(cd.meta.path)
            elif data.kind == "convo" and data.selected and data.meta and data.meta.path not in seen_paths:
                selected.append(data)
                seen_paths.add(data.meta.path)
        return selected

    def _update_node_label(self, node: TreeNode) -> None:
        """Add/remove selection marker on a node's label."""
        data: NodeData = node.data
        if not data:
            return
        label = str(node.label)
        if label.startswith("✓ ") or label.startswith("○ "):
            label = label[2:]
        if data.is_cwd and label.startswith("● "):
            label = label[2:]
        select_marker = "✓ " if data.selected else ""
        if data.is_cwd:
            styled = Text(f"{select_marker}● ", style="bold cyan")
            styled.append(label, style="bold cyan")
            node.set_label(styled)
        else:
            node.set_label(f"{select_marker}{label}")

    def action_toggle_select(self) -> None:
        tree = self.query_one("#nav-tree", Tree)
        node = tree.cursor_node
        if not node or not node.data:
            return
        data: NodeData = node.data
        data.selected = not data.selected
        self._update_node_label(node)

        # If toggling a group/project, toggle all descendants too.
        if data.kind in ("group", "project"):
            for child in self._walk_tree_nodes(node):
                cd: NodeData = child.data
                if cd:
                    cd.selected = data.selected
                    self._update_node_label(child)

        self._update_selection_count()

    def action_select_all(self) -> None:
        for node in self._walk_tree_nodes():
            data: NodeData = node.data
            if data:
                data.selected = True
                self._update_node_label(node)
        self._update_selection_count()

    def action_deselect_all(self) -> None:
        for node in self._walk_tree_nodes():
            data: NodeData = node.data
            if data:
                data.selected = False
                self._update_node_label(node)
        self._update_selection_count()

    def _estimate_tokens(self, nodes: list[NodeData]) -> str:
        """Estimate token count from file sizes (~4 chars/token)."""
        total_bytes = 0
        for nd in nodes:
            if nd.meta:
                total_bytes += _conversation_size(nd.meta.path)
        tokens = total_bytes // 4
        if tokens > 1_000_000:
            return f"~{tokens / 1_000_000:.1f}M tokens"
        if tokens > 1_000:
            return f"~{tokens // 1_000}K tokens"
        return f"~{tokens} tokens"

    def _update_selection_count(self) -> None:
        selected = self._get_selected_nodes()
        status = self.query_one("#status-bar", Static)
        model_short = self.gemini_model.replace("-preview", "").replace("[", "(").replace("]", ")")
        if selected:
            tok = self._estimate_tokens(selected)
            status.update(f" {len(selected)} selected · {tok} · A analyze · E export · M={model_short}")
        else:
            total = sum(len(p.conversations) for p in self.projects)
            status.update(f" {len(self.projects)} projects · {total} convos · S select · A analyze · M={model_short}")

    # --- Model ---

    def action_cycle_model(self) -> None:
        self._model_index = (self._model_index + 1) % len(MODELS)
        self.gemini_model = MODELS[self._model_index]
        self.notify(f"Model: {self.gemini_model}")
        self._update_selection_count()

    # --- Prompt Editor ---

    def action_edit_prompt(self) -> None:
        prompt_panel = self.query_one("#prompt-panel")
        preview_scroll = self.query_one("#preview-scroll")
        if prompt_panel.display:
            # Already open — close it
            self._save_current_prompt()
            prompt_panel.display = False
            preview_scroll.display = True
            return
        # Open editor with single prompt
        self._editing_prompt = "single"
        editor = self.query_one("#prompt-editor", TextArea)
        editor.load_text(self.custom_single_prompt)
        self.query_one("#right-title", Static).update("EDIT PROMPT (single convo)")
        prompt_panel.display = True
        preview_scroll.display = False
        editor.focus()

    def _save_current_prompt(self) -> None:
        editor = self.query_one("#prompt-editor", TextArea)
        text = editor.text
        if self._editing_prompt == "single":
            self.custom_single_prompt = text
        else:
            self.custom_multi_prompt = text

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "prompt-save":
            self._save_current_prompt()
            self.query_one("#prompt-panel").display = False
            self.query_one("#preview-scroll").display = True
            self.query_one("#right-title", Static).update("CONVERSATION")
            self.notify("Prompt saved")
        elif event.button.id == "prompt-switch":
            self._save_current_prompt()
            editor = self.query_one("#prompt-editor", TextArea)
            if self._editing_prompt == "single":
                self._editing_prompt = "multi"
                editor.load_text(self.custom_multi_prompt)
                self.query_one("#right-title", Static).update("EDIT PROMPT (multi convo)")
            else:
                self._editing_prompt = "single"
                editor.load_text(self.custom_single_prompt)
                self.query_one("#right-title", Static).update("EDIT PROMPT (single convo)")
        elif event.button.id == "prompt-reset":
            editor = self.query_one("#prompt-editor", TextArea)
            if self._editing_prompt == "single":
                self.custom_single_prompt = SINGLE_PROMPT
                editor.load_text(SINGLE_PROMPT)
            else:
                self.custom_multi_prompt = MULTI_PROMPT
                editor.load_text(MULTI_PROMPT)
            self.notify("Prompt reset to default")

    # --- Focus ---

    def action_toggle_focus(self) -> None:
        scroll = self.query_one("#preview-scroll", VerticalScroll)
        tree = self.query_one("#nav-tree", Tree)
        if tree.has_focus:
            scroll.focus()
        else:
            tree.focus()

    # --- Export ---

    def action_export(self) -> None:
        selected = self._get_selected_nodes()
        if selected:
            default = _export_stem(selected[0].meta) if len(selected) == 1 else ""
            self.push_screen(ExportNameScreen(default), lambda name: self.do_export_multi(selected, name))
        elif self.current_meta:
            default = _export_stem(self.current_meta)
            self.push_screen(ExportNameScreen(default), lambda name: self.do_export_single(self.current_meta, name))
        else:
            self.notify("Select a conversation first", severity="warning")

    @work(thread=True)
    def do_export_single(self, meta: ConversationMeta, custom_name: str = "") -> None:
        turns = parse_jsonl(meta.path)
        md = to_markdown(turns)
        out_dir = Path("output")
        out_dir.mkdir(exist_ok=True)
        filename = _export_filename(meta, custom_name)
        out_path = out_dir / filename
        out_path.write_text(md, encoding="utf-8")
        self.call_from_thread(self.notify, f"Exported to {out_path.resolve()}")
        self._last_action = "export"
        self._last_export_dir = out_dir.resolve()

    @work(thread=True)
    def do_export_multi(self, nodes: list[NodeData], custom_name: str = "") -> None:
        out_dir = Path("output")
        out_dir.mkdir(exist_ok=True)
        for nd in nodes:
            meta = nd.meta
            turns = parse_jsonl(meta.path)
            md = to_markdown(turns)
            filename = _export_filename(meta, custom_name if len(nodes) == 1 else "")
            out_path = out_dir / filename
            out_path.write_text(md, encoding="utf-8")
        self.call_from_thread(self.notify, f"Exported {len(nodes)} conversations to {out_dir.resolve()}/")
        self._last_action = "export"
        self._last_export_dir = out_dir.resolve()

    def action_export_concat(self) -> None:
        selected = self._get_selected_nodes()
        if not selected:
            self.notify("Select conversations first (S to select, Ctrl+A for all)", severity="warning")
            return
        self.do_export_concat(selected)

    @work(thread=True)
    def do_export_concat(self, nodes: list[NodeData]) -> None:
        parts = []
        for nd in nodes:
            meta = nd.meta
            turns = parse_jsonl(meta.path)
            md = to_markdown(turns)
            name = _export_stem(meta)
            date = _export_date(meta)
            header = f"# {name} ({date})\n**CWD:** {meta.cwd}\n\n"
            parts.append(header + md)

        combined = "\n\n---\n\n".join(parts)
        out_dir = ANALYSES_DIR.parent / "exports"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        first_meta = nodes[0].meta
        proj = Path(first_meta.cwd).name if first_meta and first_meta.cwd else "mixed"
        out_path = out_dir / f"{ts}-{proj}-{len(nodes)}-convos-combined.md"
        out_path.write_text(combined, encoding="utf-8")
        self.call_from_thread(self.notify, f"Combined export: {out_path.resolve()}")
        self._last_action = "export"
        self._last_export_dir = out_dir.resolve()
        self.call_from_thread(self._set_preview, f"## Exported {len(nodes)} conversations\n\nSaved to `{out_path}`\n\nSize: {len(combined):,} chars (~{len(combined)//4:,} tokens)", 0)

    def action_open_folder(self) -> None:
        """Open the relevant folder based on last action."""
        import subprocess
        if self._last_action == "analysis":
            folder = ANALYSES_DIR
        elif self._last_action == "export":
            folder = getattr(self, "_last_export_dir", ANALYSES_DIR.parent / "exports")
        else:
            folder = ANALYSES_DIR.parent  # show both
        folder.mkdir(parents=True, exist_ok=True)
        import sys
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        elif sys.platform == "win32":
            subprocess.Popen(["explorer", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])

    # --- Gemini Analysis ---

    def _check_gemini(self) -> bool:
        try:
            from .analyzer import gemini_available
        except ImportError:
            self.notify("Run: uv sync --extra ai", severity="error")
            return False
        if not gemini_available():
            self.notify("Set GEMINI_API_KEY env var", severity="error")
            return False
        return True

    def action_analyze(self) -> None:
        if self._analyzing:
            self.notify("Analysis already running — Esc to cancel", severity="warning")
            return
        if not self._check_gemini():
            return
        selected = self._get_selected_nodes()
        model = self.gemini_model
        if selected:
            self.do_analyze_multi(selected, model)
        elif self.current_meta:
            self.do_analyze_single(self.current_meta, model)
        else:
            self.notify("Select conversation(s) first", severity="warning")

    def _start_analysis(self, label: str) -> None:
        self._analyzing = True
        self.query_one("#right-title", Static).update(f"ANALYZING: {label}...")
        self._set_preview("## Analysis in progress...\n\nPress **Esc** to cancel.", 0)

    def _finish_analysis(self) -> None:
        self._analyzing = False
        self._last_action = "analysis"

    def _cancel_analysis(self) -> None:
        # Cancel all running workers
        for worker in self.workers:
            if not worker.is_finished:
                worker.cancel()
        self._analyzing = False
        self.query_one("#right-title", Static).update("CONVERSATION")
        self._set_preview("*Analysis cancelled.*", 0)
        self.notify("Analysis cancelled")

    @work(thread=True, exit_on_error=False)
    def do_analyze_single(self, meta: ConversationMeta, model: str = DEFAULT_MODEL) -> None:
        name = meta.slug or meta.uuid[:8]
        self.call_from_thread(self._start_analysis, f"{name} via {model}")
        try:
            from .analyzer import analyze_single
            turns = parse_jsonl(meta.path)
            if not self._analyzing:
                return
            result = analyze_single(turns, model=model, prompt_template=self.custom_single_prompt)
            if not self._analyzing:
                return

            ANALYSES_DIR.mkdir(parents=True, exist_ok=True)
            proj = Path(meta.cwd).name if meta.cwd else "unknown"
            path = ANALYSES_DIR / _analysis_filename(proj, 1)
            path.write_text(result, encoding="utf-8")

            header = f"## Analysis: {name}\n*Saved to {path}*\n\n---\n\n"
            self.call_from_thread(self._set_preview, header + result, 0)
            self.call_from_thread(
                lambda: self.query_one("#right-title", Static).update("GEMINI ANALYSIS")
            )
            self.call_from_thread(self._refresh_analyzed_markers)
        except Exception as e:
            if not self._analyzing:
                return
            import traceback
            tb = traceback.format_exc()
            self.call_from_thread(self._set_preview, f"## Analysis Error\n\n```\n{tb}\n```", 0)
            self.call_from_thread(self.notify, f"Analysis failed: {e}", severity="error")
        finally:
            self.call_from_thread(self._finish_analysis)

    @work(thread=True, exit_on_error=False)
    def do_analyze_multi(self, nodes: list[NodeData], model: str = DEFAULT_MODEL) -> None:
        count = len(nodes)
        self.call_from_thread(self._start_analysis, f"{count} convos via {model}")
        try:
            from .analyzer import analyze_multi
            conversations = []
            for i, nd in enumerate(nodes):
                if not self._analyzing:
                    return
                meta = nd.meta
                label = f"{meta.slug or meta.uuid[:8]} ({meta.timestamp[:10]})"
                turns = parse_jsonl(meta.path)
                conversations.append((label, turns))
                self.call_from_thread(
                    lambda i=i: self.query_one("#right-title", Static).update(
                        f"ANALYZING: loading {i+1}/{count}..."
                    )
                )

            if not self._analyzing:
                return
            self.call_from_thread(
                lambda: self.query_one("#right-title", Static).update(
                    f"ANALYZING: waiting for Gemini ({count} convos)..."
                )
            )
            result = analyze_multi(conversations, model=model, prompt_template=self.custom_multi_prompt)
            if not self._analyzing:
                return

            ANALYSES_DIR.mkdir(parents=True, exist_ok=True)
            first_meta = nodes[0].meta
            proj = Path(first_meta.cwd).name if first_meta and first_meta.cwd else "mixed"
            path = ANALYSES_DIR / _analysis_filename(proj, count)
            path.write_text(result, encoding="utf-8")

            header = f"## Cross-session Analysis ({count} conversations)\n*Saved to {path}*\n\n---\n\n"
            self.call_from_thread(self._set_preview, header + result, 0)
            self.call_from_thread(
                lambda: self.query_one("#right-title", Static).update(f"GEMINI ANALYSIS ({count} convos)")
            )
            self.call_from_thread(self._refresh_analyzed_markers)
        except Exception as e:
            if not self._analyzing:
                return
            import traceback
            tb = traceback.format_exc()
            self.call_from_thread(self._set_preview, f"## Analysis Error\n\n```\n{tb}\n```", 0)
            self.call_from_thread(self.notify, f"Analysis failed: {e}", severity="error")
        finally:
            self.call_from_thread(self._finish_analysis)

    # --- Resume ---

    def action_resume(self) -> None:
        if not self.current_meta:
            self.notify("Select a conversation first", severity="warning")
            return
        if self.current_meta.source not in _RESUMABLE_SOURCES:
            self.notify(f"Resume not supported for {self.current_meta.source.title()} conversations", severity="warning")
            return
        meta = self.current_meta
        self.push_screen(
            ResumeScreen(meta),
            lambda confirmed: self._confirm_resume(meta, confirmed),
        )

    def _confirm_resume(self, meta: ConversationMeta, confirmed: bool | None) -> None:
        if not confirmed:
            return
        self._resume_meta = meta
        self.exit()

    # --- Handoff ---

    def action_handoff(self) -> None:
        if not self.current_meta:
            self.notify("Select a conversation first", severity="warning")
            return
        self._handoff_meta = self.current_meta
        self.exit()

    # --- Search ---

    def action_search(self) -> None:
        filt = self.query_one("#filter-input", Input)
        filt.focus()

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_view_full(self) -> None:
        """Render the whole conversation, not just the tail or the matches."""
        if not self.current_meta:
            self.notify("Pick a conversation first", severity="warning")
            return
        self.request_preview(self.current_meta, full=True)

    def action_copy_id(self) -> None:
        """Put the session ID on the clipboard for a resume command."""
        if not self.current_meta:
            self.notify("Pick a conversation first", severity="warning")
            return
        self.copy_to_clipboard(self.current_meta.uuid)
        self.notify(f"Copied {self.current_meta.uuid}")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "filter-input":
            return
        tree = self.query_one("#nav-tree", Tree)

        # If the user arrowed down to a conversation, open that one.
        cursor = tree.cursor_node
        if cursor and cursor.data and cursor.data.kind == "convo" and cursor.data.meta:
            tree.select_node(cursor)
            tree.focus()
            self.current_meta = cursor.data.meta
            self.request_preview(cursor.data.meta, event.value.strip())
            return

        if not event.value.strip():
            return
        first_match = next(
            (
                node
                for node in self._walk_tree_nodes(tree.root)
                if node.data and node.data.kind == "convo"
            ),
            None,
        )
        if first_match is None:
            self.notify("No matching conversations", severity="warning")
            return
        tree.select_node(first_match)
        tree.focus()
        if first_match.data.meta:
            self.current_meta = first_match.data.meta
            self.request_preview(first_match.data.meta, event.value.strip())

    def action_cancel(self) -> None:
        if self._analyzing:
            self._cancel_analysis()
        else:
            # Clear filter and return to tree
            filt = self.query_one("#filter-input", Input)
            if filt.value:
                filt.value = ""
            self.query_one("#nav-tree", Tree).focus()

    def action_quit(self) -> None:
        self.exit()


def _pick_conversation(convos: list, cwd: str):
    """Show numbered conversation list for interactive selection. Returns meta or None."""
    from .summarize import load_summaries
    summaries = load_summaries()
    print(f"\n{len(convos)} conversations for {cwd}:\n")
    for i, c in enumerate(convos):
        ts = c.timestamp[:10] if c.timestamp else "?"
        name = c.slug or c.uuid[:8]
        summary = summaries.get(c.uuid, "")
        preview = summary[:60] if summary else (c.preview or "")[:50]
        size_str = _file_size_label(_conversation_size(c.path))
        marker = " *" if i == 0 else ""
        src_tag = f"[{c.source}]" if c.source != "claude" else ""
        src_pad = f" {src_tag:7s}" if src_tag else "        "
        print(f"  [{i + 1}] {ts}{src_pad}  {name:30s}  {size_str:>7s}  {preview}{marker}")
    print(f"\nEnter number [1-{len(convos)}] or press Enter for latest: ", end="", flush=True)
    try:
        choice = input().strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not choice:
        return convos[0]
    try:
        idx = int(choice) - 1
        if not (0 <= idx < len(convos)):
            print(f"Invalid choice: {choice}")
            return None
        return convos[idx]
    except ValueError:
        print(f"Invalid choice: {choice}")
        return None


def _print_fuzzy_preview(path: Path, max_parse_bytes: int = 4_000_000) -> None:
    """Print a responsive fzf preview, avoiding full parses for huge logs."""
    from .parser import get_meta
    from .summarize import load_summaries

    meta = get_meta(path)
    if not meta:
        print(f"Unable to read conversation metadata: {path}")
        return

    summary = load_summaries().get(meta.uuid, "")
    print(f"{meta.slug or meta.uuid[:8]}  [{meta.source}]")
    print(f"{_fmt_ts(meta.timestamp) or '?'}")
    print(f"{meta.cwd or '(unknown project)'}")
    print(f"Session: {meta.uuid}")
    if meta.git_branch:
        print(f"Branch: {meta.git_branch}")
    if summary:
        print(f"\nSummary\n{summary}")
    if meta.preview:
        print(f"\nFirst prompt\n{meta.preview}")

    size = _conversation_size(path)
    if size > max_parse_bytes:
        print(
            f"\nLarge transcript ({size / 1_000_000:.1f} MB). "
            "Press Enter to select it, then use the printed --show command "
            "for the complete conversation."
        )
        return

    turns = parse_jsonl(path, detail=DETAIL_TEXT)
    if not turns:
        return
    print("\nConversation")
    remaining = 120_000
    for turn in turns:
        if remaining <= 0:
            print("\n… preview truncated; select the session to show the rest")
            break
        text = turn.text[:remaining]
        print(f"\n## {turn.role.title()}\n{text}")
        remaining -= len(text)


def _pick_conversation_fuzzy(
    convos: list[ConversationMeta],
    initial_query: str = "",
    *,
    fzf_path: str | None = None,
    runner=None,
) -> ConversationMeta | None:
    """Select a conversation with fzf without preloading transcript bodies."""
    import shlex
    import shutil
    import subprocess
    import sys

    from .summarize import load_summaries

    executable = fzf_path or shutil.which("fzf")
    if not executable:
        print("Error: --find requires fzf (https://github.com/junegunn/fzf)")
        return None

    summaries = load_summaries()

    def field(value: object) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    rows: list[str] = []
    indexed: dict[int, ConversationMeta] = {}
    home = str(Path.home())
    for index, meta in enumerate(convos):
        indexed[index] = meta
        cwd = meta.cwd or "(unknown project)"
        display_cwd = "~" + cwd[len(home):] if cwd.startswith(home) else cwd
        summary = summaries.get(meta.uuid, "")
        description = summary or meta.preview or "(no preview)"
        searchable = " ".join(
            field(value)
            for value in (
                meta.uuid,
                meta.slug,
                meta.cwd,
                meta.git_branch,
                meta.preview,
                summary,
            )
            if value
        )
        rows.append(
            "\t".join(
                (
                    str(index),
                    field(_fmt_ts(meta.timestamp, date_only=True) or "?"),
                    field(meta.source),
                    field(display_cwd),
                    field(meta.slug or meta.uuid[:8]),
                    field(description),
                    searchable,
                    field(meta.path),
                )
            )
        )

    if not rows:
        print("No conversations found.")
        return None

    preview_command = (
        f"{shlex.quote(sys.executable)} -m agentconvos.app --peek {{8}}"
    )
    command = [
        executable,
        "--delimiter=\t",
        "--nth=2..7",
        "--with-nth=2..6",
        "--height=90%",
        "--layout=reverse",
        "--border=rounded",
        "--info=inline",
        "--cycle",
        "--no-multi",
        "--prompt=Conversations> ",
        "--header=type to fuzzy-search · enter select · ctrl-/ preview · esc cancel",
        "--preview-window=right,60%,wrap",
        "--bind=ctrl-/:toggle-preview",
        f"--preview={preview_command}",
    ]
    if initial_query:
        command.append(f"--query={initial_query}")

    run = runner or subprocess.run
    result = run(
        command,
        input="\n".join(rows) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None

    try:
        selected_index = int(result.stdout.split("\t", 1)[0])
    except ValueError:
        return None
    return indexed.get(selected_index)


def _handoff_cmd(
    source: str,
    message: str,
    extra_args: list[str] | None = None,
    yolo: bool = False,
) -> list[str]:
    """Build a handoff command for the given target CLI."""
    extra = extra_args or []
    if source == "codex":
        codex_args = ["--yolo"] if yolo else []
        return ["codex"] + codex_args + extra + [message]
    if source == "pi":
        return ["pi"] + extra + [message]
    if source == "opencode":
        permission_args = ["--auto"] if yolo else []
        return ["opencode"] + permission_args + extra + ["--prompt", message]
    if source == "agy":
        permission_args = ["--dangerously-skip-permissions"] if yolo else []
        return ["agy"] + permission_args + extra + ["--prompt-interactive", message]
    permission_args = ["--dangerously-skip-permissions"] if yolo else []
    return ["claude"] + permission_args + extra + [message]


def _handoff_agent(conversation_source: str, requested_agent: str | None, _yolo: bool) -> str:
    """Return the CLI agent to start for handoff."""
    if requested_agent:
        return requested_agent
    return conversation_source


def _resume_cmd(
    source: str,
    uuid: str,
    extra_args: list[str] | None = None,
    yolo: bool = False,
) -> list[str] | None:
    """Build a resume command, or None if the source doesn't support it."""
    extra = extra_args or []
    if source == "claude":
        permission_args = ["--dangerously-skip-permissions"] if yolo else []
        return ["claude"] + permission_args + ["-r", uuid] + extra
    if source == "codex":
        permission_args = ["--dangerously-bypass-approvals-and-sandbox"] if yolo else []
        return ["codex", "resume"] + permission_args + extra + [uuid]
    if source == "pi":
        return ["pi"] + extra + ["--session", uuid]
    if source == "agy":
        permission_args = ["--dangerously-skip-permissions"] if yolo else []
        return ["agy"] + permission_args + extra + ["--conversation", uuid]
    if source == "opencode":
        permission_args = ["--auto"] if yolo else []
        return ["opencode"] + permission_args + extra + ["-s", uuid]
    if source == "clihow":
        return ["clihow", "ask", "--thread", uuid] + extra
    return None


def _resume_description(meta: ConversationMeta) -> str:
    """Build the concrete, user-visible details for a resume confirmation."""
    agent_name = _SOURCE_STYLE.get(meta.source, (meta.source.title(), ""))[0]
    name = meta.slug or meta.uuid[:8]
    cmd = _resume_cmd(meta.source, meta.uuid)
    command = " ".join(cmd) if cmd else "Resume is not supported"
    return "\n".join(
        (
            f"{agent_name} · {name}",
            f"Date: {meta.timestamp[:19] or '(unknown)'}",
            f"CWD: {meta.cwd or '(unknown)'}",
            f"Session ID: {meta.uuid}",
            f"Command: {command}",
        )
    )


def main() -> None:
    import argparse
    import sys
    from importlib.metadata import version as package_version

    if len(sys.argv) > 1 and sys.argv[1] == "recall":
        from .recall import _DEFAULT_RECALL_BACKEND, _RECALL_BACKENDS, run_recall

        recall_parser = argparse.ArgumentParser(
            prog="agentconvos recall",
            description="Answer a question from evidence in your local conversation archive.",
        )
        recall_parser.add_argument(
            "--backend",
            choices=_RECALL_BACKENDS,
            default=_DEFAULT_RECALL_BACKEND,
            help="Retrieval backend (default: luna; agy uses Gemini 3.6 Flash)",
        )
        recall_parser.add_argument("question", nargs="+", help="Question to investigate")
        recall_args = recall_parser.parse_args(sys.argv[2:])

        return_code = run_recall(
            " ".join(recall_args.question),
            backend=recall_args.backend,
        )
        if return_code:
            raise SystemExit(return_code)
        return

    parser = argparse.ArgumentParser(
        description="Browse and analyze Claude Code, Codex, Pi, Agy, and OpenCode conversations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Commands:\n  recall [--backend {luna,agy}] "question"  Answer from evidence in the local conversation archive\n\n'
            'Run: agentconvos recall "question"'
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {package_version('agentconvos')}",
    )
    parser.add_argument("--analyze", nargs="+", metavar="ID_OR_PATH", help="Analyze conversations (JSONL paths, UUIDs, or slugs)")
    parser.add_argument("--concat", nargs="+", metavar="ID_OR_PATH", help="Export concatenated markdown (JSONL paths, UUIDs, or slugs)")
    parser.add_argument("--turns", metavar="ID_OR_PATH", help="Export one normalized conversation to stdout")
    parser.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL, help="Gemini model")
    parser.add_argument("--prompt", metavar="TEXT_OR_FILE", help="Custom analysis prompt (inline text or path to .txt/.md file). Use {content} as placeholder for conversation text, {count} for multi-convo count.")
    parser.add_argument("--detail", choices=["text", "tools", "results", "full", "thinking"], default=None, help="Detail level: text, tools, results, full, thinking (text + reasoning blocks)")
    parser.add_argument("--deep", nargs="+", metavar="ID_OR_PATH", help="Deep analysis: Pro for first chunk, Flash continues with context, Pro synthesizes. Uses full detail.")
    parser.add_argument(
        "--search",
        metavar="QUERY",
        help="Search conversation text using AND terms and quoted phrases",
    )
    parser.add_argument(
        "--ngrams",
        action="store_true",
        help="Find phrases characteristic of one agent source",
    )
    parser.add_argument(
        "--habits",
        action="store_true",
        help="Build a local HTML report of recurring reply phrases and writing patterns",
    )
    parser.add_argument(
        "--nlp",
        action="store_true",
        help="Enrich --habits with local spaCy and TextDescriptives analysis",
    )
    parser.add_argument(
        "--baseline",
        choices=["matched", "none"],
        default="none",
        help="Matched other-source comparison for --habits --nlp (default: none)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic split seed for --habits --nlp (default: 42)",
    )
    parser.add_argument(
        "--spacy-model",
        default="en_core_web_sm",
        help="Explicitly installed trained model for --habits --nlp (default: en_core_web_sm)",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="Output HTML path for --habits (JSON is written beside it)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        metavar="N",
        help="Maximum search results to return (default: 50)",
    )
    parser.add_argument(
        "-f",
        "--find",
        nargs="?",
        const="",
        default=None,
        metavar="QUERY",
        help="Open the fast interactive fuzzy conversation finder",
    )
    parser.add_argument("--list", action="store_true", help="List all projects and conversations")
    parser.add_argument("--show", nargs="+", metavar="ID_OR_PATH", help="Preview conversation (first ~10K words)")
    parser.add_argument("--peek", metavar="PATH", help=argparse.SUPPRESS)
    parser.add_argument("--open", action="store_true", help="Open in Sublime Text (use with --show or --concat)")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        metavar="ID_OR_MODE",
        help="Resume a conversation in its native CLI. With no value, resume the latest resumable session for CWD; use 'select' to choose.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the command instead of running it (use with --resume/--handoff)")
    parser.add_argument("--handoff", nargs="?", const="latest", default=None, metavar="MODE",
                        help="Export CWD conversation and start new session. Use --convo SOURCE --handoff AGENT to split source and target.")
    parser.add_argument("--handoff-agent", choices=["claude", "codex", "pi", "agy", "opencode"], metavar="AGENT",
                        help="Agent CLI to start for --handoff (defaults to selected conversation source)")
    parser.add_argument("--yolo", action="store_true",
                        help="Use the target agent's no-prompt permission mode for resume/handoff commands")
    parser.add_argument("--export-all", metavar="DIR", help="Export every conversation as individual markdown files to DIR")
    parser.add_argument("--projects-dir", nargs="+", metavar="DIR", help="Additional projects directories to scan (e.g. copied from other machines)")
    parser.add_argument("--summarize", action="store_true",
                        help="Generate missing or stale two-pass session summaries via Gemini")
    parser.add_argument("--json", action="store_true",
                        help="Output machine-readable JSON (use with --list, --search, --last, --context, --turns)")
    parser.add_argument("--source", choices=["claude", "codex", "pi", "agy", "opencode", "clihow"],
                        help="Filter by agent source")
    parser.add_argument("--convo", choices=["claude", "codex", "pi", "agy", "opencode", "clihow"],
                        help="Conversation source to use, e.g. --convo agy --handoff codex --yolo")
    parser.add_argument("--after", metavar="DATE",
                        help="Only conversations after this date (YYYY-MM-DD)")
    parser.add_argument("--before", metavar="DATE",
                        help="Only conversations before this date (YYYY-MM-DD)")
    parser.add_argument("--last", nargs="?", const=1, type=int, metavar="N",
                        help="Show last N conversations for current directory (default: 1)")
    parser.add_argument("--context", action="store_true",
                        help="Quick project digest: last 5 sessions per agent with catch-up details")
    args, remaining = parser.parse_known_args()

    def _option_was_supplied(option: str) -> bool:
        return any(
            argument == option or argument.startswith(f"{option}=")
            for argument in sys.argv[1:]
        )

    if args.nlp and not args.habits:
        parser.error("--nlp requires --habits")
    for option in ("--baseline", "--seed", "--spacy-model"):
        if _option_was_supplied(option) and not (args.habits and args.nlp):
            parser.error(f"{option} requires --habits --nlp")

    if args.ngrams and sys.argv.count("--source") != 1:
        parser.error("--ngrams requires exactly one --source")
    if args.habits and sys.argv.count("--source") != 1:
        parser.error("--habits requires exactly one --source")

    if args.source and args.convo and args.source != args.convo:
        print(f"Error: --source {args.source} conflicts with --convo {args.convo}")
        return

    # Parse extra project dirs
    _extra_dirs = [Path(d) for d in args.projects_dir] if args.projects_dir else None
    source_arg = args.source or args.convo
    _scan_kwargs = dict(
        extra_dirs=_extra_dirs,
        source=source_arg,
        after=args.after,
        before=args.before,
    )
    if args.detail is None:
        args.detail = "results" if (args.deep or args.analyze) else "text"


    if args.habits and args.nlp:
        import json as _json

        from .language_nlp import LanguageNLPError
        from .language_report import run_language_report
        from .scanner import scan_projects

        projects = scan_projects(
            extra_dirs=_extra_dirs,
            source=None if args.baseline == "matched" else args.source,
            after=args.after,
            before=args.before,
        )
        target_conversations = [
            conversation
            for project in projects
            for conversation in project.conversations
            if conversation.source == args.source
        ]
        baseline_conversations = [
            conversation
            for project in projects
            for conversation in project.conversations
            if args.baseline == "matched" and conversation.source != args.source
        ]
        output = (
            Path(args.output)
            if args.output
            else Path.home()
            / ".claude"
            / "convo-explorer"
            / "reports"
            / f"{args.source}-language-patterns.html"
        )
        try:
            report = run_language_report(
                target_conversations,
                baseline_conversations,
                source=args.source,
                after=args.after,
                before=args.before,
                seed=args.seed,
                baseline_mode=args.baseline,
                spacy_model=args.spacy_model,
                output=output,
                limit=max(1, args.limit),
            )
        except LanguageNLPError as error:
            print(str(error), file=sys.stderr)
            raise SystemExit(1) from None
        if args.json:
            print(
                _json.dumps(
                    {
                        "html": str(output),
                        "json": str(output.with_suffix(".json")),
                        "report": report.public_dict(),
                    },
                    indent=2,
                )
            )
        else:
            print(f"Wrote enriched language-pattern report: {output}")
            print(f"Wrote report data: {output.with_suffix('.json')}")
        return

    if args.habits:
        import json as _json

        from .habits import (
            analyze_habits,
            session_replies_from_projects,
            write_report,
        )
        from .ngrams import assistant_replies_from_index
        from .scanner import scan_projects

        complete_projects = scan_projects(extra_dirs=_extra_dirs)
        projects = scan_projects(
            extra_dirs=_extra_dirs,
            source=args.source,
            after=args.after,
            before=args.before,
        )
        all_conversations = [
            conversation
            for project in complete_projects
            for conversation in project.conversations
        ]
        search_index = ConversationSearchIndex()
        search_index.sync(all_conversations)
        selected_conversations = [
            conversation
            for project in projects
            for conversation in project.conversations
        ]
        indexed_replies = assistant_replies_from_index(
            search_index.path,
            selected_conversations,
        )
        replies = session_replies_from_projects(
            projects,
            source=args.source,
            indexed_replies=indexed_replies,
        )
        report = analyze_habits(
            replies,
            source=args.source,
            discovered_limit=max(1, args.limit),
        )
        output = (
            Path(args.output)
            if args.output
            else Path.home()
            / ".claude"
            / "convo-explorer"
            / "reports"
            / f"{args.source}-language-patterns.html"
        )
        write_report(report, output)
        if args.json:
            print(
                _json.dumps(
                    {
                        "html": str(output),
                        "json": str(output.with_suffix(".json")),
                        "report": report.public_dict(),
                    },
                    indent=2,
                )
            )
        else:
            print(f"Wrote language-pattern report: {output}")
            print(f"Wrote report data: {output.with_suffix('.json')}")
        return


    if args.ngrams:
        import json as _json

        from .ngrams import analyze_projects, assistant_replies_from_index
        from .scanner import scan_projects

        complete_projects = scan_projects(extra_dirs=_extra_dirs)
        if args.after or args.before:
            projects = scan_projects(
                extra_dirs=_extra_dirs,
                after=args.after,
                before=args.before,
            )
        else:
            projects = complete_projects
        all_conversations = [
            conversation
            for project in complete_projects
            for conversation in project.conversations
        ]
        search_index = ConversationSearchIndex()
        search_index.sync(all_conversations)
        selected_conversations = [
            conversation
            for project in projects
            for conversation in project.conversations
        ]
        indexed_replies = assistant_replies_from_index(
            search_index.path,
            selected_conversations,
        )
        analysis = analyze_projects(
            projects,
            args.source,
            limit=max(1, args.limit),
            indexed_replies=indexed_replies,
        )
        if args.json:
            print(_json.dumps(analysis.public_dict(), indent=2))
        else:
            print(
                f"Characteristic reply phrases for {args.source} "
                f"({analysis.target_sessions} sessions; compared with "
                f"{analysis.baseline_sessions} sessions from all other indexed sources)\n"
            )
            print(f"{'Phrase':<36} {'Occurrences':>11} {'Sessions':>9}  Distinctiveness")
            print(f"{'-' * 36} {'-' * 11} {'-' * 9}  {'-' * 34}")
            for row in analysis.phrases:
                print(
                    f"{row.phrase[:36]:<36} {row.occurrences:>11} {row.sessions:>9}  "
                    f"{row.distinctiveness_label}"
                )
        return


    if args.turns:
        import json as _json

        from .parser import get_meta

        paths = _resolve_args([args.turns], extra_dirs=_extra_dirs)
        path = paths[0]
        meta = get_meta(path)
        if meta is None:
            print(f"Error: could not read metadata from {path}")
            return
        turns = parse_jsonl(path, detail=args.detail)
        if args.json:
            size_bytes, mtime_ns = conversation_signature(path)
            print(_json.dumps({
                "conversation": {
                    "uuid": meta.uuid,
                    "slug": meta.slug,
                    "source": meta.source,
                    "timestamp": meta.timestamp,
                    "cwd": meta.cwd,
                    "file": str(path),
                    "size_bytes": size_bytes,
                    "mtime_ns": mtime_ns,
                },
                "detail": args.detail,
                "turn_count": len(turns),
                "turns": [
                    {"index": index, "role": turn.role, "text": turn.text}
                    for index, turn in enumerate(turns)
                ],
            }, indent=2))
        else:
            print(to_markdown(turns))
        return


    if args.peek:
        _print_fuzzy_preview(Path(args.peek))
        return


    if args.find is not None:
        from .scanner import scan_projects

        projects = scan_projects(**_scan_kwargs)
        conversations = [
            conversation
            for project in projects
            for conversation in project.conversations
        ]
        selected = _pick_conversation_fuzzy(
            conversations,
            initial_query=args.find,
        )
        if selected is None:
            return

        name = selected.slug or selected.uuid[:8]
        date = _fmt_ts(selected.timestamp, date_only=True) or "?"
        print(f"Selected: {date}  [{selected.source}]  {name}")
        print(f"ID: {selected.uuid}")
        print(f"CWD: {selected.cwd or '(unknown)'}")
        print(f"File: {selected.path}")
        print(f"Show: agentconvos --show {selected.uuid}")
        if _resume_cmd(selected.source, selected.uuid) is not None:
            print(f"Resume: agentconvos --resume {selected.uuid}")
        return


    if args.search:
        import sys as _sys

        from .scanner import scan_projects
        projects = scan_projects(**_scan_kwargs)
        all_conversations = [c for p in projects for c in p.conversations]
        index_conversations = all_conversations
        if source_arg or args.after or args.before:
            complete_projects = scan_projects(extra_dirs=_extra_dirs)
            index_conversations = [
                conversation
                for project in complete_projects
                for conversation in project.conversations
            ]
        search_index = ConversationSearchIndex()
        progress_visible = False
        last_reported = -100

        def report_search_index(progress: IndexSyncStats) -> None:
            nonlocal progress_visible, last_reported
            if not _sys.stderr.isatty() or progress.indexed < 10:
                return
            if progress.checked - last_reported < 100 and progress.checked != progress.total:
                return
            last_reported = progress.checked
            progress_visible = True
            print(
                f"\rUpdating turn index: {progress.checked}/{progress.total}",
                end="",
                file=_sys.stderr,
                flush=True,
            )

        sync_result = search_index.sync(
            index_conversations,
            on_progress=report_search_index,
        )
        if progress_visible:
            print(
                f"\rIndexed {sync_result.indexed} changed conversations"
                f" ({sync_result.failed} failed).{' ' * 20}",
                file=_sys.stderr,
            )
        limit = max(1, args.limit)
        hits = search_index.search_hits(args.search, all_conversations, limit=limit)
        # A full page means there are probably more; saying so beats printing
        # the cap as though it were the number of matches in the archive.
        truncated = len(hits) >= limit
        if args.json:
            import json as _json
            print(_json.dumps({
                "query": args.search,
                "total_searched": len(all_conversations),
                "limit": limit,
                "truncated": truncated,
                "hits": [
                    {
                        "uuid": h.meta.uuid,
                        "slug": h.meta.slug,
                        "source": h.meta.source,
                        "timestamp": h.meta.timestamp,
                        "cwd": h.meta.cwd,
                        "file": str(h.meta.path),
                        "turn_index": h.turn_index,
                        "role": h.role,
                        "snippet": h.snippet,
                    }
                    for h in hits
                ],
            }, indent=2))
        else:
            print(f"Searching {len(all_conversations)} conversations for \"{args.search}\"...\n")
            if not hits:
                print("No results found.")
            else:
                for hit in hits:
                    slug_part = f"  {hit.meta.slug}" if hit.meta.slug else ""
                    ts = hit.meta.timestamp[:10] if hit.meta.timestamp else "?"
                    print(f"  {ts}  {hit.meta.uuid}{slug_part}  turn {hit.turn_index+1:3d} ({hit.role:9s})  {hit.snippet}")
                if truncated:
                    print(
                        f"\nShowing the top {len(hits)} matches. "
                        f"There may be more; raise --limit to see them."
                    )
                else:
                    print(f"\n{len(hits)} matches found.")
        return

    if args.last is not None or args.context:
        import json as _json

        from .scanner import scan_projects
        from .summarize import load_summaries

        projects = scan_projects(**_scan_kwargs)
        summaries = load_summaries()
        cwd = os.path.realpath(os.getcwd())

        cwd_convos = []
        for p in projects:
            for c in p.conversations:
                if c.cwd and os.path.realpath(c.cwd) == cwd:
                    cwd_convos.append(c)
        cwd_convos.sort(key=lambda c: c.timestamp or "", reverse=True)

        if not cwd_convos:
            if args.json:
                print(_json.dumps({"project": cwd, "conversations": []}))
            else:
                print(f"No conversations found for {cwd}")
            return

        if args.context:
            selected = []
            selected_per_source: dict[str, int] = {}
            for conversation in cwd_convos:
                source_count = selected_per_source.get(conversation.source, 0)
                if source_count >= 5:
                    continue
                selected.append(conversation)
                selected_per_source[conversation.source] = source_count + 1
        else:
            selected = cwd_convos[:args.last]

        def _convo_record(c):
            size = _conversation_size(c.path)
            rec = {
                "uuid": c.uuid,
                "slug": c.slug,
                "source": c.source,
                "timestamp": c.timestamp,
                "summary": summaries.get(c.uuid, ""),
                "file": str(c.path),
                "size_bytes": size,
                "estimated_tokens": size // 4,
            }
            if args.context:
                turns = parse_jsonl(c.path, detail=DETAIL_TEXT)
                stats = get_stats(c.path)
                first_user = next((turn.text for turn in turns if turn.role == "user"), "")
                if not first_user and c.agent_path:
                    task_name = c.agent_path.rstrip("/").rsplit("/", 1)[-1]
                    first_user = f"[delegated task] {task_name} (prompt not recorded)"
                last_user = next(
                    (turn.text for turn in reversed(turns) if turn.role == "user"),
                    "",
                )
                last_agent = next(
                    (turn.text for turn in reversed(turns) if turn.role == "assistant"),
                    "",
                )
                rec.update({
                    "turn_count": len(turns),
                    "model": stats.model,
                    "effort": getattr(stats, "effort", ""),
                    "first_message": first_user,
                    "last_user_message": last_user,
                    "last_agent_message": last_agent,
                })
            if c.git_branch:
                rec["git_branch"] = c.git_branch
            return rec

        records = [_convo_record(conversation) for conversation in selected]

        if args.json:
            print(_json.dumps({
                "project": cwd,
                "total_for_project": len(cwd_convos),
                "showing": len(selected),
                "conversations": records,
            }, indent=2))
        else:
            label = "Context" if args.context else "Last"
            print(f"\n{label} for {_short_path(cwd)} ({len(cwd_convos)} total):\n")
            if args.context:
                # Wrapping wider than the terminal makes it wrap again at the
                # edge, which is the misalignment this is here to prevent.
                width = max(28, shutil.get_terminal_size((100, 24)).columns)

                def _print_context_field(label: str, text: str) -> None:
                    prefix = f"    {label:<9}"
                    continuation = " " * len(prefix)
                    complete = text or "(none)"
                    # Wrapping at the terminal edge restarts at column zero and
                    # breaks the label columns, so wrap under the label instead.
                    for index, paragraph in enumerate(complete.split("\n")):
                        wrapped = textwrap.wrap(
                            paragraph,
                            width=width,
                            initial_indent=prefix if index == 0 else continuation,
                            subsequent_indent=continuation,
                            # A long unbroken token is left whole. Splitting it
                            # would corrupt an id or a path someone copies out.
                            break_long_words=False,
                            break_on_hyphens=False,
                        ) or [(prefix if index == 0 else continuation).rstrip()]
                        for line in wrapped:
                            print(line)

                for record in records:
                    name = record["slug"] or record["uuid"][:8]
                    model = record["model"] or "?"
                    effort = record["effort"] or "?"
                    print(
                        f"  [{record['source']}]  {_fmt_ts(record['timestamp'])}  {name}"
                        f"  · {record['turn_count']} turns"
                        f"  · model={model}  · effort={effort}"
                    )
                    if record["summary"]:
                        _print_context_field("Summary:", record["summary"])
                    _print_context_field("First:", record["first_message"])
                    if record["last_user_message"] != record["first_message"]:
                        _print_context_field("You:", record["last_user_message"])
                    _print_context_field("Agent:", record["last_agent_message"])
                    print()
            else:
                for c in selected:
                    ts = _fmt_ts(c.timestamp)
                    name = c.slug or c.uuid[:8]
                    summary = summaries.get(c.uuid, "")
                    src = c.source
                    size = _conversation_size(c.path)
                    print(f"  {ts}  [{src}]  {name}  {_file_size_label(size)}")
                    if summary:
                        print(f"           {summary}")
        return

    if args.resume is not None:
        if args.resume in ("latest", "select"):
            from .scanner import scan_projects

            cwd = os.path.realpath(os.getcwd())
            projects = scan_projects(**_scan_kwargs)
            cwd_convos = [
                c
                for project in projects
                for c in project.conversations
                if c.cwd
                and os.path.realpath(c.cwd) == cwd
                and c.source in _RESUMABLE_SOURCES
            ]
            cwd_convos.sort(key=lambda c: c.timestamp or "", reverse=True)
            if not cwd_convos:
                source_note = f" from {source_arg}" if source_arg else ""
                print(f"No resumable conversations found{source_note} for {cwd}")
                return
            if args.resume == "select" and len(cwd_convos) > 1:
                meta = _pick_conversation(cwd_convos, cwd)
                if meta is None:
                    return
            else:
                meta = cwd_convos[0]
        else:
            paths = _resolve_args([args.resume], extra_dirs=_extra_dirs)
            if not paths:
                return
            from .parser import get_meta
            meta = get_meta(paths[0])
            if not meta:
                print(f"Error: could not read metadata from {paths[0]}")
                return
        cmd = _resume_cmd(meta.source, meta.uuid, remaining, yolo=args.yolo)
        if cmd is None:
            print(f"Error: resume not supported for {meta.source.title()} conversations (use handoff instead)")
            return
        name = meta.slug or meta.uuid[:8]
        print(f"Resuming: {name} ({meta.timestamp[:10]})")
        if meta.cwd:
            print(f"  cd {meta.cwd}")
        print(f"  {' '.join(cmd)}")
        if args.dry_run:
            return
        if meta.cwd and os.path.isdir(meta.cwd):
            os.chdir(meta.cwd)
        os.execvp(cmd[0], cmd)

    if args.handoff is not None:
        from .scanner import scan_projects
        cwd = os.path.realpath(os.getcwd())
        projects = scan_projects(**_scan_kwargs)
        cwd_convos = []
        for p in projects:
            for c in p.conversations:
                if c.cwd and os.path.realpath(c.cwd) == cwd:
                    cwd_convos.append(c)
        source_filter = source_arg
        target_agent_from_handoff = None
        if args.handoff in ("claude", "codex", "pi", "agy", "opencode"):
            if source_arg:
                target_agent_from_handoff = args.handoff
            else:
                source_filter = args.handoff
        if source_filter:
            cwd_convos = [c for c in cwd_convos if c.source == source_filter]
        if not cwd_convos:
            label = f" from {source_filter}" if source_filter else ""
            print(f"No conversations found{label} for {cwd}")
            return
        cwd_convos.sort(key=lambda c: c.timestamp or "", reverse=True)
        if args.handoff == "select" and len(cwd_convos) > 1:
            meta = _pick_conversation(cwd_convos, cwd)
            if meta is None:
                return
        else:
            meta = cwd_convos[0]
        out_dir = Path("output")
        out_dir.mkdir(exist_ok=True)
        turns = parse_jsonl(meta.path, detail=args.detail)
        stats = get_stats(meta.path)
        md = to_markdown(turns, stats=stats)
        filename = _export_filename(meta)
        out_path = out_dir / filename
        out_path.write_text(md, encoding="utf-8")
        name = meta.slug or meta.uuid[:8]
        print(f"Exported: {name} → {out_path}")
        message = f"Read the file {out_path.resolve()} for context from our last session, then summarize what we were working on and ask how to continue."
        target_agent = _handoff_agent(meta.source, args.handoff_agent or target_agent_from_handoff, args.yolo)
        cmd = _handoff_cmd(target_agent, message, remaining, yolo=args.yolo)
        display = " ".join(cmd[:-1]) + f' "{message}"'
        print(f"  {display}")
        if args.dry_run:
            return
        os.execvp(cmd[0], cmd)

    if args.list:
        from .scanner import scan_projects
        from .summarize import load_summaries
        projects = scan_projects(**_scan_kwargs)
        summaries = load_summaries()
        if args.json:
            import json as _json

            def _convo_dict(c):
                size = _conversation_size(c.path)
                rec = {
                    "uuid": c.uuid,
                    "slug": c.slug,
                    "source": c.source,
                    "timestamp": c.timestamp,
                    "cwd": c.cwd,
                    "preview": c.preview,
                    "summary": summaries.get(c.uuid, ""),
                    "file": str(c.path),
                    "size_bytes": size,
                    "estimated_tokens": size // 4,
                }
                if c.git_branch:
                    rec["git_branch"] = c.git_branch
                return rec

            total_convos = sum(len(p.conversations) for p in projects)
            print(_json.dumps({
                "total_projects": len(projects),
                "total_conversations": total_convos,
                "projects": [
                    {
                        "path": p.display_path,
                        "folder": p.folder_name,
                        "conversations": [_convo_dict(c) for c in p.conversations],
                    }
                    for p in projects
                ],
            }, indent=2))
        else:
            for p in projects:
                print(f"\n{p.display_path} ({len(p.conversations)} convos)")
                for c in p.conversations:
                    ts = c.timestamp[:10] if c.timestamp else "?"
                    name = c.slug or c.uuid[:8]
                    size = _conversation_size(c.path)
                    print(f"  {ts}  {name:30s}  {_file_size_label(size):>7s}  {c.preview[:50]}")
        return

    if args.show:
        paths = _resolve_args(args.show, extra_dirs=_extra_dirs)
        for p in paths:
            from .parser import get_meta
            meta = get_meta(p)
            turns = parse_jsonl(p, detail=args.detail)
            stats = get_stats(p)
            md = to_markdown(turns, stats=stats)
            # Truncate to ~10K words
            words = md.split()
            if len(words) > 10_000:
                md = " ".join(words[:10_000]) + f"\n\n... truncated ({len(words):,} words total, showing first 10,000)"
            if args.open:
                _open_in_sublime(md, meta)
            else:
                print(md)
        return

    if args.export_all:
        from .parser import get_meta
        from .scanner import scan_projects
        out_dir = Path(args.export_all)
        out_dir.mkdir(parents=True, exist_ok=True)
        projects = scan_projects(**_scan_kwargs)
        total = 0
        for proj in projects:
            for c in proj.conversations:
                try:
                    turns = parse_jsonl(c.path, detail=args.detail)
                    stats = get_stats(c.path)
                    md = to_markdown(turns, stats=stats)
                    name = _export_stem(c)
                    date = _export_date(c)
                    filename = f"{date}-{name}-{c.uuid[:8]}.md"
                    (out_dir / filename).write_text(f"# {name} ({date})\n**CWD:** {c.cwd}\n\n{md}", encoding="utf-8")
                    total += 1
                    print(f"  [{total}] {filename}")
                except Exception as e:
                    print(f"  SKIP {c.uuid[:8]}: {e}")
        print(f"\nExported {total} conversations to {out_dir}")
        return

    if args.concat:
        from pathlib import Path as P
        paths = _resolve_args(args.concat, extra_dirs=_extra_dirs)
        parts = []
        for p in paths:
            from .parser import get_meta
            meta = get_meta(p)
            turns = parse_jsonl(p, detail=args.detail)
            stats = get_stats(p)
            md = to_markdown(turns, stats=stats)
            name = _export_stem(meta) if meta else p.stem[:12]
            ts = (meta.timestamp[:10]) if meta else "?"
            cwd = meta.cwd if meta else "?"
            parts.append(f"# {name} ({ts})\n**CWD:** {cwd}\n\n{md}")
        combined = "\n\n---\n\n".join(parts)
        out_dir = ANALYSES_DIR.parent / "exports"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        proj = paths[0].parent.name.split("--")[-1].replace("-", " ").strip() or "cli"
        out_path = out_dir / f"{ts}-{proj}-{len(paths)}-convos-combined.md"
        out_path.write_text(combined, encoding="utf-8")
        print(f"Exported {len(paths)} conversations ({len(combined):,} chars, ~{len(combined)//4:,} tokens)")
        print(f"Saved to {out_path}")
        if args.open:
            _open_in_editor(out_path)
        return

    if args.analyze or args.deep:
        from .analyzer import (
            MULTI_PROMPT,
            SINGLE_PROMPT,
            analyze_deep,
            analyze_multi,
            analyze_single,
            gemini_available,
        )
        if not gemini_available():
            print("Error: set GEMINI_API_KEY env var")
            return
        # Resolve custom prompt (inline text or file path)
        custom_prompt = None
        if args.prompt:
            from pathlib import Path as P
            prompt_path = P(args.prompt)
            if prompt_path.is_file():
                custom_prompt = prompt_path.read_text(encoding="utf-8")
            else:
                custom_prompt = args.prompt
            if "{content}" not in custom_prompt:
                custom_prompt += "\n\nCONVERSATION:\n{content}"

        analyze_ids = args.deep or args.analyze
        paths = _resolve_args(analyze_ids, extra_dirs=_extra_dirs)
        def _progress(msg): print(f"  {msg}", flush=True)

        # Pre-compute output path so deep mode can save progress
        ANALYSES_DIR.mkdir(parents=True, exist_ok=True)
        proj = paths[0].parent.name.split("--")[-1].replace("-", " ").strip() or "cli"
        out_path = ANALYSES_DIR / _analysis_filename(proj, len(paths))

        if args.deep:
            # Deep mode: sequential pro→flash→pro analysis
            all_turns = []
            for p in paths:
                all_turns.extend(parse_jsonl(p, detail=args.detail))
            result = analyze_deep(all_turns, on_progress=_progress, prompt_template=custom_prompt, out_path=out_path)
        elif len(paths) == 1:
            turns = parse_jsonl(paths[0], detail=args.detail)
            prompt = custom_prompt or SINGLE_PROMPT
            result = analyze_single(turns, model=args.model, prompt_template=prompt, on_progress=_progress)
        else:
            convos = []
            for p in paths:
                meta = from_path_meta(p)
                label = meta or p.stem[:12]
                turns = parse_jsonl(p, detail=args.detail)
                convos.append((label, turns))
            prompt = custom_prompt or MULTI_PROMPT
            result = analyze_multi(convos, model=args.model, prompt_template=prompt, on_progress=_progress)

        # Save and print
        out_path.write_text(result, encoding="utf-8")
        print(result)
        print(f"\n--- Saved to {out_path} ---")
        from .analyzer import get_cost_summary
        print(f"\n--- Cost ---\n{get_cost_summary()}")
        return

    if args.summarize:
        from .scanner import scan_projects
        from .summarize import _load_api_key, summarize_all
        try:
            api_key = _load_api_key()
        except RuntimeError as e:
            print(f"Error: {e}")
            raise SystemExit(1) from e
        projects = scan_projects(**_scan_kwargs)

        import sys
        is_tty = sys.stdout.isatty()

        def on_progress(done, total, skipped, result):
            if result and result.startswith("ERROR"):
                print(f"  [{done}/{total}] {result}")
            elif result:
                print(f"  [{done}/{total}] {result[:80]}")
            elif is_tty:
                print(f"  [{done}/{total}] (cached)", end="\r", flush=True)

        print("Summarizing sessions...")
        done, skipped = summarize_all(projects, api_key, on_progress)
        print(f"\nDone. {done} processed, {skipped} already cached.")
        raise SystemExit(0)

    app = ConvoExplorer(**_scan_kwargs)
    app.run()

    # After TUI exits, check if user wants to resume a conversation
    if app._resume_meta:
        meta = app._resume_meta
        cmd = _resume_cmd(meta.source, meta.uuid, remaining)
        if cmd is None:
            print(f"Resume not supported for {meta.source.title()} — use handoff instead")
        else:
            name = meta.slug or meta.uuid[:8]
            print(f"Resuming: {name} ({meta.timestamp[:10]})")
            if meta.cwd:
                print(f"  cd {meta.cwd}")
            if meta.cwd and os.path.isdir(meta.cwd):
                os.chdir(meta.cwd)
            print(f"  {' '.join(cmd)}")
            os.execvp(cmd[0], cmd)

    if app._handoff_meta:
        meta = app._handoff_meta
        out_dir = Path("output")
        out_dir.mkdir(exist_ok=True)
        turns = parse_jsonl(meta.path)
        stats = get_stats(meta.path)
        md = to_markdown(turns, stats=stats)
        filename = _export_filename(meta)
        out_path = out_dir / filename
        out_path.write_text(md, encoding="utf-8")
        name = meta.slug or meta.uuid[:8]
        print(f"Exported: {name} → {out_path}")
        message = f"Read the file {out_path.resolve()} for context from our last session, then summarize what we were working on and ask how to continue."
        target_agent = _handoff_agent(meta.source, args.handoff_agent, args.yolo)
        cmd = _handoff_cmd(target_agent, message, remaining, yolo=args.yolo)
        display = " ".join(cmd[:-1]) + f' "{message}"'
        print(f"  {display}")
        if meta.cwd and os.path.isdir(meta.cwd):
            os.chdir(meta.cwd)
        os.execvp(cmd[0], cmd)


def _open_in_editor(path):
    """Open a file in Sublime Text (or fallback to default editor)."""
    import subprocess
    try:
        subprocess.Popen(["subl", str(path)])
    except FileNotFoundError:
        try:
            subprocess.Popen(["code", str(path)])
        except FileNotFoundError:
            os.startfile(str(path))


def _open_in_sublime(content: str, meta=None):
    """Write content to a temp file and open in Sublime Text."""
    import tempfile
    name = (meta.slug or meta.uuid[:8]) if meta else "preview"
    tmp = Path(tempfile.gettempdir()) / f"convo-{name}.md"
    tmp.write_text(content, encoding="utf-8")
    print(f"Opening in editor: {tmp}")
    _open_in_editor(tmp)


def _resolve_args(args: list[str], extra_dirs: list[Path] | None = None) -> list:
    """Resolve a mix of file paths and conversation IDs to Path objects."""
    from pathlib import Path as P
    file_paths = []
    ids_to_resolve = []
    for arg in args:
        p = P(arg)
        if p.is_file():
            file_paths.append(p)
        else:
            ids_to_resolve.append(arg)
    if ids_to_resolve:
        from .scanner import resolve_ids
        file_paths.extend(resolve_ids(ids_to_resolve, extra_dirs=extra_dirs))
    if not file_paths:
        print("Error: no conversations found for the given arguments")
        import sys
        sys.exit(1)
    return file_paths


def from_path_meta(path):
    """Quick label from a jsonl path."""
    from .parser import get_meta
    meta = get_meta(path)
    if meta:
        return f"{meta.slug or meta.uuid[:8]} ({meta.timestamp[:10]})"
    return None


if __name__ == "__main__":
    main()
