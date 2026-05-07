"""Textual TUI for browsing Claude Code conversations."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rich.text import Text

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
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

from .scanner import Project, scan_projects


def _group_key(display_path: str, source: str) -> tuple[str, str]:
    """Return (group_name, relative_label) for a project path."""
    home = str(Path.home())
    if display_path.startswith(home):
        display_path = "~" + display_path[len(home):]

    if source == "codex":
        return "[codex]", display_path

    if display_path.startswith("~/Work"):
        rest = display_path[len("~/Work/"):]
        return "~/Work", rest or display_path
    if display_path.startswith("~/."):
        parts = display_path.split("/")
        group = "/".join(parts[:2])  # e.g. "~/.claude"
        rest = "/".join(parts[2:])
        return group, rest or display_path
    if display_path.startswith("~/"):
        return "~", display_path[len("~/"):]

    return "Other", display_path


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
from .parser import ConversationMeta, parse_jsonl, to_markdown, get_stats, search_conversations, DETAIL_TEXT, DETAIL_TOOLS, DETAIL_RESULTS, DETAIL_FULL
from .analyzer import MODELS, DEFAULT_MODEL, SINGLE_PROMPT, MULTI_PROMPT


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


class ConvoExplorer(App):
    CSS = """
    #main { height: 1fr; }
    #sidebar { width: 40%; min-width: 30; max-width: 90; }
    #resize-handle {
        width: 1;
        height: 1fr;
        background: $surface-lighten-2;
        color: $text-muted;
    }
    #resize-handle:hover { background: $accent; }
    #content { width: 1fr; }
    #filter-input { dock: top; }
    #nav-tree { height: 1fr; }
    #preview-scroll { height: 1fr; }
    #preview { padding: 1 2; }
    #status-bar { dock: bottom; height: 1; background: $accent; color: $text; padding: 0 1; }
    .panel-title { dock: top; height: 1; background: $boost; padding: 0 1; text-style: bold; }
    Tree { scrollbar-size: 1 1; }
    #prompt-editor { height: 1fr; }
    #prompt-panel { height: 1fr; }
    #prompt-bar { dock: bottom; height: 3; }
    #prompt-bar Button { width: 1fr; margin: 0 1; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=False),
        Binding("e", "export", "Export MD", priority=False),
        Binding("c", "export_concat", "Export combined", priority=False),
        Binding("a", "analyze", "Analyze (Gemini)", priority=False),
        Binding("s", "toggle_select", "Select", priority=False),
        Binding("tab", "toggle_focus", "Switch panel", priority=True),
        Binding("ctrl+a", "select_all", "Select all", priority=False),
        Binding("ctrl+d", "deselect_all", "Deselect all", priority=False),
        Binding("m", "cycle_model", "Model", priority=False),
        Binding("p", "edit_prompt", "Edit prompt", priority=False),
        Binding("o", "open_folder", "Open folder", priority=False),
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("slash", "search", "Search", priority=False),
        Binding("r", "resume", "Resume in Claude", priority=False),
    ]

    TITLE = "cc-convo-explorer"

    def __init__(self, extra_dirs: list[Path] | None = None) -> None:
        super().__init__()
        self.projects: list[Project] = []
        self._extra_dirs = extra_dirs
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
        self._search_cache: dict[str, str] = {}  # uuid -> searchable text (last 10 turns)

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with Vertical(id="sidebar"):
                yield Input(placeholder="Filter convos...  (Enter for deep search)", id="filter-input")
                yield Static("PROJECTS", classes="panel-title", id="left-title")
                yield Tree("Conversations", id="nav-tree")
            yield Static("┃", id="resize-handle")
            with Vertical(id="content"):
                yield Static("PREVIEW", classes="panel-title", id="right-title")
                with VerticalScroll(id="preview-scroll"):
                    yield Markdown("*Select a project, then a conversation*", id="preview")
                with Vertical(id="prompt-panel"):
                    yield TextArea(id="prompt-editor", language="markdown")
                    with Horizontal(id="prompt-bar"):
                        yield Button("Save & Close", id="prompt-save", variant="primary")
                        yield Button("Switch Single/Multi", id="prompt-switch", variant="default")
                        yield Button("Reset Default", id="prompt-reset", variant="warning")
        yield Static("Loading...", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#prompt-panel").display = False
        tree = self.query_one("#nav-tree", Tree)
        tree.show_root = False
        tree.guide_depth = 3
        self.load_projects()
        tree.focus()

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
        self._dragging_sidebar = False

    # --- Data loading ---

    @work(thread=True)
    def load_projects(self) -> None:
        projects = scan_projects(extra_dirs=self._extra_dirs)
        # Build search cache: last 10 turns per conversation
        cache = {}
        for p in projects:
            for c in p.conversations:
                try:
                    turns = parse_jsonl(c.path)
                    tail = turns[-10:] if len(turns) > 10 else turns
                    cache[c.uuid] = "\n".join(t.text for t in tail).lower()
                except Exception:
                    cache[c.uuid] = ""
        self._search_cache = cache
        self.call_from_thread(self._populate_tree, projects)

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
        analyzed = self._get_analyzed_set()
        name_lower = name.lower().replace("\\", " ").replace("/", " ").replace("-", " ")
        return any(name_lower in a.replace("-", " ") for a in analyzed)

    def _populate_tree(self, projects: list, filter_text: str = "") -> None:
        self.projects = projects
        tree = self.query_one(Tree)
        tree.clear()
        tree.root.data = None
        ft = filter_text.strip().lower()

        summaries = {}
        try:
            from .summarize import load_summaries
            summaries = load_summaries()
        except Exception:
            pass

        cwd = os.getcwd()
        groups: dict[str, list[tuple[str, object]]] = {}
        filtered_count = 0

        for proj in projects:
            convos = proj.conversations
            if ft:
                convos = [
                    c for c in convos
                    if ft in (self._search_cache.get(c.uuid, "") or "").lower()
                    or ft in (c.slug or "").lower()
                    or ft in c.uuid.lower()
                    or ft in proj.display_path.lower()
                ]
            if not convos:
                continue

            source = convos[0].source if convos else "claude"
            gkey, rel_label = _group_key(proj.display_path, source)
            groups.setdefault(gkey, []).append((rel_label, proj, convos))
            filtered_count += len(convos)

        self.query_one("#left-title", Static).update(
            f" PROJECTS ({filtered_count})"
        )

        group_order = ["~/Work"]
        for k in sorted(groups.keys()):
            if k not in group_order and k not in ("[codex]", "Other"):
                group_order.append(k)
        if "[codex]" in groups:
            group_order.append("[codex]")
        if "Other" in groups:
            group_order.append("Other")

        for gkey in group_order:
            if gkey not in groups:
                continue
            items = groups[gkey]
            group_node = tree.root.add(
                gkey,
                data=NodeData(kind="group"),
                expand=any(
                    cwd == p.display_path or cwd.startswith(p.display_path + "/")
                    for _, p, _ in items
                ) or bool(ft),
            )

            for rel_label, proj, convos in sorted(items, key=lambda x: x[0]):
                is_cwd = cwd == proj.display_path
                date_str = convos[0].timestamp[:10] if convos else ""
                count = len(convos)

                plabel = Text()
                if is_cwd:
                    plabel.append("● ", "bold cyan")
                plabel.append(f"{rel_label}  ({count})  {date_str}")
                if self._is_analyzed(proj.folder_name):
                    plabel.append(" ★", "yellow")

                pnode = group_node.add(
                    plabel,
                    data=NodeData(kind="project", project=proj, is_cwd=is_cwd),
                    expand=is_cwd or bool(ft),
                )

                for c in convos:
                    d = c.timestamp[:10] if c.timestamp else ""
                    slug = c.slug or c.uuid[:8]
                    summary = summaries.get(c.uuid, "")
                    if summary:
                        preview = summary[:60]
                    else:
                        preview = (c.preview or "")[:45]
                    pnode.add_leaf(
                        f"  {d}  {slug}  {preview}",
                        data=NodeData(kind="convo", meta=c, project=proj),
                    )

        shown = sum(len(items) for items in groups.values())
        self.query_one("#status-bar", Static).update(
            f" {shown} projects · {filtered_count} conversations · / search · S select · Tab switch · A analyze · E export"
        )

    def _refresh_analyzed_markers(self) -> None:
        """Re-scan analyses dir and update ★ markers on project nodes."""
        tree = self.query_one("#nav-tree", Tree)
        for pnode in tree.root.children:
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
            self.current_meta = data.meta
            self.load_preview(data.meta)

    @work(thread=True)
    def load_preview(self, meta: ConversationMeta) -> None:
        turns = parse_jsonl(meta.path)
        meta.turn_count = len(turns)
        # Show last 10 turns for quick preview
        tail = turns[-10:] if len(turns) > 10 else turns
        md = to_markdown(tail)
        skipped = len(turns) - len(tail)
        header = f"## {meta.slug or meta.uuid}\n**Date:** {meta.timestamp[:19]}  \n**CWD:** {meta.cwd}\n**Turns:** {len(turns)} total"
        if skipped:
            header += f" (showing last {len(tail)})"
        header += "\n\n---\n\n"
        self.call_from_thread(self._set_preview, header + md, len(turns))

    def _set_preview(self, md: str, turn_count: int) -> None:
        self.query_one("#preview", Markdown).update(md)
        label = f"PREVIEW ({turn_count} turns)" if turn_count else "PREVIEW"
        self.query_one("#right-title", Static).update(label)
        self.query_one("#preview-scroll", VerticalScroll).scroll_home()

    # --- Filter ---

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "filter-input":
            self._populate_tree(self.projects, filter_text=event.value.lower().strip())
            # Auto-preview if filter narrows to exactly 1 conversation
            tree = self.query_one("#nav-tree", Tree)
            all_convos = [
                cnode for pnode in tree.root.children
                for cnode in pnode.children
                if cnode.data and cnode.data.kind == "convo"
            ]
            if len(all_convos) == 1:
                node = all_convos[0]
                tree.select_node(node)
                if node.data.meta:
                    self.current_meta = node.data.meta
                    self.load_preview(node.data.meta)

    # --- Multi-select ---

    def _get_selected_nodes(self) -> list[NodeData]:
        """Collect all nodes marked as selected."""
        tree = self.query_one("#nav-tree", Tree)
        selected = []
        for pnode in tree.root.children:
            pd: NodeData = pnode.data
            if pd and pd.selected:
                # Whole project selected — include all its convos
                for cnode in pnode.children:
                    cd: NodeData = cnode.data
                    if cd and cd.meta:
                        selected.append(cd)
            else:
                for cnode in pnode.children:
                    cd: NodeData = cnode.data
                    if cd and cd.selected and cd.meta:
                        selected.append(cd)
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

        # If toggling a project, toggle all its children too
        if data.kind == "project":
            for child in node.children:
                cd: NodeData = child.data
                if cd:
                    cd.selected = data.selected
                    self._update_node_label(child)

        self._update_selection_count()

    def action_select_all(self) -> None:
        tree = self.query_one("#nav-tree", Tree)
        for pnode in tree.root.children:
            pd: NodeData = pnode.data
            if pd:
                pd.selected = True
                self._update_node_label(pnode)
                for cnode in pnode.children:
                    cd: NodeData = cnode.data
                    if cd:
                        cd.selected = True
                        self._update_node_label(cnode)
        self._update_selection_count()

    def action_deselect_all(self) -> None:
        tree = self.query_one("#nav-tree", Tree)
        for pnode in tree.root.children:
            pd: NodeData = pnode.data
            if pd:
                pd.selected = False
                self._update_node_label(pnode)
                for cnode in pnode.children:
                    cd: NodeData = cnode.data
                    if cd:
                        cd.selected = False
                        self._update_node_label(cnode)
        self._update_selection_count()

    def _estimate_tokens(self, nodes: list[NodeData]) -> str:
        """Estimate token count from file sizes (~4 chars/token)."""
        total_bytes = 0
        for nd in nodes:
            if nd.meta:
                try:
                    total_bytes += nd.meta.path.stat().st_size
                except OSError:
                    pass
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
            self.query_one("#right-title", Static).update("PREVIEW")
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
        self.query_one("#right-title", Static).update("PREVIEW")
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
        if self.current_meta.source == "codex":
            self.notify("Resume not supported for Codex conversations", severity="warning")
            return
        self._resume_meta = self.current_meta
        self.exit()

    # --- Search ---

    def action_search(self) -> None:
        filt = self.query_one("#filter-input", Input)
        filt.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "filter-input" and event.value.strip():
            # Enter in filter: deep search across all conversation content
            self.do_search(event.value.strip())

    @work(thread=True)
    def do_search(self, query: str) -> None:
        self.call_from_thread(
            lambda: self.query_one("#right-title", Static).update(f"SEARCHING: \"{query}\"...")
        )
        all_paths = [c.path for p in self.projects for c in p.conversations]
        hits = search_conversations(all_paths, query)
        if not hits:
            self.call_from_thread(self._set_preview, f"## No results for \"{query}\"\n\nSearched {len(all_paths)} conversations.", 0)
            self.call_from_thread(
                lambda: self.query_one("#right-title", Static).update("SEARCH RESULTS (0)")
            )
            return
        # Format results
        lines = [f"## Search: \"{query}\"\n\n**{len(hits)} matches** across {len(all_paths)} conversations\n"]
        for hit in hits:
            name = hit.meta.slug or hit.meta.uuid[:8]
            ts = hit.meta.timestamp[:10] if hit.meta.timestamp else "?"
            lines.append(f"### {name} ({ts}) — turn {hit.turn_index + 1} ({hit.role})")
            lines.append(f"> {hit.snippet}\n")
        md = "\n".join(lines)
        self.call_from_thread(self._set_preview, md, len(hits))
        self.call_from_thread(
            lambda: self.query_one("#right-title", Static).update(f"SEARCH RESULTS ({len(hits)})")
        )

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


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Browse and analyze Claude Code and Codex conversations")
    parser.add_argument("--analyze", nargs="+", metavar="ID_OR_PATH", help="Analyze conversations (JSONL paths, UUIDs, or slugs)")
    parser.add_argument("--concat", nargs="+", metavar="ID_OR_PATH", help="Export concatenated markdown (JSONL paths, UUIDs, or slugs)")
    parser.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL, help="Gemini model")
    parser.add_argument("--prompt", metavar="TEXT_OR_FILE", help="Custom analysis prompt (inline text or path to .txt/.md file). Use {content} as placeholder for conversation text, {count} for multi-convo count.")
    parser.add_argument("--detail", choices=["text", "tools", "results", "full"], default=None, help="Detail level: text, tools, results (default for analyze), full")
    parser.add_argument("--deep", nargs="+", metavar="ID_OR_PATH", help="Deep analysis: Pro for first chunk, Flash continues with context, Pro synthesizes. Uses full detail.")
    parser.add_argument("--search", metavar="QUERY", help="Search all conversations for a string")
    parser.add_argument("--list", action="store_true", help="List all projects and conversations")
    parser.add_argument("--show", nargs="+", metavar="ID_OR_PATH", help="Preview conversation (first ~10K words)")
    parser.add_argument("--open", action="store_true", help="Open in Sublime Text (use with --show or --concat)")
    parser.add_argument("--resume", nargs=1, metavar="ID_OR_PATH", help="Resume a conversation with claude -r (add extra claude flags after --)")
    parser.add_argument("--dry-run", action="store_true", help="Print the command instead of running it (use with --resume)")
    parser.add_argument("--handoff", action="store_true", help="Export latest CWD conversation and start a new Claude session with it as context")
    parser.add_argument("--export-all", metavar="DIR", help="Export every conversation as individual markdown files to DIR")
    parser.add_argument("--projects-dir", nargs="+", metavar="DIR", help="Additional projects directories to scan (e.g. copied from other machines)")
    parser.add_argument("--summarize", action="store_true",
                        help="Generate missing session summaries via Gemini (cron-friendly)")
    args, remaining = parser.parse_known_args()

    # Parse extra project dirs
    _extra_dirs = [Path(d) for d in args.projects_dir] if args.projects_dir else None
    if args.detail is None:
        args.detail = "results" if (args.deep or args.analyze) else "text"


    if args.search:
        from .scanner import scan_projects
        projects = scan_projects(extra_dirs=_extra_dirs)
        all_paths = [c.path for p in projects for c in p.conversations]
        print(f"Searching {len(all_paths)} conversations for \"{args.search}\"...\n")
        hits = search_conversations(all_paths, args.search)
        if not hits:
            print("No results found.")
        else:
            for hit in hits:
                slug_part = f"  {hit.meta.slug}" if hit.meta.slug else ""
                ts = hit.meta.timestamp[:10] if hit.meta.timestamp else "?"
                print(f"  {ts}  {hit.meta.uuid}{slug_part}  turn {hit.turn_index+1:3d} ({hit.role:9s})  {hit.snippet}")
            print(f"\n{len(hits)} matches found.")
        return

    if args.resume:
        paths = _resolve_args(args.resume, extra_dirs=_extra_dirs)
        if not paths:
            return
        from .parser import get_meta
        meta = get_meta(paths[0])
        if not meta:
            print(f"Error: could not read metadata from {paths[0]}")
            return
        if meta.source == "codex":
            print("Error: resume not supported for Codex conversations")
            return
        # Pass any extra/unknown args straight to claude
        cmd = ["claude", "--dangerously-skip-permissions", "-r", meta.uuid] + remaining
        name = meta.slug or meta.uuid[:8]
        print(f"Resuming: {name} ({meta.timestamp[:10]})")
        if meta.cwd:
            print(f"  cd {meta.cwd}")
        print(f"  {' '.join(cmd)}")
        if args.dry_run:
            return
        if meta.cwd and os.path.isdir(meta.cwd):
            os.chdir(meta.cwd)
        os.execvp("claude", cmd)

    if args.handoff:
        from .scanner import scan_projects
        cwd = os.path.realpath(os.getcwd())
        projects = scan_projects(extra_dirs=_extra_dirs)
        cwd_convos = []
        for p in projects:
            if os.path.realpath(p.display_path) == cwd:
                cwd_convos.extend(p.conversations)
        if not cwd_convos:
            print(f"No conversations found for {cwd}")
            return
        cwd_convos.sort(key=lambda c: c.timestamp or "", reverse=True)
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
        cmd = ["claude", "--dangerously-skip-permissions"] + remaining + [message]
        display = " ".join(cmd[:-1]) + f' "{message}"'
        print(f"  {display}")
        if args.dry_run:
            return
        os.execvp("claude", cmd)

    if args.list:
        from .scanner import scan_projects
        for p in scan_projects(extra_dirs=_extra_dirs):
            print(f"\n{p.display_path} ({len(p.conversations)} convos)")
            for c in p.conversations:
                ts = c.timestamp[:10] if c.timestamp else "?"
                name = c.slug or c.uuid[:8]
                size = c.path.stat().st_size if c.path.exists() else 0
                tokens = size // 4
                if tokens >= 1_000_000:
                    tok_str = f"{tokens / 1_000_000:.1f}M"
                elif tokens >= 1000:
                    tok_str = f"{tokens // 1000}K"
                else:
                    tok_str = str(tokens)
                print(f"  {ts}  {name:30s}  ~{tok_str:>6s} tok  {c.preview[:50]}")
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
        from .scanner import scan_projects
        from .parser import get_meta
        out_dir = Path(args.export_all)
        out_dir.mkdir(parents=True, exist_ok=True)
        projects = scan_projects(extra_dirs=_extra_dirs)
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
        from .analyzer import gemini_available, analyze_single, analyze_multi, analyze_deep, SINGLE_PROMPT, MULTI_PROMPT
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
        from .summarize import summarize_all, _load_api_key
        from .scanner import scan_projects
        try:
            api_key = _load_api_key()
        except RuntimeError as e:
            print(f"Error: {e}")
            raise SystemExit(1)
        projects = scan_projects(extra_dirs=_extra_dirs)

        def on_progress(done, total, skipped, result):
            status = f"  [{done}/{total}]"
            if result and result.startswith("ERROR"):
                print(f"{status} {result}")
            elif result:
                print(f"{status} {result[:80]}")
            else:
                print(f"{status} (cached)", end="\r")

        print("Summarizing sessions...")
        done, skipped = summarize_all(projects, api_key, on_progress)
        print(f"\nDone. {done} processed, {skipped} already cached.")
        raise SystemExit(0)

    app = ConvoExplorer(extra_dirs=_extra_dirs)
    app.run()

    # After TUI exits, check if user wants to resume a conversation
    if app._resume_meta:
        meta = app._resume_meta
        name = meta.slug or meta.uuid[:8]
        cmd = ["claude", "--dangerously-skip-permissions", "-r", meta.uuid] + remaining
        print(f"Resuming: {name} ({meta.timestamp[:10]})")
        if meta.cwd:
            print(f"  cd {meta.cwd}")
        if meta.cwd and os.path.isdir(meta.cwd):
            os.chdir(meta.cwd)
        print(f"  {' '.join(cmd)}")
        os.execvp("claude", cmd)


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
        if p.exists() and p.suffix == ".jsonl":
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
