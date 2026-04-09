"""Textual TUI for browsing Claude Code conversations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
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
from .parser import ConversationMeta, parse_jsonl, to_markdown, get_stats, DETAIL_TEXT, DETAIL_TOOLS, DETAIL_RESULTS, DETAIL_FULL
from .analyzer import MODELS, DEFAULT_MODEL, SINGLE_PROMPT, MULTI_PROMPT


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
    ]

    TITLE = "convo-explorer"

    def __init__(self) -> None:
        super().__init__()
        self.projects: list[Project] = []
        self.current_meta: ConversationMeta | None = None
        self._dragging_sidebar = False
        self._model_index = 0
        self.gemini_model = MODELS[0]
        self.custom_single_prompt: str = SINGLE_PROMPT
        self.custom_multi_prompt: str = MULTI_PROMPT
        self._editing_prompt: str = "single"  # which prompt is being edited
        self._analyzing = False
        self._last_action: str = ""  # "analysis" or "export"

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with Vertical(id="sidebar"):
                yield Input(placeholder="Type to filter...", id="filter-input")
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
        projects = scan_projects()
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

    def _is_analyzed(self, analyzed: set[str], name: str) -> bool:
        """Check if any analysis file contains this name."""
        name_lower = name.lower().replace("\\", " ").replace("/", " ").replace("-", " ")
        return any(name_lower in a.replace("-", " ") for a in analyzed)

    def _populate_tree(self, projects: list[Project], filter_text: str = "") -> None:
        self.projects = projects
        tree = self.query_one("#nav-tree", Tree)
        tree.clear()
        analyzed = self._get_analyzed_set()

        total_convos = 0
        for p in projects:
            if filter_text and filter_text not in p.display_path.lower() and filter_text not in p.folder_name.lower():
                continue

            short = p.display_path
            if len(short) > 50:
                short = "..." + short[-47:]

            n = len(p.conversations)
            total_convos += n
            ts = p.conversations[0].timestamp[:10] if p.conversations else ""
            proj_name = Path(p.display_path).name if p.display_path else p.folder_name
            marker = " ★" if self._is_analyzed(analyzed, proj_name) else ""
            project_label = f"{short}  ({n})  {ts}{marker}"

            pnode = tree.root.add(
                project_label,
                data=NodeData(kind="project", project=p),
                expand=False,
            )

            for c in p.conversations:
                cts = c.timestamp[:10] if c.timestamp else "?"
                name = c.slug or c.uuid[:8]
                preview = c.preview[:45] if c.preview else ""
                convo_label = f"  {cts}  {name}  {preview}"
                pnode.add_leaf(
                    convo_label,
                    data=NodeData(kind="convo", project=p, meta=c),
                )

        shown = len([n for n in tree.root.children])
        self.query_one("#left-title", Static).update(
            f"PROJECTS ({shown})  ·  S=select  Ctrl+A=all"
        )
        self.query_one("#status-bar", Static).update(
            f" {shown} projects · {total_convos} conversations · Tab switch · S multi-select · A analyze · E export"
        )

    def _refresh_analyzed_markers(self) -> None:
        """Re-scan analyses dir and update ★ markers on project nodes."""
        analyzed = self._get_analyzed_set()
        tree = self.query_one("#nav-tree", Tree)
        for pnode in tree.root.children:
            data: NodeData = pnode.data
            if not data or data.kind != "project" or not data.project:
                continue
            proj_name = Path(data.project.display_path).name if data.project.display_path else ""
            label = str(pnode.label)
            # Strip old marker
            label = label.replace(" ★", "")
            if self._is_analyzed(analyzed, proj_name):
                label += " ★"
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
        md = to_markdown(turns)
        meta.turn_count = len(turns)
        header = f"## {meta.slug or meta.uuid}\n**Date:** {meta.timestamp[:19]}  \n**CWD:** {meta.cwd}\n\n---\n\n"
        self.call_from_thread(self._set_preview, header + md, len(turns))

    def _set_preview(self, md: str, turn_count: int) -> None:
        self.query_one("#preview", Markdown).update(md)
        label = f"PREVIEW ({turn_count} turns)" if turn_count else "PREVIEW"
        self.query_one("#right-title", Static).update(label)
        self.query_one("#preview-scroll", VerticalScroll).scroll_home()

    # --- Filter ---

    def on_input_changed(self, event: Input.Changed) -> None:
        self._populate_tree(self.projects, filter_text=event.value.lower().strip())

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
        # Strip existing marker
        if label.startswith("✓ ") or label.startswith("○ "):
            label = label[2:]
        marker = "✓ " if data.selected else ""
        node.set_label(f"{marker}{label}")

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
            self.do_export_multi(selected)
        elif self.current_meta:
            self.do_export_single(self.current_meta)
        else:
            self.notify("Select a conversation first", severity="warning")

    @work(thread=True)
    def do_export_single(self, meta: ConversationMeta) -> None:
        turns = parse_jsonl(meta.path)
        md = to_markdown(turns)
        out_dir = Path("output")
        out_dir.mkdir(exist_ok=True)
        name = meta.slug or meta.uuid[:12]
        ts = meta.timestamp[:10] if meta.timestamp else "export"
        out_path = out_dir / f"{name}_{ts}.md"
        out_path.write_text(md, encoding="utf-8")
        self.call_from_thread(self.notify, f"Exported to {out_path}")
        self._last_action = "export"

    @work(thread=True)
    def do_export_multi(self, nodes: list[NodeData]) -> None:
        out_dir = Path("output")
        out_dir.mkdir(exist_ok=True)
        for nd in nodes:
            meta = nd.meta
            turns = parse_jsonl(meta.path)
            md = to_markdown(turns)
            name = meta.slug or meta.uuid[:12]
            ts = meta.timestamp[:10] if meta.timestamp else "export"
            out_path = out_dir / f"{name}_{ts}.md"
            out_path.write_text(md, encoding="utf-8")
        self.call_from_thread(self.notify, f"Exported {len(nodes)} conversations to output/")
        self._last_action = "export"

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
            name = meta.slug or meta.uuid[:8]
            ts = meta.timestamp[:10] if meta.timestamp else "?"
            header = f"# {name} ({ts})\n**CWD:** {meta.cwd}\n\n"
            parts.append(header + md)

        combined = "\n\n---\n\n".join(parts)
        out_dir = ANALYSES_DIR.parent / "exports"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        first_meta = nodes[0].meta
        proj = Path(first_meta.cwd).name if first_meta and first_meta.cwd else "mixed"
        out_path = out_dir / f"{ts}-{proj}-{len(nodes)}-convos-combined.md"
        out_path.write_text(combined, encoding="utf-8")
        self.call_from_thread(self.notify, f"Combined export: {out_path}")
        self._last_action = "export"
        self.call_from_thread(self._set_preview, f"## Exported {len(nodes)} conversations\n\nSaved to `{out_path}`\n\nSize: {len(combined):,} chars (~{len(combined)//4:,} tokens)", 0)

    def action_open_folder(self) -> None:
        """Open the relevant folder based on last action."""
        import subprocess
        if self._last_action == "analysis":
            folder = ANALYSES_DIR
        elif self._last_action == "export":
            folder = ANALYSES_DIR.parent / "exports"
        else:
            folder = ANALYSES_DIR.parent  # show both
        folder.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["explorer", str(folder)])

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

    def action_cancel(self) -> None:
        if self._analyzing:
            self._cancel_analysis()
        else:
            # Clear filter if active
            filt = self.query_one("#filter-input", Input)
            if filt.value:
                filt.value = ""

    def action_quit(self) -> None:
        self.exit()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Browse and analyze Claude Code conversations")
    parser.add_argument("--analyze", nargs="+", metavar="ID_OR_PATH", help="Analyze conversations (JSONL paths, UUIDs, or slugs)")
    parser.add_argument("--concat", nargs="+", metavar="ID_OR_PATH", help="Export concatenated markdown (JSONL paths, UUIDs, or slugs)")
    parser.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL, help="Gemini model")
    parser.add_argument("--prompt", metavar="TEXT_OR_FILE", help="Custom analysis prompt (inline text or path to .txt/.md file). Use {content} as placeholder for conversation text, {count} for multi-convo count.")
    parser.add_argument("--detail", choices=["text", "tools", "results", "full"], default=None, help="Detail level: text, tools, results (default for analyze), full")
    parser.add_argument("--deep", nargs="+", metavar="ID_OR_PATH", help="Deep analysis: Pro for first chunk, Flash continues with context, Pro synthesizes. Uses full detail.")
    parser.add_argument("--list", action="store_true", help="List all projects and conversations")
    parser.add_argument("--show", nargs="+", metavar="ID_OR_PATH", help="Preview conversation (first ~10K words)")
    parser.add_argument("--open", action="store_true", help="Open in Sublime Text (use with --show or --concat)")
    args = parser.parse_args()
    if args.detail is None:
        args.detail = "results" if (args.deep or args.analyze) else "text"


    if args.list:
        from .scanner import scan_projects
        for p in scan_projects():
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
        paths = _resolve_args(args.show)
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

    if args.concat:
        from pathlib import Path as P
        paths = _resolve_args(args.concat)
        parts = []
        for p in paths:
            from .parser import get_meta
            meta = get_meta(p)
            turns = parse_jsonl(p, detail=args.detail)
            stats = get_stats(p)
            md = to_markdown(turns, stats=stats)
            name = (meta.slug or meta.uuid[:8]) if meta else p.stem[:12]
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
        paths = _resolve_args(analyze_ids)
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

    app = ConvoExplorer()
    app.run()


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


def _resolve_args(args: list[str]) -> list:
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
        file_paths.extend(resolve_ids(ids_to_resolve))
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
