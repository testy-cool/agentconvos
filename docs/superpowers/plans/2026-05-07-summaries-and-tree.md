# Session Summaries + Hierarchical Tree — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 1-sentence Gemini-generated session summaries and a hierarchical project tree to cc-convo-explorer.

**Architecture:** New `summarize.py` module handles Gemini flex inference calls and caching. `_populate_tree()` in `app.py` is rewritten to group projects by path prefix (`~/Work`, `~/.claude`, `[codex]`, `Other`) with summary text in leaf labels. A `--summarize` CLI flag enables cron-friendly batch summarization.

**Tech Stack:** Python 3.12+, google-genai SDK (flex inference via `service_tier`), Textual TUI

**Spec:** `docs/superpowers/specs/2026-05-07-summaries-and-tree-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/cc_convo_explorer/summarize.py` | Create | Gemini flex calls, summary caching, batch logic |
| `src/cc_convo_explorer/app.py` | Modify | Hierarchical tree, summary labels, `--summarize` CLI |
| `src/cc_convo_explorer/analyzer.py` | Modify | Rename `_load_env()` to public `load_env()` |

---

### Task 1: Create `summarize.py` — caching and API call logic

**Files:**
- Create: `src/cc_convo_explorer/summarize.py`

- [ ] **Step 1: Write the summary cache loader**

Create `src/cc_convo_explorer/summarize.py`:

```python
from __future__ import annotations

import json
import os
from pathlib import Path

from .parser import ConversationMeta, parse_jsonl, DETAIL_TEXT

SUMMARIES_DIR = Path.home() / ".claude" / "convo-explorer" / "summaries"
MODEL = "gemini-3.1-flash-lite-preview"

PROMPT = """Summarize this conversation session in one sentence, commit-message style.
Focus on what was accomplished or decided, based on these final turns.
Examples of good summaries:
- "Add flex inference summaries to convo-explorer"
- "Debug and fix Windmill flow timeout on large store batches"
- "Discuss AILookup monetization strategy and decide on verified listings"

Conversation (last turns):
{content}"""


def load_summaries() -> dict[str, str]:
    if not SUMMARIES_DIR.exists():
        return {}
    out: dict[str, str] = {}
    for f in SUMMARIES_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            out[f.stem] = data["summary"]
        except (json.JSONDecodeError, KeyError):
            continue
    return out


def _needs_summary(meta: ConversationMeta) -> bool:
    cache = SUMMARIES_DIR / f"{meta.uuid}.json"
    if not cache.exists():
        return True
    return meta.path.stat().st_mtime > cache.stat().st_mtime


def _write_cache(uuid: str, summary: str) -> None:
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone
    data = {
        "summary": summary,
        "model": MODEL,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (SUMMARIES_DIR / f"{uuid}.json").write_text(json.dumps(data, indent=2))
```

- [ ] **Step 2: Add the Gemini flex inference call**

Append to `src/cc_convo_explorer/summarize.py`:

```python
def _call_gemini_flex(prompt: str, api_key: str) -> str:
    from google import genai
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={"service_tier": "flex"},
    )
    return resp.text.strip().strip('"')


def summarize_session(meta: ConversationMeta, api_key: str) -> str:
    turns = parse_jsonl(meta.path, detail=DETAIL_TEXT, last_n=5)
    if not turns:
        return ""
    content = "\n\n".join(f"**{t.role}**: {t.text}" for t in turns)
    summary = _call_gemini_flex(PROMPT.format(content=content), api_key)
    _write_cache(meta.uuid, summary)
    return summary
```

- [ ] **Step 3: Add the batch summarization entry point**

Append to `src/cc_convo_explorer/summarize.py`:

```python
def summarize_all(
    projects: list,
    api_key: str,
    on_progress: callable | None = None,
) -> tuple[int, int]:
    done = 0
    total = sum(len(p.conversations) for p in projects)
    skipped = 0
    for project in projects:
        for meta in project.conversations:
            if not _needs_summary(meta):
                skipped += 1
                done += 1
                if on_progress:
                    on_progress(done, total, skipped, None)
                continue
            try:
                summary = summarize_session(meta, api_key)
                done += 1
                if on_progress:
                    on_progress(done, total, skipped, summary)
            except Exception as e:
                done += 1
                if on_progress:
                    on_progress(done, total, skipped, f"ERROR: {e}")
    return done, skipped
```

- [ ] **Step 4: Verify the module imports cleanly**

Run: `cd /home/testycool/Work/convo-explorer && uv run python -c "from cc_convo_explorer.summarize import load_summaries, summarize_all; print('OK')"`

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/cc_convo_explorer/summarize.py
git commit -m "Add session summary module with Gemini flex inference and caching"
```

---

### Task 2: Add `last_n` parameter to `parse_jsonl()`

The summarizer needs only the last 5 turns. `parse_jsonl()` currently returns all turns. Add a `last_n` parameter to avoid parsing entire large conversations.

**Files:**
- Modify: `src/cc_convo_explorer/parser.py:213-225`

- [ ] **Step 1: Add `last_n` parameter to `parse_jsonl()`**

In `parser.py`, the `parse_jsonl()` function signature is at line 213. Modify it:

Current (line 213-225):
```python
def parse_jsonl(path, detail=DETAIL_TEXT):
    """Return list[Turn] from a .jsonl conversation file."""
    fmt = _detect_format(path)
    if fmt == "codex":
        return _parse_jsonl_codex(path, detail)
    return _parse_jsonl_claude(path, detail)
```

Replace with:
```python
def parse_jsonl(path, detail=DETAIL_TEXT, last_n=0):
    """Return list[Turn] from a .jsonl conversation file."""
    fmt = _detect_format(path)
    if fmt == "codex":
        turns = _parse_jsonl_codex(path, detail)
    else:
        turns = _parse_jsonl_claude(path, detail)
    if last_n > 0:
        return turns[-last_n:]
    return turns
```

- [ ] **Step 2: Verify existing behavior is unchanged**

Run: `cd /home/testycool/Work/convo-explorer && uv run python -c "from cc_convo_explorer.parser import parse_jsonl; print(len(parse_jsonl(list(__import__('pathlib').Path.home().joinpath('.claude/projects').iterdir()).__next__().glob('*.jsonl').__next__())))"`

Expected: prints a number (the turn count of any conversation)

- [ ] **Step 3: Commit**

```bash
git add src/cc_convo_explorer/parser.py
git commit -m "Let parse_jsonl return only the last N turns for lightweight sampling"
```

---

### Task 3: Add `--summarize` CLI flag

**Files:**
- Modify: `src/cc_convo_explorer/app.py:885-1041` (the `main()` function)

- [ ] **Step 1: Add the argparse flag**

In `app.py`, inside `main()`, find the argparse block (around line 895-930). Add after the existing flags:

```python
    p.add_argument("--summarize", action="store_true",
                   help="Generate missing session summaries via Gemini (cron-friendly)")
```

- [ ] **Step 2: Add the dispatch logic**

In `app.py`, find the CLI dispatch section (after argparse, around line 940-1040). Add before the TUI launch (before `app = ConvoExplorer(...)`):

```python
    if args.summarize:
        from .summarize import summarize_all
        from .analyzer import _load_env
        _load_env()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("GEMINI_API_KEY not set. Check .env or environment.")
            raise SystemExit(1)
        projects = scan_projects(projects_dir=projects_dir)

        def on_progress(done, total, skipped, result):
            status = f"  [{done}/{total}]"
            if result and result.startswith("ERROR"):
                print(f"{status} {result}")
            elif result:
                print(f"{status} {result[:80]}")
            else:
                print(f"{status} (cached)", end="\r")

        print(f"Summarizing sessions...")
        done, skipped = summarize_all(projects, api_key, on_progress)
        print(f"\nDone. {done} processed, {skipped} already cached.")
        raise SystemExit(0)
```

- [ ] **Step 3: Test the CLI flag parses**

Run: `cd /home/testycool/Work/convo-explorer && uv run cc-convo-explorer --summarize --help 2>&1 | head -5`

Expected: help output showing `--summarize` flag

- [ ] **Step 4: Commit**

```bash
git add src/cc_convo_explorer/app.py
git commit -m "Expose batch session summarization as a cron-friendly CLI flag"
```

---

### Task 4: Rewrite `_populate_tree()` for hierarchical grouping

**Files:**
- Modify: `src/cc_convo_explorer/app.py:120-128` (NodeData), `src/cc_convo_explorer/app.py:268-331` (`_populate_tree()`)

- [ ] **Step 1: Add "group" kind to NodeData**

At `app.py:120-128`, the `NodeData` dataclass. Add `"group"` as a valid kind (just documentation — the field is a str, not an enum). No code change needed, but we'll use `kind="group"` for group nodes.

- [ ] **Step 2: Add the grouping helper**

Add this function above `_populate_tree()` (around line 265):

```python
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
```

- [ ] **Step 3: Rewrite `_populate_tree()` with hierarchy**

Replace the entire `_populate_tree()` method (lines 268-331) with:

```python
    def _populate_tree(self, projects: list, filter_text: str = "") -> None:
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

        self.query_one("#tree-title", Static).update(
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
```

- [ ] **Step 4: Test the TUI launches with the new tree**

Run: `cd /home/testycool/Work/convo-explorer && timeout 5 uv run cc-convo-explorer --list 2>&1 | head -30`

Expected: project listing output (the `--list` flag prints to stdout, so we can verify grouping without launching the full TUI)

- [ ] **Step 5: Commit**

```bash
git add src/cc_convo_explorer/app.py
git commit -m "Organize project tree into collapsible groups with session summaries"
```

---

### Task 5: Wire up `_load_env` for shared use

The summarize module needs `_load_env()` from `analyzer.py`. Currently it's a private function. Make it importable.

**Files:**
- Modify: `src/cc_convo_explorer/analyzer.py:127-147`

- [ ] **Step 1: Rename `_load_env` to `load_env`**

In `analyzer.py`, rename the function at line 127 from `_load_env` to `load_env`. Also update all call sites within `analyzer.py` (search for `_load_env()`  — used in `analyze_single`, `analyze_deep`, `analyze_multi`).

- [ ] **Step 2: Update the import in app.py**

In `app.py`, the `--summarize` dispatch (added in Task 3) imports `from .analyzer import _load_env`. Change to `from .analyzer import load_env` and update the call.

- [ ] **Step 3: Commit**

```bash
git add src/cc_convo_explorer/analyzer.py src/cc_convo_explorer/app.py
git commit -m "Make env loader public so summarize module can reuse it"
```

---

### Task 6: Test end-to-end

- [ ] **Step 1: Test `--summarize` with real sessions**

Run: `cd /home/testycool/Work/convo-explorer && uv run cc-convo-explorer --summarize 2>&1 | tail -5`

Expected: progress output showing sessions being summarized, ending with a "Done" line.

- [ ] **Step 2: Verify cache files were created**

Run: `ls ~/.claude/convo-explorer/summaries/ | head -10`

Expected: UUID-named `.json` files

- [ ] **Step 3: Inspect a summary**

Run: `cat ~/.claude/convo-explorer/summaries/$(ls ~/.claude/convo-explorer/summaries/ | head -1)`

Expected: JSON with `summary`, `model`, `generated_at` fields

- [ ] **Step 4: Launch TUI and verify tree**

Run: `cd /home/testycool/Work/convo-explorer && uv run cc-convo-explorer`

Verify:
- Projects are grouped under `~/Work`, `~/.claude`, `[codex]`, `Other`
- Group nodes are collapsible
- CWD project is auto-expanded
- Conversation leaves show summaries (not just previews)
- Selection, filtering, export, and analysis still work

- [ ] **Step 5: Test `--list` shows hierarchy**

Run: `cd /home/testycool/Work/convo-explorer && uv run cc-convo-explorer --list 2>&1 | head -20`

Verify the output reflects the new grouping.

- [ ] **Step 6: Final commit if any fixups needed**

```bash
git add -A
git commit -m "Fix any issues found during end-to-end testing"
```

(Skip if no fixes needed.)
