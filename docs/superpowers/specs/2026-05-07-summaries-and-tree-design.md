# Session Summaries + Hierarchical Tree

## Overview

Two features for cc-convo-explorer:
1. Auto-generated 1-sentence session summaries using Gemini 3.1 Flash Lite with flex inference
2. Hierarchical project tree grouped by path segments with `~` prefix

## Feature 1: Session Summaries

### Input

Last 5 turns of each conversation, extracted at `text` detail level (no tool output). Keeps token count minimal.

### LLM Call

- Model: `gemini-3.1-flash-lite-preview`
- Flex inference: `config={"service_tier": "flex"}` — 50% cheaper, same quality, variable latency (acceptable for background job)
- Prompt: instruct the model to produce a single commit-message-style sentence summarizing what was accomplished in the session, based on the last few turns

### Cache

- Location: `~/.claude/convo-explorer/summaries/{uuid}.json`
- Schema: `{"summary": str, "model": str, "generated_at": str (ISO 8601)}`
- One file per session. Existence = "already summarized". No invalidation needed — sessions are append-only, but if a session's mtime is newer than the summary file, re-summarize.

### CLI Entry Point

`cc-convo-explorer --summarize` — headless batch mode:
1. Scan all projects (reuse `scanner.scan_projects()`)
2. For each conversation, check if `summaries/{uuid}.json` exists and is current
3. If not, parse last 5 turns, call Gemini, write cache file
4. Print progress: `Summarized 12/47 sessions ($0.003)`

Designed to be cron-friendly. Exits 0 on success.

### TUI Integration

At startup, `_populate_tree()` loads all summary files into a `dict[str, str]` (uuid → summary text). When building conversation leaf nodes, use the summary as the label if available, fall back to the existing preview (first 120 chars of first user message).

### New Module: `summarize.py`

Separate from `analyzer.py` (which handles heavy multi-chunk analysis). Contains:
- `summarize_session(meta: ConversationMeta, api_key: str) → str` — parse last 5 turns, call Gemini, write cache, return summary
- `summarize_all(projects: list[Project], api_key: str, on_progress: Callable)` — batch entry point
- `load_summaries() → dict[str, str]` — read all cached summaries into memory
- Reuses `_load_env()` from `analyzer.py` for API key resolution

### Prompt

```
Summarize this conversation session in one sentence, commit-message style.
Focus on what was accomplished or decided, based on these final turns.
Examples of good summaries:
- "Add flex inference summaries to convo-explorer"
- "Debug and fix Windmill flow timeout on large store batches"
- "Discuss AILookup monetization strategy and decide on verified listings"

Conversation (last turns):
{content}
```

## Feature 2: Hierarchical Project Tree

### Grouping Logic

In `_populate_tree()`, after receiving the project list from the scanner:

1. Replace `os.path.expanduser("~")` prefix with `~` in each `display_path`
2. Determine group key from the normalized path:
   - Starts with `~/Work` → group `~/Work`
   - Starts with `~/.` (dotfiles) → group by first dotdir, e.g. `~/.claude`
   - Codex projects (source="codex") → group `[codex]`
   - Anything else → group `Other`
3. Within each group, the project label is the path relative to the group key (e.g. `~/Work/convo-explorer` → `convo-explorer`)
4. Groups are tree nodes (collapsible). Projects are child nodes. Conversations are leaf nodes.

### Tree Appearance

```
PROJECTS (47)
├─ ~/Work
│  ├─ convo-explorer  (5)
│  │  ├─ 2026-05-07  session-slug  Add flex inference summaries
│  │  ├─ 2026-05-06  cleanup       Remove dead code and simplify export
│  │  └─ 2026-04-14  codex-support  Browse and export Codex sessions
│  ├─ ailookup  (12)
│  │  └─ ...
│  └─ dpf-store-policy-scraper  (30)
│     └─ ...
├─ ~/.claude  (2)
│  └─ ...
├─ [codex]
│  └─ ~/Work/ailookup  (3)
│     └─ ...
└─ Other
   └─ /media/testycool/SSD/code/foo  (1)
      └─ ...
```

### Conversation Leaf Label Format

`{date}  {slug_or_uuid8}  {summary_or_preview}`

- Date: first 10 chars of ISO timestamp (YYYY-MM-DD)
- Slug: if set, otherwise first 8 chars of UUID
- Summary: from cache if available, otherwise first ~45 chars of preview

### Expansion Behavior

- Group containing CWD: auto-expanded
- Project matching CWD: auto-expanded
- Everything else: collapsed by default
- Filtering: expands matching nodes as current behavior

### No Scanner Changes

Grouping is purely a display concern in `_populate_tree()`. `scanner.py` continues to return a flat list of `Project` objects.

## Changes Summary

| File | Change |
|---|---|
| `summarize.py` (new) | Summary generation, caching, batch CLI logic |
| `app.py` | `_populate_tree()` rewritten for hierarchy + summary labels; `--summarize` CLI flag added |
| `parser.py` | No changes needed — already supports reading last N turns |
| `scanner.py` | No changes |
| `analyzer.py` | No changes (extract `_load_env` to shared util if needed) |

## Cost Estimate

- ~200 sessions × ~5 turns × ~500 chars/turn ≈ 500K chars ≈ 125K tokens input
- Output: ~200 sentences ≈ 3K tokens
- Flash Lite flex pricing: ~$0.016 input + ~$0.002 output ≈ **$0.02 total** for full backfill
- Ongoing: near-zero (only new sessions)
