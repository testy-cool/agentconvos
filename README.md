# convolog

A terminal UI for browsing, searching, resuming, and analyzing your Claude Code, Codex, and Pi conversation history.

It auto-discovers Claude Code sessions from `~/.claude/projects/`, Codex sessions from `~/.codex/sessions/`, and Pi sessions from `~/.pi/agent/sessions/`, then gives you a searchable interface across every project you've worked on.

## Screenshot

```
 convolog
 ┌─ PROJECTS (618) ───────────────┐┌─ PREVIEW (25 turns) ──────────────────────┐
 │ Filter convos... (Enter=search) ││                                            │
 │                                  ││ ## ticklish-twirling-hejlsberg             │
 │ ▼ Claude Code  (580)            ││ Date: 2026-04-06 21:11                     │
 │   ▼ ~/Work                      ││ CWD: ~/Work/my-project                     │
 │     ● my-project    (12) ★     ││ ──────────────────────────────────────────  │
 │     ► web-app       (36)        ││ ## User                                    │
 │     ► api-server   (116) ★     ││ would like to make this more profesh...    │
 │     ► cli-tool       (6)        ││                                            │
 │   ► ~                           ││ ## Assistant                               │
 │ ► Codex  (23)                   ││ Let me explore the codebase first...       │
 │ ► Pi  (15)                      ││                                            │
 │                                  ││                                            │
 ├──────────────────────────────────┤│                                            │
 │ 12 projects · 618 conversations ││                                            │
 └──────────────────────────────────┘└────────────────────────────────────────────┘
```

## Features

- **Browse all projects** — auto-discovers Claude Code, Codex, and Pi sessions
- **Agent-grouped tree** — top-level nodes per agent (colored), with path groups and projects underneath
- **Search/filter** — type to filter instantly, press Enter for deep full-text search
- **Resume & handoff** — press `R` to resume or `H` to hand off context to a new session (all agents)
- **Preview** — select any conversation to see the full user/assistant exchange
- **Multi-select** — select individual conversations, entire projects, or everything
- **Token estimation** — see estimated token count for selected conversations
- **Export** — export individual conversations or combined multi-conversation markdown
- **Gemini analysis** (optional) — analyze conversations with Google Gemini to extract patterns, preferences, and insights
- **Model picker** — cycle between Gemini models
- **Editable prompts** — customize the analysis prompt before running
- **Analyzed indicators** — projects that have been analyzed show a ★ marker
- **Resizable sidebar** — drag the divider to resize
- **CLI mode** — list, search, export, resume, and analyze without the TUI

## Install

```bash
# Install globally (recommended)
uv tool install "convolog[ai] @ git+https://github.com/testy-cool/convolog.git"

# Or clone and install locally
git clone https://github.com/testy-cool/convolog.git
cd convolog
uv sync --extra ai
```

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

## Usage

### TUI (interactive)

```bash
convolog
```

**Keyboard shortcuts:**

| Key | Action |
|-----|--------|
| `/` | Focus search/filter |
| `Enter` | Deep search (in filter) / Preview conversation (in tree) |
| `Esc` | Clear filter / Cancel analysis |
| `S` | Toggle select on current item |
| `Ctrl+A` | Select all |
| `Ctrl+D` | Deselect all |
| `R` | Resume conversation (Claude Code, Codex) |
| `H` | Handoff context to a new session |
| `E` | Export selected as individual markdown files |
| `C` | Export selected as one combined markdown file |
| `A` | Analyze with Gemini |
| `M` | Cycle Gemini model |
| `P` | Edit analysis prompt |
| `O` | Open exports/analyses folder |
| `Tab` | Switch focus between sidebar and preview |
| `Q` | Quit |

### CLI (headless)

```bash
# List all projects and conversations
convolog --list

# Search across all conversations
convolog --search "auth middleware"

# Resume a conversation by slug or UUID
convolog --resume reflective-herding-biscuit
convolog --resume 019e4488

# Handoff to a new session (exports context, launches new CLI)
convolog --handoff              # latest conversation in cwd
convolog --handoff select       # pick from list
convolog --handoff codex        # latest Codex conversation in cwd

# Export by file path, UUID prefix, or slug
convolog --concat path/to/session.jsonl
convolog --concat 315ce500
convolog --concat reflective-herding-biscuit

# Analyze with Gemini
export GEMINI_API_KEY=your-key-here
convolog --analyze 315ce5 reflective-herding --model gemini-3.1-pro-preview

# Custom analysis prompt (inline or from file)
convolog --analyze 315ce5 --prompt "List all tools used.\n\n{content}"
convolog --analyze 315ce5 --prompt my-prompt.txt

# Detail levels: text (default), tools, results, full
convolog --concat 315ce5 --detail tools     # +tool call summaries
convolog --concat 315ce5 --detail results   # +truncated tool output
convolog --concat 315ce5 --detail full      # +everything untruncated

# Filter by agent and date range
convolog --list --source codex --after 2026-05-01
convolog --search "auth" --source claude --before 2026-05-15

# JSON output for piping to other tools
convolog --list --json
convolog --list --source codex --json | jq '.projects[].conversations[].uuid'
convolog --search "middleware" --json | jq '.hits[].snippet'
```

#### Detail levels

| Level | What's included | Typical overhead |
|-------|----------------|-----------------|
| `text` | User/assistant text only | baseline |
| `tools` | + tool call summaries (Bash commands, file edits, greps) | +20-30% |
| `results` | + truncated tool output (500 chars each) | +80-100% |
| `full` | + full untruncated tool output | +300-2000% |

Exports include a stats header: model, token count, duration, tool calls, and estimated cost.

## Gemini Analysis

Set your API key:

```bash
# Environment variable
export GEMINI_API_KEY=your-key-here

# Or .env file in project directory
echo GEMINI_API_KEY=your-key-here > .env
```

Get a free key at [aistudio.google.com](https://aistudio.google.com/apikey).

Analysis extracts:
- Key decisions and their rationale
- User preferences and workflow patterns
- Problems encountered and solutions
- Recurring patterns across sessions
- Unfinished work and TODOs

Results saved to `~/.claude/convo-explorer/analyses/`.

For multi-conversation analysis, select multiple items and press `A` — Gemini finds cross-session patterns and preference evolution.

## Library API

Use as a building block in other tools:

```python
from convolog import scan_projects, parse_jsonl, get_meta, search, get_stats

# Discover all sessions across all agents
projects = scan_projects()

# Filter by agent and date
codex_recent = scan_projects(source="codex", after="2026-05-01")

# Parse a session into normalized turns
turns = parse_jsonl(projects[0].conversations[0].path)

# Search across all conversations
hits = search([c.path for p in projects for c in p.conversations], "auth middleware")

# Get token/cost stats
stats = get_stats(projects[0].conversations[0].path)
print(f"${stats.cost_estimate:.2f}, {stats.tool_calls} tool calls")
```

## File locations

| What | Where |
|------|-------|
| Claude Code logs | `~/.claude/projects/{project}/*.jsonl` |
| Codex logs | `~/.codex/sessions/YYYY/MM/DD/*.jsonl` |
| Pi logs | `~/.pi/agent/sessions/**/*.jsonl` |
| Analyses | `~/.claude/convo-explorer/analyses/` |
| Combined exports | `~/.claude/convo-explorer/exports/` |

## License

MIT
