# convo-explorer

A terminal UI for browsing, searching, exporting, and analyzing your [Claude Code](https://docs.anthropic.com/en/docs/claude-code) conversation history.

Claude Code stores every session as `.jsonl` files in `~/.claude/projects/`. This tool gives you a searchable, interactive interface to explore them all — across every project you've ever worked on.

## Screenshot

```
 convo-explorer
 ┌─ PROJECTS (49) ─────────────────┐┌─ PREVIEW (25 turns) ──────────────────────┐
 │ Filter...                        ││                                            │
 │                                  ││ ## ticklish-twirling-hejlsberg             │
 │ F:\code\ailookup      (12) ★    ││ Date: 2026-04-06T21:11:48                  │
 │ F:\code\testycool      (36)     ││ CWD: F:\code\convo-explorer                │
 │ F:\code\store-scraper (116) ★   ││ ──────────────────────────────────────────  │
 │ F:\code\taskbartimer    (6)     ││ ## User                                    │
 │ ✓ F:\code\my-project   (24)     ││ would like to make this more profesh...    │
 │ ✓ F:\code\another       (8)     ││                                            │
 │   2026-04-05  fix-auth  ...     ││ ## Assistant                               │
 │   2026-04-03  refactor  ...     ││ Let me explore the codebase first...       │
 │                                  ││                                            │
 ├──────────────────────────────────┤│                                            │
 │ 2 selected · ~850K tokens       ││                                            │
 │ S=select A=analyze E=export     ││                                            │
 │ C=combined M=model O=open       ││                                            │
 └──────────────────────────────────┘└────────────────────────────────────────────┘
```

## Features

- **Browse all projects** — auto-discovers every Claude Code project in `~/.claude/projects/`
- **Tree view** — expandable project nodes with conversation children, sorted by date
- **Search/filter** — type to filter projects or conversations instantly
- **Preview** — select any conversation to see the full user/assistant exchange
- **Multi-select** — select individual conversations, entire projects, or everything
- **Token estimation** — see estimated token count for selected conversations
- **Export** — export individual conversations or combined multi-conversation markdown
- **Gemini analysis** (optional) — analyze conversations with Google Gemini to extract patterns, preferences, and insights
- **Model picker** — cycle between `gemini-3-flash-preview`, `gemini-3.1-flash-lite-preview`, `gemini-3.1-pro-preview`
- **Editable prompts** — customize the analysis prompt before running
- **Analyzed indicators** — projects that have been analyzed show a ★ marker
- **Resizable sidebar** — drag the divider to resize
- **CLI mode** — list, export, and analyze without the TUI

## Install

```bash
# Clone
git clone https://github.com/voidxd/convo-explorer.git
cd convo-explorer

# Install (requires Python 3.13+ and uv)
uv sync

# Optional: enable Gemini analysis
uv sync --extra ai
```

## Usage

### TUI (interactive)

```bash
uv run convo-explorer
```

**Keyboard shortcuts:**

| Key | Action |
|-----|--------|
| `S` | Toggle select on current item |
| `Ctrl+A` | Select all |
| `Ctrl+D` | Deselect all |
| `Enter` | Preview conversation |
| `E` | Export selected as individual markdown files |
| `C` | Export selected as one combined markdown file |
| `A` | Analyze with Gemini |
| `M` | Cycle Gemini model |
| `P` | Edit analysis prompt |
| `O` | Open exports/analyses folder |
| `Esc` | Cancel running analysis |
| `Tab` | Switch focus between sidebar and preview |
| `Q` | Quit |

### CLI (headless)

```bash
# List all projects and conversations
uv run convo-explorer --list

# Export concatenated markdown (for use with other tools)
uv run convo-explorer --concat path/to/session.jsonl [more files...]

# Analyze with Gemini
export GEMINI_API_KEY=your-key-here
uv run convo-explorer --analyze path/to/session.jsonl --model gemini-3.1-pro-preview
```

## Gemini Analysis

Set your Gemini API key using any of these methods:

**Option 1: `.env` file (recommended for Windows)**
```bash
# Create in project directory or ~/.claude/convo-explorer/.env
echo GEMINI_API_KEY=your-key-here > .env
```

**Option 2: Environment variable**
```bash
# Linux/macOS
export GEMINI_API_KEY=your-key-here

# Windows (PowerShell)
$env:GEMINI_API_KEY="your-key-here"

# Windows (CMD)
set GEMINI_API_KEY=your-key-here
```

Get a free API key at [aistudio.google.com](https://aistudio.google.com/apikey).

Analysis extracts:

- Key decisions and their rationale
- User preferences and workflow patterns
- Problems encountered and solutions
- Recurring patterns across sessions
- Unfinished work and TODOs

Results are saved to `~/.claude/convo-explorer/analyses/` with human-readable filenames.

For multi-conversation analysis, select multiple items and press `A` — Gemini will find cross-session patterns and preference evolution.

## File locations

| What | Where |
|------|-------|
| Conversation logs | `~/.claude/projects/{project}/*.jsonl` |
| Analyses | `~/.claude/convo-explorer/analyses/` |
| Combined exports | `~/.claude/convo-explorer/exports/` |

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) (for conversation data)
- Google Gemini API key (optional, for analysis features)

## License

MIT
