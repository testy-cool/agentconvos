# convolog

Discover, query, and browse AI coding agent conversations. Works with Claude Code, Codex, and Pi.

Use as a **CLI** (`convolog --context --json`), a **Python library** (`from convolog import scan_projects`), or an **interactive TUI** (`convolog`).

## Install

```bash
uv tool install "convolog[ai] @ git+https://github.com/testy-cool/convolog.git"
```

Without Gemini analysis: drop `[ai]`. Requires Python 3.12+.

## CLI

### Project context (the fast path)

```bash
convolog --last              # most recent conversation for cwd
convolog --last 3            # last 3
convolog --context           # last 5 with summaries
convolog --context --json    # structured, for piping to other tools
```

### Search

```bash
convolog --search "auth middleware"
convolog --search "auth" --source claude --json
```

### List and filter

```bash
convolog --list
convolog --list --source codex --after 2026-05-01 --json
convolog --list --json | jq '.projects[].conversations[].summary'
```

### Resume and handoff

```bash
convolog --resume <id>              # resume in native CLI
convolog --handoff                  # export context, start new session
convolog --handoff select           # pick from list
convolog --handoff codex            # latest Codex conversation
```

### Export

```bash
convolog --concat <id>              # markdown export
convolog --concat <id> --detail tools    # include tool call summaries
convolog --concat <id> --detail full     # include everything
```

### Analyze with Gemini

Requires `GEMINI_API_KEY` env var or `.env` file. Get a key at [aistudio.google.com](https://aistudio.google.com/apikey).

```bash
convolog --analyze <id>
convolog --analyze <id1> <id2> --model gemini-3.1-pro-preview
convolog --analyze <id> --prompt "What tools were used most?"
```

### JSON output

`--json` works with `--list`, `--search`, `--last`, and `--context`. Output includes session summaries, token estimates, file paths, and UUIDs.

## Library API

```python
from convolog import scan_projects, parse_jsonl, search, get_meta, get_stats

# Discover and filter
projects = scan_projects(source="claude", after="2026-05-01")

# Parse into normalized turns
turns = parse_jsonl(projects[0].conversations[0].path)

# Search across all sessions
hits = search([c.path for p in projects for c in p.conversations], "auth")

# Token and cost stats
stats = get_stats(projects[0].conversations[0].path)
```

## TUI

```bash
convolog
```

Interactive tree grouped by agent (Claude Code, Codex, Pi) with search, multi-select, preview, export, and Gemini analysis.

| Key | Action |
|-----|--------|
| `/` | Search/filter |
| `S` | Toggle select |
| `R` | Resume session |
| `H` | Handoff to new session |
| `E` | Export markdown |
| `A` | Analyze with Gemini |
| `Tab` | Switch panels |
| `Q` | Quit |

## File locations

| What | Where |
|------|-------|
| Claude Code logs | `~/.claude/projects/{project}/*.jsonl` |
| Codex logs | `~/.codex/sessions/*.jsonl`, `~/.codex/conversations/*.json` |
| Pi logs | `~/.pi/agent/sessions/**/*.jsonl` |
| Summaries | `~/.claude/convo-explorer/summaries/` |
| Analyses | `~/.claude/convo-explorer/analyses/` |

## License

MIT
