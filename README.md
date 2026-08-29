# agentconvos

[![CI](https://github.com/testy-cool/agentconvos/actions/workflows/ci.yml/badge.svg)](https://github.com/testy-cool/agentconvos/actions/workflows/ci.yml)

Find the decision, message, or session buried in your local coding-agent history.

`agentconvos` treats the transcripts already written by Claude Code, Codex, Pi,
Agy, OpenCode, and Clihow as the source of truth. The CLI scans them, builds a
rebuildable local search index, and returns inspectable text or JSON. The TUI is
an optional human view over the same local engine; the CLI is the executable
contract for people, scripts, and agents.

## One useful loop

Search broadly, inspect the result, then tighten the query or open the matching
turns. This command was run against the repository's isolated synthetic archive;
the output below is observed output projected through `jq` to omit identity and
path fields:

```bash
agentconvos --search 'rate "fail open"' --source codex --limit 1 --json |
  jq '{query,total_searched,truncated,hit:(.hits[0]|{source,timestamp,role,snippet})}'
```

```json
{
  "query": "rate \"fail open\"",
  "total_searched": 3,
  "truncated": true,
  "hit": {
    "source": "codex",
    "timestamp": "2026-05-30T09:15:00",
    "role": "assistant",
    "snippet": "…eats a missing counter as a full bucket. I would rather fail open and log it, since a rate limiter outage should not become an API outage."
  }
}
```

The full search record also contains local file, project, and conversation
identity fields so a follow-up command can open the exact evidence. Treat that
JSON as private unless you explicitly redact it.

## Install from source

Prerequisites: Python 3.12 or newer, [uv](https://docs.astral.sh/uv/), and at
least one supported agent's local conversation history.

The project is not published on PyPI and has no packaged release yet. Install
the current public Git source explicitly:

```bash
uv tool install "agentconvos @ git+https://github.com/testy-cool/agentconvos.git"
agentconvos --version
```

For a contributor checkout instead:

```bash
git clone https://github.com/testy-cool/agentconvos.git
cd agentconvos
uv sync
uv run agentconvos --help
```

The installed command is `agentconvos`. It reads existing local transcripts;
there is no import step, server, account, or agentconvos credential.

## Common jobs

### Catch up on the current project

Run these from the project directory recorded in the conversation:

```bash
agentconvos --last 3
agentconvos --context
agentconvos --context --json
```

`--last N` gives the newest N conversations across sources. `--context` gives
up to five per source with dates, models, first/latest messages, and cached
summaries. JSON includes complete normalized message text and private local
metadata; pipe it only to tools you trust.

### Search exact evidence

```bash
agentconvos --search "auth middleware"
agentconvos --search 'auth "request id"' --source claude
agentconvos --search "auth" --source claude --limit 20 --json
```

Separate terms use AND matching across a conversation; quoted terms stay
together. Results are ranked and capped at 50 by default. A full page sets
`truncated: true`; raise `--limit` to inspect more. No matches is a successful
empty result, not an operational error.

### Inspect normalized turns

Use an identity returned by search, list, context, or the picker:

```bash
agentconvos --turns <id> --json
agentconvos --show <id>
agentconvos --concat <id> --detail tools
```

`--detail text` is the default for normalized turns and excludes tool calls,
tool results, reasoning blocks, and injected bootstrap metadata. `tools`,
`results`, `thinking`, and `full` deliberately expose progressively more.

### Preview resume or handoff

```bash
agentconvos --resume --dry-run
agentconvos --resume <id> --dry-run
agentconvos --handoff --dry-run
agentconvos --convo codex --handoff claude --dry-run
```

Resume continues the selected native session. Handoff exports normalized context
and prepares a fresh target-agent command. Without `--dry-run`, both can replace
the current process with another agent CLI. Important current boundary:
`--handoff --dry-run` still writes the local markdown export under `./output/`;
it only suppresses launching the target agent. Resume dry-run does not write.

### Measure recurring reply language

The basic reports are local and do not call a model:

```bash
agentconvos --ngrams --source claude --limit 20 --json
agentconvos --habits --source claude --output ./claude-habits.html
```

`--ngrams` compares assistant replies with other indexed sources. `--habits`
builds local HTML and JSON candidate evidence without a comparator. These are
descriptive corpus statistics, not proof of training, intent, or inherent style.

For the optional reproducible NLP report, install spaCy, TextDescriptives, and
the trained model into one environment. In a source checkout:

```bash
uv sync --extra language
uv run python -m spacy download en_core_web_sm
uv run agentconvos --habits --nlp --source claude
```

For a persistent Git-sourced tool, install the model package alongside the
`language` extra:

```bash
uv tool install --force \
  --with "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl" \
  "agentconvos[language] @ git+https://github.com/testy-cool/agentconvos.git"
```

The command never downloads a model at runtime and processes replies in one
local process. Add `--baseline matched` only for the opt-in other-source
comparison. The source-only recurring default uses a deterministic 70%/30%
project split. Descriptors are cached under the platform cache directory;
`AGENTCONVOS_LANGUAGE_CACHE` overrides that SQLite path. See
[SPEC.md](SPEC.md#deterministic-language-analysis) for the eligibility,
matching, cache invalidation, and claim rules.

### Browse interactively

```bash
agentconvos --find "auth request id"   # optional fzf picker
agentconvos                              # optional Textual TUI
```

These are thin human surfaces over the same scanner, parser, search index, and
resume/handoff primitives. `--find` requires `fzf`. The TUI adds preview,
multi-select, export, and optional analysis; scripts should use the CLI.

## Machine use and exits

`--json` produces one bounded JSON document for `--list`, `--search`,
`--ngrams`, `--habits`, `--last`, `--context`, and `--turns`. Successful JSON
goes to stdout. Search-index progress and NLP cache diagnostics go to stderr.
Human formatting may evolve; documented JSON fields and enum meanings are the
compatibility surface.

| Exit | Meaning currently safe to depend on |
|---:|---|
| `0` | Successful documented workflow, including an empty search/list/context result |
| `1` | Operational failure explicitly surfaced by recall or optional NLP setup/runtime |
| `2` | `argparse` usage or flag-validation failure |

Some older non-parser error paths still print human `Error:` text and return
`0`; do not automate those paths as machine errors yet. The exact schemas,
state changes, and compatibility limits are frozen in [SPEC.md](SPEC.md).

## Local state, privacy, and models

Agentconvos does not upload transcripts in its default scan, list, context,
search, export, phrase-analysis, or browsing paths. It does write rebuildable
metadata and search caches under `~/.claude/convo-explorer/`.

Model-backed commands are explicit:

- `--analyze`, `--deep`, and `--summarize` send selected transcript content to
  Gemini and require the `ai` extra plus configured credentials.
- `recall` sends selected evidence through an installed/authenticated Codex CLI;
  its optional Agy backend depends on a separately installed local bridge.
- `--habits --nlp` uses a local spaCy model and makes no model/API network call.

Full JSON, exports, reports, and previews can contain source text, identifiers,
and local paths. File permissions and downstream redaction remain the operator's
responsibility. Transcript content is data, not trusted instructions.

## Supported local sources

| Source | Default canonical store |
|---|---|
| Claude Code | `~/.claude/projects/{project}/*.jsonl` |
| Codex | `~/.codex/sessions/**/*.jsonl`, `~/.codex/conversations/*.json` |
| Pi | `~/.pi/agent/sessions/**/*.jsonl` |
| Agy | `~/.gemini/antigravity-cli/conversations/*.db` |
| OpenCode | `~/.local/share/opencode/opencode.db` |
| Clihow | `$CLIHOW_HOME/threads/*.jsonl` |

Unsupported or malformed records are skipped where the source parser can
identify them safely. Agentconvos does not modify the canonical transcript
stores.

## Performance and current limits

- Metadata is cached by source file size and modification time.
- First search builds a turn-level SQLite index and can take several minutes on
  a large archive. Later searches parse only new or changed conversations.
- Search returns at most `--limit` hits and does not expose pagination.
- NLP startup and descriptor work can take tens of seconds even when descriptor
  rows are cached.
- Cache and scanner files do not share one configurable root, and concurrent
  cache writers are not a documented guarantee.
- The root parser retains legacy flag-style grammar; only `recall` is currently
  a subcommand. A normalized command tree and structured error schema are deferred.

## Library and agent integration

The same normalization layer is importable:

```python
from agentconvos import get_meta, get_stats, parse_jsonl, scan_projects, search
```

`skills/agentconvos/SKILL.md` teaches an instruction-following coding agent the
safe desire paths. It does not grant permissions or transmit data by itself.

## Development

```bash
uv sync
uv run pytest -q
uv run ruff check src tests scripts
(cd tui && go test ./...)
uv build
```

README captures in `assets/` were produced with the repository's synthetic
archive generator. The behavioral contract and verification matrix live in
[SPEC.md](SPEC.md).

## License

[MIT](LICENSE)
