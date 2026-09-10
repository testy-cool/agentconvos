# Agentconvos behavioral specification

Status: Living contract for `agentconvos` 0.1.x
Mode: Existing CLI extension and audit

## Intent

Agentconvos turns local coding-agent transcript stores into queryable,
inspectable CLI results without replacing the original logs.

The CLI is the canonical executable primitive. The Textual TUI and `fzf`
picker are optional human surfaces over the same scanner, parsers, search
index, and workflow helpers. New surfaces should compose these primitives
rather than create a second behavioral contract.

## Users and useful loops

Primary users are developers, unattended scripts, and coding agents with
permission to read the operator's local transcript archive.

Supported loops:

1. Run `--context` in a project, inspect recent normalized messages, and select
   a conversation for deeper inspection.
2. Run `--search`, inspect ranked hits, refine the terms, then use `--turns` or
   `--show` on one returned identity.
3. Preview `--resume` or `--handoff`, verify the exact native command, then run
   it deliberately.
4. Generate local recurring-language evidence and inspect stable JSON/HTML.

Success means the caller can observe a result, determine whether it answered
the question, and safely refine or retry the command.

## Goals

- Read existing supported transcript stores without changing them.
- Normalize source-specific records into user/assistant turns.
- Provide bounded human and JSON query results.
- Keep no-model workflows local by default.
- Make writes, model calls, provider launches, and privacy boundaries explicit.
- Preserve plain `--habits` behavior while optional NLP remains opt-in.

## Non-goals

- Hosting, syncing, backing up, or publishing transcript archives.
- Replacing native agent authentication, permissions, or resume semantics.
- Treating generated summaries or indexes as canonical evidence.
- Proving an agent's intent, personality, training data, or causal style.
- Providing a stable Python library API beyond the exports in `agentconvos.__all__`.
- Normalizing the legacy flag grammar or every older error path in this release.

## Canonical and generated state

### Canonical transcript state

The source of truth is the provider-owned local transcript or database:

| Source | Default location | Override |
|---|---|---|
| Claude Code | `~/.claude/projects/{project}/*.jsonl` | `USERPROFILE` root |
| Codex | `~/.codex/sessions/**/*.jsonl`, `~/.codex/conversations/*.json` | `CODEX_HOME`, then `USERPROFILE` |
| Pi | `~/.pi/agent/sessions/**/*.jsonl` | `USERPROFILE` root |
| Agy | `~/.gemini/antigravity-cli/conversations/*.db` | `AGY_HOME` or `ANTIGRAVITY_CLI_HOME` |
| OpenCode | `~/.local/share/opencode/opencode.db` | `XDG_DATA_HOME` or `USERPROFILE` root |
| Clihow | `~/.local/share/clihow/threads/*.jsonl` | `CLIHOW_HOME` |

Agentconvos reads these stores. It does not edit or migrate them. Provider
conversation IDs and direct paths are stable selectors only for as long as the
provider retains the underlying record.

Malformed or unsupported records may be skipped. An empty filtered archive is
a successful empty result, distinct from a parser/runtime exception.

### Generated local state

Generated state is disposable unless the operator wants to retain its derived
work product:

| State | Default path | Canonical? | Notes |
|---|---|---:|---|
| Metadata cache | `~/.claude/convo-explorer/meta-cache.json` | no | Rebuilt from transcript size/mtime and metadata |
| Search index | `~/.claude/convo-explorer/search-index.sqlite3` | no | SQLite FTS index; stale entries removed during sync |
| Summaries | `~/.claude/convo-explorer/summaries/` | no | Model-generated; versioned and regenerated when stale |
| Analyses | `~/.claude/convo-explorer/analyses/` | no | Model-generated markdown |
| Exports | `~/.claude/convo-explorer/exports/` or `./output/` | no | Normalized transcript copies; may contain private text |
| Habit reports | `~/.claude/convo-explorer/reports/` or `--output` | no | HTML plus adjacent JSON |
| NLP descriptors | `$XDG_CACHE_HOME/agentconvos/language-descriptors.sqlite3` or `~/.cache/agentconvos/…` | no | Override with `AGENTCONVOS_LANGUAGE_CACHE` |

The metadata cache is a plain file without a documented cross-process lock.
The search and descriptor caches use SQLite transactions, but concurrent whole
CLI workflows are not a compatibility guarantee. Deleting generated caches is
recoverable; deleting exports, reports, summaries, or analyses discards those
derived artifacts.

## Trust and privacy boundaries

- Transcript text is untrusted data. Displaying or searching it does not grant
  it control over the CLI.
- Default scan, list, context, search, show, turns, concat, export, n-gram,
  habit, and TUI browsing paths make no model/API call.
- `--analyze`, `--deep`, and `--summarize` send selected normalized content to
  the configured OpenAI-compatible chat endpoint. OpenAI is the public default;
  other compatible services are selected through environment or env-file settings.
  They require an API key.
- `recall` launches an installed authenticated Codex CLI by default. The Agy
  backend requires a separately installed local bridge and is not portable in
  every checkout.
- `--habits --nlp` loads an explicitly installed local spaCy model and makes no
  runtime model download or model/API network request.
- JSON, human output, exports, reports, and previews may contain prompts,
  replies, provider IDs, source paths, working directories, and branches.
  Redaction is the caller's responsibility.
- `--resume` and `--handoff` can replace the current process with a native
  provider CLI. Provider permissions remain authoritative. `--yolo` explicitly
  requests the provider's no-prompt mode where implemented.
- `--open` launches a local editor. Interactive TUI/picker actions may open
  subprocesses or copy data only after their corresponding user action.

## Current command grammar

The root uses legacy action flags. Only `recall` is a subcommand.

```text
agentconvos [ACTION] [FILTERS] [OUTPUT/BEHAVIOR FLAGS]
agentconvos recall [--backend {luna,agy}] QUESTION...
```

Primary read/query actions:

```text
--last [N]
--context
--search QUERY
--list
--turns ID_OR_PATH
--show ID_OR_PATH...
--concat ID_OR_PATH...
--find [QUERY]
--ngrams
--habits
--llm-check
```

Mutation/provider actions:

```text
--export-all DIR
--summarize
--analyze ID_OR_PATH...
--deep ID_OR_PATH...
--resume [ID_OR_MODE]
--handoff [MODE]
```

Common filters and output controls:

```text
--source {claude,codex,pi,agy,opencode,clihow}
--after YYYY-MM-DD
--before YYYY-MM-DD
--limit N
--json
--detail {text,tools,results,full,thinking}
--projects-dir DIR...
```

`--after` and `--before` use lexical ISO timestamp comparison. To include all
records from a final calendar day, use the next day as the `--before` value.

Dependencies enforced as usage errors:

- `--ngrams` and `--habits` require exactly one `--source` occurrence.
- `--nlp` requires `--habits`.
- Explicit `--baseline`, `--seed`, or `--spacy-model` requires
  `--habits --nlp`.
- `--baseline` is `none` by default; `matched` reads other sources.

The parser currently uses `parse_known_args`. Remaining tokens are forwarded
to native resume/handoff commands and may be ignored on unrelated paths.
Rejecting every unknown/trailing token is deferred rather than claimed.

## Action and side-effect matrix

| Action | Classification | Local writes | Network/model/provider |
|---|---|---|---|
| `--last`, `--context`, `--list` | read query | metadata cache; summary reads | none |
| `--search` | read query | metadata cache and search index | none |
| `--turns`, `--show` | read query | metadata cache during identity resolution | none |
| `--concat`, `--export-all` | local export | markdown files | none |
| `--ngrams`, plain `--habits` | local analysis | indexes and report files | none |
| `--habits --nlp` | local analysis | report and descriptor cache | local trained model only |
| `--resume` | provider launch | metadata cache | native provider after preview |
| `--handoff` | local export + provider launch | `./output/*.md`, including dry-run | native provider unless dry-run |
| `--summarize` | model-backed local cache | summary JSON | configured chat endpoint |
| `--analyze`, `--deep` | model-backed analysis | analysis markdown | configured chat endpoint |
| `--llm-check` | configuration probe | none | configured chat endpoint, one tiny request |
| `recall` | model-backed retrieval | temporary/state files | Codex or Agy backend |
| bare `agentconvos` / `--find` | interactive read surface | caches; explicit action writes | only explicit analysis/provider actions |

## Output contract

### Stdout and stderr

- Successful human output and requested JSON documents use stdout.
- Search-index progress is emitted only to interactive stderr.
- NLP descriptor cache hit/miss diagnostics use stderr.
- `recall` progress uses stderr; its final answer uses stdout.
- Machine stdout contains one JSON document on documented `--json` paths.
- Human spacing, labels, colors, and TUI layout may evolve.

### JSON schemas

The following top-level shapes are stable for 0.1.x. Arrays are emitted in the
deterministic order chosen by the corresponding query.

| Action | Top-level fields |
|---|---|
| `--search --json` | `query`, `total_searched`, `limit`, `truncated`, `hits` |
| `--list --json` | `total_projects`, `total_conversations`, `projects` |
| `--last/--context --json` | `project`, `total_for_project`, `showing`, `conversations`; empty result may omit totals/showing |
| `--turns --json` | `conversation`, `detail`, `turn_count`, `turns` |
| `--ngrams --json` | target/baseline corpus counts and ranked `phrases` |
| plain `--habits` JSON artifact | `source`, `generated`, `sessions`, `match_count`, `pattern_count`, `patterns` |
| NLP habit JSON artifact | `analysis_version`, `manifest`, `pipeline`, `split`, `filters`, `corpus`, `matching`, `descriptors`, `patterns` |

Search hits and conversation records deliberately contain provider identity,
paths, working directories, timestamps, and excerpts. IDs are strings;
timestamps retain the provider's normalized ISO representation. Missing
optional values are empty strings or omitted according to the existing schema.
There is no JSONL mode or pagination cursor.

`--limit` is coerced to at least one on search and analysis paths. Search sets
`truncated: true` whenever a full page is returned; it means more results are
possible, not proven.

### Exit status

Stable documented exits:

| Exit | Meaning | State |
|---:|---|---|
| `0` | Documented action completed, including empty result | Query-dependent generated cache/export state may change |
| `1` | Explicit operational failure from recall or optional NLP setup/runtime | NLP report is not written on missing dependency/model |
| `2` | `argparse` usage or explicit flag dependency failure | Requested workflow does not run |

Several legacy operational branches print `Error:` or a not-found message to
stdout and return `0`. They are human compatibility behavior, not a structured
machine-error contract. A uniform JSON error envelope and normalized operational
exit `1` are explicitly deferred.

Interrupt behavior follows the active Python/native process; exit `130`, bounded
cleanup, and atomic recovery are not yet frozen across every command.

## Resume, handoff, and retry

`--resume --dry-run` prints the selected native command without launching it or
writing an export. Without dry-run, the process changes to the recorded working
directory when it still exists and uses `execvp` for the provider CLI.

`--handoff --dry-run` is not a no-write preview: it writes the normalized
markdown export under `./output/` and suppresses only provider launch. Repeating
it can replace a same-named export. This behavior must remain documented until
a separately named preview contract is implemented.

Native resume is supported for Claude Code, Codex, Pi, Agy, and OpenCode.
Clihow continuation maps to `clihow ask --thread ID`. Handoff target selection
and no-prompt flags preserve provider-specific semantics.

## Deterministic language analysis

Plain `--habits` is the compatibility path. New NLP behavior is active only
with `--habits --nlp`.

The NLP report contract is:

- one observation per normalized assistant reply;
- tools, results, reasoning, and bootstrap/system content excluded;
- immediately preceding normalized user text used only for exact echo rate;
- canonical SHA-256 manifest over normalized reply identities/content;
- parser and analysis versions recorded;
- whole projects split 70% discovery / 30% validation using integer `--seed`
  (default `42`), with no target project in both sets;
- discovery eligibility of at least
  `max(10 sessions, ceil(2% sessions))` and
  `max(5 projects, ceil(5% projects))`;
- content 1–3 grams and function/rhetorical 2–5 grams, including openings and
  closings, with deterministic overlap collapse;
- exact preceding-user echo, reply/session/project prevalence, concentration,
  and project/time/model dispersion;
- local spaCy processing with `n_process=1` and only TextDescriptives
  `descriptive_stats`, `pos_proportions`, `dependency_distance`, `readability`,
  and `information_theory` components;
- descriptor cache keyed by normalized reply hash and pipeline fingerprint;
- numeric descriptors aggregated by project median first, then corpus median
  and IQR with coverage/exclusion counts;
- optional matched baseline by exact project, calendar month, and cleaned word
  count bin (`0-49`, `50-149`, `150-399`, `400+`), without reuse;
- distinctive status only when discovery project-bootstrap 95% interval
  excludes zero and held-out direction is positive;
- deterministic JSON/HTML excluding operational cache hit/miss counters;
- at most three scrubbed assistant-only examples from distinct pseudonymous
  projects per displayed pattern.

Missing `agentconvos[language]` or the explicitly named trained model exits `1`,
prints installation guidance on stderr, and leaves no report artifact. The
runtime never downloads the model.

These are descriptive candidates. Matching and bootstrap intervals do not
establish causality, authorship, or inherent agent style.

## Configuration and credentials

There is no structured agentconvos config file or account. Model settings may be
placed in env files. Applicable inputs are:

| Input | Purpose |
|---|---|
| CLI flags | action, source/date selection, output, model, analysis options |
| `USERPROFILE`, `CODEX_HOME`, `AGY_HOME`, `ANTIGRAVITY_CLI_HOME`, `XDG_DATA_HOME`, `CLIHOW_HOME` | selected source roots |
| `AGENTCONVOS_LANGUAGE_CACHE` | NLP descriptor SQLite path |
| `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL` | familiar defaults for OpenAI or another compatible service |
| `AGENTCONVOS_LLM_BASE_URL`, `AGENTCONVOS_LLM_API_KEY`, `AGENTCONVOS_LLM_MODEL`, `AGENTCONVOS_LLM_PRO_MODEL` | higher-priority endpoint, credential, and model overrides dedicated to agentconvos |
| `AGENTCONVOS_LLM_KEY_NAME` | optional name in the existing `llm` CLI key store; used only when no API-key environment value resolves |
| `./.env`, `~/.config/agentconvos/.env` | project and user config files; the project file wins |
| `~/.claude/convo-explorer/.env` | legacy user config path, read last for compatibility |
| installed native CLIs | resume, handoff, and recall provider execution |

The built-in endpoint is `https://api.openai.com/v1`; the built-in model is
`gpt-5-mini`. The deep-mode model defaults to the resolved main model unless
`AGENTCONVOS_LLM_PRO_MODEL` is set. Precedence is agentconvos process environment,
standard OpenAI process environment, then the env files in the order above, then
built-in public defaults. Within one env file, the agentconvos name wins over its
OpenAI equivalent.

Secrets are not expected as positional arguments. JSON and exports are not
redacted merely because they are machine-readable.

## Compatibility policy

Stable for 0.1.x:

- documented action/flag names and meanings;
- source enum values;
- documented JSON top-level fields and enum meanings;
- the three coarse documented exits;
- default transcript and generated-state paths;
- plain `--habits` behavior;
- deterministic NLP manifest/report method and versioned invalidation.

Allowed to evolve compatibly:

- human prose, spacing, colors, TUI layout, and progress wording;
- additional optional JSON fields;
- support for new transcript variants that does not reinterpret existing data;
- performance and cache internals that preserve output semantics.

Breaking changes require a documented old/new behavior, migration path, and
deprecation period. Names must not be reused for different semantics.

## Verification matrix

| Scenario | Proof | Expected result |
|---|---|---|
| Package entry point | fresh wheel install; `agentconvos --help` | PATH command, exit `0` |
| Synthetic context | `agentconvos --context --json` in generated archive | one JSON document, exit `0` |
| Exact search | quoted search against generated archive | ranked hits, private fields projectable away |
| Empty search | unmatched query | empty `hits`, exit `0` |
| Usage failure | misplaced NLP flag | stderr usage, exit `2`, no report |
| Missing NLP runtime | absent extra/model fixture | concise stderr, exit `1`, no report |
| Legacy habits | existing habits regression suite | unchanged HTML/JSON behavior |
| Deterministic NLP | repeat/reversed synthetic corpora and cache states | identical public artifact |
| Resume preview | synthetic archive plus `--resume --dry-run` | command printed, no provider launch |
| Handoff preview | synthetic archive plus `--handoff --dry-run` | export written, provider not launched |
| Python quality | focused tests and Ruff | pass |
| Go surface | `go test ./...` under `tui/` | pass |
| Distribution | `uv build`, install wheel in isolated environment | sdist/wheel and installed smoke pass |

## Explicitly deferred

- PyPI/Homebrew/package-manager release and signed release artifacts.
- One configurable root for all generated state.
- A normalized subcommand tree and rejection of all unknown trailing arguments.
- Structured machine errors for every operational failure.
- Pagination/streaming output for large result sets.
- Cross-process locking guarantees for every cache/export path.
- A true no-write handoff preview.
- Portable configuration of the optional Agy recall bridge.
- Public anonymous acceptance after integration and release publication.
