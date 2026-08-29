---
name: agentconvos
description: Recover context from past AI coding sessions on this machine (Claude Code, Codex, Pi, Agy, OpenCode, Clihow). Use at the start of work in an unfamiliar repo, and whenever the user refers to earlier work you cannot see - "what were we doing here", "what did we decide about X", "the thing we fixed yesterday", "we already tried that", "remember when", "why is this written this way", "find that conversation", "resume where we left off". Also for handing a session to another agent.
---

# agentconvos

Your context window starts empty every session. The user's does not. This
reads the conversation archive already on their disk so you can answer from
what was actually said instead of guessing or asking them to repeat it.

Git tells you what changed. This tells you what was discussed and why, which
is usually the part that was never written down.

## Pick the command by what you need

| You need | Command | Costs money | Costs context |
|---|---|---|---|
| What happened in this directory | `agentconvos --context` | no | small |
| A conversation containing some words | `agentconvos --search "terms"` | no | small |
| A decision and its reasoning, across sessions | `agentconvos recall "question"` | **yes** | small |
| The full text of one conversation | `agentconvos --turns <id> --json` | no | **large** |
| To continue a session in its own agent | `agentconvos --resume <id>` | no | none |

Two different budgets. `recall` spends money; `--turns` spends *your context*,
and it is the only command here that can swallow a session whole. Neither is a
reason to avoid them, only to know which one you are spending.

Work down that list. `--context` answers most catch-up questions on its own,
and `--search` answers most of the rest. Reach for `recall` when the answer is
spread over several sessions or the user asks *why* something was decided.

## Catching up on a directory

```bash
agentconvos --context          # last 5 sessions per agent, with catch-up detail
agentconvos --context --json   # same, machine-readable
agentconvos --last 3           # last 3 conversations, one line each
```

`--context` reports on the **current working directory**, so run it from the
repo you are asking about. It gives you, per conversation: date, turn count,
model, effort, the first thing the user asked, their latest message, the
latest agent reply, and a cached summary when one exists.

Read it before asking the user to re-explain the project. If it comes back
empty, the work simply was not done in this directory.

## Finding a conversation

```bash
agentconvos --search "auth middleware"          # all words, anywhere in a conversation
agentconvos --search 'auth "request id"'        # quoted text is matched as a phrase
agentconvos --search "auth" --source codex --json
```

Quoting narrows a lot: unquoted `rate limit` finds thousands of conversations,
`"rate limit"` a few dozen. It is a ranked phrase match rather than a literal
grep, so a hit can occasionally be a near miss like "rate limiting". Read the
turn before quoting it back to the user as an exact match.

Search is case-insensitive and ranked, and runs against a local SQLite index,
so it is fast even over thousands of sessions. JSON hits carry `uuid`,
`source`, `timestamp`, `cwd`, `file`, `turn_index`, `role`, and `snippet` -
enough to cite a specific turn without opening the transcript. `turn_index` is
zero-based and lines up with the positions in default `--turns` output; the
human-readable listing prints the same turn as `turn 1`, and asking for
`--detail tools` shifts positions because tool-only turns appear. Cite from
JSON. `total_searched` is how many conversations were scanned, not how many
matched.

**Results are capped at 50 by default.** A broad query hits that cap easily,
so never report the number of results as the number of matches in the archive.
The text output says when it is showing only the top N, and the JSON carries
`limit` and `truncated`. Raise it with `--limit 200` when you need the real
shape of something.

The index tracks new sessions, so work from an hour ago is findable. Only the
very first search on a new machine is slow, while the index is built once.

## Inspecting reply-language patterns locally

```bash
agentconvos --habits --source claude
```

This creates a local HTML and JSON report with recurring reply phrases,
reproducible writing structures, and three short examples per row. It does not
call a model or upload conversation text. Treat its labels as candidate textual
evidence, not proof of intent, training data, or an agent's inherent personality.

For deterministic source-only recurring NLP evidence, explicitly install
`agentconvos[language]` plus `en_core_web_sm`, then run:

```bash
agentconvos --habits --nlp --source claude
```

The command never downloads a model at runtime and never calls a model API. Add
`--baseline matched` only when an opt-in, project/month/length-matched other-source
comparison is wanted. Recurring and distinctive are separate claims; without
positive held-out matched evidence and a bootstrap interval excluding zero, report
the phrase as a descriptive candidate. Plain `--habits` retains its original report.

## Asking a question the archive has to answer

```bash
agentconvos recall "Where did we decide how scraper fallbacks should work?"
agentconvos recall --backend agy "Why did we drop the Redis cache?"
```

`recall` searches iteratively, opens only promising turns, and answers with
source, date, session id, turn numbers, and project paths. Use it when a
plain search will not do: the answer spans sessions, or the user wants the
reasoning rather than the message.

It costs a real model call and needs an authenticated Codex CLI, or
`--backend agy` to route through the local AGY bridge instead. The bridge runs
locally but still calls Gemini, so either backend sends transcript excerpts to
a model provider. Try `--search` first.

## Reading a conversation without drowning in it

```bash
agentconvos --turns <id> --json                  # normalized user/assistant turns
agentconvos --turns <id> --json --detail tools   # plus the commands that ran
agentconvos --turns <id> --json --detail full    # plus every tool result
```

`--turns` defaults to text only: no tool calls, no tool results, no reasoning
blocks, no harness bootstrap noise. That default exists so you can read a
conversation without spending your context on machine chatter. Only ask for
`--detail tools` or `full` when the tool activity is the thing you need, and
know that `full` is where transcripts get enormous: one real session measured
4KB of text and 358KB with every tool result attached.

**Size it before you read it.** `estimated_tokens` in `--context --json` is
computed from the raw file, which includes all the tool traffic that the
default read strips, so it overstates a text-only read by one to two orders of
magnitude. Treat it as an upper bound, not a measurement. To find out what a
read will actually cost, measure it:

```bash
agentconvos --turns <id> --json | wc -c        # bytes; roughly 4 bytes a token
```

That also works for conversations outside the current directory, which
`--context` cannot see.

## Continuing or handing off work

```bash
agentconvos --resume <id> --dry-run   # print the command, start nothing
agentconvos --resume <id>             # continue in the agent that created it
agentconvos --handoff codex --dry-run # print the handoff command
agentconvos --handoff codex           # move this directory's context to Codex
```

Use `--dry-run` first and show the user the command. Resume replaces the
session they are in; that is their call, not yours. Note that `--dry-run`
stops the agent from launching, not the handoff export from being written.

## Sending data off the machine

Everything above is local. Three things are not:

- `agentconvos --analyze` sends conversation text to Gemini
- `agentconvos --summarize` sends conversation text to Gemini
- `agentconvos recall` sends transcript excerpts to OpenAI via the Codex CLI,
  or to Gemini when run with `--backend agy`

Conversation archives hold whatever the user has ever pasted into an agent:
credentials, customer data, unreleased work. Do not run these on someone's
archive without telling them what leaves the machine.

## Commands that write files

Two commands write plaintext copies of conversations to disk rather than
returning them to you. Nothing leaves the machine, but a new readable copy of
a private transcript now exists where there was not one before.

- `--concat` saves a combined markdown file under
  `~/.claude/convo-explorer/exports/` and prints only the path
- `--handoff` writes one into `output/` in the current directory, and does so
  even with `--dry-run`, which suppresses launching the agent but not the
  export

Neither puts the transcript on stdout, so do not reach for them to *read* a
conversation; use `--turns` for that.

## Identifiers

A conversation id is a UUID, and a prefix works. Anywhere an id is accepted a
file path works too. `--source` narrows to one agent: `claude`, `codex`, `pi`,
`agy`, `opencode`, `clihow`.

## When there is nothing to find

An empty result is an answer. Say the archive has no record of it rather than
filling the gap with a plausible story - the user can tell the difference, and
inventing history is worse than admitting the search came up empty.
