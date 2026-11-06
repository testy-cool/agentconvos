"""Every shell command an agent ran, in full, and the ones worth shortening.

``agentconvos commands`` prints the commands of one or more conversations
without the 120-character cut the transcript views use.
``agentconvos commands group`` looks for three things across them:

- programs: which program and subcommand took the most typing in total;
- repeated commands: the same command, apart from paths, numbers and IDs,
  run again and again, which an alias or a script would shorten;
- repeated runs: the same steps run one after another, inside one command or
  across consecutive ones, which a small tool could do in one go.

Claude Code, Codex and Pi conversations are read. Agy and OpenCode store
their commands differently and are skipped.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .parser import _detect_format

SUPPORTED = ("claude", "codex", "pi")


@dataclass
class Command:
    text: str
    session: str
    index: int


# ---------------------------------------------------------------------------
# Reading commands out of each engine's records
# ---------------------------------------------------------------------------

def _records(path: Path, needle: str):
    """Parse only the lines that contain *needle*; most lines hold no tool call."""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if needle in line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

def _claude_commands(path: Path) -> list[str]:
    out = []
    for rec in _records(path, '"tool_use"'):
        if rec.get("type") != "assistant":
            continue
        for block in rec.get("message", {}).get("content", []) or []:
            if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name") == "Bash":
                cmd = (block.get("input") or {}).get("command")
                if isinstance(cmd, str) and cmd.strip():
                    out.append(cmd)
    return out


def _unwrap_argv(cmd) -> str:
    """Codex logs some commands as an argv list, often ``bash -lc <script>``."""
    if isinstance(cmd, list):
        parts = [str(p) for p in cmd]
        if len(parts) == 3 and os.path.basename(parts[0]) in ("bash", "sh", "zsh") and parts[1] in ("-lc", "-c"):
            return parts[2]
        return shlex.join(parts)
    return cmd if isinstance(cmd, str) else ""


# Newer Codex sessions call tools from a JavaScript snippet, one or more
# ``exec_command({cmd: "..."})`` calls per snippet.
_JS_CMD = re.compile(r"""cmd\s*:\s*(?:"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)'|`((?:[^`\\]|\\.)*)`)""", re.S)


_JS_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "0": "\0"}


def _js_string(raw: str) -> str:
    """Undo the backslash escapes of a JavaScript string literal."""
    return re.sub(r"\\(.)", lambda m: _JS_ESCAPES.get(m.group(1), m.group(1)), raw, flags=re.S)


def _codex_commands(path: Path) -> list[str]:
    out = []
    for rec in _records(path, '"response_item"'):
        if rec.get("type") != "response_item":
            continue
        payload = rec.get("payload", {})
        kind = payload.get("type")
        if kind == "function_call" and payload.get("name") in ("exec_command", "shell", "shell_command"):
            try:
                args = json.loads(payload.get("arguments") or "{}")
            except json.JSONDecodeError:
                continue
            cmd = _unwrap_argv(args.get("cmd", args.get("command", "")))
            if cmd.strip():
                out.append(cmd)
        elif kind == "custom_tool_call":
            snippet = payload.get("input") or payload.get("arguments") or ""
            if not isinstance(snippet, str):
                continue
            for match in _JS_CMD.finditer(snippet):
                raw = next(g for g in match.groups() if g is not None)
                cmd = _js_string(raw)
                if cmd.strip():
                    out.append(cmd)
    return out


def _pi_commands(path: Path) -> list[str]:
    out = []
    for rec in _records(path, '"toolCall"'):
        if rec.get("type") != "message":
            continue
        msg = rec.get("message", {})
        if msg.get("role") != "assistant":
            continue
        for block in msg.get("content", []) or []:
            if not (isinstance(block, dict) and block.get("type") == "toolCall"):
                continue
            if str(block.get("name", "")).lower() != "bash":
                continue
            args = block.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    continue
            cmd = args.get("command") if isinstance(args, dict) else None
            if isinstance(cmd, str) and cmd.strip():
                out.append(cmd)
    return out


def session_commands(path: Path) -> list[str] | None:
    """All shell commands in one conversation, in order; None if unsupported."""
    fmt = _detect_format(path)
    if fmt == "claude":
        return _claude_commands(path)
    if fmt == "codex":
        return _codex_commands(path)
    if fmt == "pi":
        return _pi_commands(path)
    return None


# ---------------------------------------------------------------------------
# Splitting a command into steps
# ---------------------------------------------------------------------------

_SPECIAL = re.compile(r"[\\'\"`#<()\n&|;]")


def split_steps(cmd: str) -> list[str]:
    """Split a shell command at ``&&``, ``||``, ``;``, ``&`` and newlines.

    Quotes, ``$(...)`` and heredoc bodies stay inside their step. A pipeline
    stays one step, because ``foo | head`` is still the work of ``foo``.
    """
    steps: list[str] = []
    cur: list[str] = []
    quote = ""
    depth = 0
    heredocs: list[str] = []
    i, n = 0, len(cmd)

    def flush():
        text = "".join(cur).strip()
        if text:
            steps.append(text)
        cur.clear()

    while i < n:
        if not quote:
            m = _SPECIAL.search(cmd, i)
            stop = m.start() if m else n
            if stop > i:
                cur.append(cmd[i:stop])
                i = stop
                continue
        c = cmd[i]
        if quote:
            cur.append(c)
            if c == "\\" and quote != "'" and i + 1 < n:
                cur.append(cmd[i + 1])
                i += 2
                continue
            if c == quote:
                quote = ""
            i += 1
            continue
        if c == "\\" and i + 1 < n:
            cur.append(cmd[i:i + 2])
            i += 2
            continue
        if c in "'\"`":
            quote = c
            cur.append(c)
            i += 1
            continue
        if c == "#" and (i == 0 or cmd[i - 1] in " \t\n;&|("):
            while i < n and cmd[i] != "\n":
                i += 1
            continue
        if cmd.startswith("<<", i) and not cmd.startswith("<<<", i):
            m = re.match(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1", cmd[i:])
            if m:
                heredocs.append(m.group(2))
                cur.append(m.group(0))
                i += len(m.group(0))
                continue
        if c in "(":
            depth += 1
        elif c == ")" and depth:
            depth -= 1
        if depth == 0:
            if c == "\n":
                if heredocs:
                    # The heredoc bodies belong to the step that opened them.
                    rest = cmd[i + 1:].split("\n")
                    used = 0
                    body = []
                    for word in heredocs:
                        while used < len(rest):
                            line = rest[used]
                            used += 1
                            body.append(line)
                            if line.strip() == word:
                                break
                    cur.append("\n" + "\n".join(body))
                    i += 1 + sum(len(line) + 1 for line in rest[:used])
                    heredocs.clear()
                    flush()
                    continue
                flush()
                i += 1
                continue
            if cmd.startswith("&&", i) or cmd.startswith("||", i):
                flush()
                i += 2
                continue
            if c == ";" or (c == "&" and not cmd.startswith("&>", i) and (i == 0 or cmd[i - 1] not in "<>")):
                flush()
                i += 1
                continue
        cur.append(c)
        i += 1
    flush()
    return _merge_blocks(steps)


_OPENERS = {"for": "done", "while": "done", "until": "done", "select": "done", "if": "fi", "case": "esac"}
_CLOSERS = {"done", "fi", "esac"}


def _first_word(step: str) -> str:
    return step.split(None, 1)[0] if step.split() else ""


def _merge_blocks(steps: list[str]) -> list[str]:
    """Join the pieces of a ``for``/``while``/``if``/``case`` block back into one step."""
    merged: list[str] = []
    block: list[str] = []
    depth = 0
    for step in steps:
        word = _first_word(step)
        if word in _OPENERS:
            depth += 1
        if depth:
            block.append(step)
            if word in _CLOSERS:
                depth -= 1
                if depth == 0:
                    merged.append("; ".join(block))
                    block = []
            continue
        merged.append(step)
    if block:
        merged.append("; ".join(block))
    return merged


_TOKEN = re.compile(r"""(?:[^\s'"\\]|\\.|'[^']*'?|"(?:[^"\\]|\\.)*"?)+""")
_UNQUOTE = re.compile(r"""'([^']*)'?|"((?:[^"\\]|\\.)*)"?|\\(.)""")


def _words(step: str, all_lines: bool = False) -> list[str]:
    """Shell words of a step's first line, with quotes removed (close to shlex, much faster)."""
    first_line = step if all_lines else step.split("\n", 1)[0]
    return [
        _UNQUOTE.sub(lambda m: next((g for g in m.groups() if g is not None), ""), token)
        for token in _TOKEN.findall(first_line)
    ]


# Programs whose first plain word is a subcommand (``git log``, ``gh pr view``).
SUBCOMMAND_PROGRAMS = {
    "git", "gh", "herdr", "herdr-app", "hc", "docker", "podman", "uv", "npm", "pnpm",
    "yarn", "cargo", "kubectl", "systemctl", "journalctl", "minion", "agentconvos",
    "gbrain", "codex", "op", "gcloud", "az", "wmill", "tmux", "apt", "pip", "brew",
    "go", "make", "just", "vx-pane", "vx-review", "vx-index", "convo-miner", "tailscale",
    "coolify", "b2", "aws", "flyctl", "wrangler", "supabase", "terraform", "helm",
}
# Programs that run the command after them. The number is how many plain words
# (a duration, a lock file) they take before that command.
_PREFIXES = {
    "sudo": 0, "env": 0, "command": 0, "exec": 0, "nohup": 0, "time": 0, "nice": 0,
    "stdbuf": 0, "ionice": 0, "setsid": 0, "xvfb-run": 0, "timeout": 1, "flock": 1,
}
_PREFIX_FLAGS_WITH_VALUE = {"-n", "-u", "-g", "-w", "-E", "-c", "-k", "-s", "-o", "-e", "-i", "-a"}


def _strip_prefixes(words: list[str]) -> list[str]:
    """Drop ``VAR=value``, ``sudo``, ``nice -n 19``, ``flock <file>`` and the like."""
    while words:
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[0]):
            words = words[1:]
            continue
        program = os.path.basename(words[0])
        if program not in _PREFIXES:
            break
        rest = words[1:]
        while rest and rest[0].startswith("-"):
            flag = rest[0]
            rest = rest[2:] if flag in _PREFIX_FLAGS_WITH_VALUE and len(rest) > 1 else rest[1:]
        words = rest[_PREFIXES[program]:]
    return words
# Programs with two levels of subcommand (``gh pr view``, ``herdr pane split``).
NESTED_PROGRAMS = {
    "gh", "herdr", "herdr-app", "hc", "docker", "podman", "kubectl", "gcloud", "az", "op",
    "aws", "wmill", "b2", "tailscale", "coolify", "supabase", "flyctl", "wrangler", "helm",
    "systemctl", "uv", "pnpm", "npm", "codex", "gbrain", "minion",
}
NESTED_GIT = {"stash", "worktree", "remote", "submodule", "notes", "bisect"}
_PLAIN = re.compile(r"^[a-z][a-z0-9_-]*$")
# Looking around: a run made only of these is reading, not a workflow.
BASIC = {
    "cat", "ls", "sed", "rg", "grep", "head", "tail", "find", "wc", "jq", "awk", "sort",
    "cut", "file", "readlink", "realpath", "stat", "test", "[", "mkdir", "rm", "cp", "mv",
    "touch", "which", "diff", "tree", "du", "df", "ps", "uniq", "tr", "xargs", "column",
    "fd", "bat", "less", "nl", "date", "basename", "dirname", "env", "type", "command",
    "python", "python3", "node", "bash", "sh", "zsh", "chmod",
}
GLUE = {"cd", "echo", "printf", "sleep", "true", "false", ":", "set", "export", "pwd", "source", "."}


def shape(step: str) -> str:
    """Program and subcommand of a step, such as ``git log`` or ``rg``."""
    words = _words(step)
    if words and words[0] in _OPENERS:
        body = re.search(r"(?:^|;)\s*(?:do|then)\s+(.+)", step, re.S)
        inner = shape(split_steps(body.group(1))[0]) if body else ""
        return f"{words[0]}: {inner}" if inner else words[0]
    if words and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=\$\(", words[0]):
        return shape(step.split("$(", 1)[1].rstrip(")"))
    words = _strip_prefixes(words)
    if not words:
        return ""
    program = os.path.basename(words[0])
    if program in ("python", "python3", "node", "bash", "sh", "uv") and len(words) > 1:
        rest = words[1:]
        if program == "uv" and rest[0] == "run":
            rest = [w for w in rest[1:] if not w.startswith("-")]
            return "uv run " + (os.path.basename(rest[0]) if rest else "")
        script = next((w for w in rest if not w.startswith("-")), "")
        if program != "uv" and script.endswith((".py", ".js", ".mjs", ".sh")):
            return f"{program} {os.path.basename(script)}"
        if program != "uv":
            return program
    parts = [program]
    if program in SUBCOMMAND_PROGRAMS:
        rest = words[1:]
        if program == "git":
            while len(rest) >= 2 and rest[0] in ("-C", "-c"):
                rest = rest[2:]
        nested = program in NESTED_PROGRAMS or (program == "git" and rest[:1] and rest[0] in NESTED_GIT)
        for word in rest[: 2 if nested else 1]:
            if not _PLAIN.match(word):
                break
            parts.append(word)
    return " ".join(parts)


_HEX_ID = re.compile(r"^[0-9a-f]{7,}$|^[0-9a-f]{8}-[0-9a-f-]{27,}$", re.I)


def _norm_word(word: str) -> str:
    if re.fullmatch(r"-\d+", word):
        return "-<n>"
    if word.startswith("-"):
        flag, eq, value = word.partition("=")
        return flag + (eq + _norm_word(value) if eq else "")
    if re.fullmatch(r"\d+(\.\d+)?[a-z]{0,2}", word):
        return "<n>"
    if _HEX_ID.match(word):
        return "<id>"
    if any(ch.isspace() for ch in word):
        return "<text>"
    if "/" in word or word.startswith("~"):
        return "<path>"
    if len(word) > 40:
        return "<text>"
    return word


def template(cmd: str, steps: list[str] | None = None) -> str:
    """The command with its paths, numbers, IDs and long text left out."""
    out = []
    for step in steps if steps is not None else split_steps(cmd):
        # A heredoc's body is part of the command: two different inline scripts
        # are not the same command.
        words = _words(step, all_lines=True)
        out.append(" ".join(_norm_word(w) for w in words))
    return " ; ".join(out)


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)


def collect(paths: list[Path]) -> tuple[list[Command], list[Path]]:
    commands: list[Command] = []
    skipped: list[Path] = []
    for path in paths:
        found = session_commands(Path(path))
        if found is None:
            skipped.append(Path(path))
            continue
        stem = Path(path).stem
        uuid = _UUID.search(stem)
        session = uuid.group(0) if uuid else stem
        commands.extend(Command(text, session, i) for i, text in enumerate(found, start=1))
    return commands, skipped


def group(commands: list[Command], top: int = 15, min_repeats: int = 3, max_run: int = 6) -> dict:
    by_shape: dict[str, list[int]] = defaultdict(list)
    by_template: dict[str, list[Command]] = defaultdict(list)
    per_session: dict[str, list[tuple[str, str, Command]]] = defaultdict(list)

    for cmd in commands:
        steps = split_steps(cmd.text)
        by_template[template(cmd.text, steps)].append(cmd)
        for step in steps:
            sh = shape(step)
            if not sh:
                continue
            by_shape[sh].append(len(step))
            if sh.split(" ", 1)[0] not in GLUE:
                per_session[cmd.session].append((sh, step, cmd))

    programs = sorted(
        ({"shape": s, "uses": len(lens), "chars": sum(lens)} for s, lens in by_shape.items()),
        key=lambda r: r["chars"], reverse=True,
    )[:top]

    repeated = []
    for tpl, cmds in by_template.items():
        if len(cmds) < 2:
            continue
        mean = sum(len(c.text) for c in cmds) / len(cmds)
        if mean < 40:
            continue
        example = max(cmds, key=lambda c: len(c.text))
        repeated.append({
            "template": tpl, "count": len(cmds), "mean_chars": round(mean),
            "chars": sum(len(c.text) for c in cmds),
            "sessions": len({c.session for c in cmds}),
            "example": example.text, "example_session": example.session, "example_index": example.index,
        })
    repeated.sort(key=lambda r: r["chars"], reverse=True)

    # Consecutive runs: a step that repeats right after itself counts once, so
    # three rg searches in a row are not mistaken for a workflow.
    runs: Counter = Counter()
    run_sessions: dict[tuple, set] = defaultdict(set)
    run_example: dict[tuple, list] = {}
    for session, steps in per_session.items():
        collapsed: list[tuple[str, str, Command]] = []
        for item in steps:
            if not collapsed or collapsed[-1][0] != item[0]:
                collapsed.append(item)
        for size in range(2, max_run + 1):
            for start in range(len(collapsed) - size + 1):
                window = collapsed[start:start + size]
                key = tuple(w[0] for w in window)
                if len(set(key)) < 2 or all(k.split(" ", 1)[0] in BASIC for k in key):
                    continue
                runs[key] += 1
                run_sessions[key].add(session)
                run_example.setdefault(key, window)
    kept = [k for k, c in runs.items() if c >= min_repeats]
    # Keep a run only when no longer run containing it happens as often.
    best_extension: dict[tuple, int] = defaultdict(int)
    for key in kept:
        for part in (key[:-1], key[1:]):
            best_extension[part] = max(best_extension[part], runs[key])
    maximal = [key for key in kept if best_extension.get(key, 0) < runs[key]]
    maximal.sort(key=lambda k: runs[k] * len(k), reverse=True)
    sequences = [
        {
            "steps": list(key), "count": runs[key], "sessions": len(run_sessions[key]),
            "example": [w[1] for w in run_example[key]],
            "example_session": run_example[key][0][2].session,
            "example_index": run_example[key][0][2].index,
        }
        for key in maximal[:top]
    ]
    return {
        "commands": len(commands),
        "sessions": len({c.session for c in commands}),
        "programs": programs,
        "repeated_commands": repeated[:top],
        "repeated_runs": sequences,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _clip(text: str, width: int) -> str:
    one = " ".join(text.split())
    return one if len(one) <= width else one[: width - 1] + "…"


def print_commands(commands: list[Command], many_sessions: bool) -> None:
    for cmd in commands:
        label = f"{cmd.session[:8]} #{cmd.index}" if many_sessions else f"#{cmd.index}"
        print(f"# {label}")
        print(cmd.text.rstrip())
        print()


def print_groups(result: dict, width: int = 100) -> None:
    print(f"{result['commands']} commands in {result['sessions']} conversations.\n")

    print("Programs, by characters typed:")
    for row in result["programs"]:
        print(f"  {row['chars']:>8,}  {row['uses']:>5}×  {row['shape']}")

    print("\nRepeated commands (same apart from paths, numbers and IDs), by characters typed:")
    if not result["repeated_commands"]:
        print("  none")
    for row in result["repeated_commands"]:
        print(f"\n  {row['count']}× in {row['sessions']} conversations, {row['mean_chars']} characters each")
        print(f"    shape:   {_clip(row['template'], width)}")
        print(f"    example: {_clip(row['example'], width)}")
        print(f"             ({row['example_session'][:8]} #{row['example_index']})")

    print("\nRepeated runs (steps done one after another), by count × length:")
    if not result["repeated_runs"]:
        print("  none")
    for row in result["repeated_runs"]:
        print(f"\n  {row['count']}× in {row['sessions']} conversations: {' → '.join(row['steps'])}")
        for step in row["example"]:
            print(f"    {_clip(step, width)}")
        print(f"    ({row['example_session'][:8]} #{row['example_index']})")
