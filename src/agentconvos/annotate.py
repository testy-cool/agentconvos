"""Open a pass/fail review dashboard for one conversation's agent replies.

Each agent reply becomes one record in a vx-annotate job: the user message it
answered, then the reply. The reviewer marks it Pass or Fail and writes a note.
``--notes`` prints what they wrote as Markdown, ready to hand to an agent that
will turn it into changes to the reviewed agent's instructions.

The dashboard itself is the vx-annotate engine, found as ``vx-annotate`` on
PATH or through ``AGENTCONVOS_ANNOTATE_ENGINE``. This module only writes the
job folder and reads its feedback back.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import webbrowser
from pathlib import Path

from .parser import DETAIL_TEXT, ConversationMeta, Turn, get_meta, parse_jsonl

JOBS_ROOT = Path.home() / ".local/share/agentconvos/annotate"
VERDICTS = ["Pass", "Fail"]
ENGINE_ENV = "AGENTCONVOS_ANNOTATE_ENGINE"


def replies(turns: list[Turn]) -> list[tuple[str, str]]:
    """Pair each agent reply with the user message it answered.

    Consecutive assistant turns form one reply, because the reviewer judges
    what the agent said back, not how the parser split it.
    """
    pairs: list[tuple[str, list[str]]] = []
    asked = ""
    for turn in turns:
        if turn.role == "user":
            asked = turn.text
            pairs.append((asked, []))
        elif turn.role == "assistant" and turn.text.strip():
            if not pairs:
                pairs.append(("", []))
            pairs[-1][1].append(turn.text)
    return [(asked, "\n\n".join(parts)) for asked, parts in pairs if parts]


def _title(index: int, asked: str) -> str:
    plain = re.sub(r"<[^>\n]{1,200}>", " ", asked)
    first = " ".join(plain.split())[:70] or "(no user message)"
    return f"Reply {index} · {first}"


def build_items(turns: list[Turn]) -> list[dict]:
    items = []
    for index, (asked, reply) in enumerate(replies(turns), start=1):
        blocks = []
        if asked.strip():
            blocks.append({"id": "user", "type": "message", "role": "user", "title": "User", "text": asked})
        blocks.append({"id": "reply", "type": "markdown", "role": "assistant", "title": "Agent reply", "text": reply})
        items.append({"id": f"reply-{index:04d}", "title": _title(index, asked), "meta": {"reply": index}, "blocks": blocks})
    return items


def job_dir_for(meta: ConversationMeta) -> Path:
    return JOBS_ROOT / f"{meta.source}-{meta.uuid}"


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_job(path: Path, meta: ConversationMeta, job_dir: Path) -> int:
    """Write or refresh the job. Item IDs are stable, so notes survive a rerun."""
    review = job_dir / "review"
    review.mkdir(parents=True, exist_ok=True)
    items = build_items(parse_jsonl(path, detail=DETAIL_TEXT))
    label = meta.slug or meta.uuid[:8]
    config = {
        "schemaVersion": 1,
        "title": f"{meta.source} · {label} · {meta.timestamp[:10]}",
        "instructions": (
            "Mark each agent reply Pass or Fail and say why in the note. "
            "Select words in a reply to comment on that passage."
        ),
        "verdicts": VERDICTS,
        "layout": "right",
    }
    _atomic_write(review / "job.json", json.dumps(config, indent=2) + "\n")
    _atomic_write(review / "items.jsonl", "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items))
    _atomic_write(job_dir / "source.json", json.dumps({
        "file": str(path), "source": meta.source, "uuid": meta.uuid,
        "slug": meta.slug, "cwd": meta.cwd, "timestamp": meta.timestamp,
    }, indent=2) + "\n")
    return len(items)


def find_engine() -> str | None:
    explicit = os.environ.get(ENGINE_ENV)
    if explicit:
        return explicit if Path(explicit).is_file() else None
    return shutil.which("vx-annotate")


def _states(job_dir: Path) -> dict[str, dict]:
    """Latest saved feedback per item. Older rows belong to replaced evidence."""
    db = job_dir / "review" / "annotations.sqlite"
    if not db.exists():
        return {}
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT item, data FROM states").fetchall()
    finally:
        con.close()
    latest: dict[str, dict] = {}
    for item, data in rows:
        state = json.loads(data)
        if item not in latest or state.get("revision", 0) > latest[item].get("revision", 0):
            latest[item] = state
    return latest


def _quote(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) > limit:
        text = text[:limit].rstrip() + " …"
    return "\n".join("> " + line if line else ">" for line in text.splitlines())


def notes_markdown(job_dir: Path) -> str:
    source = json.loads((job_dir / "source.json").read_text())
    items = [json.loads(line) for line in (job_dir / "review/items.jsonl").read_text().splitlines() if line.strip()]
    states = _states(job_dir)
    noted = []
    for item in items:
        state = states.get(item["id"])
        if state and (state.get("verdict") or state.get("note", "").strip() or state.get("quotes")):
            noted.append((item, state))

    passed = sum(1 for _, s in noted if s.get("verdict") == "Pass")
    failed = sum(1 for _, s in noted if s.get("verdict") == "Fail")
    lines = [
        f"# Review notes: {source['source']} session {source['slug'] or source['uuid']}",
        "",
        f"- Session file: `{source['file']}`",
        f"- Agent folder: `{source['cwd']}`",
        f"- Notes on {len(noted)} of {len(items)} replies: {passed} passed, {failed} failed.",
    ]
    cwd = Path(source["cwd"]) if source.get("cwd") else None
    found = [name for name in ("AGENTS.md", "CLAUDE.md") if cwd and (cwd / name).is_file()]
    if found:
        lines.append("- Instruction files there: " + ", ".join(f"`{cwd / name}`" for name in found))
    lines.append("")
    if not noted:
        lines.append("No notes yet.")
    for item, state in noted:
        blocks = {block["id"]: block["text"] for block in item["blocks"]}
        lines += [f"## {item['title']}", "", f"**Verdict:** {state.get('verdict') or 'none'}", ""]
        if state.get("note", "").strip():
            lines += ["**Note:**", "", state["note"].strip(), ""]
        for quote in state.get("quotes", []):
            tone = f" ({quote['tone']})" if quote.get("tone") else ""
            lines += [_quote(quote.get("exact", ""), 400), "", f"Comment{tone}: {quote.get('note', '').strip()}", ""]
        if blocks.get("user"):
            lines += ["User asked:", "", _quote(blocks["user"], 300), ""]
        lines += ["Agent replied:", "", _quote(blocks["reply"], 600), ""]
    return "\n".join(lines).rstrip() + "\n"


def run_annotate(path: Path, *, port: int, notes: bool, open_browser: bool) -> int:
    meta = get_meta(path)
    if meta is None:
        print(f"Error: could not read metadata from {path}", file=sys.stderr)
        return 1
    job_dir = job_dir_for(meta)

    if notes:
        if not (job_dir / "source.json").exists():
            print(f"Error: no review exists for this session yet. Run: agentconvos annotate {meta.uuid}", file=sys.stderr)
            return 1
        sys.stdout.write(notes_markdown(job_dir))
        return 0

    engine = find_engine()
    if engine is None:
        print(
            f"Error: the vx-annotate engine was not found. Put `vx-annotate` on PATH or set {ENGINE_ENV}.",
            file=sys.stderr,
        )
        return 1
    count = write_job(path, meta, job_dir)
    print(f"{count} agent replies to review. Notes are kept in {job_dir}")
    print(f"When done, run: agentconvos annotate {meta.uuid} --notes")

    proc = subprocess.Popen([engine, str(job_dir), "--port", str(port)], stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            if open_browser and "http://127.0.0.1:" in line:
                webbrowser.open(line.rsplit(" ", 1)[-1].strip())
                open_browser = False
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
        return 0
