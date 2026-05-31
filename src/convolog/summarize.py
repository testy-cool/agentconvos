from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

from .parser import ConversationMeta, parse_jsonl, DETAIL_TEXT

SUMMARIES_DIR = Path.home() / ".claude" / "convo-explorer" / "summaries"
BIFROST_URL = "https://bifrost.voidxd.cloud/v1/chat/completions"
MODEL = "gemini/gemini-3.1-flash-lite-preview"
_LLM_KEYS = Path.home() / ".config" / "io.datasette.llm" / "keys.json"

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
    data = {
        "summary": summary,
        "model": MODEL,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (SUMMARIES_DIR / f"{uuid}.json").write_text(json.dumps(data, indent=2))


def _load_api_key() -> str:
    if not _LLM_KEYS.exists():
        raise RuntimeError(f"No llm keys at {_LLM_KEYS}")
    keys = json.loads(_LLM_KEYS.read_text())
    key = keys.get("bifrost")
    if not key:
        raise RuntimeError("No 'bifrost' key in llm keys.json")
    return key


def _call_bifrost(prompt: str, api_key: str) -> str:
    import httpx
    import time
    for attempt in range(3):
        resp = httpx.post(
            BIFROST_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,
            },
            timeout=60,
        )
        if resp.status_code == 429 and attempt < 2:
            time.sleep(10 * (attempt + 1))
            continue
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip().strip('"')
    resp.raise_for_status()
    return ""


def summarize_session(meta: ConversationMeta, api_key: str) -> str:
    turns = parse_jsonl(meta.path, detail=DETAIL_TEXT, last_n=5)
    if not turns:
        return ""
    content = "\n\n".join(f"**{t.role}**: {t.text}" for t in turns)
    summary = _call_bifrost(PROMPT.format(content=content), api_key)
    _write_cache(meta.uuid, summary)
    return summary


def summarize_all(
    projects: list,
    api_key: str | None = None,
    on_progress: callable | None = None,
) -> tuple[int, int]:
    if not api_key:
        api_key = _load_api_key()
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
