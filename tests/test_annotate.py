import json
import sqlite3

from agentconvos import annotate
from agentconvos.parser import ConversationMeta, Turn


def test_consecutive_agent_turns_form_one_reply_to_the_user_message():
    turns = [
        Turn("user", "first ask"),
        Turn("assistant", "looking"),
        Turn("assistant", "done"),
        Turn("user", "<pasted_content id=\"x\">second</pasted_content> ask"),
        Turn("assistant", "ok"),
        Turn("user", "unanswered"),
    ]
    items = annotate.build_items(turns)
    assert [item["id"] for item in items] == ["reply-0001", "reply-0002"]
    assert items[0]["blocks"][1]["text"] == "looking\n\ndone"
    assert items[1]["title"] == "Reply 2 · second ask"


def test_notes_print_verdict_note_and_quoted_comment(tmp_path, monkeypatch):
    session = tmp_path / "s.jsonl"
    session.write_text("")
    meta = ConversationMeta(path=session, uuid="u1", slug="", timestamp="2026-11-06T10:00:00", cwd=str(tmp_path), preview="")
    (tmp_path / "AGENTS.md").write_text("rules")
    monkeypatch.setattr(annotate, "parse_jsonl", lambda *a, **k: [Turn("user", "hi"), Turn("assistant", "hello there")])
    job = tmp_path / "job"
    assert annotate.write_job(session, meta, job) == 1

    db = sqlite3.connect(job / "review/annotations.sqlite")
    db.execute("CREATE TABLE states (item TEXT, hash TEXT, data TEXT)")
    state = {"revision": 2, "verdict": "Fail", "note": "too terse",
             "quotes": [{"exact": "hello", "note": "say more", "tone": "bad"}]}
    db.execute("INSERT INTO states VALUES (?,?,?)", ("reply-0001", "h", json.dumps(state)))
    db.commit()
    db.close()

    notes = annotate.notes_markdown(job)
    assert "**Verdict:** Fail" in notes
    assert "too terse" in notes
    assert "Comment (bad): say more" in notes
    assert str(tmp_path / "AGENTS.md") in notes
    assert "0 passed, 1 failed" in notes
