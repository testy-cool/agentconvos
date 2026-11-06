import json

from agentconvos import commands as cmds


def test_steps_split_at_operators_but_keep_quotes_loops_and_heredocs_whole():
    text = (
        'cd /x && for f in *; do echo "$f"; done | sort; echo "a;b" || true\n'
        "cat <<EOF\nhi; there\nEOF\nls 2>&1"
    )
    assert cmds.split_steps(text) == [
        "cd /x",
        'for f in *; do echo "$f"; done | sort',
        'echo "a;b"',
        "true",
        "cat <<EOF\nhi; there\nEOF",
        "ls 2>&1",
    ]


def test_shape_names_program_and_subcommand_past_wrappers():
    assert cmds.shape("git -C /repo log --oneline -5") == "git log"
    assert cmds.shape("nice -n 19 flock /tmp/a.lock gh pr view 3") == "gh pr view"
    assert cmds.shape("DISPLAY=:1 timeout 20s python3 tools/check.py --fast") == "python3 check.py"
    assert cmds.shape("rg -n foo src") == "rg"
    assert cmds.shape("for f in *; do herdr pane read $f; done") == "for: herdr pane read"
    assert cmds.shape("n=$(git rev-parse HEAD)") == "git rev-parse"


def test_template_drops_paths_numbers_and_ids_but_not_heredoc_bodies():
    a = cmds.template("cat /home/a/notes.md | head -40")
    b = cmds.template("cat /tmp/other.md | head -5")
    assert a == b == "cat <path> | head -<n>"
    assert cmds.template("python3 - <<'PY'\nprint(1)\nPY") != cmds.template("python3 - <<'PY'\nimport os\nPY")


def test_reads_full_commands_from_claude_codex_and_pi(tmp_path):
    long = "echo " + "x" * 300
    claude = tmp_path / "c.jsonl"
    claude.write_text(json.dumps({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Bash", "input": {"command": long}},
        {"type": "tool_use", "name": "Read", "input": {"file_path": "/a"}},
    ]}}) + "\n")
    codex = tmp_path / "rollout-2026-11-06T10-00-00-01a1b2c3-fa11-7860-89ad-ba68a8d3eb45.jsonl"
    codex.write_text("\n".join(json.dumps(r) for r in [
        {"type": "session_meta", "payload": {}},
        {"type": "response_item", "payload": {"type": "function_call", "name": "shell",
                                              "arguments": json.dumps({"command": ["bash", "-lc", "git status"]})}},
        {"type": "response_item", "payload": {"type": "custom_tool_call", "name": "exec",
                                              "input": 'tools.exec_command({cmd:"rg -n \\"a b\\" src"}); tools.exec_command({cmd:\'ls\'})'}},
    ]) + "\n")
    pi = tmp_path / "p.jsonl"
    pi.write_text("\n".join(json.dumps(r) for r in [
        {"type": "session", "version": 3},
        {"type": "message", "message": {"role": "assistant", "content": [
            {"type": "toolCall", "name": "bash", "arguments": {"command": "uv run pytest"}}]}},
    ]) + "\n")

    assert cmds.session_commands(claude) == [long]
    assert cmds.session_commands(codex) == ["git status", 'rg -n "a b" src', "ls"]
    assert cmds.session_commands(pi) == ["uv run pytest"]
    found, skipped = cmds.collect([codex])
    assert found[0].session == "01a1b2c3-fa11-7860-89ad-ba68a8d3eb45" and not skipped


def _commands(*sessions):
    return [
        cmds.Command(text, f"s{n}", i)
        for n, texts in enumerate(sessions)
        for i, text in enumerate(texts, start=1)
    ]


def test_group_finds_repeated_runs_across_consecutive_commands():
    session = [
        "git add a.py", "git diff --cached --check", "git commit -m one", "git push",
        "rg foo", "sed -n 1,5p a.py",
        "git add b.py && git diff --cached --check && git commit -m two && git push",
        "cat x", "rg bar",
        "git add c.py", "git add d.py", "git diff --cached --check", "git commit -m three", "git push",
    ]
    result = cmds.group(_commands(session), min_repeats=3)
    runs = {" → ".join(r["steps"]): r["count"] for r in result["repeated_runs"]}
    # The whole run is listed, not each of its pieces, and the repeated
    # "git add" counts once.
    assert runs == {"git add → git diff → git commit → git push": 3}


def test_group_leaves_out_runs_of_reading_commands_and_short_one_offs():
    session = ["cat a", "rg b", "cat c", "rg d", "cat e", "rg f", "ls"]
    result = cmds.group(_commands(session), min_repeats=2)
    assert result["repeated_runs"] == []
    assert result["repeated_commands"] == []


def test_group_ranks_repeated_long_commands_by_characters_typed():
    long = "herdr pane list --workspace wJE 2>&1 | python3 -c 'import json,sys; print(1)'"
    result = cmds.group(_commands([long, long], [long, "ls"]))
    top = result["repeated_commands"][0]
    assert top["count"] == 3 and top["sessions"] == 2 and top["example"] == long
