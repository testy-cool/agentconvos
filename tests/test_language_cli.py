import contextlib
import io
import json
import sys
from unittest.mock import patch

import pytest

from agentconvos.app import main
from agentconvos.language_nlp import LanguageNLPError
from agentconvos.parser import ConversationMeta
from agentconvos.scanner import Project


def _run_cli(arguments: list[str]):
    stdout = io.StringIO()
    stderr = io.StringIO()
    old_argv = sys.argv
    sys.argv = ["agentconvos", *arguments]
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            main()
    finally:
        sys.argv = old_argv
    return stdout.getvalue(), stderr.getvalue()


def test_help_documents_opt_in_language_flags():
    old_argv = sys.argv
    stream = io.StringIO()
    sys.argv = ["agentconvos", "--help"]
    try:
        with contextlib.redirect_stdout(stream), pytest.raises(SystemExit) as raised:
            main()
    finally:
        sys.argv = old_argv

    assert raised.value.code == 0
    help_text = stream.getvalue()
    assert "--nlp" in help_text
    assert "--baseline {matched,none}" in help_text
    assert "--seed SEED" in help_text
    assert "--spacy-model SPACY_MODEL" in help_text


@pytest.mark.parametrize(
    "arguments",
    [
        ["--nlp", "--source", "claude"],
        ["--baseline", "matched", "--habits", "--source", "claude"],
        ["--seed", "7", "--habits", "--source", "claude"],
        ["--spacy-model", "custom_model", "--habits", "--source", "claude"],
    ],
)
def test_language_flags_reject_misplaced_combinations(arguments):
    old_argv = sys.argv
    stream = io.StringIO()
    sys.argv = ["agentconvos", *arguments]
    try:
        with contextlib.redirect_stderr(stream), pytest.raises(SystemExit) as raised:
            main()
    finally:
        sys.argv = old_argv

    assert raised.value.code == 2
    assert "usage:" in stream.getvalue()


def test_missing_language_runtime_exits_one_without_report_artifacts(tmp_path):
    transcript = tmp_path / "one.jsonl"
    transcript.write_text(
        '\n'.join(
            (
                json.dumps({"type": "user", "message": {"content": "private prompt"}}),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {"content": [{"type": "text", "text": "A reply."}]},
                    }
                ),
            )
        ),
        encoding="utf-8",
    )
    conversation = ConversationMeta(
        path=transcript,
        uuid="session-one",
        slug="",
        timestamp="2026-08-20T12:00:00Z",
        cwd="/private/project-one",
        preview="private prompt",
        source="claude",
    )
    projects = [Project("project-one", "/private/project-one", [conversation])]
    output = tmp_path / "report.html"
    old_argv = sys.argv
    stderr = io.StringIO()
    sys.argv = [
        "agentconvos",
        "--habits",
        "--nlp",
        "--source",
        "claude",
        "--output",
        str(output),
    ]
    try:
        with (
            patch("agentconvos.scanner.scan_projects", return_value=projects),
            patch(
                "agentconvos.language_nlp.NLPAdapter.load",
                side_effect=LanguageNLPError(
                    "Install agentconvos[language] and en_core_web_sm explicitly."
                ),
            ),
            contextlib.redirect_stderr(stderr),
            pytest.raises(SystemExit) as raised,
        ):
            main()
    finally:
        sys.argv = old_argv

    assert raised.value.code == 1
    assert stderr.getvalue().strip() == (
        "Install agentconvos[language] and en_core_web_sm explicitly."
    )
    assert not output.exists()
    assert not output.with_suffix(".json").exists()


def test_nlp_cli_forwards_date_filters_and_preserves_legacy_path(tmp_path):
    output = tmp_path / "report.html"
    calls = []

    def fake_scan_projects(**kwargs):
        calls.append(kwargs)
        return []

    old_argv = sys.argv
    sys.argv = [
        "agentconvos",
        "--habits",
        "--nlp",
        "--source",
        "claude",
        "--after",
        "2026-08-01",
        "--before",
        "2026-08-02",
        "--output",
        str(output),
    ]
    try:
        with (
            patch("agentconvos.scanner.scan_projects", side_effect=fake_scan_projects),
            patch("agentconvos.language_report.run_language_report") as run_report,
        ):
            run_report.return_value.public_dict.return_value = {}
            main()
    finally:
        sys.argv = old_argv

    assert any(
        call.get("source") == "claude"
        and call.get("after") == "2026-08-01"
        and call.get("before") == "2026-08-02"
        for call in calls
    )
    assert len(calls) == 1
    run_report.assert_called_once()

    old_argv = sys.argv
    sys.argv = ["agentconvos", "--habits", "--source", "claude", "--output", str(output)]
    try:
        with (
            patch("agentconvos.scanner.scan_projects", return_value=[]),
            patch("agentconvos.habits.analyze_habits") as legacy,
            patch("agentconvos.language_report.run_language_report") as nlp_report,
            patch("agentconvos.habits.write_report"),
        ):
            legacy.return_value.public_dict.return_value = {}
            main()
    finally:
        sys.argv = old_argv

    legacy.assert_called_once()
    nlp_report.assert_not_called()
