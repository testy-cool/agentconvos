import contextlib
import io
import re
import sys
import tomllib
from pathlib import Path

import pytest

from agentconvos.app import main

ROOT = Path(__file__).parents[1]


def _help_text() -> str:
    stream = io.StringIO()
    old_argv = sys.argv
    sys.argv = ["agentconvos", "--help"]
    try:
        with contextlib.redirect_stdout(stream), pytest.raises(SystemExit) as raised:
            main()
    finally:
        sys.argv = old_argv
    assert raised.value.code == 0
    return stream.getvalue()


def test_public_docs_freeze_the_shipped_desire_paths():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    spec = (ROOT / "SPEC.md").read_text(encoding="utf-8")
    help_text = _help_text()

    for token in (
        "--context",
        "--search",
        "--json",
        "--resume",
        "--handoff",
        "--habits",
        "--nlp",
        "--find",
        "recall",
    ):
        assert token in help_text
        assert token in readme
        assert token in spec


def test_readme_installation_is_source_truthful_and_links_the_contract():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]

    assert "git+https://github.com/testy-cool/agentconvos.git" in readme
    assert "not published on PyPI" in readme
    assert "[SPEC.md](SPEC.md)" in readme
    assert "python -m spacy download en_core_web_sm" in readme
    assert "pip install 'agentconvos" not in readme
    assert "License :: OSI Approved :: MIT License" in project["classifiers"]


def test_public_markdown_has_no_private_identity_or_broken_local_links():
    documents = [ROOT / "README.md", ROOT / "SPEC.md"]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in documents)

    assert "/home/" not in combined
    assert "/Users/" not in combined
    assert "bifrost" not in combined.casefold()
    assert not re.search(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        combined,
        re.IGNORECASE,
    )

    local_links = re.findall(r"\[[^]]+\]\((?!https?://|mailto:|#)([^)]+)\)", combined)
    image_links = re.findall(r'<img[^>]+src="(?!https?://)([^"]+)"', combined)
    for target in local_links + image_links:
        assert (ROOT / target.split("#", 1)[0]).exists(), target


def test_public_docs_offer_a_portable_llm_setup():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    spec = (ROOT / "SPEC.md").read_text(encoding="utf-8")

    for document in (readme, spec):
        assert "OPENAI_API_KEY" in document
        assert "OPENAI_BASE_URL" in document
        assert "OPENAI_MODEL" in document
        assert "AGENTCONVOS_LLM_BASE_URL" in document
        assert "AGENTCONVOS_LLM_API_KEY" in document
        assert "AGENTCONVOS_LLM_KEY_NAME" in document
        assert "AGENTCONVOS_LLM_MODEL" in document
