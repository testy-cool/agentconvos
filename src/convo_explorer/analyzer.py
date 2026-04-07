"""Optional Gemini-powered conversation analysis. Requires GEMINI_API_KEY."""

from __future__ import annotations

import os
from pathlib import Path

from .parser import Turn, to_markdown

MODELS = [
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-3.1-pro-preview",
]

DEFAULT_MODEL = MODELS[0]

SINGLE_PROMPT = """Analyze this Claude Code conversation and extract:

1. **Key Decisions** — technical choices made and why
2. **User Preferences** — communication style, tool preferences, workflow patterns
3. **Problems & Solutions** — issues hit and how they were resolved
4. **Patterns** — recurring approaches, habits, or conventions
5. **Unfinished Work** — TODOs, blocked items, things left open

Be specific. Use exact names, paths, and values from the conversation.
Output as structured markdown.

CONVERSATION:
{content}"""

MULTI_PROMPT = """Analyze these {count} Claude Code conversations together and find cross-session patterns:

1. **Recurring Preferences** — what the user consistently asks for or corrects
2. **Workflow Patterns** — how they typically start sessions, structure work, make decisions
3. **Communication Style** — how they phrase requests, level of detail they expect
4. **Tool & Tech Preferences** — preferred languages, frameworks, tools, approaches
5. **Pain Points** — recurring frustrations or corrections
6. **Evolution** — how preferences or approaches changed over time

Be specific and cite which conversation each observation comes from.
Output as structured markdown.

CONVERSATIONS:
{content}"""


def _load_env():
    """Load GEMINI_API_KEY from .env files if not already set."""
    if os.environ.get("GEMINI_API_KEY"):
        return
    # Check .env in cwd, then ~/.claude/convo-explorer/.env
    candidates = [
        Path(".env"),
        Path(os.environ.get("USERPROFILE", Path.home())) / ".claude" / "convo-explorer" / ".env",
    ]
    for env_path in candidates:
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip().strip("'\"")
                if key == "GEMINI_API_KEY" and val:
                    os.environ["GEMINI_API_KEY"] = val
                    return


def gemini_available() -> bool:
    _load_env()
    return bool(os.environ.get("GEMINI_API_KEY"))


def analyze_single(turns: list[Turn], model: str = DEFAULT_MODEL, prompt_template: str = SINGLE_PROMPT) -> str:
    """Analyze a single conversation with Gemini."""
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    content = to_markdown(turns)
    prompt = prompt_template.replace("{content}", content)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    return response.text


def analyze_multi(conversations: list[tuple[str, list[Turn]]], model: str = DEFAULT_MODEL, prompt_template: str = MULTI_PROMPT) -> str:
    """Analyze multiple conversations. Each tuple is (label, turns)."""
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    parts = []
    for label, turns in conversations:
        md = to_markdown(turns)
        parts.append(f"### {label}\n{md}")

    content = "\n---\n".join(parts)
    prompt = prompt_template.replace("{count}", str(len(conversations))).replace("{content}", content)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    return response.text
