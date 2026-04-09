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

# ~300K tokens = ~1.2M chars. Conversations above this get chunked.
CHUNK_THRESHOLD_CHARS = 1_200_000
CHUNK_TARGET_CHARS = 1_200_000

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

CHUNK_PROMPT = """You are analyzing PART {chunk_num} of {total_chunks} of a large Claude Code conversation.

Analyze THIS PART and extract:

1. **Key Decisions** — technical choices made and why
2. **User Preferences** — communication style, tool preferences, workflow patterns
3. **Problems & Solutions** — issues hit and how they were resolved
4. **Patterns** — recurring approaches, habits, or conventions
5. **Unfinished Work** — TODOs, blocked items, things left open

Be specific. Use exact names, paths, and values. Note this is part {chunk_num}/{total_chunks} — focus only on what's in this part.
Output as structured markdown.

CONVERSATION (part {chunk_num}/{total_chunks}):
{content}"""

SYNTHESIS_PROMPT = """Below are {total_chunks} separate analyses of consecutive parts of ONE large Claude Code conversation.

Synthesize them into a single cohesive analysis:

1. **Key Decisions** — merge and deduplicate, preserve chronological order
2. **User Preferences** — combine, note if preferences evolved across parts
3. **Problems & Solutions** — full timeline of issues and resolutions
4. **Patterns** — recurring themes across all parts
5. **Unfinished Work** — only items not resolved in later parts

Resolve any contradictions between parts (later parts override earlier ones).
Output as structured markdown.

PART ANALYSES:
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


def _chunk_turns(turns: list[Turn], target_chars: int = CHUNK_TARGET_CHARS) -> list[list[Turn]]:
    """Split turns into chunks that fit within target_chars each."""
    chunks: list[list[Turn]] = []
    current: list[Turn] = []
    current_size = 0

    for turn in turns:
        turn_size = len(turn.text) + 20  # overhead for headers
        if current and current_size + turn_size > target_chars:
            chunks.append(current)
            current = []
            current_size = 0
        current.append(turn)
        current_size += turn_size

    if current:
        chunks.append(current)
    return chunks


def _call_gemini(client, model: str, prompt: str) -> str:
    """Single Gemini API call."""
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text


def analyze_single(
    turns: list[Turn],
    model: str = DEFAULT_MODEL,
    prompt_template: str = SINGLE_PROMPT,
    on_progress: callable = None,
) -> str:
    """Analyze a single conversation with Gemini. Auto-chunks if too large."""
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    content = to_markdown(turns)

    # Check if chunking needed
    if len(content) <= CHUNK_THRESHOLD_CHARS:
        prompt = prompt_template.replace("{content}", content)
        return _call_gemini(client, model, prompt)

    # Chunk it
    chunks = _chunk_turns(turns)
    total = len(chunks)
    if on_progress:
        on_progress(f"Large conversation ({len(content):,} chars) — splitting into {total} chunks")

    chunk_analyses = []
    for i, chunk_turns in enumerate(chunks, 1):
        if on_progress:
            on_progress(f"Analyzing chunk {i}/{total}...")
        chunk_md = to_markdown(chunk_turns)
        prompt = CHUNK_PROMPT.replace("{chunk_num}", str(i)).replace("{total_chunks}", str(total)).replace("{content}", chunk_md)
        result = _call_gemini(client, model, prompt)
        chunk_analyses.append(f"## Part {i}/{total}\n\n{result}")

    # Synthesis pass
    if on_progress:
        on_progress(f"Synthesizing {total} chunk analyses...")
    combined = "\n\n---\n\n".join(chunk_analyses)
    synth_prompt = SYNTHESIS_PROMPT.replace("{total_chunks}", str(total)).replace("{content}", combined)
    return _call_gemini(client, model, synth_prompt)


def analyze_multi(
    conversations: list[tuple[str, list[Turn]]],
    model: str = DEFAULT_MODEL,
    prompt_template: str = MULTI_PROMPT,
    on_progress: callable = None,
) -> str:
    """Analyze multiple conversations. Each tuple is (label, turns)."""
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    parts = []
    for label, turns in conversations:
        md = to_markdown(turns)
        parts.append(f"### {label}\n{md}")

    content = "\n---\n".join(parts)

    # Check if chunking needed
    if len(content) <= CHUNK_THRESHOLD_CHARS:
        prompt = prompt_template.replace("{count}", str(len(conversations))).replace("{content}", content)
        return _call_gemini(client, model, prompt)

    # Too big — analyze each conversation individually, then synthesize
    total = len(conversations)
    if on_progress:
        on_progress(f"Large batch ({len(content):,} chars) — analyzing {total} conversations individually")

    individual_analyses = []
    for i, (label, turns) in enumerate(conversations, 1):
        if on_progress:
            on_progress(f"Analyzing {i}/{total}: {label}...")
        # Each conversation might itself need chunking
        result = analyze_single(turns, model=model, on_progress=on_progress)
        individual_analyses.append(f"## {label}\n\n{result}")

    # Cross-session synthesis
    if on_progress:
        on_progress(f"Synthesizing cross-session patterns from {total} analyses...")
    combined = "\n\n---\n\n".join(individual_analyses)

    # If combined analyses still too big, summarize in stages
    if len(combined) > CHUNK_THRESHOLD_CHARS:
        # Just use the multi prompt with the analyses (they're already summaries)
        synth_prompt = MULTI_PROMPT.replace("{count}", str(total)).replace("{content}", combined[:CHUNK_THRESHOLD_CHARS])
    else:
        synth_prompt = MULTI_PROMPT.replace("{count}", str(total)).replace("{content}", combined)

    return _call_gemini(client, model, synth_prompt)
