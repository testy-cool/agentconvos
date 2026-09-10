"""Model-backed conversation analysis over any OpenAI-compatible endpoint.

The endpoint, key, and models come from ``llm_config``."""

from __future__ import annotations

from pathlib import Path

from .llm_config import DEFAULT_MODEL, LLMConfig, load_config
from .parser import Turn, to_markdown

MODELS = [
    DEFAULT_MODEL,
    "gpt-5",
]

# ~300K tokens = ~1.2M chars. Conversations above this get chunked.
CHUNK_THRESHOLD_CHARS = 1_200_000
CHUNK_TARGET_CHARS = 1_200_000

# Deep mode: ~100K tokens = ~400K chars per chunk
DEEP_CHUNK_TARGET_CHARS = 400_000

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

DEEP_FIRST_PROMPT = """You are doing an exhaustive, code-level analysis of a Claude Code conversation.
This is CHUNK 1 of {total_chunks}. You are setting the standard — be extremely thorough and detailed.

Include:
- Exact file paths, function names, class names
- Code snippets and command outputs where relevant
- Specific technical decisions with reasoning
- Problems encountered with full error context
- User preferences and corrections (exact quotes)
- Workflow patterns observed

Output as richly structured markdown with code blocks. Do NOT summarize — be exhaustive.

CONVERSATION (chunk 1/{total_chunks}):
{content}"""

DEEP_CONTINUE_PROMPT = """You are continuing an exhaustive, code-level analysis of a Claude Code conversation.
This is CHUNK {chunk_num} of {total_chunks}.

Here is the analysis from the previous chunks — continue at the same level of detail and format:

PREVIOUS ANALYSIS:
{previous}

---

Now analyze this next chunk. Continue where the previous analysis left off. Add new findings, don't repeat what's already covered. Maintain the same structure and depth.

CONVERSATION (chunk {chunk_num}/{total_chunks}):
{content}"""

DEEP_FINAL_PROMPT = """Below is a sequential, exhaustive analysis of a Claude Code conversation done in {total_chunks} passes.

Compile this into one cohesive, detailed document. Preserve all code snippets, file paths, function names, and technical details. Organize chronologically and by topic. Remove only exact duplicates — keep everything else.

SEQUENTIAL ANALYSIS:
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


def llm_available() -> bool:
    return bool(load_config().api_key)


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


# Pricing per million tokens, keyed by model name without the provider prefix
_PRICING = {
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00},
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},
    "gemini-3.1-flash-lite-preview": {"input": 0.25, "output": 1.50},
}


class _CostTracker:
    def __init__(self):
        self.calls = []

    def record(self, model: str, input_tokens: int, output_tokens: int):
        pricing = _PRICING.get(model.rsplit("/", 1)[-1], {"input": 1.0, "output": 5.0})
        cost = input_tokens * pricing["input"] / 1_000_000 + output_tokens * pricing["output"] / 1_000_000
        self.calls.append({"model": model, "input": input_tokens, "output": output_tokens, "cost": cost})

    def summary(self) -> str:
        total_in = sum(c["input"] for c in self.calls)
        total_out = sum(c["output"] for c in self.calls)
        total_cost = sum(c["cost"] for c in self.calls)
        lines = [f"  API calls: {len(self.calls)}"]
        for c in self.calls:
            lines.append(f"    {c['model']:30s}  {c['input']:>8,} in / {c['output']:>6,} out  ${c['cost']:.4f}")
        lines.append(f"  Total: {total_in:,} in / {total_out:,} out — ${total_cost:.4f}")
        return "\n".join(lines)


# Global tracker, reset per analyze call
_tracker = _CostTracker()


def get_cost_summary() -> str:
    return _tracker.summary()


REQUEST_TIMEOUT_SECONDS = 900


def _call_llm(cfg: LLMConfig, model: str, prompt: str, retries: int = 3) -> str:
    """One chat-completions request with retry on transport error or empty reply."""
    import time

    import httpx

    text = ""
    for attempt in range(retries + 1):
        try:
            resp = httpx.post(
                cfg.chat_url,
                headers={"Authorization": f"Bearer {cfg.api_key}", "User-Agent": "agentconvos"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}]},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage") or {}
            if usage:
                _tracker.record(model, usage.get("prompt_tokens") or 0, usage.get("completion_tokens") or 0)
            text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            if text.strip():
                return text
        except Exception as e:
            if attempt < retries:
                wait = 5 * (attempt + 1)
                print(f"  API error: {e} — retrying in {wait}s ({attempt + 1}/{retries})", flush=True)
                time.sleep(wait)
                continue
            raise
        if attempt < retries:
            time.sleep(2)
    return text


def analyze_single(
    turns: list[Turn],
    model: str | None = None,
    prompt_template: str = SINGLE_PROMPT,
    on_progress: callable = None,
) -> str:
    """Analyze a single conversation. Auto-chunks if too large."""
    global _tracker
    _tracker = _CostTracker()
    cfg = load_config()
    model = model or cfg.model
    content = to_markdown(turns)

    # Check if chunking needed
    if len(content) <= CHUNK_THRESHOLD_CHARS:
        prompt = prompt_template.replace("{content}", content)
        return _call_llm(cfg, model, prompt)

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
        result = _call_llm(cfg, model, prompt)
        chunk_analyses.append(f"## Part {i}/{total}\n\n{result}")

    # Synthesis pass
    if on_progress:
        on_progress(f"Synthesizing {total} chunk analyses...")
    combined = "\n\n---\n\n".join(chunk_analyses)
    synth_prompt = SYNTHESIS_PROMPT.replace("{total_chunks}", str(total)).replace("{content}", combined)
    return _call_llm(cfg, model, synth_prompt)


def analyze_deep(
    turns: list[Turn],
    on_progress: callable = None,
    prompt_template: str | None = None,
    out_path: Path | None = None,
) -> str:
    """Deep sequential analysis: pro model for first chunk and synthesis, default model in between."""
    global _tracker
    _tracker = _CostTracker()
    cfg = load_config()
    pro_model, flash_model = cfg.pro_model, cfg.model
    chunks = _chunk_turns(turns, target_chars=DEEP_CHUNK_TARGET_CHARS)
    total = len(chunks)

    def _save_progress(text: str):
        if out_path:
            out_path.write_text(text, encoding="utf-8")

    if total == 1:
        if on_progress:
            on_progress(f"Single chunk — analyzing with {pro_model}")
        content = to_markdown(chunks[0])
        prompt = (prompt_template or DEEP_FIRST_PROMPT).replace("{total_chunks}", "1").replace("{content}", content)
        result = _call_llm(cfg, pro_model, prompt)
        _save_progress(result)
        return result

    if on_progress:
        on_progress(f"Deep mode: {total} chunks (~100K tokens each). {pro_model} for chunk 1, {flash_model} for 2-{total}, {pro_model} for final synthesis.")

    # Chunk 1: Pro sets the tone
    if on_progress:
        on_progress(f"Chunk 1/{total} with {pro_model} (setting the standard)...")
    chunk_md = to_markdown(chunks[0])
    prompt = DEEP_FIRST_PROMPT.replace("{total_chunks}", str(total)).replace("{content}", chunk_md)
    running_analysis = _call_llm(cfg, pro_model, prompt)
    all_analyses = [f"## Chunk 1/{total}\n\n{running_analysis}"]
    _save_progress("\n\n---\n\n".join(all_analyses) + "\n\n---\n\n*Synthesis pending...*")

    # Chunks 2..N: Flash continues with previous analysis as context
    for i, chunk_turns in enumerate(chunks[1:], 2):
        if on_progress:
            on_progress(f"Chunk {i}/{total} with {flash_model} (continuing with prior context)...")
        chunk_md = to_markdown(chunk_turns)
        prev_context = running_analysis
        if len(prev_context) > 200_000:
            prev_context = prev_context[-200_000:]
        prompt = DEEP_CONTINUE_PROMPT.replace("{chunk_num}", str(i)).replace("{total_chunks}", str(total)).replace("{previous}", prev_context).replace("{content}", chunk_md)
        result = _call_llm(cfg, flash_model, prompt)
        running_analysis = result
        all_analyses.append(f"## Chunk {i}/{total}\n\n{result}")
        _save_progress("\n\n---\n\n".join(all_analyses) + "\n\n---\n\n*Synthesis pending...*")

    # Final synthesis with Pro
    if on_progress:
        on_progress(f"Final synthesis with {pro_model}...")
    combined = "\n\n---\n\n".join(all_analyses)
    # If combined is too big for one pass, just use the last running analysis
    if len(combined) > CHUNK_THRESHOLD_CHARS:
        combined = combined[:CHUNK_THRESHOLD_CHARS]
    synth_prompt = DEEP_FINAL_PROMPT.replace("{total_chunks}", str(total)).replace("{content}", combined)
    return _call_llm(cfg, pro_model, synth_prompt)


def analyze_multi(
    conversations: list[tuple[str, list[Turn]]],
    model: str | None = None,
    prompt_template: str = MULTI_PROMPT,
    on_progress: callable = None,
) -> str:
    """Analyze multiple conversations. Each tuple is (label, turns)."""
    global _tracker
    _tracker = _CostTracker()
    cfg = load_config()
    model = model or cfg.model

    parts = []
    for label, turns in conversations:
        md = to_markdown(turns)
        parts.append(f"### {label}\n{md}")

    content = "\n---\n".join(parts)

    # Check if chunking needed
    if len(content) <= CHUNK_THRESHOLD_CHARS:
        prompt = prompt_template.replace("{count}", str(len(conversations))).replace("{content}", content)
        return _call_llm(cfg, model, prompt)

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

    return _call_llm(cfg, model, synth_prompt)
