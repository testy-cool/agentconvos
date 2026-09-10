"""Where model-backed commands send their requests.

Agentconvos works with OpenAI by default and with any service that implements
the OpenAI chat-completions API. Standard ``OPENAI_*`` variables provide the
shortest setup. ``AGENTCONVOS_LLM_*`` variables override them for a dedicated
endpoint or model. The same names work in a project ``.env`` or the user config
file at ``~/.config/agentconvos/.env``.

    AGENTCONVOS_LLM_BASE_URL   OpenAI-compatible base URL
    AGENTCONVOS_LLM_API_KEY    bearer token for that endpoint
    AGENTCONVOS_LLM_KEY_NAME   optional key name stored by the llm CLI
    AGENTCONVOS_LLM_MODEL      model for analyze, summarize, and deep-mode chunks
    AGENTCONVOS_LLM_PRO_MODEL  stronger model for deep-mode first chunk and synthesis
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_PRO_MODEL = DEFAULT_MODEL

ENV_BASE_URL = "AGENTCONVOS_LLM_BASE_URL"
ENV_API_KEY = "AGENTCONVOS_LLM_API_KEY"
ENV_MODEL = "AGENTCONVOS_LLM_MODEL"
ENV_PRO_MODEL = "AGENTCONVOS_LLM_PRO_MODEL"
ENV_KEY_NAME = "AGENTCONVOS_LLM_KEY_NAME"
_ENV_ALIASES = {
    ENV_BASE_URL: (ENV_BASE_URL, "OPENAI_BASE_URL"),
    ENV_API_KEY: (ENV_API_KEY, "OPENAI_API_KEY"),
    ENV_MODEL: (ENV_MODEL, "OPENAI_MODEL"),
    ENV_PRO_MODEL: (ENV_PRO_MODEL,),
    ENV_KEY_NAME: (ENV_KEY_NAME,),
}

_LLM_KEYS = Path.home() / ".config" / "io.datasette.llm" / "keys.json"


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    pro_model: str
    key_source: str

    @property
    def chat_url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    def masked_key(self) -> str:
        return "configured" if self.api_key else ""

    def describe(self) -> str:
        return "\n".join(
            [
                f"  endpoint:  {self.chat_url}",
                f"  model:     {self.model}",
                f"  pro model: {self.pro_model}",
                f"  api key:   {self.masked_key() or '(none)'}  [{self.key_source}]",
            ]
        )


def env_file_candidates() -> list[Path]:
    home = Path.home()
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", home / ".config"))
    return [
        Path(".env"),
        config_home / "agentconvos" / ".env",
        home / ".claude" / "convo-explorer" / ".env",
    ]


def _read_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip("'\"")
        if key and val:
            out[key] = val
    return out


def _from_env_files() -> tuple[dict[str, str], dict[str, str]]:
    """Return normalized settings from env files, with the first file winning."""
    values: dict[str, str] = {}
    sources: dict[str, str] = {}
    for path in env_file_candidates():
        if not path.is_file():
            continue
        file_values = _read_env_file(path)
        for setting, aliases in _ENV_ALIASES.items():
            if setting in values:
                continue
            for alias in aliases:
                if file_values.get(alias):
                    values[setting] = file_values[alias]
                    sources[setting] = str(path)
                    break
    return values, sources


def _named_llm_key(name: str) -> str:
    try:
        keys = json.loads(_LLM_KEYS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    return str(keys.get(name) or "")


def load_config() -> LLMConfig:
    """Resolve settings from environment, env files, then public defaults."""
    file_values, file_sources = _from_env_files()

    def pick(name: str, default: str) -> tuple[str, str]:
        for alias in _ENV_ALIASES[name]:
            if os.environ.get(alias):
                return os.environ[alias], f"${alias}"
        if file_values.get(name):
            return file_values[name], file_sources[name]
        return default, "default"

    base_url, _ = pick(ENV_BASE_URL, DEFAULT_BASE_URL)
    model, _ = pick(ENV_MODEL, DEFAULT_MODEL)
    pro_model, _ = pick(ENV_PRO_MODEL, "")
    pro_model = pro_model or model

    api_key, key_source = pick(ENV_API_KEY, "")
    key_name, _ = pick(ENV_KEY_NAME, "")
    if not api_key and key_name:
        api_key = _named_llm_key(key_name)
        key_source = f"{_LLM_KEYS} [{key_name}]" if api_key else "not found"
    if not api_key:
        key_source = "not found"

    return LLMConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        pro_model=pro_model,
        key_source=key_source,
    )


def missing_key_message() -> str:
    return (
        "No LLM API key configured. Set OPENAI_API_KEY for the default OpenAI endpoint. "
        "For another OpenAI-compatible service, set AGENTCONVOS_LLM_API_KEY, "
        "AGENTCONVOS_LLM_BASE_URL, and AGENTCONVOS_LLM_MODEL in the environment, "
        "./.env, or ~/.config/agentconvos/.env. Existing llm CLI users may set "
        "AGENTCONVOS_LLM_KEY_NAME instead of copying a stored key."
    )
