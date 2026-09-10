import json
from unittest.mock import patch

import pytest

from agentconvos import analyzer, llm_config
from agentconvos.app import ConvoExplorer
from agentconvos.llm_config import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_PRO_MODEL,
    LLMConfig,
    load_config,
)

_ALL_VARS = (
    "AGENTCONVOS_LLM_BASE_URL",
    "AGENTCONVOS_LLM_API_KEY",
    "AGENTCONVOS_LLM_MODEL",
    "AGENTCONVOS_LLM_PRO_MODEL",
    "AGENTCONVOS_LLM_KEY_NAME",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
)


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for name in _ALL_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(llm_config, "_LLM_KEYS", tmp_path / "keys.json", raising=False)
    monkeypatch.setattr(
        llm_config, "env_file_candidates", lambda: [tmp_path / ".env", tmp_path / "home.env"]
    )
    return tmp_path


def test_defaults_are_public_and_report_missing_key(clean_env):
    cfg = load_config()
    assert DEFAULT_BASE_URL == "https://api.openai.com/v1"
    assert DEFAULT_MODEL == "gpt-5-mini"
    assert DEFAULT_PRO_MODEL == DEFAULT_MODEL
    assert cfg.base_url == DEFAULT_BASE_URL
    assert cfg.chat_url == DEFAULT_BASE_URL + "/chat/completions"
    assert cfg.model == DEFAULT_MODEL
    assert cfg.pro_model == DEFAULT_PRO_MODEL
    assert cfg.api_key == ""
    assert cfg.key_source == "not found"
    assert not analyzer.llm_available()


def test_openai_api_key_is_the_zero_config_fallback(clean_env, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    cfg = load_config()
    assert cfg.api_key == "sk-openai"
    assert cfg.key_source == "$OPENAI_API_KEY"
    assert cfg.base_url == DEFAULT_BASE_URL
    assert cfg.model == DEFAULT_MODEL
    assert analyzer.llm_available()


def test_named_llm_key_is_an_optional_secret_store_fallback(clean_env, monkeypatch):
    (clean_env / "keys.json").write_text(json.dumps({"work-gateway": "stored-key"}))
    monkeypatch.setenv("AGENTCONVOS_LLM_KEY_NAME", "work-gateway")
    cfg = load_config()
    assert cfg.api_key == "stored-key"
    assert cfg.key_source.endswith("keys.json [work-gateway]")


def test_project_env_file_overrides_user_file_and_accepts_openai_names(clean_env):
    (clean_env / "home.env").write_text(
        "AGENTCONVOS_LLM_API_KEY=from-home\nAGENTCONVOS_LLM_MODEL=home-model\n"
    )
    (clean_env / ".env").write_text(
        "# comment\nOPENAI_API_KEY='from-cwd'\nOPENAI_MODEL=project-model\n"
    )
    cfg = load_config()
    assert cfg.api_key == "from-cwd"
    assert cfg.key_source.endswith(".env")
    assert cfg.model == "project-model"
    assert cfg.pro_model == "project-model"


def test_environment_overrides_everything(clean_env, monkeypatch):
    (clean_env / ".env").write_text("AGENTCONVOS_LLM_API_KEY=from-file\n")
    monkeypatch.setenv("OPENAI_API_KEY", "from-openai-env")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://openai-env/v1")
    monkeypatch.setenv("OPENAI_MODEL", "openai-model")
    monkeypatch.setenv("AGENTCONVOS_LLM_API_KEY", "from-env")
    monkeypatch.setenv("AGENTCONVOS_LLM_BASE_URL", "http://localhost:4000/v1/")
    monkeypatch.setenv("AGENTCONVOS_LLM_MODEL", "main-model")
    monkeypatch.setenv("AGENTCONVOS_LLM_PRO_MODEL", "big-model")
    cfg = load_config()
    assert cfg.api_key == "from-env"
    assert cfg.key_source == "$AGENTCONVOS_LLM_API_KEY"
    assert cfg.chat_url == "http://localhost:4000/v1/chat/completions"
    assert cfg.model == "main-model"
    assert cfg.pro_model == "big-model"


def test_openai_compatible_environment_can_replace_endpoint_and_model(clean_env, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "provider-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.example/v1/")
    monkeypatch.setenv("OPENAI_MODEL", "provider/model")
    cfg = load_config()
    assert cfg.chat_url == "https://provider.example/v1/chat/completions"
    assert cfg.api_key == "provider-key"
    assert cfg.model == "provider/model"
    assert cfg.pro_model == "provider/model"


def test_tui_starts_with_the_configured_model(clean_env, monkeypatch):
    monkeypatch.setenv("AGENTCONVOS_LLM_MODEL", "provider/model")
    app = ConvoExplorer()
    assert app.gemini_model == "provider/model"


def test_custom_endpoint_tui_does_not_offer_openai_models(clean_env, monkeypatch):
    monkeypatch.setenv("AGENTCONVOS_LLM_BASE_URL", "https://provider.example/v1")
    monkeypatch.setenv("AGENTCONVOS_LLM_MODEL", "provider/model")
    app = ConvoExplorer()
    assert app._analysis_models == ["provider/model"]


def test_user_config_uses_xdg_path_and_keeps_legacy_path(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(llm_config.Path, "home", lambda: tmp_path / "home")
    assert llm_config.env_file_candidates() == [
        llm_config.Path(".env"),
        tmp_path / "xdg" / "agentconvos" / ".env",
        tmp_path / "home" / ".claude" / "convo-explorer" / ".env",
    ]


def test_missing_key_message_is_provider_neutral():
    message = llm_config.missing_key_message()
    assert "OPENAI_API_KEY" in message
    assert "AGENTCONVOS_LLM_API_KEY" in message
    assert "AGENTCONVOS_LLM_BASE_URL" in message
    assert "bifrost" not in message.casefold()


def test_describe_masks_the_key(clean_env, monkeypatch):
    monkeypatch.setenv("AGENTCONVOS_LLM_API_KEY", "sk-provider-0123456789abcdef")
    text = load_config().describe()
    assert "sk-provider-0123456789abcdef" not in text
    assert "sk-provider" not in text
    assert "api key:   configured" in text
    assert DEFAULT_MODEL in text


class _Response:
    def __init__(self, content, usage=None, status=200):
        self._content = content
        self._usage = usage
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return {
            "choices": [{"message": {"content": self._content}}],
            "usage": self._usage or {},
        }


def _cfg(**overrides):
    base = dict(
        base_url="http://gw/v1", api_key="k", model="vertex/m", pro_model="vertex/p", key_source="test"
    )
    base.update(overrides)
    return LLMConfig(**base)


def test_call_llm_sends_openai_chat_request_and_records_usage():
    analyzer._tracker = analyzer._CostTracker()
    with patch("httpx.post", return_value=_Response("hello", {"prompt_tokens": 10, "completion_tokens": 4})) as post:
        out = analyzer._call_llm(_cfg(), "vertex/gemini-3-flash-preview", "hi")
    assert out == "hello"
    kwargs = post.call_args.kwargs
    assert post.call_args.args[0] == "http://gw/v1/chat/completions"
    assert kwargs["headers"]["Authorization"] == "Bearer k"
    assert kwargs["json"] == {
        "model": "vertex/gemini-3-flash-preview",
        "messages": [{"role": "user", "content": "hi"}],
    }
    assert "max_tokens" not in kwargs["json"]
    call = analyzer._tracker.calls[0]
    assert (call["input"], call["output"]) == (10, 4)
    assert call["cost"] == pytest.approx(10 * 0.5 / 1e6 + 4 * 3.0 / 1e6)


def test_call_llm_retries_an_empty_reply_once_then_returns_text():
    with (
        patch("httpx.post", side_effect=[_Response(""), _Response("second")]) as post,
        patch("time.sleep"),
    ):
        assert analyzer._call_llm(_cfg(), "vertex/m", "hi") == "second"
    assert post.call_count == 2


def test_analyze_single_uses_configured_model_when_none_given(clean_env, monkeypatch):
    monkeypatch.setenv("AGENTCONVOS_LLM_API_KEY", "k")
    monkeypatch.setenv("AGENTCONVOS_LLM_MODEL", "custom/model")
    turns = [analyzer.Turn("user", "hello"), analyzer.Turn("assistant", "hi")]
    with patch("httpx.post", return_value=_Response("analysis")) as post:
        assert analyzer.analyze_single(turns) == "analysis"
    assert post.call_args.kwargs["json"]["model"] == "custom/model"


def test_deep_mode_uses_pro_model_for_single_chunk(clean_env, monkeypatch):
    monkeypatch.setenv("AGENTCONVOS_LLM_API_KEY", "k")
    monkeypatch.setenv("AGENTCONVOS_LLM_PRO_MODEL", "custom/pro")
    turns = [analyzer.Turn("user", "hello")]
    with patch("httpx.post", return_value=_Response("deep")) as post:
        assert analyzer.analyze_deep(turns) == "deep"
    assert post.call_args.kwargs["json"]["model"] == "custom/pro"
