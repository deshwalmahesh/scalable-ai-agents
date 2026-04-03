from __future__ import annotations

import pytest
from pydantic import ValidationError

from voice_agent.settings import AppSettings, GEMINI_OPENAI_BASE_URL, validate_settings


def build_settings(**overrides: object) -> AppSettings:
    values = {
        "LIVEKIT_URL": "wss://example.livekit.cloud",
        "LIVEKIT_API_KEY": "livekit-key",
        "LIVEKIT_API_SECRET": "livekit-secret",
        "TWILIO_ACCOUNT_SID": "AC123",
        "TWILIO_AUTH_TOKEN": "twilio-token",
        "DEEPGRAM_API_KEY": "deepgram-key",
        "OPENAI_BASE_URL": "https://api.example.com/v1",
        "OPENAI_API_KEY": "openai-key",
        "OPENAI_MODEL": "meta-llama",
        "CHROMA_COLLECTION_NAME": "test-collection",
    }
    values.update(overrides)
    return AppSettings(**values)


def test_settings_accept_valid_values() -> None:
    settings = build_settings(LOG_LEVEL="debug", API_PORT=9090, VOICE_RUNTIME_MODE="gemini_live")

    assert settings.log_level == "DEBUG"
    assert settings.api_port == 9090
    assert settings.voice_runtime_mode == "gemini_live"
    assert settings.embedding_provider == "local"
    assert settings.chroma_provider == "local"
    assert settings.chroma_persist_dir.as_posix().endswith("data/chroma")


def test_settings_reject_invalid_urls() -> None:
    with pytest.raises(ValidationError):
        build_settings(OPENAI_BASE_URL="api.example.com")


def test_settings_reject_invalid_log_level() -> None:
    with pytest.raises(ValidationError):
        build_settings(LOG_LEVEL="verbose")


def test_settings_reject_invalid_runtime_mode() -> None:
    with pytest.raises(ValidationError):
        build_settings(VOICE_RUNTIME_MODE="queue_everything")


def test_settings_reject_invalid_embedding_provider() -> None:
    with pytest.raises(ValidationError):
        build_settings(EMBEDDING_PROVIDER="hash")


def test_settings_require_jina_key_for_remote_embeddings() -> None:
    with pytest.raises(ValidationError):
        build_settings(EMBEDDING_PROVIDER="remote", JINA_API_KEY=None)


def test_settings_reject_invalid_chroma_provider() -> None:
    with pytest.raises(ValidationError):
        build_settings(CHROMA_PROVIDER="cluster")


def test_settings_require_remote_chroma_credentials() -> None:
    with pytest.raises(ValidationError):
        build_settings(
            CHROMA_PROVIDER="remote",
            CHROMA_API_KEY=None,
            CHROMA_TENANT=None,
            CHROMA_DATABASE=None,
        )


def test_settings_accept_gemini_openai_compat_fallback() -> None:
    settings = build_settings(
        OPENAI_BASE_URL=None,
        OPENAI_API_KEY=None,
        OPENAI_MODEL=None,
        GEMINI_API_KEY="gemini-key",
        GEMINI_LLM_MODEL_NAME="gemini-2.5-flash",
    )

    assert settings.openai_api_key == "gemini-key"
    assert settings.openai_model == "gemini-2.5-flash"
    assert settings.openai_base_url == GEMINI_OPENAI_BASE_URL


def test_settings_require_llm_configuration_without_env_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keys = [
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "DEEPGRAM_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "GEMINI_API_KEY",
        "GEMINI_LLM_MODEL_NAME",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(ValidationError) as exc:
        AppSettings(_env_file=None)

    assert "LIVEKIT_URL" in str(exc.value)
