from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
ALLOWED_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
ALLOWED_VOICE_RUNTIME_MODES = {"cascade", "gemini_live"}
ALLOWED_TTS_PROVIDERS = {"deepgram", "cartesia"}
ALLOWED_EMBEDDING_PROVIDERS = {"local", "remote"}
ALLOWED_CHROMA_PROVIDERS = {"local", "remote"}
ALLOWED_REDIS_SCHEMES = ("redis://", "rediss://")


class AppSettings(BaseSettings):
    """Central application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    livekit_url: str = Field(..., alias="LIVEKIT_URL")
    livekit_api_key: str = Field(..., alias="LIVEKIT_API_KEY")
    livekit_api_secret: str = Field(..., alias="LIVEKIT_API_SECRET")
    twilio_account_sid: str = Field(..., alias="TWILIO_ACCOUNT_SID")
    twilio_auth_token: str = Field(..., alias="TWILIO_AUTH_TOKEN")
    twilio_phone_number: str | None = Field(default=None, alias="TWILIO_PHONE_NUMBER")
    twilio_sip_domain: str | None = Field(default=None, alias="TWILIO_SIP_DOMAIN")
    sip_inbound_trunk_name: str = Field(
        default="policy-support-inbound",
        alias="SIP_INBOUND_TRUNK_NAME",
    )
    sip_dispatch_rule_name: str = Field(
        default="policy-support-dispatch",
        alias="SIP_DISPATCH_RULE_NAME",
    )
    sip_room_prefix: str = Field(default="call-", alias="SIP_ROOM_PREFIX")
    deepgram_api_key: str = Field(..., alias="DEEPGRAM_API_KEY")
    cartesia_api_key: str | None = Field(default=None, alias="CARTESIA_API_KEY")
    tts_provider: str = Field(default="deepgram", alias="TTS_PROVIDER")
    chroma_provider: str = Field(default="local", alias="CHROMA_PROVIDER")
    chroma_host: str = Field(default="api.trychroma.com", alias="CHROMA_HOST")
    chroma_api_key: str | None = Field(default=None, alias="CHROMA_API_KEY")
    chroma_tenant: str | None = Field(default=None, alias="CHROMA_TENANT")
    chroma_database: str | None = Field(default=None, alias="CHROMA_DATABASE")
    jina_api_key: str | None = Field(default=None, alias="JINA_API_KEY")
    embedding_provider: str = Field(default="local", alias="EMBEDDING_PROVIDER")
    jina_embedding_model: str = Field(
        default="jina-embeddings-v5-text-small",
        alias="JINA_EMBEDDING_MODEL",
    )
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str | None = Field(default=None, alias="OPENAI_MODEL")
    fallback_openai_base_url: str | None = Field(
        default=None,
        alias="FALLBACK_OPENAI_BASE_URL",
    )
    fallback_openai_api_key: str | None = Field(
        default=None,
        alias="FALLBACK_OPENAI_API_KEY",
    )
    fallback_openai_model: str | None = Field(
        default=None,
        alias="FALLBACK_OPENAI_MODEL",
    )
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_llm_model_name: str | None = Field(
        default=None, alias="GEMINI_LLM_MODEL_NAME"
    )
    gemini_audio_model_name: str | None = Field(
        default=None, alias="GEMINI_AUDIO_MODEL_NAME"
    )
    gemini_embedding_model_name: str | None = Field(
        default=None, alias="GEMINI_EMBEDDING_MODEL_NAME"
    )
    chroma_persist_dir: Path = Field(
        default=Path("./data/chroma"), alias="CHROMA_PERSIST_DIR"
    )
    turn_metrics_path: Path = Field(
        default=Path("./data/turn_metrics.jsonl"),
        alias="TURN_METRICS_PATH",
    )
    active_sessions_path: Path = Field(
        default=Path("./data/active_sessions.json"),
        alias="ACTIVE_SESSIONS_PATH",
    )
    redis_url: str | None = Field(default=None, alias="REDIS_URL")
    active_sessions_key: str = Field(
        default="voice-agent:active-sessions",
        alias="ACTIVE_SESSIONS_KEY",
    )
    kb_collection_name: str = Field(
        default="test-collection",
        validation_alias=AliasChoices("CHROMA_COLLECTION_NAME", "KB_COLLECTION_NAME"),
    )
    kb_source_pdf_url: str = Field(
        default="https://www.rbcinsurance.com/samplepolicy/pdf/your-term-sample-policy.pdf",
        alias="KB_SOURCE_PDF_URL",
    )
    agent_name: str = Field(default="policy-support-agent", alias="AGENT_NAME")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8080, alias="API_PORT", ge=1, le=65535)
    worker_port: int = Field(default=8081, alias="WORKER_PORT", ge=0, le=65535)
    worker_max_concurrent_sessions: int = Field(
        default=1,
        alias="WORKER_MAX_CONCURRENT_SESSIONS",
        ge=1,
        le=1000,
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    voice_runtime_mode: str = Field(
        default="cascade",
        alias="VOICE_RUNTIME_MODE",
    )
    http_timeout_seconds: float = Field(
        default=15.0, alias="HTTP_TIMEOUT_SECONDS", gt=0
    )
    agent_max_retries: int = Field(default=0, alias="AGENT_MAX_RETRIES", ge=0, le=5)
    kb_embedding_weight: float = Field(
        default=0.7,
        alias="KB_EMBEDDING_WEIGHT",
        ge=0.0,
        le=1.0,
    )
    kb_lexical_weight: float = Field(
        default=0.3,
        alias="KB_LEXICAL_WEIGHT",
        ge=0.0,
        le=1.0,
    )
    kb_min_combined_score: float = Field(
        default=0.45,
        alias="KB_MIN_COMBINED_SCORE",
        ge=0.0,
        le=1.0,
    )
    kb_min_vector_score: float = Field(
        default=0.5,
        alias="KB_MIN_VECTOR_SCORE",
        ge=0.0,
        le=1.0,
    )
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    gemini_live_model: str | None = Field(default=None, alias="GEMINI_LIVE_MODEL")
    gemini_live_voice: str = Field(default="Puck", alias="GEMINI_LIVE_VOICE")
    gemini_live_include_thoughts: bool = Field(
        default=False,
        alias="GEMINI_LIVE_INCLUDE_THOUGHTS",
    )
    gemini_live_modalities: str = Field(
        default="AUDIO",
        alias="GEMINI_LIVE_MODALITIES",
    )
    load_test_concurrency: int = Field(
        default=10, alias="LOAD_TEST_CONCURRENCY", ge=1, le=100
    )
    load_test_turns: int = Field(default=2, alias="LOAD_TEST_TURNS", ge=1, le=20)

    @field_validator("livekit_url", "openai_base_url")
    @classmethod
    def validate_urls(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value.startswith(("http://", "https://", "ws://", "wss://")):
            raise ValueError("must be an absolute URL")
        return value.rstrip("/")

    @field_validator("redis_url")
    @classmethod
    def validate_redis_url(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value.startswith(ALLOWED_REDIS_SCHEMES):
            raise ValueError("REDIS_URL must start with redis:// or rediss://")
        return value.rstrip("/")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in ALLOWED_LOG_LEVELS:
            allowed = ", ".join(sorted(ALLOWED_LOG_LEVELS))
            raise ValueError(f"log level must be one of: {allowed}")
        return normalized

    @field_validator("voice_runtime_mode")
    @classmethod
    def validate_voice_runtime_mode(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in ALLOWED_VOICE_RUNTIME_MODES:
            allowed = ", ".join(sorted(ALLOWED_VOICE_RUNTIME_MODES))
            raise ValueError(f"VOICE_RUNTIME_MODE must be one of: {allowed}")
        return normalized

    @field_validator("tts_provider")
    @classmethod
    def validate_tts_provider(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in ALLOWED_TTS_PROVIDERS:
            allowed = ", ".join(sorted(ALLOWED_TTS_PROVIDERS))
            raise ValueError(f"TTS_PROVIDER must be one of: {allowed}")
        return normalized

    @field_validator("embedding_provider")
    @classmethod
    def validate_embedding_provider(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in ALLOWED_EMBEDDING_PROVIDERS:
            allowed = ", ".join(sorted(ALLOWED_EMBEDDING_PROVIDERS))
            raise ValueError(f"EMBEDDING_PROVIDER must be one of: {allowed}")
        return normalized

    @field_validator("chroma_provider")
    @classmethod
    def validate_chroma_provider(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in ALLOWED_CHROMA_PROVIDERS:
            allowed = ", ".join(sorted(ALLOWED_CHROMA_PROVIDERS))
            raise ValueError(f"CHROMA_PROVIDER must be one of: {allowed}")
        return normalized

    @model_validator(mode="after")
    def apply_llm_fallbacks(self) -> "AppSettings":
        total_weight = self.kb_embedding_weight + self.kb_lexical_weight
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(
                "KB_EMBEDDING_WEIGHT and KB_LEXICAL_WEIGHT must sum to 1.0"
            )
        if self.embedding_provider == "remote" and not self.jina_api_key:
            raise ValueError(
                "JINA_API_KEY is required when EMBEDDING_PROVIDER=remote"
            )
        if self.chroma_provider == "remote":
            missing_chroma = []
            if not self.chroma_api_key:
                missing_chroma.append("CHROMA_API_KEY")
            if not self.chroma_tenant:
                missing_chroma.append("CHROMA_TENANT")
            if not self.chroma_database:
                missing_chroma.append("CHROMA_DATABASE")
            if missing_chroma:
                raise ValueError(
                    "Missing remote Chroma configuration: " + ", ".join(missing_chroma)
                )
        self.google_api_key = self.google_api_key or self.gemini_api_key
        self.gemini_live_model = self.gemini_live_model or self.gemini_audio_model_name
        self.openai_api_key = self.openai_api_key or self.gemini_api_key
        self.openai_model = self.openai_model or self.gemini_llm_model_name
        if not self.openai_base_url and self.gemini_api_key:
            self.openai_base_url = GEMINI_OPENAI_BASE_URL

        fallback_values = (
            self.fallback_openai_base_url,
            self.fallback_openai_api_key,
            self.fallback_openai_model,
        )
        if any(value is not None for value in fallback_values) and not all(fallback_values):
            raise ValueError(
                "Set FALLBACK_OPENAI_BASE_URL, FALLBACK_OPENAI_API_KEY, and "
                "FALLBACK_OPENAI_MODEL together to enable LLM failover"
            )

        missing = []
        if not self.openai_base_url:
            missing.append("OPENAI_BASE_URL")
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if not self.openai_model:
            missing.append("OPENAI_MODEL")
        if missing:
            raise ValueError(
                "Missing LLM configuration. Set OPENAI_* or compatible GEMINI_* vars: "
                + ", ".join(missing)
            )
        return self


def load_settings() -> AppSettings:
    """Load settings from the current environment."""

    return AppSettings()


def validate_settings() -> tuple[bool, str]:
    """Validate configuration without raising framework-specific errors upstream."""

    try:
        load_settings()
    except ValidationError as exc:
        return False, str(exc)
    return True, "ok"
