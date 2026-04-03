from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
ALLOWED_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


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
    deepgram_api_key: str = Field(..., alias="DEEPGRAM_API_KEY")
    chroma_host: str = Field(default="api.trychroma.com", alias="CHROMA_HOST")
    chroma_api_key: str | None = Field(default=None, alias="CHROMA_API_KEY")
    chroma_tenant: str | None = Field(default=None, alias="CHROMA_TENANT")
    chroma_database: str | None = Field(default=None, alias="CHROMA_DATABASE")
    jina_api_key: str | None = Field(default=None, alias="JINA_API_KEY")
    jina_embedding_model: str = Field(
        default="jina-embeddings-v5-text-small",
        alias="JINA_EMBEDDING_MODEL",
    )
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str | None = Field(default=None, alias="OPENAI_MODEL")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_llm_model_name: str | None = Field(
        default=None, alias="GEMINI_LLM_MODEL_NAME"
    )
    chroma_persist_dir: Path = Field(
        default=Path("./data/chroma"), alias="CHROMA_PERSIST_DIR"
    )
    turn_metrics_path: Path = Field(
        default=Path("./data/turn_metrics.jsonl"),
        alias="TURN_METRICS_PATH",
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
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    http_timeout_seconds: float = Field(
        default=15.0, alias="HTTP_TIMEOUT_SECONDS", gt=0
    )
    agent_max_retries: int = Field(default=2, alias="AGENT_MAX_RETRIES", ge=0, le=5)
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

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in ALLOWED_LOG_LEVELS:
            allowed = ", ".join(sorted(ALLOWED_LOG_LEVELS))
            raise ValueError(f"log level must be one of: {allowed}")
        return normalized

    @model_validator(mode="after")
    def apply_llm_fallbacks(self) -> "AppSettings":
        total_weight = self.kb_embedding_weight + self.kb_lexical_weight
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(
                "KB_EMBEDDING_WEIGHT and KB_LEXICAL_WEIGHT must sum to 1.0"
            )
        self.openai_api_key = self.openai_api_key or self.gemini_api_key
        self.openai_model = self.openai_model or self.gemini_llm_model_name
        if not self.openai_base_url and self.gemini_api_key:
            self.openai_base_url = GEMINI_OPENAI_BASE_URL

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
