from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Env-driven config. The only place os.environ is read."""

    model_config = SettingsConfigDict(env_file=(".env", "../.env"), extra="ignore")

    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # OpenAI-compatible provider. Defaults to Gemini; swap base_url + models + key for any other.
    gemini_api_key: str | None = None
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    chat_model: str = "gemini-3.1-flash-lite"
    # User-selectable chat models (per-chat dropdown). Same provider/key — swap the model only.
    # Full BYOK (arbitrary model + key + base_url) is intentionally out of scope; change the list
    # or the provider via env when needed.
    available_chat_models: str = "gemini-3.5-flash,gemini-3.1-flash-lite,gemini-2.5-flash"
    embed_model: str = "gemini-embedding-2"
    embed_dim: int = 768  # requested; real collection size is auto-detected at startup
    llm_timeout_s: int = 60
    llm_max_retries: int = 4  # SDK retries 429/5xx respecting Retry-After (rate-limit resilience)
    llm_breaker_threshold: int = 8  # consecutive provider failures before the breaker opens
    llm_breaker_cooldown_s: int = 30  # fail-fast window while open (avoids retry storms)
    llm_temperature: float = 0.0

    # Ingestion-time defaults (per-chat + per-upload overridable). 768 chars ~= 190 tokens —
    # tighter chunks give more precise retrieval/citations; ~17% overlap keeps cross-boundary context.
    chunk_size: int = 768
    chunk_overlap: int = 128
    embed_concurrency: int = 8

    # Query-time defaults (per-chat overridable via ChatSettings).
    retrieval_top_k: int = 5
    hybrid_prefetch: int = 20  # candidates per branch (dense + BM25) before RRF fusion
    bm25_model: str = "Qdrant/bm25"
    # Coarse floor on best-hit cosine similarity. NOT a universal constant — it depends on
    # the embedding model/corpus, so it is user-tunable per chat and the prompt is the
    # primary grounding enforcer. This floor only rejects obvious total misses.
    retrieval_min_score: float = 0.3
    # Online eval: an LLM-as-Judge scores each generated answer for groundedness/source/
    # correctness and returns the verdict in the same request (a trailing SSE event). Costs
    # one extra LLM call per answered query — disable to save cost.
    judge_enabled: bool = True
    history_turns: int = 6  # recent turns kept verbatim (coreference window)
    context_token_budget: int = 8000  # assembled-prompt cap; older turns get summarized past this
    chars_per_token: int = 4  # English pre-send token estimate (OpenAI rule of thumb)

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "documents"

    redis_url: str = "redis://localhost:6379/0"

    ingest_executor: Literal["inline", "queue"] = "inline"  # queue = offload to arq worker
    worker_max_jobs: int = 10
    ingest_timeout_s: int = 600
    ingest_max_tries: int = 3
    max_queue_depth: int = 0  # load-shed uploads with 503 past this backlog (0 = disabled)

    max_question_chars: int = 4000  # reject oversized prompts (DoS guard)

    database_url: str = "sqlite+aiosqlite:///./dev.db"
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_timeout: int = 30
    db_pool_recycle: int = 1800
    db_pool_pre_ping: bool = True

    jwt_secret: str = "CHANGE_ME_IN_PROD"  # startup refuses this default in prod
    jwt_lifetime_seconds: int = 60 * 60 * 24 * 7

    max_upload_files: int = 10
    max_upload_mb: int = 20
    max_json_body_kb: int = 256  # non-multipart body cap (guards JSON endpoints from OOM)

    # Security
    min_password_length: int = 8
    auth_max_per_min: int = 10  # per-IP throttle on login/register (credential-stuffing guard)

    cors_origins: str = "http://localhost:5173,http://localhost,http://127.0.0.1:5173"

    @field_validator("database_url")
    @classmethod
    def _async_db_url(cls, v: str) -> str:
        """PaaS (Railway/Heroku/Supabase) inject `postgres://` / `postgresql://` URLs, but the
        async engine needs the asyncpg driver. Normalize so a platform DATABASE_URL works as-is."""
        if v.startswith("postgres://"):
            v = "postgresql://" + v[len("postgres://") :]
        if v.startswith("postgresql://"):
            v = "postgresql+asyncpg://" + v[len("postgresql://") :]
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
