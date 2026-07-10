"""Per-chat tunable RAG settings. Defaults come from global config; a user overrides a
subset per chat. One contract shared by the API (validation) and the chat service."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field

from app.config import get_settings


class ChatSettings(BaseModel):
    """Resolved settings (concrete values, validated ranges)."""

    top_k: int = Field(ge=1, le=20)
    min_score: float = Field(ge=0.0, le=1.0)
    history_turns: int = Field(ge=0, le=50)
    context_token_budget: int = Field(ge=1000, le=200_000)
    temperature: float = Field(ge=0.0, le=2.0)
    model: str  # chat model for this chat (validated against config.available_chat_models at use)
    # Ingest-time knobs applied to NEW uploads made from this chat (existing docs keep their chunking).
    chunk_size: int = Field(ge=200, le=4000)
    chunk_overlap: int = Field(ge=0, le=1000)
    document_ids: list[UUID] | None = None


class ChatSettingsPatch(BaseModel):
    """Partial update from the API; unset fields keep their current value."""

    top_k: int | None = Field(default=None, ge=1, le=20)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    history_turns: int | None = Field(default=None, ge=0, le=50)
    context_token_budget: int | None = Field(default=None, ge=1000, le=200_000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    model: str | None = None
    chunk_size: int | None = Field(default=None, ge=200, le=4000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=1000)
    document_ids: list[UUID] | None = None


def available_models() -> list[str]:
    """Selectable chat models (single source: config)."""
    return [m.strip() for m in get_settings().available_chat_models.split(",") if m.strip()]


def resolve_settings(stored: dict | None) -> ChatSettings:
    s = get_settings()
    values: dict = {
        "top_k": s.retrieval_top_k,
        "min_score": s.retrieval_min_score,
        "history_turns": s.history_turns,
        "context_token_budget": s.context_token_budget,
        "temperature": s.llm_temperature,
        "model": s.chat_model,
        "chunk_size": s.chunk_size,
        "chunk_overlap": s.chunk_overlap,
        "document_ids": None,
    }
    values.update({k: v for k, v in (stored or {}).items() if v is not None})
    # A stored model no longer offered falls back to the default (never send an invalid model).
    if values["model"] not in available_models():
        values["model"] = s.chat_model
    return ChatSettings(**values)
