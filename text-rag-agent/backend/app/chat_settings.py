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
    document_ids: list[UUID] | None = None


class ChatSettingsPatch(BaseModel):
    """Partial update from the API; unset fields keep their current value."""

    top_k: int | None = Field(default=None, ge=1, le=20)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    history_turns: int | None = Field(default=None, ge=0, le=50)
    context_token_budget: int | None = Field(default=None, ge=1000, le=200_000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    document_ids: list[UUID] | None = None


def resolve_settings(stored: dict | None) -> ChatSettings:
    s = get_settings()
    values: dict = {
        "top_k": s.retrieval_top_k,
        "min_score": s.retrieval_min_score,
        "history_turns": s.history_turns,
        "context_token_budget": s.context_token_budget,
        "temperature": s.llm_temperature,
        "document_ids": None,
    }
    values.update({k: v for k, v in (stored or {}).items() if v is not None})
    return ChatSettings(**values)
