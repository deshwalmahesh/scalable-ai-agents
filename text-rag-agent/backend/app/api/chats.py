"""Chat endpoints: CRUD, message history, and the streaming grounded-answer query."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.chat_settings import ChatSettingsPatch, resolve_settings
from app.config import get_settings
from app.db import get_async_session
from app.db.models import UserDB
from app.db.repos import (
    create_chat,
    delete_chat,
    get_chat,
    list_chats,
    list_messages,
    update_chat_settings,
)
from app.quota import (
    ConcurrencyLimitExceeded,
    QuotaExceeded,
    acquire_slot,
    enforce_quota,
    release_slot,
)
from app.services.chat_service import stream_answer
from app.users import current_active_user

router = APIRouter(prefix="/chats", tags=["chats"])


class ChatCreate(BaseModel):
    title: str = "New chat"


class QueryBody(BaseModel):
    question: str = Field(min_length=1)
    document_ids: list[UUID] | None = None


def _to_response(chat) -> dict:
    return {
        "id": str(chat.id),
        "title": chat.title,
        "settings": resolve_settings(chat.settings).model_dump(mode="json"),
        "has_summary": bool(chat.summary),
        "created_at": chat.created_at.isoformat(),
        "updated_at": chat.updated_at.isoformat(),
    }


@router.post("", status_code=status.HTTP_201_CREATED)
async def create(
    body: ChatCreate,
    user: Annotated[UserDB, Depends(current_active_user)],
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> dict:
    return _to_response(await create_chat(session, user_id=user.id, title=body.title))


@router.get("")
async def list_mine(
    user: Annotated[UserDB, Depends(current_active_user)],
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> list[dict]:
    return [_to_response(c) for c in await list_chats(session, user_id=user.id)]


@router.get("/{chat_id}")
async def get_one(
    chat_id: UUID,
    user: Annotated[UserDB, Depends(current_active_user)],
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> dict:
    chat = await get_chat(session, chat_id=chat_id, user_id=user.id)
    if chat is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "chat not found")
    return _to_response(chat)


@router.patch("/{chat_id}")
async def patch(
    chat_id: UUID,
    body: ChatSettingsPatch,
    user: Annotated[UserDB, Depends(current_active_user)],
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> dict:
    updates = body.model_dump(mode="json", exclude_none=True)
    chat = await update_chat_settings(session, chat_id=chat_id, user_id=user.id, settings=updates)
    if chat is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "chat not found")
    return _to_response(chat)


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete(
    chat_id: UUID,
    user: Annotated[UserDB, Depends(current_active_user)],
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> None:
    if not await delete_chat(session, chat_id=chat_id, user_id=user.id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "chat not found")


@router.get("/{chat_id}/messages")
async def messages(
    chat_id: UUID,
    user: Annotated[UserDB, Depends(current_active_user)],
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> list[dict]:
    if await get_chat(session, chat_id=chat_id, user_id=user.id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "chat not found")
    return [
        {
            "id": str(m.id),
            "role": m.role,
            "content": m.content,
            "citations": m.citations,
            "verdict": m.verdict,
            "created_at": m.created_at.isoformat(),
        }
        for m in await list_messages(session, chat_id=chat_id)
    ]


@router.post("/{chat_id}/query")
async def query(
    chat_id: UUID,
    body: QueryBody,
    user: Annotated[UserDB, Depends(current_active_user)],
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> EventSourceResponse:
    s = get_settings()
    if len(body.question) > s.max_question_chars:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "question too long")
    if await get_chat(session, chat_id=chat_id, user_id=user.id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "chat not found")

    try:
        await enforce_quota(user.id, user.plan)
        await acquire_slot(user.id, user.plan)
    except QuotaExceeded as exc:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS, f"Daily token limit reached ({exc.used}/{exc.cap})."
        ) from exc
    except ConcurrencyLimitExceeded as exc:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS, f"Too many concurrent queries (limit {exc.cap})."
        ) from exc

    async def gen():
        try:
            async for event in stream_answer(
                user_id=user.id,
                chat_id=chat_id,
                question=body.question,
                document_ids=body.document_ids,
            ):
                yield event
        finally:
            await release_slot(user.id, user.plan)

    return EventSourceResponse(gen())
