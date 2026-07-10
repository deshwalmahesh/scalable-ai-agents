"""Owner-scoped async queries. Every read/write filters by the owning user_id."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ChatDB, DocumentBlobDB, DocumentDB, MessageDB


# --- documents ---------------------------------------------------------------
async def create_document(
    s: AsyncSession, *, user_id: UUID, filename: str, mime: str, size: int, content_hash: str | None = None
) -> DocumentDB:
    doc = DocumentDB(
        user_id=user_id, filename=filename, mime=mime, size=size, status="pending", content_hash=content_hash
    )
    s.add(doc)
    await s.commit()
    await s.refresh(doc)
    return doc


async def get_document_by_hash(s: AsyncSession, *, user_id: UUID, content_hash: str) -> DocumentDB | None:
    """Most recent document with this content for the user (for upload de-duplication)."""
    return (
        (
            await s.execute(
                select(DocumentDB)
                .where(DocumentDB.user_id == user_id, DocumentDB.content_hash == content_hash)
                .order_by(DocumentDB.created_at.desc())
            )
        )
        .scalars()
        .first()
    )


async def get_document(s: AsyncSession, *, doc_id: UUID, user_id: UUID) -> DocumentDB | None:
    return (
        await s.execute(select(DocumentDB).where(DocumentDB.id == doc_id, DocumentDB.user_id == user_id))
    ).scalar_one_or_none()


async def list_documents(s: AsyncSession, *, user_id: UUID) -> list[DocumentDB]:
    return list(
        (
            await s.execute(
                select(DocumentDB).where(DocumentDB.user_id == user_id).order_by(DocumentDB.created_at.desc())
            )
        ).scalars()
    )


async def set_document_status(
    s: AsyncSession, *, doc_id: UUID, status: str, chunk_count: int | None = None, error: str | None = None
) -> None:
    doc = await s.get(DocumentDB, doc_id)
    if doc is None:
        return
    doc.status = status
    if chunk_count is not None:
        doc.chunk_count = chunk_count
    doc.error = error
    await s.commit()


async def delete_document(s: AsyncSession, *, doc_id: UUID, user_id: UUID) -> bool:
    doc = await get_document(s, doc_id=doc_id, user_id=user_id)
    if doc is None:
        return False
    await s.delete(doc)
    await s.commit()
    return True


# --- transient upload blobs (shared DB, not local disk) ----------------------
async def save_blob(s: AsyncSession, *, document_id: UUID, data: bytes) -> None:
    s.add(DocumentBlobDB(document_id=document_id, data=data))
    await s.commit()


async def load_blob(s: AsyncSession, *, document_id: UUID) -> bytes | None:
    row = await s.get(DocumentBlobDB, document_id)
    return row.data if row else None


async def delete_blob(s: AsyncSession, *, document_id: UUID) -> None:
    row = await s.get(DocumentBlobDB, document_id)
    if row is not None:
        await s.delete(row)
        await s.commit()


# --- chats -------------------------------------------------------------------
async def create_chat(s: AsyncSession, *, user_id: UUID, title: str) -> ChatDB:
    chat = ChatDB(user_id=user_id, title=title)
    s.add(chat)
    await s.commit()
    await s.refresh(chat)
    return chat


async def get_chat(s: AsyncSession, *, chat_id: UUID, user_id: UUID) -> ChatDB | None:
    return (
        await s.execute(select(ChatDB).where(ChatDB.id == chat_id, ChatDB.user_id == user_id))
    ).scalar_one_or_none()


async def list_chats(s: AsyncSession, *, user_id: UUID) -> list[ChatDB]:
    return list(
        (
            await s.execute(
                select(ChatDB).where(ChatDB.user_id == user_id).order_by(ChatDB.updated_at.desc())
            )
        ).scalars()
    )


async def update_chat_settings(
    s: AsyncSession, *, chat_id: UUID, user_id: UUID, settings: dict
) -> ChatDB | None:
    chat = await get_chat(s, chat_id=chat_id, user_id=user_id)
    if chat is None:
        return None
    chat.settings = {**(chat.settings or {}), **settings}
    await s.commit()
    await s.refresh(chat)
    return chat


async def set_chat_memory(s: AsyncSession, *, chat_id: UUID, summary: str, summary_upto) -> None:
    chat = await s.get(ChatDB, chat_id)
    if chat is not None:
        chat.summary = summary
        chat.summary_upto = summary_upto
        await s.commit()


async def delete_chat(s: AsyncSession, *, chat_id: UUID, user_id: UUID) -> bool:
    chat = await get_chat(s, chat_id=chat_id, user_id=user_id)
    if chat is None:
        return False
    await s.delete(chat)
    await s.commit()
    return True


# --- messages ----------------------------------------------------------------
async def add_message(
    s: AsyncSession, *, chat_id: UUID, role: str, content: str, citations: list | None = None
) -> MessageDB:
    msg = MessageDB(chat_id=chat_id, role=role, content=content, citations=citations)
    s.add(msg)
    await s.commit()
    await s.refresh(msg)
    return msg


async def set_message_analysis(s: AsyncSession, *, message_id: UUID, citations: list, verdict: dict) -> None:
    """Persist the post-stream analysis: enriched citations (with used_quote) + judge verdict."""
    msg = await s.get(MessageDB, message_id)
    if msg is not None:
        msg.citations = citations
        msg.verdict = verdict
        await s.commit()


async def list_messages(s: AsyncSession, *, chat_id: UUID) -> list[MessageDB]:
    return list(
        (
            await s.execute(
                select(MessageDB).where(MessageDB.chat_id == chat_id).order_by(MessageDB.created_at)
            )
        ).scalars()
    )


async def delete_last_exchange(s: AsyncSession, *, chat_id: UUID) -> str | None:
    """Remove the last user message and everything after it (its assistant answer), for editing.
    Returns the removed user message's content, or None if there is no user message."""
    msgs = await list_messages(s, chat_id=chat_id)  # oldest first
    last_user = next((i for i in range(len(msgs) - 1, -1, -1) if msgs[i].role == "user"), None)
    if last_user is None:
        return None
    content = msgs[last_user].content
    for m in msgs[last_user:]:
        await s.delete(m)
    await s.commit()
    return content


async def messages_since(s: AsyncSession, *, chat_id: UUID, after) -> list[MessageDB]:
    """Messages newer than `after` (all if None), oldest first. Bounded in practice: once
    the unsummarized set exceeds the token budget it is folded into the summary and `after`
    advances, so this never grows without limit."""
    stmt = select(MessageDB).where(MessageDB.chat_id == chat_id)
    if after is not None:
        stmt = stmt.where(MessageDB.created_at > after)
    return list((await s.execute(stmt.order_by(MessageDB.created_at))).scalars())
