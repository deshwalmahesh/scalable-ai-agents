"""Document endpoints: upload (multi-file, async ingest), list, status, delete."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_async_session
from app.db.models import UserDB
from app.db.repos import delete_blob, delete_document, get_document, list_documents
from app.rag.extract import SUPPORTED, extension
from app.rag.vectorstore import get_vector_store
from app.redis_client import get_redis
from app.services.ingest_service import stage_and_enqueue
from app.users import current_active_user

log = structlog.get_logger()

router = APIRouter(prefix="/documents", tags=["documents"])


def _to_response(doc) -> dict:
    return {
        "id": str(doc.id),
        "filename": doc.filename,
        "status": doc.status,
        "chunk_count": doc.chunk_count,
        "error": doc.error,
        "created_at": doc.created_at.isoformat(),
    }


async def _check_load(request: Request) -> None:
    s = get_settings()
    if s.max_queue_depth <= 0 or s.ingest_executor != "queue":
        return
    try:
        from arq.constants import default_queue_name

        depth = await get_redis().zcard(default_queue_name)
    except Exception as exc:  # noqa: BLE001 — never fail-closed on the load probe
        log.warning("load_probe.unavailable", error=str(exc))
        return
    if depth >= s.max_queue_depth:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "The server is busy processing uploads right now. Please try again in a moment.",
        )


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload(
    request: Request,
    files: list[UploadFile],
    user: Annotated[UserDB, Depends(current_active_user)],
    session: Annotated[AsyncSession, Depends(get_async_session)],
    chunk_size: Annotated[int | None, Query(ge=200, le=4000)] = None,
    chunk_overlap: Annotated[int | None, Query(ge=0, le=1000)] = None,
) -> list[dict]:
    s = get_settings()
    if len(files) > s.max_upload_files:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"You can upload at most {s.max_upload_files} files at once.",
        )
    await _check_load(request)

    max_bytes = s.max_upload_mb * 1024 * 1024
    out: list[dict] = []
    for f in files:
        if extension(f.filename or "") not in SUPPORTED:
            raise HTTPException(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                f'"{f.filename}" isn\'t a supported file type. Upload a PDF, DOCX, DOC, TXT, or MD file.',
            )
        try:
            doc = await stage_and_enqueue(
                session,
                user_id=user.id,
                file=f,
                max_bytes=max_bytes,
                arq_pool=request.app.state.arq_pool,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        except ValueError as exc:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE,
                f'"{f.filename}" is larger than the {s.max_upload_mb} MB limit.',
            ) from exc
        out.append(_to_response(doc))
    return out


@router.get("")
async def list_mine(
    user: Annotated[UserDB, Depends(current_active_user)],
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> list[dict]:
    return [_to_response(d) for d in await list_documents(session, user_id=user.id)]


@router.get("/{doc_id}")
async def get_one(
    doc_id: UUID,
    user: Annotated[UserDB, Depends(current_active_user)],
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> dict:
    doc = await get_document(session, doc_id=doc_id, user_id=user.id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "That document could not be found.")
    return _to_response(doc)


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete(
    doc_id: UUID,
    user: Annotated[UserDB, Depends(current_active_user)],
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> None:
    doc = await get_document(session, doc_id=doc_id, user_id=user.id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "That document could not be found.")
    await get_vector_store().delete_document(user_id=user.id, document_id=doc_id)
    await delete_blob(session, document_id=doc_id)
    await delete_document(session, doc_id=doc_id, user_id=user.id)
