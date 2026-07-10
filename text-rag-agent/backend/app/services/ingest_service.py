"""Document ingestion dispatch. One processor reused by both the inline (dev) and arq
(prod) paths so status transitions, cleanup, and error handling never drift.

The raw upload is staged as a DB blob (shared across replicas, no local disk), read by
the worker, and deleted once ingested — portable to any managed Postgres."""

from __future__ import annotations

import asyncio
import hashlib
from uuid import UUID

import structlog
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_session_factory
from app.db.models import DocumentDB
from app.db.repos import (
    create_document,
    delete_blob,
    get_document_by_hash,
    load_blob,
    save_blob,
    set_document_status,
)
from app.rag import pipeline

log = structlog.get_logger()

_READ_CHUNK = 1024 * 1024


async def _read_bounded(file: UploadFile, max_bytes: int) -> bytes:
    """Read an upload into memory, bounded — never load an unbounded body."""
    buf = bytearray()
    while chunk := await file.read(_READ_CHUNK):
        buf.extend(chunk)
        if len(buf) > max_bytes:
            raise ValueError("file exceeds size limit")
    return bytes(buf)


async def stage_and_enqueue(
    session: AsyncSession,
    *,
    user_id: UUID,
    file: UploadFile,
    max_bytes: int,
    arq_pool,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> DocumentDB:
    """Persist the document row + raw bytes, then enqueue ingestion. Raises ValueError if
    the file exceeds max_bytes (caller maps to 413)."""
    data = await _read_bounded(file, max_bytes)

    # De-dup: identical content already indexed (or in flight) for this user is reused instead
    # of re-embedding. A prior failed attempt is NOT reused, so retries still work.
    content_hash = hashlib.sha256(data).hexdigest()
    existing = await get_document_by_hash(session, user_id=user_id, content_hash=content_hash)
    if existing is not None and existing.status in ("pending", "processing", "ready"):
        log.info("document.duplicate_skipped", document_id=str(existing.id), filename=existing.filename)
        return existing

    doc = await create_document(
        session,
        user_id=user_id,
        filename=file.filename or "upload",
        mime=file.content_type or "",
        size=len(data),
        content_hash=content_hash,
    )
    await save_blob(session, document_id=doc.id, data=data)
    await enqueue_ingest(
        doc_id=doc.id,
        user_id=user_id,
        filename=doc.filename,
        arq_pool=arq_pool,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    log.info("document.uploaded", document_id=str(doc.id), user_id=str(user_id), size=len(data))
    return doc


async def process_document(
    doc_id: UUID,
    user_id: UUID,
    filename: str,
    *,
    is_final_attempt: bool,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> None:
    """Load the staged bytes, ingest, update status, and drop the blob. On failure the
    blob is kept for a retry unless this is the final attempt."""
    sf = get_session_factory()
    async with sf() as s:
        await set_document_status(s, doc_id=doc_id, status="processing")
        data = await load_blob(s, document_id=doc_id)
    if data is None:
        async with sf() as s:
            await set_document_status(s, doc_id=doc_id, status="failed", error="staged file missing")
        return

    try:
        count = await pipeline.ingest_document(
            user_id=user_id,
            document_id=doc_id,
            filename=filename,
            data=data,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except Exception as exc:
        log.error("ingest.failed", document_id=str(doc_id), error=str(exc))
        async with sf() as s:
            await set_document_status(s, doc_id=doc_id, status="failed", error=str(exc)[:500])
            if is_final_attempt:
                await delete_blob(s, document_id=doc_id)
        raise

    async with sf() as s:
        await set_document_status(s, doc_id=doc_id, status="ready", chunk_count=count)
        await delete_blob(s, document_id=doc_id)


async def enqueue_ingest(
    *,
    doc_id: UUID,
    user_id: UUID,
    filename: str,
    arq_pool,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> None:
    if get_settings().ingest_executor == "queue" and arq_pool is not None:
        await arq_pool.enqueue_job(
            "ingest_job", str(doc_id), str(user_id), filename, chunk_size, chunk_overlap
        )
        return

    async def _inline() -> None:
        try:
            await process_document(
                doc_id,
                user_id,
                filename,
                is_final_attempt=True,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        except Exception:  # noqa: BLE001 — already logged + persisted; background task
            pass

    asyncio.create_task(_inline())
