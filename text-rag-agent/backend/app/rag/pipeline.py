"""RAG orchestration: ingest a staged document, and retrieve grounded sources for a query.

Ingestion is idempotent (points for the doc are cleared before upsert so an arq retry
never duplicates vectors) and RAM-bounded (embed + upsert per batch, bounded concurrency).
"""

from __future__ import annotations

import asyncio
import time
from uuid import UUID

import structlog

from app.config import get_settings
from app.llm import get_llm
from app.rag.chunk import Chunk, chunk_pages
from app.rag.extract import extract_text
from app.rag.vectorstore import Hit, get_vector_store

log = structlog.get_logger()

_EMBED_BATCH = 64


async def _embed_and_upsert(
    sem: asyncio.Semaphore, *, user_id: UUID, document_id: UUID, filename: str, batch: list[Chunk]
) -> None:
    llm = get_llm()
    async with sem:  # bound concurrent embedding calls
        vectors = await llm.embed_documents([c.text for c in batch])
    await get_vector_store().upsert(
        user_id=user_id, document_id=document_id, filename=filename, chunks=batch, vectors=vectors
    )


async def ingest_document(
    *,
    user_id: UUID,
    document_id: UUID,
    filename: str,
    data: bytes,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> int:
    pages = await extract_text(data, filename)
    chunks = chunk_pages(pages, chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError("no extractable text in document")

    await get_vector_store().delete_document(user_id=user_id, document_id=document_id)

    t0 = time.monotonic()
    sem = asyncio.Semaphore(get_settings().embed_concurrency)
    batches = [chunks[i : i + _EMBED_BATCH] for i in range(0, len(chunks), _EMBED_BATCH)]
    await asyncio.gather(
        *(
            _embed_and_upsert(sem, user_id=user_id, document_id=document_id, filename=filename, batch=b)
            for b in batches
        )
    )
    log.info(
        "ingest.done",
        document_id=str(document_id),
        chunks=len(chunks),
        elapsed_ms=int((time.monotonic() - t0) * 1000),
    )
    return len(chunks)


async def retrieve(
    *,
    user_id: UUID,
    question: str,
    top_k: int,
    min_score: float,
    document_ids: list[UUID] | None = None,
) -> tuple[list[Hit], bool]:
    """Return (hits, grounded). grounded is False when the best hit is below the score
    gate — the caller then refuses without calling the LLM."""
    query_vector = await get_llm().embed_query(question)
    hits = await get_vector_store().search(
        user_id=user_id,
        query_text=question,
        query_vector=query_vector,
        top_k=top_k,
        document_ids=document_ids,
    )
    # hits are RRF-ordered, so the best cosine may not be first — gate on the max.
    grounded = bool(hits) and max(h.score for h in hits) >= min_score
    return hits, grounded
