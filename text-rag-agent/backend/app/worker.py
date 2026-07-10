"""arq worker: durable, bounded, horizontally-scalable document ingestion.

Run: `arq app.worker.WorkerSettings`. Scale by running more worker processes/replicas
against the same Redis — max_jobs caps per-worker concurrency (backpressure).
"""

from __future__ import annotations

from uuid import UUID

import structlog
from arq.connections import RedisSettings

from app.config import get_settings
from app.llm import aclose_llm
from app.logging import configure_logging
from app.rag.vectorstore import aclose_vector_store
from app.redis_client import aclose_redis
from app.services.ingest_service import process_document

log = structlog.get_logger()


async def ingest_job(
    ctx: dict,
    doc_id: str,
    user_id: str,
    filename: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> None:
    is_final = ctx["job_try"] >= get_settings().ingest_max_tries
    await process_document(
        UUID(doc_id),
        UUID(user_id),
        filename,
        is_final_attempt=is_final,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


async def _startup(ctx: dict) -> None:
    s = get_settings()
    configure_logging(s.log_level, json_logs=(s.app_env == "prod"))
    log.info("worker.startup")


async def _shutdown(ctx: dict) -> None:
    await aclose_vector_store()
    await aclose_llm()
    await aclose_redis()


class WorkerSettings:
    functions = [ingest_job]
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    max_jobs = get_settings().worker_max_jobs
    job_timeout = get_settings().ingest_timeout_s
    max_tries = get_settings().ingest_max_tries
    on_startup = _startup
    on_shutdown = _shutdown
