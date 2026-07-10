"""Liveness + readiness. Liveness stays dependency-free so a transient blip never
restart-loops the pod; readiness gates traffic on DB + Redis + Qdrant."""

from __future__ import annotations

from fastapi import APIRouter, Response, status
from sqlalchemy import text

from app.db import get_session_factory
from app.rag.vectorstore import get_vector_store
from app.redis_client import get_redis

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready")
async def ready(response: Response) -> dict:
    checks: dict[str, str] = {}
    ok = True
    for name, probe in (
        ("db", _db_ok),
        ("redis", _redis_ok),
        ("qdrant", _qdrant_ok),
    ):
        try:
            await probe()
            checks[name] = "ok"
        except Exception as exc:  # noqa: BLE001
            checks[name] = f"error: {type(exc).__name__}"
            ok = False
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ok" if ok else "degraded", "checks": checks}


async def _db_ok() -> None:
    async with get_session_factory()() as s:
        await s.execute(text("SELECT 1"))


async def _redis_ok() -> None:
    await get_redis().ping()


async def _qdrant_ok() -> None:
    await get_vector_store().ping()
