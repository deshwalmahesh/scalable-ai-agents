"""Async SQLAlchemy engine, session factory, and Base. SQLite (dev/tests) or Postgres (prod)."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import structlog
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

log = structlog.get_logger()


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        s = get_settings()
        kwargs: dict = {}
        if not s.database_url.startswith("sqlite"):
            kwargs.update(
                pool_size=s.db_pool_size,
                max_overflow=s.db_max_overflow,
                pool_timeout=s.db_pool_timeout,
                pool_recycle=s.db_pool_recycle,
                pool_pre_ping=s.db_pool_pre_ping,
            )
        _engine = create_async_engine(s.database_url, **kwargs)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _session_factory


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with get_session_factory()() as session:
        yield session


async def create_all() -> None:
    from app.db import models  # noqa: F401 — register tables on Base.metadata

    log.info("db.create_all", url=get_settings().database_url)
    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
