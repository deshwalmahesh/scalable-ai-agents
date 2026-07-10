"""Process-wide async Redis singleton (rate-limit store + arq broker + quota + cache)."""

from __future__ import annotations

from redis.asyncio import Redis

from app.config import get_settings

_redis: Redis | None = None


def get_redis() -> Redis:
    global _redis
    if _redis is None:
        _redis = Redis.from_url(
            get_settings().redis_url,
            decode_responses=True,
            socket_timeout=5,  # command timeout — never block a request on a hung Redis
            socket_connect_timeout=5,
            health_check_interval=30,
        )
    return _redis


async def aclose_redis() -> None:
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None
