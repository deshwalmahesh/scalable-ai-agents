"""Multi-tenant fairness: per-user daily token quota + concurrent-query cap (Redis).

Both are best-effort guardrails, not hard ledgers: token accounting is eventually
consistent and the concurrency counter self-heals via TTL if a decrement is ever lost
to a crash — a tenant can never get permanently locked out.
"""

from __future__ import annotations

import structlog

from app.db import utcnow
from app.plans import limits_for
from app.redis_client import get_redis

log = structlog.get_logger()

_QUOTA_TTL_S = 60 * 60 * 48
_SLOT_TTL_S = 900  # > max query wall-clock; heals leaked slots from crashes


class QuotaExceeded(Exception):
    def __init__(self, used: int, cap: int) -> None:
        self.used, self.cap = used, cap
        super().__init__(f"daily tokens {used} >= cap {cap}")


class ConcurrencyLimitExceeded(Exception):
    def __init__(self, cap: int) -> None:
        self.cap = cap
        super().__init__(f"concurrent-query cap {cap} reached")


def _day_key(user_id) -> str:
    return f"quota:tokens:{user_id}:{utcnow():%Y%m%d}"


async def usage_today(user_id) -> int:
    try:
        val = await get_redis().get(_day_key(user_id))
        return int(val) if val else 0
    except Exception as exc:  # noqa: BLE001 — metering must never block a request
        log.warning("quota.read_failed", user_id=str(user_id), error=str(exc))
        return 0


async def add_usage(user_id, tokens: int) -> None:
    if tokens <= 0:
        return
    try:
        r = get_redis()
        key = _day_key(user_id)
        await r.incrby(key, tokens)
        await r.expire(key, _QUOTA_TTL_S)
    except Exception as exc:  # noqa: BLE001
        log.warning("quota.incr_failed", user_id=str(user_id), error=str(exc))


async def enforce_quota(user_id, plan: str | None) -> None:
    cap = limits_for(plan).daily_tokens
    if cap <= 0:
        return
    used = await usage_today(user_id)
    if used >= cap:
        raise QuotaExceeded(used, cap)


async def acquire_slot(user_id, plan: str | None) -> None:
    """Reserve a concurrent-query slot for the user, or raise ConcurrencyLimitExceeded.
    Fails open if Redis is unreachable — a cache outage must not break every query."""
    cap = limits_for(plan).max_concurrent
    if cap <= 0:
        return
    try:
        r = get_redis()
        key = f"active:{user_id}"
        active = await r.incr(key)
        await r.expire(key, _SLOT_TTL_S)
    except Exception as exc:  # noqa: BLE001 — Redis down must not block queries
        log.warning("quota.acquire_failed", user_id=str(user_id), error=str(exc))
        return
    if active > cap:
        await r.decr(key)
        raise ConcurrencyLimitExceeded(cap)


async def release_slot(user_id, plan: str | None) -> None:
    if limits_for(plan).max_concurrent <= 0:
        return
    try:
        await get_redis().decr(f"active:{user_id}")
    except Exception as exc:  # noqa: BLE001 — TTL heals a lost decrement
        log.warning("quota.release_failed", user_id=str(user_id), error=str(exc))
