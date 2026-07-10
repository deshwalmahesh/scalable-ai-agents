"""Per-plan tenant limits. Single source for concurrency + daily-token policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlanLimits:
    max_concurrent: int  # in-flight queries per user; 0 = unlimited
    daily_tokens: int  # tokens/day per user; 0 = unlimited


PLANS: dict[str, PlanLimits] = {
    "free": PlanLimits(max_concurrent=2, daily_tokens=100_000),
    "paid": PlanLimits(max_concurrent=10, daily_tokens=0),
}


def limits_for(plan: str | None) -> PlanLimits:
    return PLANS.get(plan or "free", PLANS["free"])
