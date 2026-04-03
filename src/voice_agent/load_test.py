from __future__ import annotations

import asyncio
import statistics

from voice_agent.data.evaluation_queries import EVALUATION_QUERIES
from voice_agent.livekit_smoke import run_smoke
from voice_agent.settings import load_settings


async def run_single_session(question: str) -> dict[str, object]:
    return await run_smoke(
        question_text=question,
        room_name_prefix="voice-agent-load-test",
    )


async def run_load() -> dict[str, float | int]:
    settings = load_settings()
    semaphore = asyncio.Semaphore(settings.load_test_concurrency)
    results: list[dict[str, object]] = []
    failures = 0

    async def bounded_session(index: int) -> None:
        nonlocal failures
        question = EVALUATION_QUERIES[index % len(EVALUATION_QUERIES)]
        async with semaphore:
            try:
                results.append(await run_single_session(question))
            except Exception:
                failures += 1

    total_sessions = settings.load_test_concurrency * settings.load_test_turns
    await asyncio.gather(
        *(bounded_session(index) for index in range(total_sessions))
    )

    successful_latencies = [
        float(result["response_latency_ms"])
        for result in results
        if result.get("round_trip_success") and result.get("response_latency_ms") is not None
    ]
    p95_ms = 0.0
    if successful_latencies:
        sorted_latencies = sorted(successful_latencies)
        p95_index = max(0, int(len(sorted_latencies) * 0.95) - 1)
        p95_ms = round(sorted_latencies[p95_index], 2)

    return {
        "sessions": total_sessions,
        "successes": len(successful_latencies),
        "failures": failures + (total_sessions - len(results)),
        "avg_ms": round(statistics.mean(successful_latencies), 2)
        if successful_latencies
        else 0.0,
        "p95_ms": p95_ms,
    }


def main() -> None:
    result = asyncio.run(run_load())
    print(
        "sessions={sessions} successes={successes} failures={failures} avg_ms={avg_ms} p95_ms={p95_ms}".format(
            **result
        )
    )
