from __future__ import annotations

import asyncio
import statistics

from voice_agent.data.evaluation_queries import EVALUATION_CASES
from voice_agent.livekit_smoke import run_smoke
from voice_agent.settings import load_settings


async def run_single_session(case: dict[str, object]) -> dict[str, object]:
    return await run_smoke(
        question_text=str(case["question"]),
        room_name_prefix="voice-agent-load-test",
        expected_keywords=list(case.get("expected_keywords", [])),
        forbidden_keywords=list(case.get("forbidden_keywords", [])),
    )


def _summarize_results(
    *,
    concurrency: int,
    total_sessions: int,
    results: list[dict[str, object]],
    failures: int,
    failure_examples: list[str],
) -> dict[str, float | int | str]:
    successful_latencies = [
        float(result["response_latency_ms"])
        for result in results
        if result.get("round_trip_success") and result.get("response_latency_ms") is not None
    ]
    successful_count = len(successful_latencies)
    failed_completed_sessions = len(results) - successful_count
    p95_ms = 0.0
    if successful_latencies:
        sorted_latencies = sorted(successful_latencies)
        p95_index = max(0, int(len(sorted_latencies) * 0.95) - 1)
        p95_ms = round(sorted_latencies[p95_index], 2)

    summary: dict[str, float | int | str] = {
        "concurrency": concurrency,
        "sessions": total_sessions,
        "successes": successful_count,
        "failures": failures + failed_completed_sessions,
        "avg_ms": round(statistics.mean(successful_latencies), 2)
        if successful_latencies
        else 0.0,
        "p95_ms": p95_ms,
    }
    if failure_examples:
        summary["failure_examples"] = " | ".join(failure_examples)
    return summary


async def run_load(
    *,
    concurrency: int | None = None,
    turns: int | None = None,
) -> dict[str, float | int | str]:
    settings = load_settings()
    load_concurrency = concurrency or settings.load_test_concurrency
    load_turns = turns or settings.load_test_turns
    semaphore = asyncio.Semaphore(load_concurrency)
    results: list[dict[str, object]] = []
    failures = 0
    failure_examples: list[str] = []

    async def bounded_session(index: int) -> None:
        nonlocal failures
        case = EVALUATION_CASES[index % len(EVALUATION_CASES)]
        async with semaphore:
            try:
                results.append(await run_single_session(case))
            except Exception as exc:
                failures += 1
                if len(failure_examples) < 3:
                    failure_examples.append(f"{type(exc).__name__}: {exc}")

    total_sessions = load_concurrency * load_turns
    await asyncio.gather(
        *(bounded_session(index) for index in range(total_sessions))
    )
    return _summarize_results(
        concurrency=load_concurrency,
        total_sessions=total_sessions,
        results=results,
        failures=failures,
        failure_examples=failure_examples,
    )


async def run_stepped_load(
    levels: list[int],
    *,
    turns: int | None = None,
) -> list[dict[str, float | int | str]]:
    return [await run_load(concurrency=level, turns=turns) for level in levels]


def main() -> None:
    result = asyncio.run(run_load())
    print(
        "concurrency={concurrency} sessions={sessions} successes={successes} failures={failures} avg_ms={avg_ms} p95_ms={p95_ms}{failure_suffix}".format(
            failure_suffix=(
                f" failure_examples={result['failure_examples']}"
                if "failure_examples" in result
                else ""
            ),
            **result
        )
    )
