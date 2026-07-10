from __future__ import annotations

import os

import pytest

from voice_agent.data.evaluation_queries import EVALUATION_CASES, EVALUATION_QUERIES
from voice_agent.livekit_smoke import run_multi_turn_smoke, run_smoke


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_E2E") != "1",
    reason="set RUN_E2E=1 to run the full live voice end-to-end test",
)


async def test_live_voice_agent_refund_flow() -> None:
    refund_case = EVALUATION_CASES[0]
    result = await run_smoke(
        question_text=EVALUATION_QUERIES[0],
        expected_keywords=list(refund_case["expected_keywords"]),
        forbidden_keywords=list(refund_case["forbidden_keywords"]),
    )

    assert result["round_trip_success"] is True
    response_text = str(result["response_text"]).lower()

    assert "premium" in response_text
    assert "refund" in response_text or "refunded" in response_text
    assert "10" in response_text or "ten" in response_text
    assert "30" not in response_text
    assert "thirty" not in response_text


@pytest.mark.skipif(
    os.getenv("RUN_MULTI_TURN_E2E") != "1",
    reason="set RUN_MULTI_TURN_E2E=1 to run the live multi-turn audio test",
)
async def test_live_voice_agent_multi_turn_audio_flow() -> None:
    result = await run_multi_turn_smoke(
        [
            {
                "question_text": EVALUATION_QUERIES[0],
                "expected_keywords": ["ten", "premium", "refund"],
                "forbidden_keywords": ["thirty", "30"],
            },
            {
                "question_text": EVALUATION_QUERIES[1],
                "expected_keywords": ["grace", "31", "days"],
            },
            {
                "question_text": EVALUATION_QUERIES[3],
                "expected_keywords": ["reinstate", "two", "years"],
                "forbidden_keywords": ["three", "five"],
            },
            {
                "question_text": "What is the CEO of the insurer?",
                "expected_keywords": ["policy", "human", "representative"],
            },
        ]
    )

    assert result["round_trip_success"] is True
    assert len(result["turns"]) == 4

    first_turn = result["turns"][0]
    second_turn = result["turns"][1]
    third_turn = result["turns"][2]
    fourth_turn = result["turns"][3]

    assert first_turn["round_trip_success"] is True
    assert second_turn["round_trip_success"] is True
    assert third_turn["round_trip_success"] is True
    assert fourth_turn["round_trip_success"] is True

    assert "ten" in str(first_turn["response_text"]).lower() or "10" in str(
        first_turn["response_text"]
    ).lower()
    assert "grace" in str(second_turn["response_text"]).lower()
    assert "31" in str(second_turn["response_text"]).lower() or "thirty one" in str(
        second_turn["response_text"]
    ).lower()
    assert "reinstate" in str(third_turn["response_text"]).lower()
    assert "two" in str(third_turn["response_text"]).lower() or "2" in str(
        third_turn["response_text"]
    ).lower()
    unsupported_text = str(fourth_turn["response_text"]).lower()
    assert "policy" in unsupported_text
    assert "human" in unsupported_text or "representative" in unsupported_text

    quality_summary = result["quality_summary"]
    assert quality_summary["turn_count"] == 4
    assert quality_summary["successful_turns"] == 4
    assert quality_summary["turns_with_network_quality"] >= 1
    assert quality_summary["avg_response_latency_ms"] is not None
