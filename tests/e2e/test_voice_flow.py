from __future__ import annotations

import os

import pytest

from voice_agent.data.evaluation_queries import EVALUATION_QUERIES
from voice_agent.livekit_smoke import run_smoke


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_E2E") != "1",
    reason="set RUN_E2E=1 to run the full live voice end-to-end test",
)


async def test_live_voice_agent_refund_flow() -> None:
    result = await run_smoke(question_text=EVALUATION_QUERIES[0])

    assert result["round_trip_success"] is True
    assert "premium" in str(result["response_text"]).lower()
