from __future__ import annotations

import os

import pytest

from voice_agent.deepgram_smoke import run_smoke as run_deepgram_smoke
from voice_agent.kb import KnowledgeBase
from voice_agent.settings import load_settings


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_SMOKE") != "1",
    reason="set RUN_SMOKE=1 to run real provider smoke tests",
)


async def test_deepgram_round_trip_smoke() -> None:
    result = await run_deepgram_smoke()

    assert result["round_trip_success"] is True


def test_live_chroma_retrieval_smoke() -> None:
    settings = load_settings()
    kb = KnowledgeBase.from_settings(settings)

    results = kb.search(
        "If I return the policy within ten days, do I get my premium back?",
        limit=3,
    )

    assert results
    assert results[0].metadata["source_url"].startswith("https://")
