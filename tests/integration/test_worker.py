from __future__ import annotations

from pathlib import Path

from livekit.plugins import openai

from voice_agent.kb import KnowledgeBase, KnowledgeDocument
from voice_agent.settings import AppSettings
from voice_agent.worker import SupportAgent, create_llm, create_runtime


TEST_DOCUMENTS = [
    KnowledgeDocument(
        id="free-look",
        title="Ten day policy examination and refund",
        content=(
            "If you return the policy within ten days of receiving it, "
            "the policy is treated as never issued and any premium paid is refunded."
        ),
        metadata={"source_url": "https://example.com/policy.pdf"},
    )
]


def build_settings(tmp_path: Path, **overrides: object) -> AppSettings:
    values = {
        "LIVEKIT_URL": "wss://example.livekit.cloud",
        "LIVEKIT_API_KEY": "livekit-key",
        "LIVEKIT_API_SECRET": "livekit-secret",
        "TWILIO_ACCOUNT_SID": "AC123",
        "TWILIO_AUTH_TOKEN": "twilio-token",
        "DEEPGRAM_API_KEY": "deepgram-key",
        "OPENAI_BASE_URL": "https://api.example.com/v1",
        "OPENAI_API_KEY": "openai-key",
        "OPENAI_MODEL": "model-name",
        "CHROMA_PERSIST_DIR": str(tmp_path / "chroma"),
        "CHROMA_COLLECTION_NAME": "test-collection",
    }
    values.update(overrides)
    return AppSettings(**values)


def test_create_llm_uses_openai_compatible_plugin(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)

    llm = create_llm(settings)

    assert isinstance(llm, openai.LLM)


def test_create_runtime_uses_configured_agent_name(tmp_path: Path) -> None:
    settings = build_settings(tmp_path, AGENT_NAME="support-agent")
    kb = KnowledgeBase(tmp_path / "chroma", "test-collection")

    runtime = create_runtime(settings=settings, knowledge_base=kb)

    assert runtime.agent_name == "support-agent"


async def test_support_agent_tool_returns_matches(tmp_path: Path) -> None:
    kb = KnowledgeBase(tmp_path / "chroma", "test-collection")
    kb.upsert_documents(TEST_DOCUMENTS)
    agent = SupportAgent(kb)

    result = await agent.search_support_knowledge(
        None,
        "If I return the policy within ten days, do I get my premium back?",
    )

    assert result["matches"]
