from __future__ import annotations

from pathlib import Path

from livekit.plugins import google, openai
from livekit.agents import llm

from voice_agent.kb import KnowledgeBase, KnowledgeDocument
from voice_agent.settings import AppSettings
from voice_agent.worker import (
    SupportAgent,
    create_gemini_live_model,
    create_llm,
    create_runtime,
)


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


def test_create_runtime_uses_capacity_based_load_function(tmp_path: Path) -> None:
    settings = build_settings(tmp_path, WORKER_MAX_CONCURRENT_SESSIONS=4)

    runtime = create_runtime(settings=settings)

    assert runtime.server._load_threshold == 1.0
    assert runtime.server.load_fnc is not None


def test_create_gemini_live_model_uses_google_plugin(tmp_path: Path) -> None:
    settings = build_settings(
        tmp_path,
        VOICE_RUNTIME_MODE="gemini_live",
        GOOGLE_API_KEY="google-key",
        GEMINI_LIVE_MODEL="gemini-3.1-flash-live-preview",
    )

    model = create_gemini_live_model(settings)

    assert isinstance(model, google.realtime.RealtimeModel)


async def test_support_agent_injects_grounding_into_turn_context(tmp_path: Path) -> None:
    kb = KnowledgeBase(tmp_path / "chroma", "test-collection")
    kb.upsert_documents(TEST_DOCUMENTS)
    agent = SupportAgent(kb)
    turn_ctx = llm.ChatContext.empty()
    new_message = turn_ctx.add_message(
        role="user",
        content="If I return the policy within ten days, do I get my premium back?",
    )

    await agent.on_user_turn_completed(turn_ctx, new_message)

    assert len(turn_ctx.messages()) == 2
    assert "Grounding for this turn only" in str(turn_ctx.messages()[-1].text_content)
