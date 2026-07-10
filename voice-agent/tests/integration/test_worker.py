from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import types

import pytest
from prometheus_client import CollectorRegistry

from livekit.plugins import deepgram, google, openai
from livekit.agents import CloseEvent, CloseReason, ErrorEvent, llm, stt, tts

from voice_agent.kb import (
    KnowledgeBase,
    KnowledgeDocument,
    RetrievalService,
    UNSUPPORTED_POLICY_RESPONSE,
)
from voice_agent.load_test import run_load, run_single_session, run_stepped_load
from voice_agent.observability import ActiveSessionsStore, VoiceAgentMetrics
from voice_agent.settings import AppSettings
from voice_agent.worker import (
    SessionFailurePolicy,
    cancel_watchdog,
    classify_upstream_source,
    create_session_connect_options,
    prewarm,
    handle_session_close_event,
    handle_session_error_event,
    record_stt_timeout,
    start_watchdog,
    stt_timeout_seconds_for_transcript,
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


class FailingRetrievalService:
    def should_ground(self, query: str) -> bool:
        return True

    async def async_ground(self, query: str, limit: int = 1) -> object:
        raise RuntimeError("retrieval unavailable")


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def say(
        self,
        text: str,
        *,
        allow_interruptions: bool,
        add_to_chat_ctx: bool,
    ) -> None:
        self.calls.append(
            {
                "text": text,
                "allow_interruptions": allow_interruptions,
                "add_to_chat_ctx": add_to_chat_ctx,
            }
        )


class FakeJobProcess:
    def __init__(self) -> None:
        self.userdata: dict[str, object] = {}


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

    model = create_llm(settings)

    assert isinstance(model, openai.LLM)
    assert model._opts.temperature == 0
    assert model._client.max_retries == 0
    assert model._client.timeout.connect == settings.http_timeout_seconds
    assert model._client.timeout.read == settings.http_timeout_seconds
    assert model._client.timeout.write == settings.http_timeout_seconds
    assert model._client.timeout.pool == settings.http_timeout_seconds


def test_create_session_connect_options_disables_live_llm_retries(tmp_path: Path) -> None:
    settings = build_settings(tmp_path, HTTP_TIMEOUT_SECONDS=7.5)

    options = create_session_connect_options(settings)

    assert options.llm_conn_options.max_retry == 0
    assert options.llm_conn_options.timeout == 7.5
    assert options.stt_conn_options.timeout == 7.5
    assert options.tts_conn_options.timeout == 7.5


def test_create_llm_uses_fallback_adapter_when_fallback_is_configured(
    tmp_path: Path,
) -> None:
    settings = build_settings(
        tmp_path,
        FALLBACK_OPENAI_BASE_URL="https://fallback.example.com/v1",
        FALLBACK_OPENAI_API_KEY="fallback-key",
        FALLBACK_OPENAI_MODEL="fallback-model",
    )

    model = create_llm(settings)

    assert isinstance(model, llm.FallbackAdapter)


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


def test_prewarm_warms_local_embeddings_before_live_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warmed: list[str] = []

    class DummyRetrieval:
        def count(self) -> int:
            return 41

    settings = build_settings(tmp_path, EMBEDDING_PROVIDER="local")
    monkeypatch.setattr("voice_agent.worker.get_runtime_settings", lambda: settings)
    monkeypatch.setattr("voice_agent.worker.get_runtime_retrieval", lambda: DummyRetrieval())
    monkeypatch.setattr("voice_agent.worker.warm_local_model", lambda: warmed.append("ok"))
    monkeypatch.setattr("voice_agent.worker.silero.VAD.load", lambda: "vad")

    proc = FakeJobProcess()

    prewarm(proc)  # type: ignore[arg-type]

    assert warmed == ["ok"]
    assert proc.userdata["vad"] == "vad"


def test_create_runtime_resets_stale_active_session_store(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    store = ActiveSessionsStore(Path(settings.active_sessions_path))
    store.write(3)

    create_runtime(settings=settings)

    assert store.read() == 0


def test_create_runtime_does_not_reset_shared_redis_active_session_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRedisClient:
        def __init__(self) -> None:
            self.values: dict[str, int] = {"voice-agent:test": 3}

        def get(self, key: str) -> str | None:
            value = self.values.get(key)
            return None if value is None else str(value)

        def set(self, key: str, value: int) -> None:
            self.values[key] = int(value)

        def incr(self, key: str) -> int:
            current = self.values.get(key, 0) + 1
            self.values[key] = current
            return current

        def eval(self, script: str, numkeys: int, key: str) -> int:
            current = max(0, self.values.get(key, 0) - 1)
            self.values[key] = current
            return current

    fake_client = FakeRedisClient()

    class FakeRedis:
        @staticmethod
        def from_url(url: str, decode_responses: bool = True) -> FakeRedisClient:
            assert url == "redis://example:6379/0"
            assert decode_responses is True
            return fake_client

    monkeypatch.setitem(sys.modules, "redis", types.SimpleNamespace(Redis=FakeRedis))
    settings = build_settings(
        tmp_path,
        REDIS_URL="redis://example:6379/0",
        ACTIVE_SESSIONS_KEY="voice-agent:test",
    )

    create_runtime(settings=settings)

    assert fake_client.values["voice-agent:test"] == 3


def test_classify_upstream_source_maps_provider_types(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)

    assert (
        classify_upstream_source(
            deepgram.STT(model="nova-3", language="en-US", api_key="deepgram-key")
        )
        == "stt"
    )
    assert (
        classify_upstream_source(
            deepgram.TTS(model="aura-2-asteria-en", api_key="deepgram-key")
        )
        == "tts"
    )
    assert (
        classify_upstream_source(
            openai.LLM(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )
        )
        == "llm"
    )


def test_handle_session_error_event_records_component_metric() -> None:
    metrics = VoiceAgentMetrics(registry=CollectorRegistry())
    event = ErrorEvent(
        error=stt.STTError(
            timestamp=1.0,
            label="deepgram",
            error=RuntimeError("boom"),
            recoverable=False,
        ),
        source=deepgram.STT(model="nova-3", language="en-US", api_key="deepgram-key"),
    )

    component = handle_session_error_event(metrics, event)

    assert component == "stt"
    rendered = metrics.render_latest().decode()
    assert 'voice_agent_upstream_failures_total{component="stt"} 1.0' in rendered


def test_handle_session_close_event_records_failure_only_when_error_present() -> None:
    metrics = VoiceAgentMetrics(registry=CollectorRegistry())

    no_error_component = handle_session_close_event(
        metrics,
        CloseEvent(reason=CloseReason.TASK_COMPLETED, error=None),
    )
    error_component = handle_session_close_event(
        metrics,
        CloseEvent(
            reason=CloseReason.ERROR,
            error=tts.TTSError(
                timestamp=1.0,
                label="cartesia",
                error=RuntimeError("tts failed"),
                recoverable=False,
            ),
        ),
    )

    assert no_error_component is None
    assert error_component == "session"
    rendered = metrics.render_latest().decode()
    assert 'voice_agent_upstream_failures_total{component="session"} 1.0' in rendered


def test_record_stt_timeout_records_metric() -> None:
    metrics = VoiceAgentMetrics(registry=CollectorRegistry())

    component = record_stt_timeout(metrics, "hello there")

    assert component == "stt_timeout"
    rendered = metrics.render_latest().decode()
    assert 'voice_agent_upstream_failures_total{component="stt_timeout"} 1.0' in rendered


def test_stt_timeout_grows_with_partial_utterance_length() -> None:
    short_timeout = stt_timeout_seconds_for_transcript("hello")
    long_timeout = stt_timeout_seconds_for_transcript(
        "How long is the grace period if I miss a premium payment"
    )

    assert short_timeout == pytest.approx(3.3)
    assert long_timeout > short_timeout
    assert long_timeout <= 6.0


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
    agent = SupportAgent(RetrievalService(kb))
    turn_ctx = llm.ChatContext.empty()
    new_message = turn_ctx.add_message(
        role="user",
        content="If I return the policy within ten days, do I get my premium back?",
    )

    await agent.on_user_turn_completed(turn_ctx, new_message)

    assert len(turn_ctx.messages()) == 2
    assert "Grounding for this turn only" in str(turn_ctx.messages()[-1].text_content)


async def test_support_agent_skips_retrieval_for_small_talk(tmp_path: Path) -> None:
    kb = KnowledgeBase(tmp_path / "chroma", "test-collection")
    kb.upsert_documents(TEST_DOCUMENTS)
    agent = SupportAgent(RetrievalService(kb))
    turn_ctx = llm.ChatContext.empty()
    new_message = turn_ctx.add_message(role="user", content="hello")

    await agent.on_user_turn_completed(turn_ctx, new_message)

    assert len(turn_ctx.messages()) == 2
    assert "small talk or a connection check" in str(turn_ctx.messages()[-1].text_content)


async def test_support_agent_combines_split_user_turns_for_grounding(tmp_path: Path) -> None:
    kb = KnowledgeBase(tmp_path / "chroma", "test-collection")
    kb.upsert_documents(TEST_DOCUMENTS)
    agent = SupportAgent(RetrievalService(kb))
    turn_ctx = llm.ChatContext.empty()
    turn_ctx.add_message(
        role="user",
        content="If I return the policy within ten days, do I get my",
    )
    turn_ctx.add_message(
        role="developer",
        content="Grounding for this turn only. Answer using policy facts.",
    )
    new_message = turn_ctx.add_message(role="user", content="premium back?")

    await agent.on_user_turn_completed(turn_ctx, new_message)

    assert "If I return the policy within ten days, do I get my premium back?" in str(
        turn_ctx.messages()[-1].text_content
    )


async def test_support_agent_degrades_explicitly_when_retrieval_fails() -> None:
    agent = SupportAgent(FailingRetrievalService())
    turn_ctx = llm.ChatContext.empty()
    new_message = turn_ctx.add_message(
        role="user",
        content="What is the refund policy?",
    )

    await agent.on_user_turn_completed(turn_ctx, new_message)

    assert len(turn_ctx.messages()) == 2
    assert "Knowledge retrieval is temporarily unavailable" in str(
        turn_ctx.messages()[-1].text_content
    )


async def test_support_agent_replaces_old_turn_scoped_guidance(tmp_path: Path) -> None:
    kb = KnowledgeBase(tmp_path / "chroma", "test-collection")
    kb.upsert_documents(TEST_DOCUMENTS)
    agent = SupportAgent(RetrievalService(kb))
    turn_ctx = llm.ChatContext.empty()

    first_message = turn_ctx.add_message(
        role="user",
        content="If I return the policy within ten days, do I get my premium back?",
    )
    await agent.on_user_turn_completed(turn_ctx, first_message)
    turn_ctx.add_message(role="assistant", content="Yes, if you return it within ten days.")

    second_message = turn_ctx.add_message(
        role="user",
        content="How long is the grace period if I miss a premium payment?",
    )
    await agent.on_user_turn_completed(turn_ctx, second_message)

    developer_messages = [
        message
        for message in turn_ctx.messages()
        if message.role == "developer"
    ]
    assert len(developer_messages) == 1
    assert "How long is the grace period if I miss a premium payment?" in str(
        developer_messages[0].text_content
    )
    assert "If I return the policy within ten days" not in str(
        developer_messages[0].text_content
    )


async def test_support_agent_llm_node_uses_direct_grounded_answer(tmp_path: Path) -> None:
    kb = KnowledgeBase(tmp_path / "chroma", "test-collection")
    kb.upsert_documents(TEST_DOCUMENTS)
    agent = SupportAgent(RetrievalService(kb))
    turn_ctx = llm.ChatContext.empty()

    new_message = turn_ctx.add_message(
        role="user",
        content="If I return the policy within ten days, do I get my premium back?",
    )
    await agent.on_user_turn_completed(turn_ctx, new_message)

    result = agent.llm_node(turn_ctx, [], object())

    assert isinstance(result, str)
    assert "ten day period" in result or "ten days" in result
    assert "premium" in result
    developer_message = turn_ctx.messages()[-1]
    assert developer_message.role == "developer"
    assert developer_message.extra["turn_scoped_guidance"] is True
    assert isinstance(developer_message.extra["direct_answer"], str)


async def test_support_agent_llm_node_uses_direct_grounded_answer_for_reinstatement(
    tmp_path: Path,
) -> None:
    kb = KnowledgeBase(tmp_path / "chroma", "test-collection")
    kb.upsert_documents(
        [
            KnowledgeDocument(
                id="reinstatement",
                title="Policy reinstatement after lapse",
                content=(
                    "The policy may be reinstated subject to the following conditions: "
                    "1. You must make a written request for reinstatement within two years after the end of the grace period. "
                    "2. Every insured person aged sixteen or older must provide evidence of insurability."
                ),
                metadata={"source_url": "https://example.com/policy.pdf"},
            )
        ]
    )
    agent = SupportAgent(RetrievalService(kb))
    turn_ctx = llm.ChatContext.empty()

    new_message = turn_ctx.add_message(
        role="user",
        content="Can I reinstate the policy after it ends because I stopped paying premiums?",
    )
    await agent.on_user_turn_completed(turn_ctx, new_message)

    result = agent.llm_node(turn_ctx, [], object())

    assert isinstance(result, str)
    assert "two years" in result
    assert "evidence of insurability" not in result


async def test_support_agent_llm_node_keeps_direct_answer_in_turn_context(
    tmp_path: Path,
) -> None:
    kb = KnowledgeBase(tmp_path / "chroma", "test-collection")
    kb.upsert_documents(TEST_DOCUMENTS)
    agent = SupportAgent(RetrievalService(kb))
    turn_ctx = llm.ChatContext.empty()

    new_message = turn_ctx.add_message(
        role="user",
        content="If I return the policy within ten days, do I get my premium back?",
    )
    await agent.on_user_turn_completed(turn_ctx, new_message)

    first = agent.llm_node(turn_ctx, [], object())
    second = agent.llm_node(turn_ctx, [], object())

    assert isinstance(first, str)
    assert first == second


async def test_support_agent_llm_node_recovers_direct_answer_when_guidance_is_missing(
    tmp_path: Path,
) -> None:
    kb = KnowledgeBase(tmp_path / "chroma", "test-collection")
    kb.upsert_documents(
        [
            KnowledgeDocument(
                id="reinstatement",
                title="Policy reinstatement after lapse",
                content=(
                    "The policy may be reinstated subject to the following conditions: "
                    "1. You must make a written request for reinstatement within two years after the end of the grace period. "
                    "2. Every insured person aged sixteen or older must provide evidence of insurability."
                ),
                metadata={"source_url": "https://example.com/policy.pdf"},
            )
        ]
    )
    agent = SupportAgent(RetrievalService(kb))
    turn_ctx = llm.ChatContext.empty()
    turn_ctx.add_message(
        role="user",
        content="Can I reinstate the policy after it ends because I stopped paying premiums?",
    )

    result = agent.llm_node(turn_ctx.copy(), [], object())
    if asyncio.iscoroutine(result):
        result = await result

    assert isinstance(result, str)
    assert "two years" in result


async def test_support_agent_llm_node_uses_direct_answer_for_unsupported_policy_query(
    tmp_path: Path,
) -> None:
    kb = KnowledgeBase(tmp_path / "chroma", "test-collection")
    agent = SupportAgent(RetrievalService(kb))
    turn_ctx = llm.ChatContext.empty()

    new_message = turn_ctx.add_message(
        role="user",
        content="What is the CEO of the insurer?",
    )
    await agent.on_user_turn_completed(turn_ctx, new_message)

    result = agent.llm_node(turn_ctx, [], object())

    assert result == UNSUPPORTED_POLICY_RESPONSE
    developer_message = turn_ctx.messages()[-1]
    assert developer_message.extra["direct_answer"] == UNSUPPORTED_POLICY_RESPONSE


async def test_support_agent_llm_node_recovers_unsupported_policy_answer_without_guidance(
    tmp_path: Path,
) -> None:
    kb = KnowledgeBase(tmp_path / "chroma", "test-collection")
    agent = SupportAgent(RetrievalService(kb))
    turn_ctx = llm.ChatContext.empty()
    turn_ctx.add_message(
        role="user",
        content="What is the CEO of the insurer?",
    )

    result = agent.llm_node(turn_ctx.copy(), [], object())
    if asyncio.iscoroutine(result):
        result = await result

    assert result == UNSUPPORTED_POLICY_RESPONSE


@pytest.mark.asyncio
async def test_run_load_surfaces_failure_examples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummySettings:
        load_test_concurrency = 2
        load_test_turns = 1

    async def failing_run_single_session(case: dict[str, object]) -> dict[str, object]:
        raise RuntimeError(f"failed for {case['question']}")

    monkeypatch.setattr("voice_agent.load_test.load_settings", lambda: DummySettings())
    monkeypatch.setattr(
        "voice_agent.load_test.run_single_session",
        failing_run_single_session,
    )

    result = await run_load()

    assert result["failures"] == 2
    assert result["successes"] == 0
    assert "failure_examples" in result
    assert "RuntimeError: failed for" in str(result["failure_examples"])


@pytest.mark.asyncio
async def test_run_load_respects_explicit_concurrency_and_turn_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummySettings:
        load_test_concurrency = 99
        load_test_turns = 99

    async def successful_run_single_session(case: dict[str, object]) -> dict[str, object]:
        return {
            "question": case["question"],
            "round_trip_success": True,
            "response_latency_ms": 125.0,
        }

    monkeypatch.setattr("voice_agent.load_test.load_settings", lambda: DummySettings())
    monkeypatch.setattr(
        "voice_agent.load_test.run_single_session",
        successful_run_single_session,
    )

    result = await run_load(concurrency=2, turns=3)

    assert result["concurrency"] == 2
    assert result["sessions"] == 6
    assert result["successes"] == 6
    assert result["failures"] == 0
    assert result["avg_ms"] == 125.0


@pytest.mark.asyncio
async def test_run_stepped_load_returns_one_summary_per_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummySettings:
        load_test_concurrency = 1
        load_test_turns = 1

    async def successful_run_single_session(case: dict[str, object]) -> dict[str, object]:
        return {
            "question": case["question"],
            "round_trip_success": True,
            "response_latency_ms": 100.0,
        }

    monkeypatch.setattr("voice_agent.load_test.load_settings", lambda: DummySettings())
    monkeypatch.setattr(
        "voice_agent.load_test.run_single_session",
        successful_run_single_session,
    )

    result = await run_stepped_load([1, 3], turns=2)

    assert [summary["concurrency"] for summary in result] == [1, 3]
    assert [summary["sessions"] for summary in result] == [2, 6]
    assert all(summary["successes"] == summary["sessions"] for summary in result)


@pytest.mark.asyncio
async def test_run_single_session_uses_case_specific_expected_keywords(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_run_smoke(
        *,
        question_text: str,
        room_name_prefix: str,
        expected_keywords: list[str] | None = None,
        forbidden_keywords: list[str] | None = None,
    ) -> dict[str, object]:
        captured["question_text"] = question_text
        captured["room_name_prefix"] = room_name_prefix
        captured["expected_keywords"] = expected_keywords
        captured["forbidden_keywords"] = forbidden_keywords
        return {
            "round_trip_success": True,
            "response_latency_ms": 100.0,
        }

    monkeypatch.setattr("voice_agent.load_test.run_smoke", fake_run_smoke)

    await run_single_session(
        {
            "question": "How long is the grace period if I miss a premium payment?",
            "expected_keywords": ["grace", "31", "days"],
            "forbidden_keywords": ["thirty"],
        }
    )

    assert captured == {
        "question_text": "How long is the grace period if I miss a premium payment?",
        "room_name_prefix": "voice-agent-load-test",
        "expected_keywords": ["grace", "31", "days"],
        "forbidden_keywords": ["thirty"],
    }


@pytest.mark.asyncio
async def test_watchdog_triggers_callback_once() -> None:
    fired: list[str] = []

    task = start_watchdog(
        None,
        timeout_seconds=0.01,
        on_timeout=lambda: fired.append("timeout"),
    )

    await task

    assert fired == ["timeout"]


@pytest.mark.asyncio
async def test_cancel_watchdog_prevents_timeout_callback() -> None:
    fired: list[str] = []

    task = start_watchdog(
        None,
        timeout_seconds=0.05,
        on_timeout=lambda: fired.append("timeout"),
    )
    cancel_watchdog(task)
    await asyncio.sleep(0)

    assert fired == []


@pytest.mark.asyncio
async def test_session_failure_policy_repompts_then_escalates_for_stt_timeouts() -> None:
    session = FakeSession()
    policy = SessionFailurePolicy(session)  # type: ignore[arg-type]

    first = policy.handle_stt_timeout("hello")
    second = policy.handle_stt_timeout("hello again")

    assert first is not None
    assert second is not None
    await asyncio.gather(first, second)

    assert session.calls[0]["text"] == "I didn't catch that. Please say that again."
    assert "human representative" in str(session.calls[1]["text"])


@pytest.mark.asyncio
async def test_session_failure_policy_resets_stt_timeout_escalation_after_success() -> None:
    session = FakeSession()
    policy = SessionFailurePolicy(session)  # type: ignore[arg-type]

    first = policy.handle_stt_timeout("hello")
    assert first is not None
    await first

    policy.reset_stt_timeout_count()

    second = policy.handle_stt_timeout("hello again")
    assert second is not None
    await second

    assert [call["text"] for call in session.calls] == [
        "I didn't catch that. Please say that again.",
        "I didn't catch that. Please say that again.",
    ]


@pytest.mark.asyncio
async def test_session_failure_policy_announces_llm_failure_once() -> None:
    session = FakeSession()
    policy = SessionFailurePolicy(session)  # type: ignore[arg-type]

    first = policy.handle_llm_failure()
    second = policy.handle_llm_failure()

    assert first is not None
    assert second is None
    await first

    assert len(session.calls) == 1
    assert "trouble responding" in str(session.calls[0]["text"])


@pytest.mark.asyncio
async def test_session_failure_policy_retries_tts_once_per_content() -> None:
    session = FakeSession()
    policy = SessionFailurePolicy(session)  # type: ignore[arg-type]

    first = policy.handle_tts_failure("Answer text")
    second = policy.handle_tts_failure("Answer text")
    third = policy.handle_tts_failure("Different answer text")

    assert first is not None
    assert second is None
    assert third is not None
    await asyncio.gather(first, third)

    assert [call["text"] for call in session.calls] == [
        "Answer text",
        "Different answer text",
    ]
