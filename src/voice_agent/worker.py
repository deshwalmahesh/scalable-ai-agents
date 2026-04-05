from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import ssl

import aiohttp
import certifi
from dotenv import load_dotenv
from google.genai import types as google_types
import httpx
from livekit import rtc
from livekit.agents import (
    APIConnectOptions,
    Agent,
    AgentServer,
    AgentSession,
    CloseEvent,
    ConversationItemAddedEvent,
    ErrorEvent,
    JobContext,
    JobProcess,
    cli,
    llm,
    room_io,
    stt as agent_stt,
    tts as agent_tts,
    UserInputTranscribedEvent,
)
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.plugins import cartesia, deepgram, google, noise_cancellation, openai, silero

from voice_agent.kb import (
    KnowledgeBase,
    RetrievalService,
    UNSUPPORTED_POLICY_RESPONSE,
    select_grounding_snippet,
)
from voice_agent.observability import (
    ActiveSessionsStore,
    LiveKitTurnMetricsCollector,
    RetrievalLatencySnapshot,
    TurnMetricsLog,
    VoiceAgentMetrics,
)
from voice_agent.prompts import build_system_prompt
from voice_agent.retrieval import warm_local_model
from voice_agent.settings import AppSettings, load_settings

load_dotenv(".env")
_ORIGINAL_CREATE_DEFAULT_CONTEXT = ssl.create_default_context
_ORIGINAL_CLIENT_SESSION_INIT = aiohttp.ClientSession.__init__


def _certifi_default_context(*args: object, **kwargs: object) -> ssl.SSLContext:
    if "cafile" not in kwargs and "capath" not in kwargs and "cadata" not in kwargs:
        kwargs["cafile"] = certifi.where()
    return _ORIGINAL_CREATE_DEFAULT_CONTEXT(*args, **kwargs)


def _patched_client_session_init(
    self: aiohttp.ClientSession,
    *args: object,
    **kwargs: object,
) -> None:
    if "connector" not in kwargs:
        kwargs["connector"] = aiohttp.TCPConnector(
            ssl=_certifi_default_context(),
        )
    _ORIGINAL_CLIENT_SESSION_INIT(self, *args, **kwargs)


# aiohttp's own docs recommend supplying a TCPConnector with a certifi-backed
# SSL context when the system CA store is incomplete. LiveKit's AgentServer
# does not expose session injection, so we patch ClientSession's constructor
# to inject that connector when one is not already provided.
ssl.create_default_context = _certifi_default_context
aiohttp.ClientSession.__init__ = _patched_client_session_init
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

_RUNTIME_SETTINGS: AppSettings | None = None
_RUNTIME_KB: KnowledgeBase | None = None
_RUNTIME_RETRIEVAL: RetrievalService | None = None
_RUNTIME_METRICS: VoiceAgentMetrics | None = None
logger = logging.getLogger(__name__)
STT_FINAL_TIMEOUT_SECONDS = 3.0
STT_FINAL_TIMEOUT_PER_WORD_SECONDS = 0.3
STT_FINAL_TIMEOUT_MAX_SECONDS = 6.0
TURN_SCOPED_GUIDANCE_PREFIXES = (
    "Grounding for this turn only.",
    "This user turn is conversational small talk or a connection check.",
    "Knowledge retrieval is temporarily unavailable.",
)


@dataclass(slots=True)
class WorkerRuntime:
    server: AgentServer
    agent_name: str


class SessionFailurePolicy:
    def __init__(self, session: AgentSession) -> None:
        self._session = session
        self._closed = False
        self._pending_tasks: set[asyncio.Task[None]] = set()
        self._stt_timeout_count = 0
        self._llm_failure_announced = False
        self._last_tts_retry_content: str | None = None

    def close(self) -> None:
        self._closed = True
        for task in list(self._pending_tasks):
            task.cancel()

    @property
    def closed(self) -> bool:
        return self._closed

    def handle_stt_timeout(self, transcript: str) -> asyncio.Task[None] | None:
        if self._closed:
            return None
        self._stt_timeout_count += 1
        if self._stt_timeout_count == 1:
            return self._schedule_say("I didn't catch that. Please say that again.")
        return self._schedule_say(
            "I'm having trouble hearing you right now. Please try again later or speak with a human representative."
        )

    def reset_stt_timeout_count(self) -> None:
        self._stt_timeout_count = 0

    def handle_llm_failure(self) -> asyncio.Task[None] | None:
        if self._closed or self._llm_failure_announced:
            return None
        self._llm_failure_announced = True
        return self._schedule_say(
            "I'm having trouble responding right now. Please try again or speak with a human representative."
        )

    def handle_tts_failure(self, content: str) -> asyncio.Task[None] | None:
        normalized = content.strip()
        if self._closed or not normalized:
            return None
        if self._last_tts_retry_content == normalized:
            return None
        self._last_tts_retry_content = normalized
        return self._schedule_say(normalized)

    def _schedule_say(self, text: str) -> asyncio.Task[None]:
        task = asyncio.create_task(self._run_say(text))
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
        return task

    async def _run_say(self, text: str) -> None:
        try:
            await self._session.say(
                text,
                allow_interruptions=True,
                add_to_chat_ctx=False,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Failure-policy speech output failed text=%s", text)


def classify_upstream_source(source: object) -> str:
    if isinstance(source, agent_stt.STT):
        return "stt"
    if isinstance(source, agent_tts.TTS):
        return "tts"
    if isinstance(source, llm.LLM):
        return "llm"
    if isinstance(source, llm.RealtimeModel):
        return "realtime"
    return "session"


def cancel_watchdog(task: asyncio.Task[None] | None) -> None:
    if task is not None and not task.done():
        task.cancel()


def start_watchdog(
    current_task: asyncio.Task[None] | None,
    *,
    timeout_seconds: float,
    on_timeout: Callable[[], None],
) -> asyncio.Task[None]:
    cancel_watchdog(current_task)

    async def watchdog() -> None:
        try:
            await asyncio.sleep(timeout_seconds)
            on_timeout()
        except asyncio.CancelledError:
            return

    return asyncio.create_task(watchdog())


def record_stt_timeout(metrics: VoiceAgentMetrics, transcript: str) -> str:
    component = "stt_timeout"
    metrics.record_upstream_failure(component)
    logger.warning(
        "STT final transcript timeout component=%s last_partial_transcript=%s",
        component,
        transcript,
    )
    return component


def stt_timeout_seconds_for_transcript(transcript: str) -> float:
    word_count = len([word for word in transcript.strip().split() if word])
    adaptive_timeout = STT_FINAL_TIMEOUT_SECONDS + (
        word_count * STT_FINAL_TIMEOUT_PER_WORD_SECONDS
    )
    return min(STT_FINAL_TIMEOUT_MAX_SECONDS, adaptive_timeout)


class SupportAgent(Agent):
    def __init__(
        self,
        retrieval_service: RetrievalService,
        metrics: VoiceAgentMetrics | None = None,
    ) -> None:
        super().__init__(
            instructions=build_system_prompt(),
            allow_interruptions=True,
            # The earlier 0.15/0.6 split longer policy questions into multiple user turns
            # during live audio runs. Keep endpointing responsive, but give natural pauses
            # inside one sentence enough room to finish before replying.
            min_endpointing_delay=0.35,
            max_endpointing_delay=1.2,
        )
        self._retrieval_service = retrieval_service
        self._metrics = metrics

    async def on_user_turn_completed(
        self,
        turn_ctx: llm.ChatContext,
        new_message: llm.ChatMessage,
    ) -> None:
        query = self._build_grounding_query(turn_ctx, new_message)
        if not query:
            return

        if not self._retrieval_service.should_ground(query):
            logger.info("Knowledge retrieval skipped query=%s reason=non_policy_turn", query)
            self._replace_turn_scoped_guidance(
                turn_ctx,
                content=(
                    "This user turn is conversational small talk or a connection check. "
                    "Answer briefly and naturally without citing policy facts."
                ),
                direct_answer=None,
            )
            return

        logger.info("Knowledge retrieval started query=%s", query)
        try:
            retrieval = await self._retrieval_service.async_ground(query, limit=1)
        except Exception:
            if self._metrics is not None:
                self._metrics.record_upstream_failure("retrieval")
            logger.exception("Knowledge retrieval failed query=%s", query)
            self._replace_turn_scoped_guidance(
                turn_ctx,
                content=(
                    "Knowledge retrieval is temporarily unavailable. Do not invent policy facts. "
                    "Apologize briefly and ask the caller to try again or speak with a human agent."
                ),
                direct_answer=None,
            )
            return
        trace = retrieval.trace
        if self._metrics is not None:
            self._metrics.record_retrieval_latency(
                RetrievalLatencySnapshot(
                    embedding_ms=trace.embedding_ms,
                    vector_query_ms=trace.vector_query_ms,
                    document_load_ms=trace.document_load_ms,
                    rerank_ms=trace.rerank_ms,
                    total_ms=trace.total_ms,
                )
            )
        logger.info(
            "Knowledge retrieval completed query=%s match_count=%s match_titles=%s retrieval_trace=%s",
            query,
            len(retrieval.matches),
            [match.title[:120] for match in retrieval.matches],
            {
                "mode": trace.mode,
                "embedding_ms": trace.embedding_ms,
                "vector_query_ms": trace.vector_query_ms,
                "document_load_ms": trace.document_load_ms,
                "rerank_ms": trace.rerank_ms,
                "total_ms": trace.total_ms,
            },
        )
        self._replace_turn_scoped_guidance(
            turn_ctx,
            content=retrieval.grounding_message,
            direct_answer=(
                select_grounding_snippet(query, retrieval.matches[0].content)
                if retrieval.matches
                else UNSUPPORTED_POLICY_RESPONSE
            ),
        )
        logger.info(
            "Prepared direct grounded answer query=%s direct_answer=%s",
            query,
            self._turn_scoped_direct_answer(turn_ctx),
        )

    @staticmethod
    def _replace_turn_scoped_guidance(
        turn_ctx: llm.ChatContext,
        *,
        content: str,
        direct_answer: str | None,
    ) -> None:
        filtered_items: list[llm.ChatItem] = []
        for item in turn_ctx.items:
            if item.type != "message" or item.role != "developer":
                filtered_items.append(item)
                continue
            text = (item.text_content or "").strip()
            if item.extra.get("turn_scoped_guidance") is True or text.startswith(
                TURN_SCOPED_GUIDANCE_PREFIXES
            ):
                continue
            filtered_items.append(item)

        turn_ctx.items = filtered_items
        turn_ctx.add_message(
            role="developer",
            content=content,
            extra={
                "turn_scoped_guidance": True,
                "direct_answer": direct_answer,
            },
        )

    @staticmethod
    def _turn_scoped_direct_answer(chat_ctx: llm.ChatContext) -> str | None:
        for item in reversed(chat_ctx.items):
            if item.type != "message" or item.role != "developer":
                continue
            if item.extra.get("turn_scoped_guidance") is not True:
                continue
            direct_answer = item.extra.get("direct_answer")
            if isinstance(direct_answer, str) and direct_answer.strip():
                return direct_answer
            return None
        return None

    def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: object,
    ) -> object:
        direct_answer = self._turn_scoped_direct_answer(chat_ctx)
        if direct_answer:
            logger.info("Using direct grounded answer direct_answer=%s", direct_answer)
            return direct_answer

        async def resolve_reply() -> object:
            latest_user_message = self._latest_user_message(chat_ctx)
            if latest_user_message is not None:
                query = self._build_grounding_query(chat_ctx, latest_user_message)
                if query and self._retrieval_service.should_ground(query):
                    logger.info("Recovering grounded answer in llm_node query=%s", query)
                    retrieval = await self._retrieval_service.async_ground(query, limit=1)
                    self._replace_turn_scoped_guidance(
                        chat_ctx,
                        content=retrieval.grounding_message,
                        direct_answer=(
                            select_grounding_snippet(query, retrieval.matches[0].content)
                            if retrieval.matches
                            else UNSUPPORTED_POLICY_RESPONSE
                        ),
                    )
                    recovered_direct_answer = self._turn_scoped_direct_answer(chat_ctx)
                    if recovered_direct_answer:
                        logger.info(
                            "Using recovered direct grounded answer direct_answer=%s",
                            recovered_direct_answer,
                        )
                        return recovered_direct_answer

            result = super(SupportAgent, self).llm_node(chat_ctx, tools, model_settings)
            if asyncio.iscoroutine(result):
                return await result
            return result

        return resolve_reply()

    @staticmethod
    def _latest_user_message(chat_ctx: llm.ChatContext) -> llm.ChatMessage | None:
        for message in reversed(chat_ctx.messages()):
            if message.role == "user" and (message.text_content or "").strip():
                return message
        return None

    @staticmethod
    def _build_grounding_query(
        turn_ctx: llm.ChatContext,
        new_message: llm.ChatMessage,
    ) -> str:
        latest_text = (new_message.text_content or "").strip()
        if not latest_text:
            return ""

        fragments: list[str] = []
        seen_new_message = False
        for message in reversed(turn_ctx.messages()):
            if message is new_message:
                seen_new_message = True
            if not seen_new_message:
                continue
            if message.role == "assistant":
                break
            if message.role != "user":
                continue
            text = (message.text_content or "").strip()
            if text:
                fragments.append(text)

        fragments.reverse()
        return " ".join(fragments).strip()


def create_tts(settings: AppSettings) -> deepgram.TTS | cartesia.TTS:
    if settings.tts_provider == "cartesia":
        logger.info("Using Cartesia TTS provider")
        return cartesia.TTS()
    logger.info("Using Deepgram TTS provider model=aura-2-asteria-en")
    return deepgram.TTS(model="aura-2-asteria-en")


def _create_openai_compatible_llm(
    *,
    model: str,
    api_key: str,
    base_url: str,
    timeout_seconds: float,
    max_retries: int,
) -> openai.LLM:
    return openai.LLM(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        timeout=httpx.Timeout(timeout_seconds),
        max_retries=max_retries,
    )


def create_llm(settings: AppSettings) -> llm.LLM:
    # This is intentionally OpenAI-compatible so the same code works with Gemini,
    # Cerebras, Groq, or any provider that speaks the OpenAI chat-completions shape.
    primary = _create_openai_compatible_llm(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout_seconds=settings.http_timeout_seconds,
        max_retries=settings.agent_max_retries,
    )
    if not settings.fallback_openai_base_url:
        return primary

    fallback = _create_openai_compatible_llm(
        model=settings.fallback_openai_model,
        api_key=settings.fallback_openai_api_key,
        base_url=settings.fallback_openai_base_url,
        timeout_seconds=settings.http_timeout_seconds,
        max_retries=settings.agent_max_retries,
    )
    logger.info(
        "Configured LLM fallback primary_model=%s fallback_model=%s",
        settings.openai_model,
        settings.fallback_openai_model,
    )
    return llm.FallbackAdapter(
        [primary, fallback],
        attempt_timeout=settings.http_timeout_seconds,
        max_retry_per_llm=0,
    )


def configure_llm_failover_logging(
    model: llm.LLM,
    metrics: VoiceAgentMetrics,
) -> None:
    if not isinstance(model, llm.FallbackAdapter):
        return

    @model.on("llm_availability_changed")
    def on_llm_availability_changed(event: llm.AvailabilityChangedEvent) -> None:
        metrics.record_llm_availability_change(
            provider=event.llm.provider,
            model=event.llm.model,
            available=event.available,
        )
        log_fn = logger.info if event.available else logger.warning
        log_fn(
            "LLM availability changed provider=%s model=%s available=%s",
            event.llm.provider,
            event.llm.model,
            event.available,
        )


def handle_session_error_event(
    metrics: VoiceAgentMetrics,
    event: ErrorEvent,
) -> str:
    component = classify_upstream_source(event.source)
    metrics.record_upstream_failure(component)
    logger.exception(
        "Agent session error component=%s error_type=%s error=%s",
        component,
        type(event.error).__name__,
        event.error,
    )
    return component


def handle_session_close_event(
    metrics: VoiceAgentMetrics,
    event: CloseEvent,
) -> str | None:
    if event.error is None:
        logger.info("Agent session closed reason=%s", event.reason)
        return None
    component = "session"
    metrics.record_upstream_failure(component)
    logger.warning(
        "Agent session closed with error reason=%s error_type=%s error=%s",
        event.reason,
        type(event.error).__name__,
        event.error,
    )
    return component


def create_gemini_live_model(settings: AppSettings) -> google.realtime.RealtimeModel:
    modalities = [
        getattr(google_types.Modality, modality.strip().upper())
        for modality in settings.gemini_live_modalities.split(",")
        if modality.strip()
    ]
    realtime_kwargs: dict[str, object] = {
        "model": settings.gemini_live_model,
        "api_key": settings.google_api_key,
        "voice": settings.gemini_live_voice,
        "modalities": modalities,
        "instructions": build_system_prompt(),
    }
    if not settings.gemini_live_include_thoughts:
        realtime_kwargs["thinking_config"] = google_types.ThinkingConfig(
            include_thoughts=False,
        )
    return google.realtime.RealtimeModel(**realtime_kwargs)


def create_session_connect_options(settings: AppSettings) -> SessionConnectOptions:
    timeout = settings.http_timeout_seconds
    return SessionConnectOptions(
        llm_conn_options=APIConnectOptions(max_retry=0, timeout=timeout),
        stt_conn_options=APIConnectOptions(timeout=timeout),
        tts_conn_options=APIConnectOptions(timeout=timeout),
    )


def prewarm(proc: JobProcess) -> None:
    settings = get_runtime_settings()
    retrieval_service = get_runtime_retrieval()
    document_count = retrieval_service.count()
    if settings.embedding_provider == "local":
        warm_local_model()
    if settings.voice_runtime_mode == "cascade":
        proc.userdata["vad"] = silero.VAD.load()
    logger.info("Worker prewarm resolved knowledge documents document_count=%s", document_count)


def get_runtime_settings() -> AppSettings:
    global _RUNTIME_SETTINGS
    if _RUNTIME_SETTINGS is None:
        _RUNTIME_SETTINGS = load_settings()
    return _RUNTIME_SETTINGS


def get_runtime_kb() -> KnowledgeBase:
    global _RUNTIME_KB
    if _RUNTIME_KB is None:
        _RUNTIME_KB = KnowledgeBase.from_settings(get_runtime_settings())
    return _RUNTIME_KB


def get_runtime_retrieval() -> RetrievalService:
    global _RUNTIME_RETRIEVAL
    if _RUNTIME_RETRIEVAL is None:
        _RUNTIME_RETRIEVAL = RetrievalService(get_runtime_kb())
    return _RUNTIME_RETRIEVAL


def get_runtime_metrics() -> VoiceAgentMetrics:
    global _RUNTIME_METRICS
    if _RUNTIME_METRICS is None:
        settings = get_runtime_settings()
        _RUNTIME_METRICS = VoiceAgentMetrics(
            turn_metrics_log=TurnMetricsLog(Path(settings.turn_metrics_path)),
            active_sessions_store=ActiveSessionsStore(
                Path(settings.active_sessions_path),
                redis_url=settings.redis_url,
                redis_key=settings.active_sessions_key,
            ),
        )
    return _RUNTIME_METRICS


async def handle_session(ctx: JobContext) -> None:
    settings = get_runtime_settings()
    retrieval_service = get_runtime_retrieval()
    metrics = get_runtime_metrics()
    logger.info(
        "Starting agent session room_name=%s agent_name=%s kb_documents=%s",
        ctx.room.name,
        settings.agent_name,
        retrieval_service.count(),
    )
    active_sessions = metrics.increment_active_sessions()
    logger.info(
        "Active session started room_name=%s active_sessions=%s",
        ctx.room.name,
        active_sessions,
    )
    if settings.voice_runtime_mode == "gemini_live":
        session = AgentSession(
            llm=create_gemini_live_model(settings),
            conn_options=create_session_connect_options(settings),
        )
        logger.info(
            "Configured Gemini Live session room_name=%s model=%s voice=%s modalities=%s",
            ctx.room.name,
            settings.gemini_live_model,
            settings.gemini_live_voice,
            settings.gemini_live_modalities,
        )
    else:
        llm_model = create_llm(settings)
        configure_llm_failover_logging(llm_model, metrics)
        session = AgentSession(
            stt=deepgram.STT(
                model="nova-3",
                language="en-US",
                interim_results=True,
                smart_format=False,
            ),
            llm=llm_model,
            tts=create_tts(settings),
            vad=ctx.proc.userdata["vad"],
            turn_detection="vad",
            conn_options=create_session_connect_options(settings),
        )
    failure_policy = SessionFailurePolicy(session)
    stt_watchdog: asyncio.Task[None] | None = None
    last_partial_transcript = ""
    last_assistant_content = ""
    session_closed = False
    metrics_collector = LiveKitTurnMetricsCollector(metrics)

    def on_stt_timeout() -> None:
        if session_closed or failure_policy.closed:
            return
        record_stt_timeout(metrics, last_partial_transcript)
        failure_policy.handle_stt_timeout(last_partial_transcript)

    @session.on("metrics_collected")
    def on_metrics_collected(event: object) -> None:
        metric = getattr(event, "metrics", None)
        if metric is not None:
            metrics_collector.observe(metric)

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(event: UserInputTranscribedEvent) -> None:
        nonlocal stt_watchdog, last_partial_transcript
        logger.info(
            "User input transcribed transcript=%s is_final=%s",
            event.transcript,
            event.is_final,
        )
        transcript = event.transcript.strip()
        if event.is_final:
            last_partial_transcript = ""
            cancel_watchdog(stt_watchdog)
            stt_watchdog = None
            failure_policy.reset_stt_timeout_count()
            return
        if transcript:
            last_partial_transcript = transcript
            stt_watchdog = start_watchdog(
                stt_watchdog,
                timeout_seconds=stt_timeout_seconds_for_transcript(transcript),
                on_timeout=on_stt_timeout,
            )

    @session.on("conversation_item_added")
    def on_conversation_item_added(event: ConversationItemAddedEvent) -> None:
        nonlocal last_assistant_content
        item = event.item
        role = getattr(item, "role", "unknown")
        content = getattr(item, "text_content", None)
        if content is None:
            content = str(item)
        logger.info(
            "Conversation item added role=%s content=%s",
            role,
            content,
        )
        if role == "assistant" and str(content).strip():
            last_assistant_content = str(content)

    @session.on("error")
    def on_error(event: ErrorEvent) -> None:
        component = handle_session_error_event(metrics, event)
        if component == "stt":
            cancel_watchdog(stt_watchdog)
            failure_policy.handle_stt_timeout(last_partial_transcript)
        elif component == "tts":
            failure_policy.handle_tts_failure(last_assistant_content)
        elif component in {"llm", "realtime"}:
            failure_policy.handle_llm_failure()

    @session.on("close")
    def on_close(event: CloseEvent) -> None:
        nonlocal session_closed
        session_closed = True
        handle_session_close_event(metrics, event)
        cancel_watchdog(stt_watchdog)
        failure_policy.close()

    try:
        await session.start(
            room=ctx.room,
            agent=SupportAgent(retrieval_service, metrics=metrics),
            room_options=room_io.RoomOptions(
                audio_input=room_io.AudioInputOptions(
                    noise_cancellation=lambda params: (
                        noise_cancellation.BVCTelephony()
                        if params.participant.kind
                        == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                        else noise_cancellation.BVC()
                    )
                )
            ),
        )
    except Exception:
        metrics.record_upstream_failure("session_start")
        metrics.record_call_setup_failure()
        raise
    logger.info(
        "Skipping explicit startup greeting room_name=%s runtime_mode=%s",
        ctx.room.name,
        settings.voice_runtime_mode,
    )


async def handle_session_end(ctx: JobContext) -> None:
    metrics = get_runtime_metrics()
    active_sessions = metrics.decrement_active_sessions()
    logger.info(
        "Active session ended room_name=%s active_sessions=%s",
        ctx.room.name,
        active_sessions,
    )


def create_runtime(
    settings: AppSettings | None = None,
    knowledge_base: KnowledgeBase | None = None,
) -> WorkerRuntime:
    global _RUNTIME_SETTINGS, _RUNTIME_KB, _RUNTIME_RETRIEVAL, _RUNTIME_METRICS
    resolved_settings = settings or load_settings()
    active_sessions_store = ActiveSessionsStore(
        Path(resolved_settings.active_sessions_path),
        redis_url=resolved_settings.redis_url,
        redis_key=resolved_settings.active_sessions_key,
    )
    # The local file-backed counter can inherit stale state between worker runs.
    # Reset it on startup for honest local validation, but do not blindly zero a
    # shared Redis counter when multiple workers may already be live.
    if active_sessions_store.uses_local_file:
        active_sessions_store.write(0)
    _RUNTIME_SETTINGS = resolved_settings
    _RUNTIME_KB = knowledge_base
    _RUNTIME_RETRIEVAL = None
    _RUNTIME_METRICS = VoiceAgentMetrics(
        turn_metrics_log=TurnMetricsLog(Path(resolved_settings.turn_metrics_path)),
        active_sessions_store=active_sessions_store,
    )

    server_kwargs: dict[str, object] = {
        "port": resolved_settings.worker_port,
        "load_threshold": 1.0,
    }
    if resolved_settings.embedding_provider == "local":
        server_kwargs["num_idle_processes"] = 1
        server_kwargs["initialize_process_timeout"] = 30.0

    server = AgentServer(**server_kwargs)
    max_sessions = resolved_settings.worker_max_concurrent_sessions

    def worker_load(current_server: AgentServer) -> float:
        active_jobs = len(current_server.active_jobs)
        return active_jobs / max_sessions

    server.setup_fnc = prewarm
    server.load_fnc = worker_load
    server.rtc_session(
        agent_name=resolved_settings.agent_name,
        on_session_end=handle_session_end,
    )(handle_session)

    return WorkerRuntime(server=server, agent_name=resolved_settings.agent_name)


def run_worker() -> None:
    settings = load_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    runtime = create_runtime()
    cli.run_app(runtime.server)
