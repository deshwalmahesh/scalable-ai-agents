from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import ssl

import aiohttp
import certifi
from dotenv import load_dotenv
from google.genai import types as google_types
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ConversationItemAddedEvent,
    JobContext,
    JobProcess,
    cli,
    llm,
    room_io,
    UserInputTranscribedEvent,
)
from livekit.plugins import cartesia, deepgram, google, noise_cancellation, openai, silero

from voice_agent.kb import KnowledgeBase, RetrievalService
from voice_agent.observability import (
    ActiveSessionsStore,
    LiveKitTurnMetricsCollector,
    RetrievalLatencySnapshot,
    TurnMetricsLog,
    VoiceAgentMetrics,
)
from voice_agent.prompts import build_system_prompt
from voice_agent.settings import AppSettings, load_settings

load_dotenv(".env")
_ORIGINAL_CREATE_DEFAULT_CONTEXT = ssl.create_default_context
_ORIGINAL_CLIENT_SESSION = aiohttp.ClientSession


def _certifi_default_context(*args: object, **kwargs: object) -> ssl.SSLContext:
    if "cafile" not in kwargs and "capath" not in kwargs and "cadata" not in kwargs:
        kwargs["cafile"] = certifi.where()
    return _ORIGINAL_CREATE_DEFAULT_CONTEXT(*args, **kwargs)


class _CertifiClientSession(_ORIGINAL_CLIENT_SESSION):
    def __init__(self, *args: object, **kwargs: object) -> None:
        if "connector" not in kwargs:
            kwargs["connector"] = aiohttp.TCPConnector(
                ssl=_certifi_default_context(),
            )
        super().__init__(*args, **kwargs)


# aiohttp's own docs recommend supplying a TCPConnector with a certifi-backed
# SSL context when the system CA store is incomplete. LiveKit's AgentServer
# does not expose session injection, so we patch ClientSession with a subclass
# that preserves aiohttp's class interface while supplying that connector.
ssl.create_default_context = _certifi_default_context
aiohttp.ClientSession = _CertifiClientSession
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

_RUNTIME_SETTINGS: AppSettings | None = None
_RUNTIME_KB: KnowledgeBase | None = None
_RUNTIME_RETRIEVAL: RetrievalService | None = None
_RUNTIME_METRICS: VoiceAgentMetrics | None = None
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WorkerRuntime:
    server: AgentServer
    agent_name: str


class SupportAgent(Agent):
    def __init__(
        self,
        retrieval_service: RetrievalService,
        metrics: VoiceAgentMetrics | None = None,
    ) -> None:
        super().__init__(
            instructions=build_system_prompt(),
            allow_interruptions=True,
            min_endpointing_delay=0.15,
            max_endpointing_delay=0.6,
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
            turn_ctx.add_message(
                role="developer",
                content=(
                    "This user turn is conversational small talk or a connection check. "
                    "Answer briefly and naturally without citing policy facts."
                ),
            )
            return

        logger.info("Knowledge retrieval started query=%s", query)
        retrieval = await self._retrieval_service.async_ground(query, limit=1)
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
        turn_ctx.add_message(
            role="developer",
            content=retrieval.grounding_message,
        )

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


def create_llm(settings: AppSettings) -> openai.LLM:
    # This is intentionally OpenAI-compatible so the same code works with Gemini,
    # vLLM, or any provider that speaks the OpenAI chat-completions shape.
    return openai.LLM(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


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


def prewarm(proc: JobProcess) -> None:
    retrieval_service = get_runtime_retrieval()
    document_count = retrieval_service.count()
    if get_runtime_settings().voice_runtime_mode == "cascade":
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
            active_sessions_store=ActiveSessionsStore(Path(settings.active_sessions_path)),
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
        )
        logger.info(
            "Configured Gemini Live session room_name=%s model=%s voice=%s modalities=%s",
            ctx.room.name,
            settings.gemini_live_model,
            settings.gemini_live_voice,
            settings.gemini_live_modalities,
        )
    else:
        session = AgentSession(
            stt=deepgram.STT(
                model="nova-3",
                language="en-US",
                interim_results=True,
                smart_format=False,
            ),
            llm=create_llm(settings),
            tts=create_tts(settings),
            vad=ctx.proc.userdata["vad"],
            turn_detection="vad",
        )
    metrics_collector = LiveKitTurnMetricsCollector(metrics)

    @session.on("metrics_collected")
    def on_metrics_collected(event: object) -> None:
        metric = getattr(event, "metrics", None)
        if metric is not None:
            metrics_collector.observe(metric)

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(event: UserInputTranscribedEvent) -> None:
        logger.info(
            "User input transcribed transcript=%s is_final=%s",
            event.transcript,
            event.is_final,
        )

    @session.on("conversation_item_added")
    def on_conversation_item_added(event: ConversationItemAddedEvent) -> None:
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
    if settings.voice_runtime_mode == "cascade":
        await session.say(
            "Hello, this is the policy support assistant. How can I help you today?",
            allow_interruptions=True,
            add_to_chat_ctx=False,
        )
        logger.info("Initial agent greeting sent room_name=%s", ctx.room.name)
    else:
        logger.info(
            "Skipping explicit greeting for Gemini Live room_name=%s",
            ctx.room.name,
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
    _RUNTIME_SETTINGS = resolved_settings
    _RUNTIME_KB = knowledge_base
    _RUNTIME_RETRIEVAL = None
    _RUNTIME_METRICS = VoiceAgentMetrics(
        turn_metrics_log=TurnMetricsLog(Path(resolved_settings.turn_metrics_path)),
        active_sessions_store=ActiveSessionsStore(
            Path(resolved_settings.active_sessions_path)
        ),
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
