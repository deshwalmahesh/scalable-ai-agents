from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import ssl

import aiohttp
import certifi
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ConversationItemAddedEvent,
    FunctionToolsExecutedEvent,
    JobContext,
    JobProcess,
    UserInputTranscribedEvent,
    RunContext,
    cli,
    function_tool,
    room_io,
)
from livekit.plugins import deepgram, noise_cancellation, openai, silero

from voice_agent.kb import KnowledgeBase
from voice_agent.observability import (
    LiveKitTurnMetricsCollector,
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


def _certifi_client_session(*args: object, **kwargs: object) -> aiohttp.ClientSession:
    if "connector" not in kwargs:
        kwargs["connector"] = aiohttp.TCPConnector(
            ssl=_certifi_default_context(),
        )
    return _ORIGINAL_CLIENT_SESSION(*args, **kwargs)


# LiveKit worker uses aiohttp/websocket connections under the hood. On this
# machine the system trust store is incomplete, so we patch both the default SSL
# context and aiohttp's default client session to use certifi's CA bundle
# without disabling verification.
ssl.create_default_context = _certifi_default_context
aiohttp.ClientSession = _certifi_client_session
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

_RUNTIME_SETTINGS: AppSettings | None = None
_RUNTIME_KB: KnowledgeBase | None = None
_RUNTIME_METRICS: VoiceAgentMetrics | None = None
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WorkerRuntime:
    server: AgentServer
    agent_name: str

class SupportAgent(Agent):
    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        super().__init__(instructions=build_system_prompt())
        self._knowledge_base = knowledge_base

    @function_tool()
    async def search_support_knowledge(
        self,
        context: RunContext,
        query: str,
    ) -> dict[str, object]:
        """Search the policy knowledge base using the caller's own wording.

        Args:
            query: Pass the caller's question with minimal cleanup. Preserve numbers,
                dates, and core terms instead of rewriting it into a search phrase.
        """

        del context
        logger.info("Knowledge tool invoked query=%s", query)
        matches = self._knowledge_base.search(query, limit=2)
        logger.info(
            "Knowledge tool returning matches query=%s match_count=%s match_titles=%s",
            query,
            len(matches),
            [match.title[:120] for match in matches],
        )
        return {
            "supported": bool(matches),
            "matches": [
                {
                    "title": match.title,
                    "content": match.content,
                    "category": match.metadata.get("category", ""),
                    "source_url": match.metadata.get("source_url", ""),
                    "score": match.score,
                    "vector_score": match.vector_score,
                    "lexical_score": match.lexical_score,
                }
                for match in matches
            ],
            "message": (
                "No grounded policy excerpts were found for this question."
                if not matches
                else ""
            ),
        }


def create_llm(settings: AppSettings) -> openai.LLM:
    # This is intentionally OpenAI-compatible so the same code works with Gemini,
    # vLLM, or any provider that speaks the OpenAI chat-completions shape.
    return openai.LLM(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


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


def get_runtime_metrics() -> VoiceAgentMetrics:
    global _RUNTIME_METRICS
    if _RUNTIME_METRICS is None:
        settings = get_runtime_settings()
        _RUNTIME_METRICS = VoiceAgentMetrics(
            turn_metrics_log=TurnMetricsLog(Path(settings.turn_metrics_path))
        )
    return _RUNTIME_METRICS


async def handle_session(ctx: JobContext) -> None:
    settings = get_runtime_settings()
    knowledge_base = get_runtime_kb()
    metrics = get_runtime_metrics()
    logger.info(
        "Starting agent session room_name=%s agent_name=%s kb_documents=%s",
        ctx.room.name,
        settings.agent_name,
        knowledge_base.count(),
    )
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=create_llm(settings),
        tts=deepgram.TTS(model="aura-2-asteria-en"),
        vad=ctx.proc.userdata["vad"],
        # VAD-based turn detection is simpler and avoids a separate model download
        # while still supporting interruption and spoken turn-taking.
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

    @session.on("function_tools_executed")
    def on_function_tools_executed(event: FunctionToolsExecutedEvent) -> None:
        logger.info(
            "Function tools executed count=%s",
            len(event.function_calls),
        )

    await session.start(
        room=ctx.room,
        agent=SupportAgent(knowledge_base),
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
    await session.say(
        "Hello, this is the policy support assistant. How can I help you today?",
        allow_interruptions=True,
        add_to_chat_ctx=False,
    )
    logger.info("Initial agent greeting sent room_name=%s", ctx.room.name)


def create_runtime(
    settings: AppSettings | None = None,
    knowledge_base: KnowledgeBase | None = None,
) -> WorkerRuntime:
    global _RUNTIME_SETTINGS, _RUNTIME_KB, _RUNTIME_METRICS
    resolved_settings = settings or load_settings()
    _RUNTIME_SETTINGS = resolved_settings
    _RUNTIME_KB = knowledge_base
    if _RUNTIME_KB is None:
        _RUNTIME_KB = get_runtime_kb()
    _RUNTIME_METRICS = VoiceAgentMetrics(
        turn_metrics_log=TurnMetricsLog(Path(resolved_settings.turn_metrics_path))
    )

    server = AgentServer()
    server.setup_fnc = prewarm
    server.rtc_session(agent_name=resolved_settings.agent_name)(handle_session)

    return WorkerRuntime(server=server, agent_name=resolved_settings.agent_name)


def run_worker() -> None:
    settings = load_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    runtime = create_runtime()
    cli.run_app(runtime.server)
