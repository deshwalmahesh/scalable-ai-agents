from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel

from voice_agent.kb import KnowledgeBase, KnowledgeMatch
from voice_agent.observability import (
    ActiveSessionsStore,
    TurnMetricsLog,
    VoiceAgentMetrics,
)
from voice_agent.settings import AppSettings, load_settings
from voice_agent.telephony import (
    LiveKitTelephonyManager,
    SIPDispatchRuleSummary,
    SIPInboundTrunkSummary,
    SIPProvisionResult,
    SIPTelephonyStatus,
)


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    kb_documents: int
    checks: dict[str, str]


class SearchResponse(BaseModel):
    query: str
    matches: list[dict[str, str | int | float]]


class SIPInboundTrunkResponse(BaseModel):
    trunk_id: str
    name: str
    numbers: list[str]


class SIPDispatchRuleResponse(BaseModel):
    rule_id: str
    name: str
    trunk_ids: list[str]
    room_prefix: str
    agent_names: list[str]


class TelephonyStatusResponse(BaseModel):
    phone_number: str | None
    sip_domain: str | None
    inbound_trunk: SIPInboundTrunkResponse | None
    dispatch_rule: SIPDispatchRuleResponse | None


class TelephonyProvisionResponse(TelephonyStatusResponse):
    created_inbound_trunk: bool
    created_dispatch_rule: bool


@dataclass(slots=True)
class AppState:
    settings: AppSettings
    metrics: VoiceAgentMetrics
    knowledge_base: KnowledgeBase
    telephony_manager: LiveKitTelephonyManager


def build_readiness_payload(state: AppState) -> ReadinessResponse:
    kb_documents = state.knowledge_base.count()
    checks = {
        "config": "ok",
        "knowledge_base": "ok" if kb_documents > 0 else "empty",
        "twilio_phone_number": "ok" if state.settings.twilio_phone_number else "missing",
        "twilio_sip_domain": "ok" if state.settings.twilio_sip_domain else "missing",
    }
    status_text = "ok" if all(value == "ok" for value in checks.values()) else "degraded"
    return ReadinessResponse(
        status=status_text,
        kb_documents=kb_documents,
        checks=checks,
    )


def build_state(
    settings: AppSettings | None = None,
    metrics: VoiceAgentMetrics | None = None,
    knowledge_base: KnowledgeBase | None = None,
    telephony_manager: LiveKitTelephonyManager | None = None,
) -> AppState:
    resolved_settings = settings or load_settings()
    resolved_metrics = metrics or VoiceAgentMetrics(
        turn_metrics_log=TurnMetricsLog(Path(resolved_settings.turn_metrics_path)),
        active_sessions_store=ActiveSessionsStore(
            Path(resolved_settings.active_sessions_path),
            redis_url=resolved_settings.redis_url,
            redis_key=resolved_settings.active_sessions_key,
        ),
    )
    resolved_kb = knowledge_base or KnowledgeBase.from_settings(resolved_settings)
    return AppState(
        settings=resolved_settings,
        metrics=resolved_metrics,
        knowledge_base=resolved_kb,
        telephony_manager=telephony_manager or LiveKitTelephonyManager(resolved_settings),
    )


def create_app(
    settings: AppSettings | None = None,
    metrics: VoiceAgentMetrics | None = None,
    knowledge_base: KnowledgeBase | None = None,
    telephony_manager: LiveKitTelephonyManager | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.voice_agent = build_state(
            settings=settings,
            metrics=metrics,
            knowledge_base=knowledge_base,
            telephony_manager=telephony_manager,
        )
        yield

    app = FastAPI(title="voice-agent-control-plane", lifespan=lifespan)

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/readyz", response_model=ReadinessResponse)
    async def readyz(response: Response) -> ReadinessResponse:
        payload = build_readiness_payload(app.state.voice_agent)
        if payload.status != "ok":
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return payload

    @app.get("/metrics")
    async def metrics_endpoint() -> Response:
        return Response(
            content=app.state.voice_agent.metrics.render_latest(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.get("/kb/search", response_model=SearchResponse)
    async def kb_search(q: str, limit: int = 3) -> SearchResponse:
        matches = app.state.voice_agent.knowledge_base.search(q, limit=limit)
        return SearchResponse(
            query=q,
            matches=[serialize_match(match) for match in matches],
        )

    @app.get("/telephony/sip/status", response_model=TelephonyStatusResponse)
    async def telephony_sip_status() -> TelephonyStatusResponse:
        status_payload = await app.state.voice_agent.telephony_manager.fetch_status()
        return serialize_telephony_status(status_payload)

    @app.post("/telephony/sip/provision", response_model=TelephonyProvisionResponse)
    async def telephony_sip_provision() -> TelephonyProvisionResponse:
        try:
            provisioned = await app.state.voice_agent.telephony_manager.provision()
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        return serialize_telephony_provision(provisioned)

    return app


def serialize_match(match: KnowledgeMatch) -> dict[str, str | int | float]:
    return {
        "id": match.id,
        "title": match.title,
        "content": match.content,
        "score": match.score,
        "vector_score": match.vector_score,
        "lexical_score": match.lexical_score,
        "category": match.metadata.get("category", ""),
        "source_url": match.metadata.get("source_url", ""),
    }


def serialize_telephony_status(
    status_payload: SIPTelephonyStatus,
) -> TelephonyStatusResponse:
    return TelephonyStatusResponse(
        phone_number=status_payload.phone_number,
        sip_domain=status_payload.sip_domain,
        inbound_trunk=serialize_inbound_trunk(status_payload.inbound_trunk),
        dispatch_rule=serialize_dispatch_rule(status_payload.dispatch_rule),
    )


def serialize_telephony_provision(
    provisioned: SIPProvisionResult,
) -> TelephonyProvisionResponse:
    status_payload = serialize_telephony_status(provisioned.status)
    return TelephonyProvisionResponse(
        **status_payload.model_dump(),
        created_inbound_trunk=provisioned.created_inbound_trunk,
        created_dispatch_rule=provisioned.created_dispatch_rule,
    )


def serialize_inbound_trunk(
    trunk: SIPInboundTrunkSummary | None,
) -> SIPInboundTrunkResponse | None:
    if trunk is None:
        return None
    return SIPInboundTrunkResponse(
        trunk_id=trunk.trunk_id,
        name=trunk.name,
        numbers=trunk.numbers,
    )


def serialize_dispatch_rule(
    rule: SIPDispatchRuleSummary | None,
) -> SIPDispatchRuleResponse | None:
    if rule is None:
        return None
    return SIPDispatchRuleResponse(
        rule_id=rule.rule_id,
        name=rule.name,
        trunk_ids=rule.trunk_ids,
        room_prefix=rule.room_prefix,
        agent_names=rule.agent_names,
    )
