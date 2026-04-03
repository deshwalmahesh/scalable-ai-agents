from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, Response, status
from pydantic import BaseModel

from voice_agent.kb import KnowledgeBase, KnowledgeMatch
from voice_agent.observability import TurnMetricsLog, VoiceAgentMetrics
from voice_agent.settings import AppSettings, load_settings


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    kb_documents: int
    checks: dict[str, str]


class SearchResponse(BaseModel):
    query: str
    matches: list[dict[str, str | int | float]]


@dataclass(slots=True)
class AppState:
    settings: AppSettings
    metrics: VoiceAgentMetrics
    knowledge_base: KnowledgeBase


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
) -> AppState:
    resolved_settings = settings or load_settings()
    resolved_metrics = metrics or VoiceAgentMetrics(
        turn_metrics_log=TurnMetricsLog(Path(resolved_settings.turn_metrics_path))
    )
    resolved_kb = knowledge_base or KnowledgeBase.from_settings(resolved_settings)
    return AppState(
        settings=resolved_settings,
        metrics=resolved_metrics,
        knowledge_base=resolved_kb,
    )


def create_app(
    settings: AppSettings | None = None,
    metrics: VoiceAgentMetrics | None = None,
    knowledge_base: KnowledgeBase | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.voice_agent = build_state(
            settings=settings,
            metrics=metrics,
            knowledge_base=knowledge_base,
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
