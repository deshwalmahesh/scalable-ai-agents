from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from voice_agent.api import create_app
from voice_agent.kb import KnowledgeBase, KnowledgeDocument
from voice_agent.observability import VoiceAgentMetrics
from voice_agent.settings import AppSettings


TEST_DOCUMENTS = [
    KnowledgeDocument(
        id="free-look",
        title="Ten day policy examination and refund",
        content=(
            "You have ten days from the date you receive the policy to examine it. "
            "If you return it within that ten day period, the policy is deemed never issued "
            "and any premium paid will be refunded."
        ),
        metadata={"source_url": "https://example.com/policy.pdf", "page_number": "1"},
    ),
    KnowledgeDocument(
        id="grace-period",
        title="Grace period for paying premiums",
        content=(
            "If a premium is not paid by its due date, the policy is in default. "
            "A grace period of thirty-one days after the premium due date is allowed."
        ),
        metadata={"source_url": "https://example.com/policy.pdf", "page_number": "8"},
    ),
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


def create_test_client(settings: AppSettings) -> TestClient:
    kb = KnowledgeBase(Path(settings.chroma_persist_dir), settings.kb_collection_name)
    kb.upsert_documents(TEST_DOCUMENTS)
    app = create_app(
        settings=settings,
        metrics=VoiceAgentMetrics(),
        knowledge_base=kb,
    )
    return TestClient(app)


def test_healthz_returns_ok(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)

    with create_test_client(settings) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_is_degraded_when_telephony_config_is_missing(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)

    with create_test_client(settings) as client:
        response = client.get("/readyz")

    body = response.json()
    assert response.status_code == 503
    assert body["status"] == "degraded"
    assert body["kb_documents"] == 2
    assert body["checks"]["knowledge_base"] == "ok"
    assert body["checks"]["twilio_phone_number"] == "missing"


def test_readyz_is_ok_when_minimum_runtime_inputs_are_present(tmp_path: Path) -> None:
    settings = build_settings(
        tmp_path,
        TWILIO_PHONE_NUMBER="+15555550100",
        TWILIO_SIP_DOMAIN="voice.example.twilio.com",
    )

    with create_test_client(settings) as client:
        response = client.get("/readyz")

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "ok"
    assert body["checks"]["twilio_phone_number"] == "ok"
    assert body["checks"]["twilio_sip_domain"] == "ok"


def test_metrics_endpoint_exposes_prometheus_text(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)

    with create_test_client(settings) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert "voice_agent_active_sessions" in response.text


def test_kb_search_returns_ranked_matches(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)

    with create_test_client(settings) as client:
        response = client.get(
            "/kb/search",
            params={"q": "return the policy within ten days and get premium back", "limit": 2},
        )

    body = response.json()
    assert response.status_code == 200
    assert "ten days" in body["query"]
    assert body["matches"]
    assert any(
        "refund" in match["title"].lower() or "ten day" in match["title"].lower()
        for match in body["matches"]
    )
