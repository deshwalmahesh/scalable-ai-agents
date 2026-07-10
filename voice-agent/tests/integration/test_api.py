from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from voice_agent.api import create_app
from voice_agent.kb import KnowledgeBase, KnowledgeDocument
from voice_agent.observability import ActiveSessionsStore, VoiceAgentMetrics
from voice_agent.settings import AppSettings
from voice_agent.telephony import (
    SIPDispatchRuleSummary,
    SIPInboundTrunkSummary,
    SIPProvisionResult,
    SIPTelephonyStatus,
)


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


class FakeTelephonyManager:
    def __init__(
        self,
        *,
        status_payload: SIPTelephonyStatus | None = None,
        provisioned_payload: SIPProvisionResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self._status_payload = status_payload
        self._provisioned_payload = provisioned_payload
        self._error = error

    async def fetch_status(self) -> SIPTelephonyStatus:
        if self._status_payload is None:
            raise AssertionError("status payload was not configured")
        return self._status_payload

    async def provision(self) -> SIPProvisionResult:
        if self._error is not None:
            raise self._error
        if self._provisioned_payload is None:
            raise AssertionError("provision payload was not configured")
        return self._provisioned_payload


def test_healthz_returns_ok(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)

    with create_test_client(settings) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_is_degraded_when_telephony_config_is_missing(tmp_path: Path) -> None:
    settings = build_settings(
        tmp_path,
        TWILIO_PHONE_NUMBER=None,
        TWILIO_SIP_DOMAIN=None,
    )

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


def test_metrics_endpoint_reads_shared_active_session_count(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    store = ActiveSessionsStore(Path(settings.active_sessions_path))
    store.write(2)
    kb = KnowledgeBase(Path(settings.chroma_persist_dir), settings.kb_collection_name)
    kb.upsert_documents(TEST_DOCUMENTS)
    app = create_app(
        settings=settings,
        metrics=VoiceAgentMetrics(active_sessions_store=store),
        knowledge_base=kb,
    )

    with TestClient(app) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert "voice_agent_active_sessions 2.0" in response.text


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


def test_telephony_status_reports_current_sip_resources(tmp_path: Path) -> None:
    settings = build_settings(
        tmp_path,
        TWILIO_PHONE_NUMBER="+15555550100",
        TWILIO_SIP_DOMAIN="voice.example.twilio.com",
    )
    kb = KnowledgeBase(Path(settings.chroma_persist_dir), settings.kb_collection_name)
    kb.upsert_documents(TEST_DOCUMENTS)
    app = create_app(
        settings=settings,
        metrics=VoiceAgentMetrics(),
        knowledge_base=kb,
        telephony_manager=FakeTelephonyManager(
            status_payload=SIPTelephonyStatus(
                phone_number=settings.twilio_phone_number,
                sip_domain=settings.twilio_sip_domain,
                inbound_trunk=SIPInboundTrunkSummary(
                    trunk_id="TRUNK123",
                    name="policy-support-inbound",
                    numbers=["+15555550100"],
                ),
                dispatch_rule=SIPDispatchRuleSummary(
                    rule_id="RULE123",
                    name="policy-support-dispatch",
                    trunk_ids=["TRUNK123"],
                    room_prefix="call-",
                    agent_names=["policy-support-agent"],
                ),
            )
        ),
    )

    with TestClient(app) as client:
        response = client.get("/telephony/sip/status")

    assert response.status_code == 200
    assert response.json()["inbound_trunk"]["trunk_id"] == "TRUNK123"
    assert response.json()["dispatch_rule"]["agent_names"] == ["policy-support-agent"]


def test_telephony_provision_returns_created_resources(tmp_path: Path) -> None:
    settings = build_settings(
        tmp_path,
        TWILIO_PHONE_NUMBER="+15555550100",
        TWILIO_SIP_DOMAIN="voice.example.twilio.com",
    )
    kb = KnowledgeBase(Path(settings.chroma_persist_dir), settings.kb_collection_name)
    kb.upsert_documents(TEST_DOCUMENTS)
    app = create_app(
        settings=settings,
        metrics=VoiceAgentMetrics(),
        knowledge_base=kb,
        telephony_manager=FakeTelephonyManager(
            provisioned_payload=SIPProvisionResult(
                status=SIPTelephonyStatus(
                    phone_number=settings.twilio_phone_number,
                    sip_domain=settings.twilio_sip_domain,
                    inbound_trunk=SIPInboundTrunkSummary(
                        trunk_id="TRUNK123",
                        name="policy-support-inbound",
                        numbers=["+15555550100"],
                    ),
                    dispatch_rule=SIPDispatchRuleSummary(
                        rule_id="RULE123",
                        name="policy-support-dispatch",
                        trunk_ids=["TRUNK123"],
                        room_prefix="call-",
                        agent_names=["policy-support-agent"],
                    ),
                ),
                created_inbound_trunk=True,
                created_dispatch_rule=True,
            )
        ),
    )

    with TestClient(app) as client:
        response = client.post("/telephony/sip/provision")

    assert response.status_code == 200
    assert response.json()["created_inbound_trunk"] is True
    assert response.json()["created_dispatch_rule"] is True


def test_telephony_provision_returns_bad_request_for_missing_number(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    kb = KnowledgeBase(Path(settings.chroma_persist_dir), settings.kb_collection_name)
    kb.upsert_documents(TEST_DOCUMENTS)
    app = create_app(
        settings=settings,
        metrics=VoiceAgentMetrics(),
        knowledge_base=kb,
        telephony_manager=FakeTelephonyManager(
            error=ValueError("TWILIO_PHONE_NUMBER is required to provision SIP resources")
        ),
    )

    with TestClient(app) as client:
        response = client.post("/telephony/sip/provision")

    assert response.status_code == 400
    assert "TWILIO_PHONE_NUMBER" in response.json()["detail"]
