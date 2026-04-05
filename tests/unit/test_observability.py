from __future__ import annotations

from pathlib import Path
import sys
import types

from livekit.agents.metrics import EOUMetrics, LLMMetrics, TTSMetrics
from livekit.rtc import stats as rtc_stats
from prometheus_client import CollectorRegistry
import pytest

from voice_agent.livekit_smoke import (
    _extract_network_quality,
    _summarize_multi_turn_result,
)
from voice_agent.observability import (
    ActiveSessionsStore,
    LiveKitTurnMetricsCollector,
    NetworkQualitySnapshot,
    RetrievalLatencySnapshot,
    TurnMetricsLog,
    TurnLatencySnapshot,
    VoiceAgentMetrics,
    average_latency_ms,
)


def test_record_turn_latency_uses_documented_formula() -> None:
    metrics = VoiceAgentMetrics(registry=CollectorRegistry())

    total = metrics.record_turn_latency(
        TurnLatencySnapshot(
            end_of_utterance_delay_ms=150.0,
            llm_ttft_ms=180.0,
            tts_ttfb_ms=120.0,
        )
    )

    assert total == 450.0
    rendered = metrics.render_latest().decode()
    assert "voice_agent_turn_latency_ms_sum 450.0" in rendered


def test_record_network_quality_updates_estimated_mos() -> None:
    metrics = VoiceAgentMetrics(registry=CollectorRegistry())

    mos = metrics.record_network_quality(
        NetworkQualitySnapshot(jitter_ms=18.0, packet_loss_fraction=0.01)
    )

    assert mos is not None
    assert 1.0 <= mos <= 4.5
    rendered = metrics.render_latest().decode()
    assert "voice_agent_estimated_mos" in rendered


def test_extract_network_quality_uses_actual_rtp_stats() -> None:
    report = rtc_stats.RtcStats()
    report.inbound_rtp.rtc.id = "inbound-1"
    report.inbound_rtp.received.jitter = 0.018
    report.inbound_rtp.received.packets_received = 980
    report.inbound_rtp.received.packets_lost = 20

    summary = _extract_network_quality([report])

    assert summary["jitter_ms"] == 18.0
    assert summary["packets_received"] == 980
    assert summary["packets_lost"] == 20
    assert summary["packet_loss_fraction"] == 0.02
    assert summary["estimated_mos"] is not None


def test_summarize_multi_turn_result_surfaces_submission_quality_summary() -> None:
    summary = _summarize_multi_turn_result(
        [
            {
                "round_trip_success": True,
                "response_latency_ms": 1200.0,
                "network_quality": {
                    "jitter_ms": 10.0,
                    "packet_loss_fraction": 0.01,
                    "estimated_mos": 4.12,
                },
            },
            {
                "round_trip_success": False,
                "response_latency_ms": 1600.0,
                "network_quality": {
                    "jitter_ms": 25.0,
                    "packet_loss_fraction": 0.03,
                    "estimated_mos": 3.55,
                },
            },
        ]
    )

    assert summary["turn_count"] == 2
    assert summary["successful_turns"] == 1
    assert summary["avg_response_latency_ms"] == 1400.0
    assert summary["worst_response_latency_ms"] == 1600.0
    assert summary["worst_jitter_ms"] == 25.0
    assert summary["worst_packet_loss_fraction"] == 0.03
    assert summary["lowest_estimated_mos"] == 3.55
    assert summary["turns_with_network_quality"] == 2


def test_record_retrieval_latency_exposes_stage_metrics() -> None:
    metrics = VoiceAgentMetrics(registry=CollectorRegistry())

    total = metrics.record_retrieval_latency(
        RetrievalLatencySnapshot(
            embedding_ms=100.0,
            vector_query_ms=80.0,
            document_load_ms=40.0,
            rerank_ms=20.0,
            total_ms=240.0,
        )
    )

    assert total == 240.0
    rendered = metrics.render_latest().decode()
    assert "voice_agent_retrieval_total_ms" in rendered


def test_explicit_failure_metrics_are_exposed() -> None:
    metrics = VoiceAgentMetrics(registry=CollectorRegistry())

    metrics.record_upstream_failure("retrieval")
    metrics.record_llm_availability_change(
        provider="example.com",
        model="fallback-model",
        available=False,
    )

    rendered = metrics.render_latest().decode()
    assert 'voice_agent_upstream_failures_total{component="retrieval"} 1.0' in rendered
    assert (
        'voice_agent_llm_availability_changes_total{model="fallback-model",provider="example.com",state="unavailable"} 1.0'
        in rendered
    )


def test_average_latency_ms_returns_zero_for_empty_input() -> None:
    assert average_latency_ms([]) == 0.0


def test_average_latency_ms_returns_mean_for_snapshots() -> None:
    snapshots = [
        TurnLatencySnapshot(100.0, 100.0, 100.0),
        TurnLatencySnapshot(200.0, 100.0, 100.0),
    ]

    assert average_latency_ms(snapshots) == 350.0


def test_turn_metrics_log_summary_is_exposed_in_metrics(tmp_path: Path) -> None:
    metrics_log = TurnMetricsLog(tmp_path / "turn_metrics.jsonl")
    metrics = VoiceAgentMetrics(
        registry=CollectorRegistry(),
        turn_metrics_log=metrics_log,
    )
    metrics.record_turn_latency(
        TurnLatencySnapshot(
            end_of_utterance_delay_ms=100.0,
            llm_ttft_ms=150.0,
            tts_ttfb_ms=120.0,
        )
    )
    metrics.record_turn_latency(
        TurnLatencySnapshot(
            end_of_utterance_delay_ms=200.0,
            llm_ttft_ms=180.0,
            tts_ttfb_ms=140.0,
        )
    )

    rendered = metrics.render_latest().decode()

    assert "voice_agent_turn_latency_avg_ms" in rendered
    assert "voice_agent_turn_latency_p95_ms" in rendered


def test_active_sessions_store_is_shared_via_metrics_render(tmp_path: Path) -> None:
    store = ActiveSessionsStore(tmp_path / "active_sessions.json")
    metrics = VoiceAgentMetrics(
        registry=CollectorRegistry(),
        active_sessions_store=store,
    )

    assert metrics.increment_active_sessions() == 1
    assert store.read() == 1

    rendered = metrics.render_latest().decode()

    assert "voice_agent_active_sessions" in rendered
    assert "voice_agent_active_sessions 1.0" in rendered
    assert metrics.decrement_active_sessions() == 0


def test_active_sessions_store_supports_redis_backing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeRedisClient:
        def __init__(self) -> None:
            self.values: dict[str, int] = {}

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

    store = ActiveSessionsStore(
        tmp_path / "active_sessions.json",
        redis_url="redis://example:6379/0",
        redis_key="voice-agent:test",
    )

    assert store.uses_local_file is False
    assert store.read() == 0
    assert store.increment() == 1
    assert store.increment() == 2
    assert store.decrement() == 1
    assert store.write(7) == 7
    assert store.read() == 7


def test_livekit_turn_metrics_collector_records_completed_turn(tmp_path: Path) -> None:
    metrics = VoiceAgentMetrics(
        registry=CollectorRegistry(),
        turn_metrics_log=TurnMetricsLog(tmp_path / "turn_metrics.jsonl"),
    )
    collector = LiveKitTurnMetricsCollector(metrics)

    assert (
        collector.observe(
            EOUMetrics(
                timestamp=1.0,
                end_of_utterance_delay=0.15,
                transcription_delay=0.05,
                on_user_turn_completed_delay=0.01,
                speech_id="speech-1",
            )
        )
        is None
    )
    assert (
        collector.observe(
            LLMMetrics(
                label="llm",
                request_id="req-1",
                timestamp=2.0,
                duration=0.4,
                ttft=0.2,
                cancelled=False,
                completion_tokens=10,
                prompt_tokens=5,
                prompt_cached_tokens=0,
                total_tokens=15,
                tokens_per_second=30.0,
                speech_id="speech-1",
            )
        )
        is None
    )

    total_ms = collector.observe(
        TTSMetrics(
            label="tts",
            request_id="req-2",
            timestamp=3.0,
            ttfb=0.12,
            duration=0.5,
            audio_duration=1.5,
            cancelled=False,
            characters_count=32,
            streamed=False,
            speech_id="speech-1",
        )
    )

    assert total_ms == 470.0
