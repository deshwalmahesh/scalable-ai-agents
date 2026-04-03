from __future__ import annotations

from pathlib import Path

from livekit.agents.metrics import EOUMetrics, LLMMetrics, TTSMetrics
from prometheus_client import CollectorRegistry

from voice_agent.observability import (
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
