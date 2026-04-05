from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import logging
from pathlib import Path
import threading
import tempfile
import time
from typing import Iterable

from livekit.agents.metrics import EOUMetrics, LLMMetrics, TTSMetrics
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class NetworkQualitySnapshot:
    jitter_ms: float | None = None
    packet_loss_fraction: float | None = None


@dataclass(slots=True)
class TurnLatencySnapshot:
    end_of_utterance_delay_ms: float
    llm_ttft_ms: float
    tts_ttfb_ms: float

    @property
    def total_ms(self) -> float:
        return (
            self.end_of_utterance_delay_ms + self.llm_ttft_ms + self.tts_ttfb_ms
        )


@dataclass(slots=True)
class TurnMetricsLogRecord:
    timestamp: float
    end_of_utterance_delay_ms: float
    llm_ttft_ms: float
    tts_ttfb_ms: float

    @property
    def total_ms(self) -> float:
        return (
            self.end_of_utterance_delay_ms + self.llm_ttft_ms + self.tts_ttfb_ms
        )


@dataclass(slots=True)
class RetrievalLatencySnapshot:
    embedding_ms: float
    vector_query_ms: float
    document_load_ms: float
    rerank_ms: float
    total_ms: float


class TurnMetricsLog:
    """Small JSONL log shared between worker and API processes."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, snapshot: TurnLatencySnapshot, timestamp: float) -> None:
        payload = {
            "timestamp": timestamp,
            "end_of_utterance_delay_ms": snapshot.end_of_utterance_delay_ms,
            "llm_ttft_ms": snapshot.llm_ttft_ms,
            "tts_ttfb_ms": snapshot.tts_ttfb_ms,
        }
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def read(self) -> list[TurnMetricsLogRecord]:
        if not self._path.exists():
            return []

        records: list[TurnMetricsLogRecord] = []
        with self._path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                records.append(TurnMetricsLogRecord(**payload))
        return records


class ActiveSessionsStore:
    """Shared active-session counter with a local-file fallback for development."""

    def __init__(
        self,
        path: Path,
        *,
        redis_url: str | None = None,
        redis_key: str = "voice-agent:active-sessions",
    ) -> None:
        self._path = path
        self._redis_url = redis_url
        self._redis_key = redis_key
        self._redis_client: object | None = None
        self._lock = threading.Lock()
        if self.uses_local_file and not self._path.exists():
            self.write(0)

    @property
    def uses_local_file(self) -> bool:
        return self._redis_url is None

    def _get_redis_client(self) -> object:
        if self._redis_url is None:
            raise RuntimeError("Redis client requested without a configured REDIS_URL")
        if self._redis_client is None:
            try:
                redis_module = importlib.import_module("redis")
            except ImportError as exc:
                raise RuntimeError(
                    "REDIS_URL is configured but the 'redis' package is not installed"
                ) from exc
            self._redis_client = redis_module.Redis.from_url(
                self._redis_url,
                decode_responses=True,
            )
        return self._redis_client

    def read(self) -> int:
        if self._redis_url is not None:
            raw_value = self._get_redis_client().get(self._redis_key)
            if raw_value is None:
                return 0
            try:
                return max(0, int(raw_value))
            except (TypeError, ValueError):
                return 0
        if not self._path.exists():
            return 0
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return 0
        return max(0, int(payload.get("active_sessions", 0)))

    def write(self, value: int) -> int:
        normalized = max(0, int(value))
        if self._redis_url is not None:
            self._get_redis_client().set(self._redis_key, normalized)
            return normalized
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self._path.parent,
            prefix=f"{self._path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(json.dumps({"active_sessions": normalized}))
            temp_path = Path(handle.name)
        temp_path.replace(self._path)
        return normalized

    def increment(self) -> int:
        if self._redis_url is not None:
            return int(self._get_redis_client().incr(self._redis_key))
        with self._lock:
            return self.write(self.read() + 1)

    def decrement(self) -> int:
        if self._redis_url is not None:
            return int(
                self._get_redis_client().eval(
                    """
                    local current = redis.call('GET', KEYS[1])
                    if not current then
                      redis.call('SET', KEYS[1], 0)
                      return 0
                    end
                    current = tonumber(current)
                    if current <= 0 then
                      redis.call('SET', KEYS[1], 0)
                      return 0
                    end
                    current = current - 1
                    redis.call('SET', KEYS[1], current)
                    return current
                    """,
                    1,
                    self._redis_key,
                )
            )
        with self._lock:
            return self.write(max(0, self.read() - 1))


@dataclass(slots=True)
class PartialTurnMetrics:
    end_of_utterance_delay_ms: float | None = None
    llm_ttft_ms: float | None = None
    tts_ttfb_ms: float | None = None

    def is_complete(self) -> bool:
        return (
            self.end_of_utterance_delay_ms is not None
            and self.llm_ttft_ms is not None
            and self.tts_ttfb_ms is not None
        )

    def to_snapshot(self) -> TurnLatencySnapshot:
        if not self.is_complete():
            raise ValueError("turn metrics are incomplete")
        return TurnLatencySnapshot(
            end_of_utterance_delay_ms=self.end_of_utterance_delay_ms or 0.0,
            llm_ttft_ms=self.llm_ttft_ms or 0.0,
            tts_ttfb_ms=self.tts_ttfb_ms or 0.0,
        )


class VoiceAgentMetrics:
    """Owns Prometheus metrics and small aggregation helpers."""

    def __init__(
        self,
        registry: CollectorRegistry | None = None,
        turn_metrics_log: TurnMetricsLog | None = None,
        active_sessions_store: ActiveSessionsStore | None = None,
    ) -> None:
        self.registry = registry or CollectorRegistry()
        self.turn_metrics_log = turn_metrics_log
        self.active_sessions_store = active_sessions_store
        self._active_session_count = (
            active_sessions_store.read() if active_sessions_store is not None else 0
        )
        self._active_session_lock = threading.Lock()
        self.call_setup_failures = Counter(
            "voice_agent_call_setup_failures_total",
            "Number of failed call setup attempts.",
            registry=self.registry,
        )
        self.retry_attempts = Counter(
            "voice_agent_retry_attempts_total",
            "Number of retry attempts across upstream integrations.",
            ["component"],
            registry=self.registry,
        )
        self.upstream_failures = Counter(
            "voice_agent_upstream_failures_total",
            "Number of upstream integration failures observed by component.",
            ["component"],
            registry=self.registry,
        )
        self.llm_availability_changes = Counter(
            "voice_agent_llm_availability_changes_total",
            "Number of LLM availability changes observed by explicit failover logic.",
            ["provider", "model", "state"],
            registry=self.registry,
        )
        self.active_sessions = Gauge(
            "voice_agent_active_sessions",
            "Current number of active agent sessions.",
            registry=self.registry,
        )
        self.end_of_utterance_delay_ms = Histogram(
            "voice_agent_end_of_utterance_delay_ms",
            "Delay between end of caller utterance and pipeline reaction.",
            registry=self.registry,
        )
        self.llm_ttft_ms = Histogram(
            "voice_agent_llm_ttft_ms",
            "LLM time to first token in milliseconds.",
            registry=self.registry,
        )
        self.tts_ttfb_ms = Histogram(
            "voice_agent_tts_ttfb_ms",
            "TTS time to first byte in milliseconds.",
            registry=self.registry,
        )
        self.turn_latency_ms = Histogram(
            "voice_agent_turn_latency_ms",
            "Estimated conversational round-trip latency in milliseconds.",
            registry=self.registry,
        )
        self.turn_latency_avg_ms = Gauge(
            "voice_agent_turn_latency_avg_ms",
            "Average conversational round-trip latency in milliseconds from shared worker turn logs.",
            registry=self.registry,
        )
        self.turn_latency_p95_ms = Gauge(
            "voice_agent_turn_latency_p95_ms",
            "95th percentile conversational round-trip latency in milliseconds from shared worker turn logs.",
            registry=self.registry,
        )
        self.retrieval_embedding_ms = Histogram(
            "voice_agent_retrieval_embedding_ms",
            "Knowledge retrieval query embedding latency in milliseconds.",
            registry=self.registry,
        )
        self.retrieval_vector_query_ms = Histogram(
            "voice_agent_retrieval_vector_query_ms",
            "Knowledge retrieval vector query latency in milliseconds.",
            registry=self.registry,
        )
        self.retrieval_document_load_ms = Histogram(
            "voice_agent_retrieval_document_load_ms",
            "Knowledge retrieval document load latency in milliseconds.",
            registry=self.registry,
        )
        self.retrieval_rerank_ms = Histogram(
            "voice_agent_retrieval_rerank_ms",
            "Knowledge retrieval rerank latency in milliseconds.",
            registry=self.registry,
        )
        self.retrieval_total_ms = Histogram(
            "voice_agent_retrieval_total_ms",
            "Knowledge retrieval total latency in milliseconds.",
            registry=self.registry,
        )
        self.first_audio_latency_ms = Histogram(
            "voice_agent_first_audio_latency_ms",
            "Latency in milliseconds from caller turn completion to first agent audio frame.",
            registry=self.registry,
        )
        self.full_response_latency_ms = Histogram(
            "voice_agent_full_response_latency_ms",
            "Latency in milliseconds from caller turn completion to completed agent spoken response.",
            registry=self.registry,
        )
        self.llm_ttft_avg_ms = Gauge(
            "voice_agent_llm_ttft_avg_ms",
            "Average LLM time-to-first-token in milliseconds from shared worker turn logs.",
            registry=self.registry,
        )
        self.tts_ttfb_avg_ms = Gauge(
            "voice_agent_tts_ttfb_avg_ms",
            "Average TTS time-to-first-byte in milliseconds from shared worker turn logs.",
            registry=self.registry,
        )
        self.end_of_utterance_delay_avg_summary_ms = Gauge(
            "voice_agent_end_of_utterance_delay_avg_summary_ms",
            "Average end-of-utterance delay in milliseconds from shared worker turn logs.",
            registry=self.registry,
        )
        self.jitter_ms = Gauge(
            "voice_agent_network_jitter_ms",
            "Observed RTP jitter in milliseconds.",
            registry=self.registry,
        )
        self.packet_loss_fraction = Gauge(
            "voice_agent_packet_loss_fraction",
            "Observed packet loss fraction.",
            registry=self.registry,
        )
        self.estimated_mos = Gauge(
            "voice_agent_estimated_mos",
            "Estimated MOS score derived from jitter and packet loss.",
            registry=self.registry,
        )

    def record_turn_latency(self, snapshot: TurnLatencySnapshot) -> float:
        self.end_of_utterance_delay_ms.observe(snapshot.end_of_utterance_delay_ms)
        self.llm_ttft_ms.observe(snapshot.llm_ttft_ms)
        self.tts_ttfb_ms.observe(snapshot.tts_ttfb_ms)
        self.turn_latency_ms.observe(snapshot.total_ms)
        if self.turn_metrics_log is not None:
            self.turn_metrics_log.append(snapshot, timestamp=time.time())
        logger.info(
            "Recorded turn latency snapshot end_of_utterance_delay_ms=%s llm_ttft_ms=%s tts_ttfb_ms=%s total_ms=%s",
            snapshot.end_of_utterance_delay_ms,
            snapshot.llm_ttft_ms,
            snapshot.tts_ttfb_ms,
            snapshot.total_ms,
        )
        return snapshot.total_ms

    def record_retrieval_latency(self, snapshot: RetrievalLatencySnapshot) -> float:
        self.retrieval_embedding_ms.observe(snapshot.embedding_ms)
        self.retrieval_vector_query_ms.observe(snapshot.vector_query_ms)
        self.retrieval_document_load_ms.observe(snapshot.document_load_ms)
        self.retrieval_rerank_ms.observe(snapshot.rerank_ms)
        self.retrieval_total_ms.observe(snapshot.total_ms)
        logger.info(
            "Recorded retrieval latency embedding_ms=%s vector_query_ms=%s document_load_ms=%s rerank_ms=%s total_ms=%s",
            snapshot.embedding_ms,
            snapshot.vector_query_ms,
            snapshot.document_load_ms,
            snapshot.rerank_ms,
            snapshot.total_ms,
        )
        return snapshot.total_ms

    def record_first_audio_latency(self, latency_ms: float) -> float:
        self.first_audio_latency_ms.observe(latency_ms)
        logger.info("Recorded first audio latency latency_ms=%s", latency_ms)
        return latency_ms

    def record_full_response_latency(self, latency_ms: float) -> float:
        self.full_response_latency_ms.observe(latency_ms)
        logger.info("Recorded full response latency latency_ms=%s", latency_ms)
        return latency_ms

    def record_retry(self, component: str) -> None:
        self.retry_attempts.labels(component=component).inc()

    def record_upstream_failure(self, component: str) -> None:
        self.upstream_failures.labels(component=component).inc()

    def record_llm_availability_change(
        self,
        *,
        provider: str,
        model: str,
        available: bool,
    ) -> None:
        self.llm_availability_changes.labels(
            provider=provider,
            model=model,
            state="available" if available else "unavailable",
        ).inc()

    def record_call_setup_failure(self) -> None:
        self.call_setup_failures.inc()

    def set_active_sessions(self, value: int) -> None:
        with self._active_session_lock:
            self._active_session_count = max(0, value)
            if self.active_sessions_store is not None:
                self._active_session_count = self.active_sessions_store.write(
                    self._active_session_count
                )
            self.active_sessions.set(self._active_session_count)

    def increment_active_sessions(self) -> int:
        with self._active_session_lock:
            if self.active_sessions_store is not None:
                self._active_session_count = self.active_sessions_store.increment()
            else:
                self._active_session_count += 1
            self.active_sessions.set(self._active_session_count)
            return self._active_session_count

    def decrement_active_sessions(self) -> int:
        with self._active_session_lock:
            if self.active_sessions_store is not None:
                self._active_session_count = self.active_sessions_store.decrement()
            else:
                self._active_session_count = max(0, self._active_session_count - 1)
            self.active_sessions.set(self._active_session_count)
            return self._active_session_count

    def record_network_quality(self, snapshot: NetworkQualitySnapshot) -> float | None:
        if snapshot.jitter_ms is not None:
            self.jitter_ms.set(snapshot.jitter_ms)
        if snapshot.packet_loss_fraction is not None:
            self.packet_loss_fraction.set(snapshot.packet_loss_fraction)

        mos = self.estimate_mos(
            jitter_ms=snapshot.jitter_ms,
            packet_loss_fraction=snapshot.packet_loss_fraction,
        )
        if mos is not None:
            self.estimated_mos.set(mos)
        return mos

    def render_latest(self) -> bytes:
        self.sync_summary_metrics()
        return generate_latest(self.registry)

    def sync_summary_metrics(self) -> None:
        if self.active_sessions_store is not None:
            self.active_sessions.set(self.active_sessions_store.read())
        if self.turn_metrics_log is None:
            return

        records = self.turn_metrics_log.read()
        if not records:
            return

        turn_latencies = [record.total_ms for record in records]
        llm_ttfts = [record.llm_ttft_ms for record in records]
        tts_ttfbs = [record.tts_ttfb_ms for record in records]
        eou_delays = [record.end_of_utterance_delay_ms for record in records]
        self.turn_latency_avg_ms.set(sum(turn_latencies) / len(turn_latencies))
        self.turn_latency_p95_ms.set(percentile(turn_latencies, 0.95))
        self.llm_ttft_avg_ms.set(sum(llm_ttfts) / len(llm_ttfts))
        self.tts_ttfb_avg_ms.set(sum(tts_ttfbs) / len(tts_ttfbs))
        self.end_of_utterance_delay_avg_summary_ms.set(
            sum(eou_delays) / len(eou_delays)
        )

    @staticmethod
    def estimate_mos(
        jitter_ms: float | None,
        packet_loss_fraction: float | None,
    ) -> float | None:
        if jitter_ms is None or packet_loss_fraction is None:
            return None

        jitter_penalty = min(max(jitter_ms, 0.0) * 0.02, 1.0)
        packet_loss_percent = min(max(packet_loss_fraction, 0.0), 1.0) * 100.0
        packet_loss_penalty = min(packet_loss_percent * 0.08, 2.2)
        mos = 4.4 - jitter_penalty - packet_loss_penalty
        return round(max(1.0, min(4.5, mos)), 2)


class LiveKitTurnMetricsCollector:
    """Correlates LiveKit session metrics into per-turn latency snapshots."""

    def __init__(self, metrics: VoiceAgentMetrics) -> None:
        self._metrics = metrics
        self._pending: dict[str, PartialTurnMetrics] = {}

    def observe(self, metric: EOUMetrics | LLMMetrics | TTSMetrics | object) -> float | None:
        speech_id = getattr(metric, "speech_id", None)
        if not speech_id:
            return None

        partial = self._pending.setdefault(speech_id, PartialTurnMetrics())
        if isinstance(metric, EOUMetrics):
            partial.end_of_utterance_delay_ms = metric.end_of_utterance_delay * 1000
        elif isinstance(metric, LLMMetrics):
            partial.llm_ttft_ms = metric.ttft * 1000
        elif isinstance(metric, TTSMetrics):
            partial.tts_ttfb_ms = metric.ttfb * 1000
        else:
            return None

        if not partial.is_complete():
            return None

        snapshot = partial.to_snapshot()
        self._pending.pop(speech_id, None)
        return self._metrics.record_turn_latency(snapshot)


def average_latency_ms(snapshots: Iterable[TurnLatencySnapshot]) -> float:
    values = [snapshot.total_ms for snapshot in snapshots]
    if not values:
        return 0.0
    return sum(values) / len(values)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = max(0, int(len(sorted_values) * quantile) - 1)
    return sorted_values[index]

