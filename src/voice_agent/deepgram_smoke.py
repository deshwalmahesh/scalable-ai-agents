from __future__ import annotations

import asyncio
import json
import logging
import re
import ssl

import aiohttp
import certifi
from livekit.plugins import deepgram

from voice_agent.settings import load_settings

logger = logging.getLogger(__name__)


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


async def run_smoke() -> dict[str, object]:
    settings = load_settings()
    logger.info("Starting Deepgram smoke run")
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_ctx))
    tts_metrics: list[dict[str, float]] = []
    stt_metrics: list[dict[str, float]] = []
    stt = deepgram.STT(
        model="nova-3",
        language="en-US",
        api_key=settings.deepgram_api_key,
        http_session=session,
    )
    tts = deepgram.TTS(
        model="aura-2-asteria-en",
        api_key=settings.deepgram_api_key,
        http_session=session,
    )
    stt.on(
        "metrics_collected",
        lambda metric: stt_metrics.append(
            {
                "duration_ms": round(metric.duration * 1000, 2),
                "audio_duration_ms": round(metric.audio_duration * 1000, 2),
            }
        ),
    )
    tts.on(
        "metrics_collected",
        lambda metric: tts_metrics.append(
            {
                "ttfb_ms": round(metric.ttfb * 1000, 2),
                "audio_duration_ms": round(metric.audio_duration * 1000, 2),
                "duration_ms": round(metric.duration * 1000, 2),
            }
        ),
    )
    test_text = (
        "Hello from the policy support Deepgram smoke test. "
        "If you return the policy within ten days, any premium paid will be refunded."
    )

    try:
        tts_stream = tts.synthesize(test_text)
        frames = []
        async for event in tts_stream:
            frames.append(event.frame)
        await tts_stream.aclose()
        if not frames:
            raise RuntimeError("Deepgram TTS returned no audio frames")

        transcription = await stt.recognize(frames, language="en-US")
        transcript_text = transcription.alternatives[0].text if transcription.alternatives else ""
        normalized_transcript = _normalize_text(transcript_text)
        expected_keywords = ["policy", "ten", "premium", "refunded"]
        keyword_hits = [keyword for keyword in expected_keywords if keyword in normalized_transcript]
        logger.info(
            "Deepgram smoke transcription complete frame_count=%s transcript_text=%s keyword_hits=%s",
            len(frames),
            transcript_text,
            keyword_hits,
        )
    finally:
        await stt.aclose()
        await tts.aclose()
        await session.close()

    return {
        "stt_initialized": True,
        "tts_initialized": True,
        "ingested_text": test_text,
        "tts_audio_frames_seen": len(frames),
        "tts_audio_duration_ms": round(sum(frame.duration for frame in frames) * 1000, 2),
        "stt_transcript": transcript_text,
        "keyword_hits": keyword_hits,
        "round_trip_success": len(keyword_hits) >= 4,
        "tts_metrics": tts_metrics,
        "stt_metrics": stt_metrics,
    }


def main() -> None:
    settings = load_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    result = asyncio.run(run_smoke())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
