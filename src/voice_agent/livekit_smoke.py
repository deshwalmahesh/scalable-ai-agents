from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import ssl
import time
from typing import Any

import aiohttp
import certifi
from livekit import api, rtc
from livekit.protocol.agent_dispatch import CreateAgentDispatchRequest
from livekit.protocol.room import CreateRoomRequest, DeleteRoomRequest
from livekit.rtc._proto.room_pb2 import TrackPublishOptions

from voice_agent.deepgram_smoke import _normalize_text
from voice_agent.settings import AppSettings, load_settings
from livekit.plugins import deepgram

logger = logging.getLogger(__name__)


def _build_room_token(settings: AppSettings, room_name: str, identity: str, name: str) -> str:
    return (
        api.AccessToken(
            api_key=settings.livekit_api_key,
            api_secret=settings.livekit_api_secret,
        )
        .with_identity(identity)
        .with_name(name)
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )


def _keyword_hits(text: str, expected_keywords: list[str]) -> list[str]:
    normalized = _normalize_text(text)
    return [keyword for keyword in expected_keywords if keyword in normalized]


def _frame_peak_level(frame: rtc.AudioFrame) -> int:
    return max((abs(sample) for sample in frame.data), default=0)


async def _wait_for_agent_participant(room: rtc.Room, timeout_seconds: float) -> rtc.RemoteParticipant:
    agent_kind = rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
    for participant in room.remote_participants.values():
        if participant.kind == agent_kind:
            return participant

    loop = asyncio.get_running_loop()
    future: asyncio.Future[rtc.RemoteParticipant] = loop.create_future()

    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant) -> None:
        if participant.kind == agent_kind and not future.done():
            future.set_result(participant)

    return await asyncio.wait_for(future, timeout=timeout_seconds)


def _describe_track_publications(
    participant: rtc.RemoteParticipant,
) -> list[dict[str, str]]:
    publications: list[dict[str, str]] = []
    for publication in participant.track_publications.values():
        publications.append(
            {
                "sid": publication.sid,
                "name": publication.name,
                "kind": str(publication.kind),
                "source": str(publication.source),
                "subscribed": str(publication.subscribed),
            }
        )
    return publications


async def _wait_for_agent_audio_track(
    participant: rtc.RemoteParticipant,
    timeout_seconds: float,
) -> rtc.Track | None:
    started = time.perf_counter()
    while time.perf_counter() - started < timeout_seconds:
        for publication in participant.track_publications.values():
            track = publication.track
            if track is None:
                continue
            if publication.kind == rtc.TrackKind.KIND_AUDIO:
                logger.info(
                    "Resolved agent audio track sid=%s name=%s source=%s",
                    publication.sid,
                    publication.name,
                    publication.source,
                )
                return track
        await asyncio.sleep(0.1)
    logger.info(
        "Timed out waiting for agent audio track participant=%s timeout_seconds=%s publications=%s",
        participant.identity,
        timeout_seconds,
        _describe_track_publications(participant),
    )
    return None


async def _publish_question_audio(
    room: rtc.Room,
    tts: deepgram.TTS,
    text: str,
) -> dict[str, Any]:
    logger.info("Publishing client question audio question_text=%s", text)
    source = rtc.AudioSource(sample_rate=tts.sample_rate, num_channels=tts.num_channels)
    track = rtc.LocalAudioTrack.create_audio_track("smoke-client-mic", source)
    publication = await room.local_participant.publish_track(
        track,
        TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
    )

    tts_stream = tts.synthesize(text)
    frame_count = 0
    audio_duration_ms = 0.0
    try:
        async for event in tts_stream:
            frame_count += 1
            audio_duration_ms += event.frame.duration * 1000
            await source.capture_frame(event.frame)
        await source.wait_for_playout()
        await asyncio.sleep(1.5)
    finally:
        await tts_stream.aclose()
        await room.local_participant.unpublish_track(publication.sid)
        await source.aclose()

    return {
        "published_frames": frame_count,
        "published_audio_duration_ms": round(audio_duration_ms, 2),
    }


async def _collect_stream_utterance(
    stream: rtc.AudioStream,
    *,
    hard_timeout_seconds: float,
    silence_timeout_seconds: float,
) -> dict[str, Any]:
    frames: list[rtc.AudioFrame] = []
    started = time.perf_counter()
    first_non_silent_at: float | None = None
    trailing_silence_ms = 0.0

    while time.perf_counter() - started < hard_timeout_seconds:
        timeout = silence_timeout_seconds if first_non_silent_at is not None else hard_timeout_seconds
        try:
            event = await asyncio.wait_for(stream.__anext__(), timeout=timeout)
        except asyncio.TimeoutError:
            break
        except StopAsyncIteration:
            logger.info(
                "Audio stream ended before utterance completed hard_timeout_seconds=%s silence_timeout_seconds=%s",
                hard_timeout_seconds,
                silence_timeout_seconds,
            )
            break
        frame = event.frame
        peak_level = _frame_peak_level(frame)
        if peak_level >= 500:
            if first_non_silent_at is None:
                first_non_silent_at = time.perf_counter()
            trailing_silence_ms = 0.0
            frames.append(frame)
            continue

        if first_non_silent_at is None:
            continue

        trailing_silence_ms += frame.duration * 1000
        if trailing_silence_ms >= silence_timeout_seconds * 1000:
            break
        frames.append(frame)

    result = {
        "frames": frames,
        "first_non_silent_at": first_non_silent_at,
    }
    logger.info(
        "Collected stream utterance frame_count=%s first_non_silent_seen=%s hard_timeout_seconds=%s silence_timeout_seconds=%s",
        len(frames),
        first_non_silent_at is not None,
        hard_timeout_seconds,
        silence_timeout_seconds,
    )
    return result


async def run_smoke(
    question_text: str = "If I return the policy within ten days, do I get my premium back?",
    room_name_prefix: str = "voice-agent-conversation",
    expected_keywords: list[str] | None = None,
) -> dict[str, Any]:
    settings = load_settings()
    keywords = expected_keywords or ["10", "ten", "premium", "refund"]
    logger.info(
        "Starting LiveKit smoke run question_text=%s expected_keywords=%s",
        question_text,
        keywords,
    )
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    http_session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_ctx))
    room_name = f"{room_name_prefix}-{time.time_ns()}"
    room = rtc.Room()
    lkapi = api.LiveKitAPI(
        url=settings.livekit_url,
        api_key=settings.livekit_api_key,
        api_secret=settings.livekit_api_secret,
        session=http_session,
    )
    tts_metrics: list[dict[str, float]] = []
    stt_metrics: list[dict[str, float]] = []
    tts = deepgram.TTS(
        model="aura-2-asteria-en",
        api_key=settings.deepgram_api_key,
        http_session=http_session,
    )
    stt = deepgram.STT(
        model="nova-3",
        language="en-US",
        api_key=settings.deepgram_api_key,
        http_session=http_session,
    )
    tts.on(
        "metrics_collected",
        lambda metric: tts_metrics.append(
            {
                "ttfb_ms": round(metric.ttfb * 1000, 2),
                "duration_ms": round(metric.duration * 1000, 2),
                "audio_duration_ms": round(metric.audio_duration * 1000, 2),
            }
        ),
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

    response_text = ""
    response_keyword_hits: list[str] = []
    agent_participant: rtc.RemoteParticipant | None = None
    question_started_at = 0.0
    first_response_frame_at: float | None = None
    published_audio: dict[str, Any] = {}
    response_frames: list[rtc.AudioFrame] = []
    stream: rtc.AudioStream | None = None

    try:
        await lkapi.room.create_room(CreateRoomRequest(name=room_name))
        logger.info("Created LiveKit room room_name=%s", room_name)
        client_token = _build_room_token(
            settings=settings,
            room_name=room_name,
            identity="smoke-client",
            name="smoke-client",
        )
        await room.connect(settings.livekit_url, client_token)
        logger.info("Connected smoke client to room room_name=%s", room_name)
        dispatch = await lkapi.agent_dispatch.create_dispatch(
            CreateAgentDispatchRequest(
                agent_name=settings.agent_name,
                room=room_name,
            )
        )
        logger.info(
            "Created agent dispatch room_name=%s dispatch_id=%s",
            room_name,
            dispatch.id,
        )
        agent_participant = await _wait_for_agent_participant(room, timeout_seconds=30)
        logger.info(
            "Agent joined room room_name=%s agent_identity=%s",
            room_name,
            agent_participant.identity,
        )
        logger.info(
            "Agent track publications on join room_name=%s tracks=%s",
            room_name,
            _describe_track_publications(agent_participant),
        )

        @room.on("track_published")
        def on_track_published(
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            if participant.identity != agent_participant.identity:
                return
            logger.info(
                "Agent track published room_name=%s sid=%s name=%s kind=%s source=%s subscribed=%s",
                room_name,
                publication.sid,
                publication.name,
                publication.kind,
                publication.source,
                publication.subscribed,
            )

        greeting_track = await _wait_for_agent_audio_track(
            agent_participant,
            timeout_seconds=3,
        )
        if greeting_track is not None:
            stream = rtc.AudioStream.from_track(
                track=greeting_track,
                sample_rate=24000,
                num_channels=1,
                frame_size_ms=20,
            )
            greeting = await _collect_stream_utterance(
                stream,
                hard_timeout_seconds=12,
                silence_timeout_seconds=1.2,
            )
            logger.info(
                "Initial greeting probe room_name=%s frame_count=%s",
                room_name,
                len(greeting["frames"]),
            )
            await stream.aclose()
            stream = None
            await asyncio.sleep(0.75)
        else:
            logger.info("No initial agent audio track yet room_name=%s", room_name)

        question_started_at = time.perf_counter()
        published_audio = await _publish_question_audio(room, tts, question_text)
        response_track = await _wait_for_agent_audio_track(
            agent_participant,
            timeout_seconds=15,
        )
        if response_track is None:
            raise RuntimeError("No agent audio track was published after the client question")
        stream = rtc.AudioStream.from_track(
            track=response_track,
            sample_rate=24000,
            num_channels=1,
            frame_size_ms=20,
        )
        collect_task = asyncio.create_task(
            _collect_stream_utterance(
                stream,
                hard_timeout_seconds=30,
                silence_timeout_seconds=2.2,
            )
        )
        response_audio = await collect_task
        response_frames = response_audio["frames"]
        first_response_frame_at = response_audio["first_non_silent_at"]
        logger.info(
            "Received agent response audio room_name=%s response_frame_count=%s published_audio=%s",
            room_name,
            len(response_frames),
            published_audio,
        )

        if not response_frames:
            raise RuntimeError("No agent audio was received after publishing the client question")

        transcription = await stt.recognize(response_frames, language="en-US")
        response_text = transcription.alternatives[0].text if transcription.alternatives else ""
        response_keyword_hits = _keyword_hits(response_text, keywords)
        logger.info(
            "Transcribed agent response room_name=%s response_text=%s keyword_hits=%s",
            room_name,
            response_text,
            response_keyword_hits,
        )
    finally:
        await stt.aclose()
        await tts.aclose()
        if stream is not None:
            await stream.aclose()
        if room.isconnected():
            await room.disconnect()
        with contextlib.suppress(Exception):
            await lkapi.room.delete_room(DeleteRoomRequest(room=room_name))
        await lkapi.aclose()
        await http_session.close()

    response_audio_duration_ms = round(
        sum(frame.duration for frame in response_frames) * 1000,
        2,
    )
    return {
        "room_name": room_name,
        "agent_identity": agent_participant.identity if agent_participant else "",
        "dispatch_created": True,
        "question_text": question_text,
        "published_audio": published_audio,
        "agent_response_frames": len(response_frames),
        "agent_response_audio_duration_ms": response_audio_duration_ms,
        "response_text": response_text,
        "response_keyword_hits": response_keyword_hits,
        "response_latency_ms": (
            round((first_response_frame_at - question_started_at) * 1000, 2)
            if question_started_at and first_response_frame_at is not None
            else None
        ),
        "round_trip_success": len(response_keyword_hits) >= max(1, len(keywords) // 2),
        "tts_metrics": tts_metrics,
        "stt_metrics": stt_metrics,
        "llm_model": settings.openai_model,
        "llm_base_url": settings.openai_base_url,
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
