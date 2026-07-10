"""OpenAI-compatible LLM client (chat streaming + embeddings). One reused async client.

Defaults to Gemini's OpenAI endpoint; swap base_url/models/key in config for any
OpenAI-compatible provider. The SDK handles transient-error retries and timeouts.
Query embeddings are cached in Redis so repeated questions skip the embed API call.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, cast

import structlog
from openai import AsyncOpenAI

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

from app.config import get_settings
from app.errors import PROVIDER_UNAVAILABLE, RATE_LIMITED, classify
from app.redis_client import get_redis

log = structlog.get_logger()

_EMBED_CACHE_TTL_S = 60 * 60 * 24  # query-embedding cache


class ProviderUnavailable(Exception):
    """Breaker is open — fail fast instead of piling requests on a failing provider."""

    status_code = 503


class _Breaker:
    """Consecutive-failure breaker: opens after `threshold` provider failures, rejects for
    `cooldown` seconds, then lets a trial through."""

    def __init__(self, threshold: int, cooldown_s: int) -> None:
        self._threshold, self._cooldown = threshold, cooldown_s
        self._fails = 0
        self._open_until = 0.0

    def allow(self) -> bool:
        return not (self._fails >= self._threshold and time.monotonic() < self._open_until)

    def record(self, exc: BaseException | None) -> None:
        if exc is None:
            self._fails = 0
            return
        if classify(exc).code in (RATE_LIMITED, PROVIDER_UNAVAILABLE):
            self._fails += 1
            if self._fails >= self._threshold:
                self._open_until = time.monotonic() + self._cooldown


class LLM:
    def __init__(self) -> None:
        s = get_settings()
        self._client = AsyncOpenAI(
            api_key=s.gemini_api_key or "EMPTY",
            base_url=s.llm_base_url,
            timeout=s.llm_timeout_s,
            max_retries=s.llm_max_retries,
        )
        self._chat_model = s.chat_model
        self._embed_model = s.embed_model
        self._embed_dim = s.embed_dim
        self._temperature = s.llm_temperature
        self._breaker = _Breaker(s.llm_breaker_threshold, s.llm_breaker_cooldown_s)

    def _guard(self) -> None:
        if not self._breaker.allow():
            raise ProviderUnavailable("LLM provider circuit is open")

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        self._guard()
        kwargs: dict = {"model": self._embed_model, "input": texts}
        if self._embed_dim:
            kwargs["dimensions"] = self._embed_dim  # ignored by providers that don't support it
        try:
            resp = await self._client.embeddings.create(**kwargs)
        except Exception as exc:
            self._breaker.record(exc)
            raise
        self._breaker.record(None)
        return [d.embedding for d in resp.data]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts)

    async def embed_query(self, text: str) -> list[float]:
        key = f"emb:{self._embed_model}:{self._embed_dim}:{hashlib.sha256(text.encode()).hexdigest()}"
        r = get_redis()
        try:
            cached = await r.get(key)
            if cached:
                return json.loads(cached)
        except Exception as exc:  # noqa: BLE001 — cache is best-effort
            log.warning("embed_cache.read_failed", error=str(exc))
        vec = (await self._embed([text]))[0]
        try:
            await r.set(key, json.dumps(vec), ex=_EMBED_CACHE_TTL_S)
        except Exception as exc:  # noqa: BLE001
            log.warning("embed_cache.write_failed", error=str(exc))
        return vec

    async def complete(self, messages: list[dict]) -> str:
        self._guard()
        try:
            resp = await self._client.chat.completions.create(
                model=self._chat_model,
                messages=cast("list[ChatCompletionMessageParam]", messages),
                temperature=self._temperature,
            )
        except Exception as exc:
            self._breaker.record(exc)
            raise
        self._breaker.record(None)
        return resp.choices[0].message.content or ""

    async def stream_chat(
        self, messages: list[dict], usage_out: dict | None = None, temperature: float | None = None
    ) -> AsyncIterator[str]:
        """Yield answer text deltas. Final token usage is written into usage_out."""
        self._guard()
        try:
            stream = await self._client.chat.completions.create(
                model=self._chat_model,
                messages=cast("list[ChatCompletionMessageParam]", messages),
                temperature=self._temperature if temperature is None else temperature,
                stream=True,
                stream_options={"include_usage": True},
            )
        except Exception as exc:
            self._breaker.record(exc)
            raise
        self._breaker.record(None)
        async for chunk in stream:
            if chunk.usage and usage_out is not None:
                usage_out["input_tokens"] = chunk.usage.prompt_tokens
                usage_out["output_tokens"] = chunk.usage.completion_tokens
                usage_out["total_tokens"] = chunk.usage.total_tokens
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def aclose(self) -> None:
        await self._client.close()


_llm: LLM | None = None


def get_llm() -> LLM:
    global _llm
    if _llm is None:
        _llm = LLM()
    return _llm


async def aclose_llm() -> None:
    global _llm
    if _llm is not None:
        await _llm.aclose()
        _llm = None
