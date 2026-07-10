"""Streaming grounded answer: assemble memory -> retrieve -> gate -> stream -> persist.

Memory: recent turns are kept verbatim up to the per-chat token budget; older turns are
folded into a rolling summary (watermark-tracked, so they are never re-summarized or lost).
The DB session is never held during the LLM token stream.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator
from uuid import UUID

import structlog

from app.chat_settings import ChatSettings, resolve_settings
from app.config import get_settings
from app.db import get_session_factory
from app.db.models import MessageDB
from app.db.repos import add_message, get_chat, messages_since, set_chat_memory, set_message_analysis
from app.errors import classify
from app.llm import get_llm
from app.quota import add_usage
from app.rag.judge import judge_answer
from app.rag.pipeline import retrieve
from app.rag.prompts import NO_ANSWER, SUMMARY_SYSTEM, build_context, build_messages, hits_to_citations

log = structlog.get_logger()


def _sse(event: str, data) -> dict:
    return {"event": event, "data": json.dumps(data)}


def _est_tokens(text: str) -> int:
    return len(text) // get_settings().chars_per_token + 1


def _apply_used_spans(citations: list[dict], used: dict) -> list[dict]:
    """Attach `used_quote` to each citation: the verbatim span the answer drew from that source
    (located case-insensitively in the snippet). Missing / not-found → no highlight (graceful)."""
    for c in citations:
        quote = (used or {}).get(str(c["n"]))
        c["used_quote"] = None
        if isinstance(quote, str) and quote.strip():
            snippet = c.get("snippet", "")
            i = snippet.lower().find(quote.strip().lower())
            if i != -1:
                c["used_quote"] = snippet[i : i + len(quote.strip())]
    return citations


def split_window(pending: list[MessageDB], settings: ChatSettings) -> tuple[list[MessageDB], list[MessageDB]]:
    """Split unsummarized messages into (verbatim_window, to_summarize).

    Keeps the newest `history_turns` messages regardless (coreference floor), then keeps
    more recent ones while they fit the history token budget. Everything older is returned
    for summarization so it is preserved, not dropped."""
    s = get_settings()
    reserve = settings.top_k * s.chunk_size // s.chars_per_token + 1024  # sources + answer headroom
    budget = max(1000, settings.context_token_budget - reserve)
    window: list[MessageDB] = []
    tokens = 0
    for m in reversed(pending):
        if len(window) >= settings.history_turns and tokens + _est_tokens(m.content) > budget:
            break
        window.append(m)
        tokens += _est_tokens(m.content)
    window.reverse()
    to_summarize = pending[: len(pending) - len(window)]
    return window, to_summarize


async def _summarize(prior: str | None, msgs: list[MessageDB]) -> str:
    convo = "\n".join(f"{m.role}: {m.content}" for m in msgs)
    user = (f"Existing summary:\n{prior}\n\n" if prior else "") + f"New messages:\n{convo}"
    return await get_llm().complete(
        [{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": user}]
    )


async def stream_answer(
    *, user_id: UUID, chat_id: UUID, question: str, document_ids: list[UUID] | None
) -> AsyncIterator[dict]:
    sf = get_session_factory()
    async with sf() as s:
        chat = await get_chat(s, chat_id=chat_id, user_id=user_id)
        if chat is None:  # deleted between the route check and here
            yield _sse("error", {"message": "That chat could not be found."})
            return
        settings = resolve_settings(chat.settings)
        prior_summary = chat.summary
        pending = await messages_since(s, chat_id=chat_id, after=chat.summary_upto)
        await add_message(s, chat_id=chat_id, role="user", content=question)

    window, to_summarize = split_window(pending, settings)
    summary_text = prior_summary
    if to_summarize:
        summary_text = await _summarize(prior_summary, to_summarize)
        async with sf() as s:
            await set_chat_memory(
                s, chat_id=chat_id, summary=summary_text, summary_upto=to_summarize[-1].created_at
            )

    scope = document_ids if document_ids is not None else settings.document_ids
    t0 = time.monotonic()
    hits, grounded = await retrieve(
        user_id=user_id,
        question=question,
        top_k=settings.top_k,
        min_score=settings.min_score,
        document_ids=scope,
    )
    retrieval_ms = int((time.monotonic() - t0) * 1000)

    if not grounded:
        async with sf() as s:
            msg = await add_message(s, chat_id=chat_id, role="assistant", content=NO_ANSWER, citations=[])
        log.info("answer.no_match", chat_id=str(chat_id), retrieval_ms=retrieval_ms, hits=len(hits))
        yield _sse("token", NO_ANSWER)
        yield _sse("done", {"message_id": str(msg.id), "citations": [], "grounded": False})
        return

    citations = hits_to_citations(hits)
    messages = build_messages(question, hits, window, summary_text)
    usage: dict = {}
    parts: list[str] = []
    t1 = time.monotonic()
    try:
        async for delta in get_llm().stream_chat(
            messages, usage, temperature=settings.temperature, model=settings.model
        ):
            parts.append(delta)
            yield _sse("token", delta)
    except Exception as exc:  # noqa: BLE001 — stream already 200; surface as an SSE error event
        log.error("answer.stream_failed", chat_id=str(chat_id), error=str(exc))
        yield _sse("error", {"message": classify(exc).user_message})
        return

    answer = "".join(parts)
    # A source is "cited" iff the answer references its [n] — the answer is the source of truth
    # for what was used (the judge only adds the exact highlight span, if it can locate one).
    cited_ns = {int(x) for x in re.findall(r"\[(\d+)\]", answer)}
    for c in citations:
        c["cited"] = c["n"] in cited_ns
    # Fallback when the provider omits streaming usage, so quota still meters.
    total_tokens = usage.get("total_tokens") or (
        _est_tokens("".join(m["content"] for m in messages)) + _est_tokens(answer)
    )
    async with sf() as s:
        msg = await add_message(s, chat_id=chat_id, role="assistant", content=answer, citations=citations)
    await add_usage(user_id, total_tokens)
    log.info(
        "answer.done",
        chat_id=str(chat_id),
        retrieval_ms=retrieval_ms,
        generation_ms=int((time.monotonic() - t1) * 1000),
        tokens=total_tokens,
        sources=len(citations),
    )
    yield _sse("done", {"message_id": str(msg.id), "citations": citations, "grounded": True, "usage": usage})

    # Online eval: judge the answer AFTER it (and its citations) have been delivered, so the
    # extra LLM call never delays what the user reads. Verdict arrives as a trailing event.
    if get_settings().judge_enabled:
        try:
            history_text = "\n".join(f"{m.role}: {m.content}" for m in window)
            verdict = await judge_answer(
                question=question, context=build_context(hits), history=history_text, answer=answer
            )
            # Line-level provenance: highlight the exact span each source contributed.
            used = verdict.pop("used", {})
            citations = _apply_used_spans(citations, used)
            async with sf() as s:
                await set_message_analysis(s, message_id=msg.id, citations=citations, verdict=verdict)
            log.info(
                "answer.judged",
                chat_id=str(chat_id),
                source=verdict.get("source"),
                grounded=verdict.get("grounded"),
                correct=verdict.get("correct"),
                spans=sum(1 for c in citations if c.get("used_quote")),
            )
            yield _sse("verdict", {"message_id": str(msg.id), "citations": citations, **verdict})
        except Exception as exc:  # noqa: BLE001 — analysis is best-effort; never fail a delivered answer
            log.warning("answer.judge_failed", chat_id=str(chat_id), error=str(exc))
