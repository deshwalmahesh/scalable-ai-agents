"""Grounded-QA prompt contract. The model answers only from retrieved sources and
must emit NO_ANSWER verbatim when the sources don't contain the answer."""

from __future__ import annotations

from app.rag.vectorstore import Hit

NO_ANSWER = "I couldn't find that in your documents."

SUMMARY_SYSTEM = (
    "You maintain a running summary of a conversation. Merge the new messages into the "
    "existing summary. Preserve user goals, named entities, decisions, and facts already "
    "established. Be concise and factual; do not invent anything."
)

_SYSTEM = (
    "You are a document assistant. Answer the question using ONLY the numbered sources "
    "below. Do not use any outside knowledge. If the answer is not contained in the "
    f'sources, reply with exactly: "{NO_ANSWER}" and nothing else. When you use a source, '
    "cite it inline as [n]."
)


def _location(hit: Hit) -> str:
    return hit.filename + (f" p.{hit.page}" if hit.page else "")


def build_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] ({_location(h)}) {h.text}" for i, h in enumerate(hits, start=1))


def build_messages(question: str, hits: list[Hit], history: list, summary: str | None = None) -> list[dict]:
    system = f"{_SYSTEM}\n\nSources:\n{build_context(hits)}"
    if summary:
        system += f"\n\nConversation summary so far:\n{summary}"
    messages = [{"role": "system", "content": system}]
    messages += [{"role": m.role, "content": m.content} for m in history]
    messages.append({"role": "user", "content": question})
    return messages


def hits_to_citations(hits: list[Hit]) -> list[dict]:
    return [
        {
            "n": i,
            "document_id": h.document_id,
            "filename": h.filename,
            "page": h.page,
            "score": round(h.score, 4),
            "snippet": h.text,
        }
        for i, h in enumerate(hits, start=1)
    ]
