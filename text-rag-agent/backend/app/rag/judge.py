"""LLM-as-Judge: score a generated answer for source, groundedness, and correctness.

Given the question, the retrieved context, and the prior conversation, the judge decides
where the answer's content came from (context / history / both / none), whether every
claim is supported (no fabrication), and whether it correctly answers the question. Used
online (per query, verdict returned in the same request) and by the offline eval harness.
"""

from __future__ import annotations

import json

import structlog

from app.llm import get_llm

log = structlog.get_logger()

JUDGE_SYSTEM = (
    "You are a strict evaluator of a document-grounded assistant. You are given a user "
    "question, the numbered CONTEXT passages retrieved for it, the prior CONVERSATION, and "
    "the assistant's ANSWER. Decide, only from what you are given:\n"
    '- source: where the answer\'s content comes from — "context" (the passages), '
    '"history" (the prior conversation), "both", or "none" (a refusal or empty answer).\n'
    "- grounded: true only if every claim in the answer is supported by the context or the "
    "conversation. A refusal is grounded.\n"
    "- correct: true if the answer correctly and adequately answers the question given the "
    "available information. If the answer is a refusal, correct is true when the information "
    "was genuinely absent from both the context and the conversation (a correct refusal).\n"
    "- reason: one short sentence.\n"
    "- used: an object mapping EVERY source number the answer cites inline as [n] to the SHORTEST "
    "EXACT substring COPIED VERBATIM from that source (same characters, casing, and punctuation) "
    "that supports the answer. Cover every cited [n]; if a cited source was paraphrased, still pick "
    "the closest verbatim phrase from it. Omit only sources the answer never cites. Empty object "
    "for a refusal.\n"
    'Reply with ONLY a JSON object: {"source": "...", "grounded": bool, "correct": bool, '
    '"reason": "...", "used": {"1": "verbatim quote", ...}}'
)


def parse_verdict(raw: str) -> dict:
    """Extract the verdict JSON from a model reply that may be fenced or padded."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1].removeprefix("json").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return {"source": "?", "grounded": False, "correct": False, "reason": "unparseable verdict"}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {"source": "?", "grounded": False, "correct": False, "reason": "unparseable verdict"}


async def judge_answer(*, question: str, context: str, history: str, answer: str) -> dict:
    """Return {source, grounded, correct, reason}. One extra LLM call."""
    user = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context or '(nothing retrieved)'}\n\n"
        f"CONVERSATION:\n{history or '(none)'}\n\n"
        f"ANSWER:\n{answer}"
    )
    raw = await get_llm().complete(
        [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]
    )
    return parse_verdict(raw)
