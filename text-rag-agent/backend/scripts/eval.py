"""Offline RAG evaluation harness (batch regression companion to the online judge).

Runs a gold Q/A set through the *real* pipeline (ingest -> retrieve -> gate -> answer),
then scores each answer with the SAME `judge_answer` used online in the request path — so
this measures the real system and shares one judge definition. Reports where the answer
came from (context / history / both / none), groundedness, and correctness.

The online judge (per-query verdict returned in the response) lives in the request path;
this script is for repeatable batch checks over a curated gold set.

Run (needs GEMINI_API_KEY + a reachable Qdrant + Redis, like the integration tests):

    cd backend && uv run python scripts/eval.py            # default dataset
    cd backend && uv run python scripts/eval.py --dataset scripts/eval_dataset.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

# Make `app` importable when run as `python scripts/eval.py` from the backend dir.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.llm import aclose_llm, get_llm  # noqa: E402
from app.rag.judge import judge_answer  # noqa: E402
from app.rag.pipeline import ingest_document, retrieve  # noqa: E402
from app.rag.prompts import NO_ANSWER, build_context, build_messages  # noqa: E402
from app.rag.vectorstore import aclose_vector_store, get_vector_store  # noqa: E402
from app.redis_client import aclose_redis  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Msg:
    """Minimal stand-in for a stored message; `build_messages` only reads role/content."""

    role: str
    content: str


async def _answer(user_id: UUID, case: dict, top_k: int, min_score: float) -> tuple[str, str]:
    """Run the real retrieve -> gate -> answer path. Returns (answer, retrieved_context)."""
    history = [Msg(m["role"], m["content"]) for m in case.get("history", [])]
    hits, grounded = await retrieve(
        user_id=user_id, question=case["question"], top_k=top_k, min_score=min_score
    )
    context = build_context(hits)
    if not grounded:
        return NO_ANSWER, context
    messages = build_messages(case["question"], hits, history)
    return await get_llm().complete(messages), context


async def _judge(case: dict, answer: str, context: str) -> dict:
    history = "\n".join(f"{m['role']}: {m['content']}" for m in case.get("history", []))
    return await judge_answer(question=case["question"], context=context, history=history, answer=answer)


async def run(dataset_path: Path, top_k: int, min_score: float) -> int:
    spec = json.loads(dataset_path.read_text())
    doc_path = REPO_ROOT / spec["document"]
    user_id, document_id = uuid4(), uuid4()

    print(f"Ingesting {doc_path.name} (eval tenant {user_id}) ...")
    n_chunks = await ingest_document(
        user_id=user_id, document_id=document_id, filename=doc_path.name, data=doc_path.read_bytes()
    )
    print(f"  {n_chunks} chunks indexed.\n")

    rows: list[dict] = []
    try:
        for case in spec["cases"]:
            answer, context = await _answer(user_id, case, top_k, min_score)
            verdict = await _judge(case, answer, context)
            refused = answer.strip() == NO_ANSWER
            expected_refuse = case["expect"] == "refuse"
            # A case passes when the answer is correct AND grounded, and the refuse/answer
            # decision matched the expectation.
            ok = (
                bool(verdict.get("correct"))
                and bool(verdict.get("grounded"))
                and (refused == expected_refuse)
            )
            rows.append({"case": case, "answer": answer, "verdict": verdict, "refused": refused, "ok": ok})
    finally:
        await get_vector_store().delete_document(user_id=user_id, document_id=document_id)

    _report(rows)
    passed = sum(r["ok"] for r in rows)
    return 0 if passed == len(rows) else 1


def _report(rows: list[dict]) -> None:
    print("=" * 78)
    for r in rows:
        c, v = r["case"], r["verdict"]
        mark = "PASS" if r["ok"] else "FAIL"
        act = "refused" if r["refused"] else "answered"
        print(f"[{mark}] {c['id']}  (expect={c['expect']}, got={act})")
        print(f"       Q: {c['question']}")
        print(f"       A: {r['answer'][:160].strip()}")
        print(
            f"       judge: source={v.get('source')} grounded={v.get('grounded')} "
            f"correct={v.get('correct')} — {v.get('reason')}"
        )
        if c.get("note"):
            print(f"       note: {c['note']}")
        print("-" * 78)

    n = len(rows)
    passed = sum(r["ok"] for r in rows)
    grounded = sum(bool(r["verdict"].get("grounded")) for r in rows)
    hallucinated = sum(not r["verdict"].get("grounded") and not r["refused"] for r in rows)
    refuse_cases = [r for r in rows if r["case"]["expect"] == "refuse"]
    refuse_ok = sum(r["refused"] for r in refuse_cases)
    print(
        f"SUMMARY: {passed}/{n} passed | grounded {grounded}/{n} | "
        f"answered-but-ungrounded (hallucinations) {hallucinated} | "
        f"correct refusals {refuse_ok}/{len(refuse_cases)}"
    )
    print("=" * 78)


def _selfcheck() -> None:
    """ponytail: one runnable check on the shared verdict parser."""
    from app.rag.judge import parse_verdict

    fenced = '```json\n{"source":"context","grounded":true,"correct":true,"reason":"x"}\n```'
    padded = 'noise {"source":"none","grounded":false,"correct":false,"reason":"y"} tail'
    assert parse_verdict(fenced)["grounded"] is True
    assert parse_verdict(padded)["source"] == "none"
    assert parse_verdict("garbage")["grounded"] is False
    print("selfcheck ok")


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG eval harness with LLM-as-Judge.")
    ap.add_argument("--dataset", default=str(Path(__file__).with_name("eval_dataset.json")))
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--min-score", type=float, default=0.3)
    ap.add_argument("--selfcheck", action="store_true", help="run the JSON-parser self-check and exit")
    args = ap.parse_args()

    if args.selfcheck:
        _selfcheck()
        return

    async def _main() -> int:
        try:
            return await run(Path(args.dataset), args.top_k, args.min_score)
        finally:
            await aclose_llm()
            await aclose_vector_store()
            await aclose_redis()

    sys.exit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
