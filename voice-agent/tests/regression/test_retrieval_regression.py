from __future__ import annotations

from pathlib import Path

from voice_agent.kb import KnowledgeBase, KnowledgeDocument


def build_kb(tmp_path: Path) -> KnowledgeBase:
    kb = KnowledgeBase(tmp_path / "chroma", "support")
    kb.upsert_documents(
        [
            KnowledgeDocument(
                id="free-look",
                title="Ten day policy examination and refund",
                content=(
                    "If you return the policy within ten days of receiving it, "
                    "the policy is treated as never issued and any premium paid is refunded."
                ),
                metadata={"source_url": "https://example.com/policy.pdf"},
            ),
            KnowledgeDocument(
                id="grace-period",
                title="Grace period for paying premiums",
                content="A grace period of thirty-one days is allowed after the premium due date.",
                metadata={"source_url": "https://example.com/policy.pdf"},
            ),
        ]
    )
    return kb


def test_free_look_query_keeps_refund_clause_first(tmp_path: Path) -> None:
    kb = build_kb(tmp_path)

    result = kb.search("refund premium 10 day free look period", limit=1)

    assert result[0].id == "free-look"


def test_grace_period_query_keeps_grace_period_doc_first(tmp_path: Path) -> None:
    kb = build_kb(tmp_path)

    result = kb.search("How long is the grace period if I miss a premium payment?", limit=1)

    assert result[0].id == "grace-period"
