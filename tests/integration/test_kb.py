from __future__ import annotations

from pathlib import Path

from voice_agent.kb import KnowledgeBase, KnowledgeDocument


TEST_DOCUMENTS = [
    KnowledgeDocument(
        id="free-look",
        title="Ten day policy examination and refund",
        content=(
            "You have ten days from the date you receive the policy to examine it. "
            "If you return it within that ten day period, the policy is deemed never issued "
            "and any premium paid will be refunded."
        ),
        metadata={"source_url": "https://example.com/policy.pdf", "page_number": "1"},
    ),
    KnowledgeDocument(
        id="grace-period",
        title="Grace period for paying premiums",
        content=(
            "If a premium is not paid by its due date, the policy is in default. "
            "A grace period of thirty-one days after the premium due date is allowed."
        ),
        metadata={"source_url": "https://example.com/policy.pdf", "page_number": "8"},
    ),
]


def build_kb(tmp_path: Path) -> KnowledgeBase:
    kb = KnowledgeBase(tmp_path / "chroma", "support")
    kb.upsert_documents(TEST_DOCUMENTS)
    return kb


def test_search_returns_free_look_policy_first(tmp_path: Path) -> None:
    kb = build_kb(tmp_path)

    results = kb.search(
        "If I return the policy within ten days, do I get my premium back?",
        limit=2,
    )

    assert results
    assert results[0].id == "free-look"


def test_search_returns_empty_list_for_blank_query(tmp_path: Path) -> None:
    kb = build_kb(tmp_path)

    assert kb.search("   ") == []


def test_upsert_documents_updates_existing_record(tmp_path: Path) -> None:
    kb = KnowledgeBase(tmp_path / "chroma", "support")
    kb.upsert_documents(
        [
            KnowledgeDocument(
                id="policy",
                title="Policy",
                content="Original policy text",
                metadata={"category": "policy"},
            )
        ]
    )
    kb.upsert_documents(
        [
            KnowledgeDocument(
                id="policy",
                title="Policy",
                content="Updated policy text",
                metadata={"category": "policy"},
            )
        ]
    )

    result = kb.search("updated policy", limit=1)

    assert kb.count() == 1
    assert result[0].content == "Updated policy text"


def test_search_returns_source_metadata(tmp_path: Path) -> None:
    kb = build_kb(tmp_path)

    results = kb.search("How long is the grace period?", limit=1)

    assert results
    assert results[0].metadata["source_url"] == "https://example.com/policy.pdf"
