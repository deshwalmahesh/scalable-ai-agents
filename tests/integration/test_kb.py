from __future__ import annotations

from pathlib import Path

from voice_agent.kb import (
    KnowledgeBase,
    KnowledgeDocument,
    RetrievalService,
    build_grounding_message,
    clean_grounding_snippet_text,
    select_grounding_snippet,
)


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


def test_search_keeps_grace_period_out_of_top_results_for_cancel_refund_query(
    tmp_path: Path,
) -> None:
    kb = build_kb(tmp_path)

    results = kb.search(
        "If I cancel and return the policy within ten days, do I get my premium back?",
        limit=1,
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


def test_search_with_trace_reports_timing_metadata(tmp_path: Path) -> None:
    kb = build_kb(tmp_path)

    results, trace = kb.search_with_trace("How long is the grace period?", limit=1)

    assert results
    assert trace.mode in {"lexical", "hybrid"}
    assert trace.total_ms >= 0.0


async def test_retrieval_service_returns_grounding_message(tmp_path: Path) -> None:
    kb = build_kb(tmp_path)
    retrieval = RetrievalService(kb)

    result = await retrieval.async_ground(
        "If I return the policy within ten days, do I get my premium back?",
        limit=1,
    )

    assert result.matches
    assert "Grounding for this turn only" in result.grounding_message
    assert result.trace.total_ms >= 0.0


def test_grounding_message_requires_exact_supported_values() -> None:
    message = build_grounding_message(
        query="If I cancel and return the policy within ten days, do I get my premium back?",
        matches=[
            TEST_DOCUMENTS[0],
        ],
    )

    assert "repeat that exact value" in message
    assert "do not broaden it into a range" in message
    assert "answer the supported part" in message
    assert "state that condition directly" in message


def test_select_grounding_snippet_prefers_exact_answer_sentence() -> None:
    content = (
        "This policy contains standard explanatory language for the owner. "
        "If you return it within that ten day period, the policy is deemed never issued "
        "and any premium paid will be refunded. "
        "General policy notices follow after this section."
    )

    snippet = select_grounding_snippet(
        "If I cancel and return the policy within ten days, do I get my premium back?",
        content,
    )

    assert "ten day period" in snippet
    assert "premium paid will be refunded" in snippet
    assert "General policy notices follow" not in snippet


def test_build_grounding_message_uses_trimmed_snippet_for_large_match() -> None:
    message = build_grounding_message(
        query="If I cancel and return the policy within ten days, do I get my premium back?",
        matches=[
            KnowledgeDocument(
                id="free-look-large",
                title="Free look policy text",
                content=(
                    "Opening page boilerplate for the policy owner. "
                    "If you return it within that ten day period, the policy is deemed never issued "
                    "and any premium paid will be refunded. "
                    "Closing legal boilerplate and notices."
                ),
                metadata={"source_url": "https://example.com/policy.pdf"},
            )
        ],
    )

    assert "premium paid will be refunded" in message
    assert "Opening page boilerplate" not in message
    assert "Closing legal boilerplate" not in message


def test_select_grounding_snippet_keeps_following_clause_for_lead_in() -> None:
    content = (
        "The policy may be reinstated subject to the following conditions: "
        "It must be reinstated within five years after the due date of the first unpaid premium. "
        "Additional boilerplate follows after this section."
    )

    snippet = select_grounding_snippet(
        "Can I reinstate the policy after it ends because I stopped paying premiums?",
        content,
    )

    assert "within five years" in snippet
    assert "Additional boilerplate follows" not in snippet


def test_select_grounding_snippet_splits_numbered_policy_conditions() -> None:
    content = (
        "The policy may be reinstated subject to the following conditions: "
        "1. You must make a written request for reinstatement within two years after the end of the grace period. "
        "2. Every insured person aged sixteen or older must provide evidence of insurability."
    )

    snippet = select_grounding_snippet(
        "Can I reinstate the policy after it ends because I stopped paying premiums?",
        content,
    )

    assert "within two years" in snippet
    assert "evidence of insurability" not in snippet


def test_clean_grounding_snippet_text_trims_heading_preamble() -> None:
    snippet = clean_grounding_snippet_text(
        "D 3 Reinstating your Policy after lapse If this policy terminates due to non-payment, it may be reinstated within two years."
    )

    assert snippet == "If this policy terminates due to non-payment, it may be reinstated within two years."
