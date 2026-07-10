from __future__ import annotations

from voice_agent.utils.doc_ingestion import clean_page_text, split_into_paragraphs


def test_clean_page_text_removes_table_of_contents_page() -> None:
    page = (
        "Policy Schedule 2\n"
        "A Terms used in this Policy 5\n"
        "B When this Policy is in force 7\n"
        "C Benefits provided by this Policy 8\n"
        "D Paying your Premiums 10\n"
        "E Policy options 12\n"
        "F Information about this Policy 15\n"
        "G General provisions 16\n"
        "H Claims information 17\n"
        "Provincial amendments 17\n"
    )

    cleaned = clean_page_text(page)

    assert cleaned == ""


def test_split_into_paragraphs_breaks_policy_sections() -> None:
    page = (
        "D 2 Grace period for paying your Premiums\n"
        "If any Premium is not paid by its due date, this Policy is in default. "
        "We allow a grace period of thirty-one days after the premium due date. "
        "If the premium is still unpaid at the end of the grace period, the policy terminates automatically. "
        "If the insured dies during the grace period, the outstanding premium is deducted from the death benefit.\n"
        "D 3 Reinstating your Policy after it has ended due to non-payment of Premium\n"
        "If this Policy terminates due to non-payment of Premium, it may be reinstated "
        "within two years after the end of the grace period. "
        "A written request is required, each life insured must still be insurable, "
        "and unpaid premiums plus interest must be paid before reinstatement is approved.\n"
    )

    paragraphs = split_into_paragraphs(page)

    assert len(paragraphs) == 2
    assert paragraphs[0].startswith("D 2 Grace period")
    assert paragraphs[1].startswith("D 3 Reinstating")
