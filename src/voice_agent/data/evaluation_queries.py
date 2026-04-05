from __future__ import annotations

EVALUATION_CASES = [
    {
        "question": "If I cancel and return the policy within ten days, do I get my premium back?",
        "expected_keywords": ["ten", "10", "premium", "refund"],
        "forbidden_keywords": ["thirty", "30"],
    },
    {
        "question": "How long is the grace period if I miss a premium payment?",
        "expected_keywords": ["grace", "31", "days"],
        "forbidden_keywords": [],
    },
    {
        "question": "When will the company refuse to pay the death benefit?",
        "expected_keywords": ["suicide", "two", "years"],
        "forbidden_keywords": [],
    },
    {
        "question": "Can I reinstate the policy after it ends because I stopped paying premiums?",
        "expected_keywords": ["reinstate", "two", "years"],
        "forbidden_keywords": ["three", "3", "five", "5"],
    },
    {
        "question": "How do I contact the insurer if I have questions about the policy?",
        "expected_keywords": ["contact", "1-800", "claims"],
        "forbidden_keywords": [],
    },
]

EVALUATION_QUERIES = [case["question"] for case in EVALUATION_CASES]
