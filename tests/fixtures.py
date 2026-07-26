"""Shared synthetic corpus for the acceptance tests. No network, no model."""
from __future__ import annotations

from typing import Dict

DOCS: Dict[str, str] = {
    "doc_aaa": (
        "Aurora Systems reported quarterly revenue of 412 million dollars on 14 March 2024. "
        "The company said its cloud division grew 22 percent year over year. "
        "Chief executive Dana Reyes attributed the result to enterprise contracts. "
        "Aurora Systems also opened a research office in Lisbon during the quarter."),
    "doc_bbb": (
        "Borealis Metals announced on 2 April 2024 that it would acquire Cinder Mining "
        "for 1.8 billion dollars. Analysts said the deal would double Borealis output. "
        "The transaction is expected to close in the fourth quarter of 2024. "
        "Borealis Metals is headquartered in Winnipeg."),
    "doc_ccc": (
        "Cinder Mining operates three open pit sites in northern Ontario. "
        "Its 2023 annual report listed 940 employees and 310 million dollars in revenue. "
        "Cinder Mining had been searching for a buyer since November 2023."),
}

QUESTION = {
    "question_id": "q_test000001",
    "question": "How much did Borealis Metals agree to pay for Cinder Mining, and how "
                "many employees did Cinder Mining report?",
    "answer": "1.8 billion dollars; 940 employees",
    "question_type": "inference_query",
    "width": 2,
    "documents": [
        {"document_id": "doc_bbb", "identity": "u/bbb", "title": "Borealis",
         "url": "u/bbb", "source": "t", "body_sha256": "x",
         "gold_facts": ["Borealis Metals announced it would acquire Cinder Mining for "
                        "1.8 billion dollars."]},
        {"document_id": "doc_ccc", "identity": "u/ccc", "title": "Cinder",
         "url": "u/ccc", "source": "t", "body_sha256": "x",
         "gold_facts": ["Cinder Mining listed 940 employees in its 2023 annual report."]},
    ],
}

GOLD = {
    "fact_match_threshold": 0.6,
    "questions": {
        QUESTION["question_id"]: {
            "question": QUESTION["question"],
            "stage": "stage_a", "width": 2, "question_type": "inference_query",
            "answer": QUESTION["answer"], "answer_aliases": [],
            "needs_manual_review": False,
            "facts": [
                {"fact_id": "doc_bbb#f1", "document_id": "doc_bbb",
                 "text": "Borealis Metals announced it would acquire Cinder Mining for "
                         "1.8 billion dollars.",
                 "aliases": [], "required": True, "needs_manual_review": False},
                {"fact_id": "doc_ccc#f1", "document_id": "doc_ccc",
                 "text": "Cinder Mining listed 940 employees in its 2023 annual report.",
                 "aliases": [], "required": True, "needs_manual_review": False},
            ],
        }
    },
}


def documents_json() -> Dict[str, Dict[str, str]]:
    return {did: {"document_id": did, "body": body, "title": did, "url": f"u/{did}"}
            for did, body in DOCS.items()}
