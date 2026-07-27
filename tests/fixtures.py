"""Shared synthetic corpus and gate fixtures for the tests. No network, no model."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

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


# --------------------------------------------------------------------------
# gate fixtures
#
# `evaluate_gates` now has fourteen gates, several of which fail on absent
# rather than contradictory data (a parity gate over an empty slot map is a
# FAILURE, not a vacuous pass). Hand-rolling "provenance that passes" inside
# each test therefore drifts: a test that means to exercise ONE gate ends up
# asserting on a report that failed for five unrelated reasons. These builders
# produce a genuinely all-green input set that a test then breaks in exactly
# one place.
# --------------------------------------------------------------------------

PRECISIONS: Tuple[str, str, str] = ("F16", "Q8_0", "Q4_K_M")

F16_SHA = "a" * 64
Q8_SHA = "b" * 64
Q4_SHA = "c" * 64
SAMPLING_HASH = "5a" * 32
CONDITION = "fixed_verified_evidence"


def passing_provenance() -> Dict[str, Any]:
    """Provenance that satisfies model_hashes_recorded, common_source_checkpoint,
    embedding_is_real_model and gpu_offload_identical_across_precisions.

    Note `derived_from_f16_sha256`: the gate no longer accepts a truthy
    `common_source` note, it compares each quantized variant's recorded parent
    against the F16 actually in use.
    """
    return {
        "models": {
            "F16": {"sha256": F16_SHA},
            "Q8_0": {"sha256": Q8_SHA, "derived_from_f16_sha256": F16_SHA},
            "Q4_K_M": {"sha256": Q4_SHA, "derived_from_f16_sha256": F16_SHA},
        },
        "common_source": {"f16_gguf_sha256": F16_SHA,
                          "hf_repo": "Qwen/Qwen2.5-3B-Instruct",
                          "hf_revision": "aa8e7253"},
        "embedding": {"embed_is_real_model": True,
                      "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
                      "embed_revision": "c9745ed1"},
        "blocks": {p: {"gpu_offload": {"layers_offloaded": 37, "layers_total": 37}}
                   for p in PRECISIONS},
    }


def passing_events_and_predictions(
        n_questions: int, precisions: Sequence[str] = PRECISIONS,
        document_ids: Sequence[str] = ("doc_bbb",), condition: str = CONDITION
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Real-inference events + predictions for `n_questions` at every precision.

    Emits BOTH a coordinator event and a document_agent event per question, so
    `coordinator_prompt_parity` and `prompt_hash_parity` have slots to compare.
    A prompt hash keyed on the question (not the precision) is the property the
    frozen coordinator and the frozen evidence exist to guarantee.
    """
    events: List[Dict[str, Any]] = []
    preds: List[Dict[str, Any]] = []
    for i in range(n_questions):
        qid = f"q{i}"
        for precision in precisions:
            events.append({
                "precision": precision, "backend": "llama-server",
                "is_fixture": False, "dry_run": False, "generated_tokens": 20,
                "condition": condition, "role": "coordinator", "question_id": qid,
                "document_id": "-", "prompt_hash": f"coord_hash_{qid}",
                "sampling_hash": SAMPLING_HASH,
            })
            for did in document_ids:
                events.append({
                    "precision": precision, "backend": "llama-server",
                    "is_fixture": False, "dry_run": False, "generated_tokens": 40,
                    "condition": condition, "role": "document_agent",
                    "question_id": qid, "document_id": did,
                    "prompt_hash": f"agent_hash_{qid}_{did}",
                    "sampling_hash": SAMPLING_HASH,
                })
            preds.append({
                "question_id": qid, "precision": precision, "condition": condition,
                "is_fixture": False, "dry_run": False,
                "evidence_chunk_ids": [f"{d}::c0000" for d in document_ids],
                # The frozen coordinator record every precision loaded. Byte
                # identical across precisions and stamped with the precision
                # that produced it -- that is what `coordinator_frozen_and_shared`
                # and `coordinator_frozen_at_f16` check, since with freezing
                # there is only ever ONE coordinator generation per question and
                # a cross-precision prompt-hash comparison would pass vacuously.
                "coordinator": {
                    "shared_tasks": [{"task_id": "t1", "instruction": f"extract for {qid}",
                                      "required_fields": ["entity"]}],
                    "synthesis_directive": f"merge findings for {qid}",
                    "frozen": True,
                    "provenance": {"produced_by_precision": "F16",
                                   "prompt_hash": f"coord_hash_{qid}",
                                   "coordinator_fallback": False},
                },
            })
    return events, preds


def passing_gate_report(n_questions: int = 6, required_questions: int = 6):
    """An all-green GateReport, built by running the real gate evaluator.

    Tests that need "the gates passed" must NOT fabricate `GateReport(gates=[])`:
    `all([])` is True, so an empty report used to satisfy `report.passed` and
    quietly bypass the entire "no science without gates" property that
    `scientific_outcome` is built on.
    """
    from gates import evaluate_gates  # imported lazily: tests set sys.path first

    events, preds = passing_events_and_predictions(n_questions)
    report = evaluate_gates(events, preds, passing_provenance(),
                            required_questions=required_questions,
                            condition=CONDITION, verify_model_files=False)
    assert report.passed, report.failures()   # the fixture itself is load-bearing
    return report
