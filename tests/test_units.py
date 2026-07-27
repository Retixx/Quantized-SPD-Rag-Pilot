"""Unit tests for scoring, chunking, JSON repair and the outcome map."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics as M
from fixtures import DOCS, passing_gate_report
from gates import (MIN_INFORMATIVE_PAIRS, MIN_PAIRS_PER_WIDTH, GateReport,
                   GateResult, calibration_status, scientific_outcome)
from retrieval import chunk_document
from schemas import CoordinatorOutput, extract_json_object, parse_into


def test_chunking_is_deterministic_and_bounded():
    a = chunk_document("doc_aaa", DOCS["doc_aaa"], 60, 10)
    b = chunk_document("doc_aaa", DOCS["doc_aaa"], 60, 10)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert all(c.document_id == "doc_aaa" for c in a)
    assert len({c.chunk_id for c in a}) == len(a)
    assert chunk_document("d", "") == []


def test_json_extraction_handles_fences_and_prose():
    assert extract_json_object('{"a":1}') == {"a": 1}
    assert extract_json_object('```json\n{"a":2}\n```') == {"a": 2}
    assert extract_json_object('here you go: {"a":3} hope that helps') == {"a": 3}
    assert extract_json_object('nested {"a":{"b":4}} tail') == {"a": {"b": 4}}
    assert extract_json_object("no json here") is None


def test_parse_into_reports_validation_errors():
    ok = parse_into(CoordinatorOutput, '{"shared_tasks":[],"synthesis_directive":"x"}')
    assert ok.ok and ok.model_obj["synthesis_directive"] == "x"
    bad = parse_into(CoordinatorOutput, '{"shared_tasks": 5}')
    assert not bad.ok and "validation_error" in bad.error


def test_answer_scoring_gives_partial_credit():
    assert M.answer_score("1.8 billion dollars", "1.8 billion dollars")["exact_match"] == 1.0
    partial = M.answer_score("about 1.8 billion dollars in cash",
                             "1.8 billion dollars")
    assert partial["exact_match"] == 0.0
    assert partial["contains"] == 1.0 and partial["score"] == 1.0
    weak = M.answer_score("roughly two billion", "1.8 billion dollars")
    assert 0.0 < weak["score"] < 1.0


def test_fact_matching_penalises_wrong_numbers():
    gold = "Cinder Mining listed 940 employees in its 2023 annual report."
    right = M.fact_match_score(gold, "Cinder Mining reported 940 employees in 2023.")
    wrong = M.fact_match_score(gold, "Cinder Mining reported 512 employees in 2019.")
    assert right > wrong
    assert M.fact_supported(gold, "Cinder Mining listed 940 employees in its 2023 "
                                  "annual report.")


def test_synthesis_loss_definition():
    agents = [{"document_id": "d1", "supported_facts": [
        {"fact": "Borealis will acquire Cinder Mining for 1.8 billion dollars.",
         "chunk_ids": ["c1"], "citation_valid": True},
        {"fact": "Cinder Mining listed 940 employees in its 2023 annual report.",
         "chunk_ids": ["c2"], "citation_valid": True}]}]
    gold = [{"fact_id": "f1", "document_id": "d1",
             "text": "Borealis will acquire Cinder Mining for 1.8 billion dollars."},
            {"fact_id": "f2", "document_id": "d1",
             "text": "Cinder Mining listed 940 employees in its 2023 annual report."}]
    both = M.synthesis_metrics(
        agents, "Borealis will acquire Cinder Mining for 1.8 billion dollars and "
                "Cinder Mining listed 940 employees in its 2023 annual report.", gold)
    assert both["facts_available_before_synthesis"] == 2
    assert both["synthesis_loss"] == 0.0
    dropped = M.synthesis_metrics(
        agents, "Borealis will acquire Cinder Mining for 1.8 billion dollars.", gold)
    assert dropped["facts_preserved_after_synthesis"] == 1
    assert dropped["synthesis_loss"] == 0.5


def test_uncited_and_unsupported_claim_rates_are_different_measures():
    """`uncited_claim_rate` is format compliance; `unsupported_claim_rate` is
    entailment by the cited chunk. Both are None for an empty list, because a
    rate over zero claims is undefined -- 0.0 read as "no unsupported claims",
    i.e. an agent that emitted nothing scored as perfectly faithful."""
    facts = [{"fact": "a", "citation_valid": True},
             {"fact": "b", "citation_valid": False}]
    assert M.uncited_claim_rate(facts) == 0.5
    assert M.uncited_claim_rate([]) is None
    assert M.unsupported_claim_rate([]) is None
    # with no chunk texts it falls back to the format measure
    assert M.unsupported_claim_rate(facts) == 0.5

    # with chunk texts it measures whether the CITED chunk entails the claim:
    # a well-formed citation to a chunk that says something else is unsupported.
    chunks = {"c1": "Cinder Mining listed 940 employees in its 2023 annual report.",
              "c2": "Borealis Metals is headquartered in Winnipeg."}
    cited = [
        {"fact": "Cinder Mining listed 940 employees in its 2023 annual report.",
         "chunk_ids": ["c1"], "citation_valid": True},
        {"fact": "Cinder Mining listed 12000 employees in its 2023 annual report.",
         "chunk_ids": ["c2"], "citation_valid": True},
    ]
    assert M.uncited_claim_rate(cited) == 0.0        # both are formally cited
    assert M.unsupported_claim_rate(cited, chunks) == 0.5   # one is not entailed


def test_orchestration_branch_failure():
    agents = [{"document_id": "d1", "supported_facts": [{"fact": "x"}],
               "retrieved_chunk_ids": ["c1", "c2"], "generated_tokens": 5},
              {"document_id": "d2", "supported_facts": [],
               "insufficient_evidence": True, "retrieved_chunk_ids": ["c2"],
               "generated_tokens": 3}]
    o = M.orchestration_metrics(agents, ["d1", "d2"], [0.9, 0.1])
    assert o["n_agents_succeeded"] == 1
    assert o["any_required_branch_failed"] is True
    assert o["required_document_coverage"] == 0.5
    assert o["duplicate_evidence_chunks"] == 1


def test_calibration_bands():
    """Bands are half-open upward, so every value lands in exactly one and the
    table matches configs/stage_a.yaml: 0.70 is a PASS, not MARGINAL."""
    assert calibration_status(0.30).startswith("BLOCKED")
    assert calibration_status(0.4999).startswith("BLOCKED")
    assert calibration_status(0.50).startswith("MARGINAL")
    assert calibration_status(0.60).startswith("MARGINAL")
    assert calibration_status(0.6999).startswith("MARGINAL")
    assert calibration_status(0.70) == "CALIBRATION PASS"
    assert calibration_status(0.80) == "CALIBRATION PASS"
    assert calibration_status(0.99).startswith("POSSIBLE CEILING")
    assert calibration_status(1.0).startswith("POSSIBLE CEILING")
    # No real F16 results is not a band; it is an absence.
    assert calibration_status(None) == "UNAVAILABLE — no real F16 results"


def test_empty_gate_report_is_not_a_pass():
    """`all([])` is True. A GateReport with no gates therefore used to satisfy
    `.passed`, which is a total bypass of 'no science without gates' -- and a
    test in this very file used to rely on it."""
    assert GateReport(gates=[]).passed is False
    assert GateReport(gates=[]).failures() == ["no gates were evaluated"]
    assert GateReport(gates=[GateResult("g", True)]).passed is True
    assert GateReport(gates=[GateResult("g", True),
                            GateResult("h", False)]).passed is False
    # and scientific_outcome must refuse to look at the numbers
    out = scientific_outcome({"f16_atomic_fact_recall": 0.9}, GateReport(gates=[]))
    assert out["scientific_outcome"] == "SCIENTIFIC_RECOMMENDATION_NOT_AVAILABLE"


def test_scientific_outcomes_map_as_specified():
    # Genuinely passing gates, produced by the real evaluator over real-shaped
    # inputs -- NOT GateReport(gates=[]), which is a vacuous pass.
    passing = passing_gate_report()
    assert passing.passed

    floor = scientific_outcome({"f16_atomic_fact_recall": 0.30}, passing)
    assert floor["scientific_outcome"] == "NO_GO_MODEL_FLOOR"

    ceiling = scientific_outcome({"f16_atomic_fact_recall": 0.99,
                                  "q8_atomic_fact_recall": 0.98,
                                  "q4_atomic_fact_recall": 0.97}, passing)
    assert ceiling["scientific_outcome"] == "NO_GO_EVALUATION_CEILING"

    # Compounding is the headline claim, so it must carry an interval on the
    # growth contrast AND enough paired questions in every width cell.
    compounding_summary = {
        "f16_atomic_fact_recall": 0.80, "q8_atomic_fact_recall": 0.78,
        "q4_atomic_fact_recall": 0.55, "f16_q4_gap_overall": 0.25,
        "f16_q8_gap_overall": 0.02,
        "f16_q4_gap_by_width": {"2": 0.02, "3": 0.15, "4": 0.35},
        "f16_q4_gap_by_width_n": {"2": 4, "3": 4, "4": 4},
        "f16_q4_gap_growth_ci": {"mean": 0.33, "lo": 0.12, "hi": 0.52},
        "f16_q4_gap_ci": {"lo": 0.10, "hi": 0.40, "n_informative": 10}}
    compounding = scientific_outcome(compounding_summary, passing)
    assert compounding["scientific_outcome"] == "GO_COMPOUNDING_DEGRADATION"

    # ... and it degrades to WEAK_GO when the interval includes zero,
    thin_ci = dict(compounding_summary,
                   f16_q4_gap_growth_ci={"mean": 0.33, "lo": -0.02, "hi": 0.60})
    assert scientific_outcome(thin_ci, passing)["scientific_outcome"] == "WEAK_GO_EXPAND"
    # ... or when a width cell is too thin to move a point estimate honestly.
    thin_cell = dict(compounding_summary,
                     f16_q4_gap_by_width_n={"2": 4, "3": MIN_PAIRS_PER_WIDTH - 1,
                                            "4": 4})
    assert scientific_outcome(thin_cell, passing)["scientific_outcome"] == "WEAK_GO_EXPAND"

    q8point = scientific_outcome({
        "f16_atomic_fact_recall": 0.80, "q8_atomic_fact_recall": 0.79,
        "q4_atomic_fact_recall": 0.60, "f16_q4_gap_overall": 0.20,
        "f16_q8_gap_overall": 0.01,
        "f16_q4_gap_by_width": {"2": 0.20, "4": 0.20},
        "f16_q4_gap_by_width_n": {"2": 4, "4": 4},
        "f16_q4_gap_ci": {"lo": 0.10, "hi": 0.30, "n_informative": 8}}, passing)
    assert q8point["scientific_outcome"] == "GO_Q8_OPERATING_POINT"

    robust_summary = {
        "f16_atomic_fact_recall": 0.82, "q8_atomic_fact_recall": 0.81,
        "q4_atomic_fact_recall": 0.81, "f16_q4_gap_overall": 0.01,
        "f16_q8_gap_overall": 0.01,
        "f16_q4_gap_by_width": {"2": 0.01, "4": 0.01},
        "f16_q4_gap_by_width_n": {"2": 4, "4": 4},
        "f16_q4_gap_ci": {"lo": -0.04, "hi": 0.04, "n_informative": 8}}
    robust = scientific_outcome(robust_summary, passing)
    assert robust["scientific_outcome"] == "GO_DECOMPOSITION_ROBUSTNESS"

    # An equivalence claim needs informative pairs. The identical interval
    # backed by all-ties is zero information, not the strongest possible result.
    uninformative = dict(robust_summary,
                         f16_q4_gap_ci={"lo": -0.04, "hi": 0.04,
                                        "n_informative": MIN_INFORMATIVE_PAIRS - 1})
    assert scientific_outcome(uninformative,
                              passing)["scientific_outcome"] == "WEAK_GO_EXPAND"

    weak = scientific_outcome({
        "f16_atomic_fact_recall": 0.75, "q8_atomic_fact_recall": 0.74,
        "q4_atomic_fact_recall": 0.70, "f16_q4_gap_overall": 0.05,
        "f16_q8_gap_overall": 0.01,
        "f16_q4_gap_by_width": {"2": 0.04, "4": 0.06},
        "f16_q4_gap_by_width_n": {"2": 4, "4": 4},
        "f16_q4_gap_ci": {"lo": -0.30, "hi": 0.40, "n_informative": 8}}, passing)
    assert weak["scientific_outcome"] == "WEAK_GO_EXPAND"


def test_max_ci_width_defaults_to_twice_the_margin():
    """The old default was a hardcoded 0.30 -- six times the declared 0.05
    margin, so [-0.28, +0.02] counted as narrow enough for an equivalence
    claim."""
    passing = passing_gate_report()
    summary = {
        "f16_atomic_fact_recall": 0.82, "q8_atomic_fact_recall": 0.81,
        "q4_atomic_fact_recall": 0.81, "f16_q4_gap_overall": -0.13,
        "f16_q8_gap_overall": 0.01,
        "f16_q4_gap_by_width_n": {"2": 4, "4": 4},
        "f16_q4_gap_ci": {"lo": -0.26, "hi": 0.02, "n_informative": 8}}
    # width 0.28 > 2 * 0.05 -> no robustness claim
    assert scientific_outcome(summary, passing)["scientific_outcome"] != \
        "GO_DECOMPOSITION_ROBUSTNESS"
    # the old permissive ceiling, passed explicitly, would have allowed it
    assert scientific_outcome(summary, passing, max_ci_width=0.30)[
        "scientific_outcome"] == "GO_DECOMPOSITION_ROBUSTNESS"


def test_bootstrap_ci_is_deterministic_and_paired():
    sys.path.insert(0, str(ROOT / "scripts"))
    from analyze import paired_bootstrap_ci
    diffs = [0.1, -0.05, 0.2, 0.0, 0.15, -0.1, 0.05, 0.3]
    a = paired_bootstrap_ci(diffs, resamples=2000)
    b = paired_bootstrap_ci(diffs, resamples=2000)
    assert a == b
    assert a["lo"] <= a["mean"] <= a["hi"]
    assert paired_bootstrap_ci([], 100)["n"] == 0
    assert paired_bootstrap_ci([0.1, 0.2], 100)["lo"] is None
    # non-zero pairs are counted, so a downstream verdict can tell "we learned
    # nothing" apart from "the two are equivalent"
    assert a["n_informative"] == 7          # one of the eight diffs is 0.0
    assert paired_bootstrap_ci([0.0] * 8, 500)["n_informative"] == 0


def test_non_inferiority_is_distinct_from_not_significant():
    """A narrow interval is necessary but NOT sufficient: `verdict` also
    requires that enough paired differences were non-zero, so a stage where
    every pair tied cannot produce the strongest possible label."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from analyze import verdict
    tight = verdict({"lo": -0.03, "hi": 0.04, "n": 8, "n_informative": 6}, 0.05)
    assert tight["label"] == "NON_INFERIOR_WITHIN_MARGIN"
    wide = verdict({"lo": -0.40, "hi": 0.02, "n": 8, "n_informative": 6}, 0.05)
    assert wide["label"] == "NOT_SIGNIFICANT_BUT_CI_TOO_WIDE"
    noisy = verdict({"lo": -0.20, "hi": 0.25, "n": 8, "n_informative": 6}, 0.05)
    assert noisy["label"] == "NOT_STATISTICALLY_SIGNIFICANT"
    bad = verdict({"lo": 0.10, "hi": 0.30, "n": 8, "n_informative": 6}, 0.05)
    assert bad["label"] == "DEGRADATION_DETECTED"

    # the SAME tight interval with too few informative pairs is degenerate
    degenerate = verdict({"lo": -0.03, "hi": 0.04, "n": 8,
                          "n_informative": MIN_INFORMATIVE_PAIRS - 1}, 0.05)
    assert degenerate["label"] == "NOT_SIGNIFICANT_BUT_CI_TOO_WIDE"
    assert degenerate["degenerate"] is True
    # an absent n_informative reads as 0, i.e. unproven, never as non-inferior
    assert verdict({"lo": -0.03, "hi": 0.04}, 0.05)["label"] != \
        "NON_INFERIOR_WITHIN_MARGIN"
    # no interval at all is INDETERMINATE, not a pass
    assert verdict({"lo": None, "hi": None, "note": "n<3"}, 0.05)["label"] == \
        "INDETERMINATE"
