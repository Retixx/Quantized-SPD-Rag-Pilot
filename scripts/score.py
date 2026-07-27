#!/usr/bin/env python3
"""Score a stage's predictions. Label-blind, partial credit, no API judge.

The scoring functions in src/metrics.py never see a precision label. When
manual adjudication is needed, `--emit-adjudication` writes a shuffled,
de-labelled worksheet plus a separate key file.

Two aggregation rules, both of which the previous version broke:

  * **None is excluded, not zeroed.** `mean()` returned 0.0 for an empty list
    and averaged `None`-valued metrics as if they were 0.0, so an unscorable
    question read as a perfect-loss / total-miss instead of being dropped.
  * **Errored and harness rows never enter an aggregate.** Filtering only on
    `is_fixture`/`dry_run` let a row whose generation returned a 502 be scored
    as a model failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import metrics as M  # noqa: E402
from config import results_dir  # noqa: E402
from gates import calibration_status  # noqa: E402
from logging_utils import read_json, write_json  # noqa: E402


def resolve_stage_dir(rd: Path, stage: str):
    """Return `(results_root, stage_dir)`, following the harness quarantine.

    `run_stage.py --dry-run` / `--backend fixture` write to
    `<results>/_harness/<stage>/`. Without this, the documented flow
    (`run_stage --dry-run` then `score.py --stage stage_a`) exits with
    "no predictions at .../stage_a/predictions.json" and no hint that the rows
    are one directory over.
    """
    if (rd / stage / "predictions.json").exists():
        return rd, rd / stage
    harness = rd / "_harness"
    if (harness / stage / "predictions.json").exists():
        print(f"[score] NOTE reading the quarantined harness run at "
              f"{harness / stage}. These rows can never produce a scientific "
              "result; they are watermarked and excluded from every aggregate.")
        return harness, harness / stage
    return rd, rd / stage


def mean(vals: Iterable[Optional[float]]) -> Optional[float]:
    """Mean over the values that exist. None when nothing is scorable."""
    xs = [float(v) for v in vals if v is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def weighted_mean(pairs: Sequence[Any]) -> Optional[float]:
    """Mean of (value, weight) pairs, skipping None values.

    Used for claim rates so an agent that emitted one fact does not count as
    much as one that emitted twenty.
    """
    num = den = 0.0
    for value, weight in pairs:
        if value is None or not weight:
            continue
        num += float(value) * float(weight)
        den += float(weight)
    return round(num / den, 4) if den else None


def score_prediction(pred: Dict[str, Any], gold: Dict[str, Any],
                     fact_threshold: float, branch_threshold: float,
                     chunk_texts: Optional[Dict[str, str]] = None
                     ) -> Dict[str, Any]:
    gold_facts = gold.get("facts") or []
    by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for gf in gold_facts:
        by_doc[gf["document_id"]].append(gf)

    agents = pred.get("agent_outputs") or []
    branch: List[Dict[str, Any]] = []
    recalls: List[Optional[float]] = []
    docs_without_gold: List[str] = []
    for a in agents:
        did = a.get("document_id")
        gold_for_doc = by_doc.get(did, [])
        if not gold_for_doc:
            # The manifest derives document identity from url-then-title; any
            # drift between the id an agent was dispatched with and the id in
            # the rubric used to silently score recall 0.0 and drag the branch
            # down with no warning anywhere.
            docs_without_gold.append(str(did))
        r = M.per_agent_recall(a, gold_for_doc, fact_threshold, chunk_texts)
        r["document_id"] = did
        r["branch_success"] = M.branch_success(r["recall"], branch_threshold)
        r["insufficient_evidence"] = bool(a.get("insufficient_evidence"))
        r["hallucinated_citations"] = int(a.get("hallucinated_citations") or 0)
        r["n_facts"] = len(a.get("supported_facts") or [])
        branch.append(r)
        recalls.append(r["recall"])

    orch = M.orchestration_metrics(agents, pred.get("required_document_ids") or [],
                                   recalls, branch_threshold)
    final = pred.get("final_answer", "")
    synth = M.synthesis_metrics(agents, final, gold_facts, fact_threshold)
    ans = M.answer_score(final, gold.get("answer", ""), gold.get("answer_aliases") or [])
    pred_fact_texts = [f.get("fact", "") for a in agents
                       for f in (a.get("supported_facts") or [])]
    prec = M.atomic_fact_precision(pred_fact_texts, gold_facts, fact_threshold)
    contra = M.unsourced_numbers(
        final, agents,
        (pred.get("synthesis") or {}).get("unresolved_conflicts") or [],
        question=pred.get("question", ""))

    return {
        "question_id": pred["question_id"], "precision": pred["precision"],
        "condition": pred["condition"], "stage": pred.get("stage"),
        "width": pred.get("width"), "question_type": pred.get("question_type"),
        "is_fixture": bool(pred.get("is_fixture")),
        "dry_run": bool(pred.get("dry_run")),
        "generation_error": pred.get("generation_error"),
        "final_answer": final, "gold_answer": gold.get("answer", ""),
        "answer_score": ans,
        "answer_scorable": bool(ans.get("scorable")),
        "atomic_fact_recall_final": synth["atomic_fact_recall_final"],
        "atomic_fact_recall_agents": synth["atomic_fact_recall_agents"],
        "atomic_fact_precision": prec["precision"],
        "final_completeness": synth["final_completeness"],
        "synthesis_loss": synth["synthesis_loss"],
        "facts_available_before_synthesis": synth["facts_available_before_synthesis"],
        "facts_preserved_after_synthesis": synth["facts_preserved_after_synthesis"],
        "contradictions_introduced": contra["contradictions_introduced"],
        "unsourced_numbers": contra["unsourced_numbers"],
        # Weighted by the number of facts each branch emitted.
        "uncited_claim_rate": weighted_mean(
            [(b["uncited_claim_rate"], b["n_facts"]) for b in branch]),
        "unsupported_claim_rate": weighted_mean(
            [(b["unsupported_claim_rate"], b["n_facts"]) for b in branch]),
        "branch": branch, "orchestration": orch,
        "evidence_chunk_ids": pred.get("evidence_chunk_ids") or [],
        "wall_s": pred.get("wall_s"),
        "document_agent_wall_s": pred.get("document_agent_wall_s"),
        "partially_resumed": bool(pred.get("partially_resumed")),
        "coordinator_fallback": bool(
            (pred.get("coordinator") or {}).get("coordinator_fallback")),
        "coordinator_frozen_from": (pred.get("coordinator") or {}).get("frozen_from"),
        "synthesis_parse_ok": bool((pred.get("synthesis") or {}).get("parse_ok", True)),
        "n_forced_finalize": orch.get("n_forced_finalize", 0),
        "documents_without_gold_facts": docs_without_gold,
        "n_gold_facts": len(gold_facts),
    }


def emit_adjudication(scored: List[Dict[str, Any]], out_dir: Path) -> None:
    """De-labelled worksheet: the adjudicator cannot see which precision
    produced which answer.

    `sample_id` is keyed on a fresh random salt written ONLY to the key file.
    The old id was `sha256("adj|{question_id}|{precision}|{condition}")[:12]`
    over a three-element input space with the question_id visible in the
    worksheet -- recoverable by brute force in three hashes. The row order was
    also a deterministic function of the precision label.
    """
    salt = secrets.token_hex(16)
    # Never ask a human to adjudicate watermarked fixture output.
    rows = [s for s in scored if not s["is_fixture"] and not s["dry_run"]]

    def sid_for(s: Dict[str, Any]) -> str:
        return hashlib.sha256(
            f"{salt}|{s['question_id']}|{s['precision']}|{s['condition']}".encode()
        ).hexdigest()[:12]

    ordered = sorted(rows, key=sid_for)
    items, key = [], []
    for s in ordered:
        sid = sid_for(s)
        items.append({"sample_id": sid, "question_id": s["question_id"],
                      "gold_answer": s["gold_answer"],
                      "system_answer": s["final_answer"],
                      "human_answer_correct": None,
                      "human_facts_recalled": None,
                      "notes": ""})
        key.append({"sample_id": sid, "question_id": s["question_id"],
                    "precision": s["precision"], "condition": s["condition"]})
    write_json(out_dir / "adjudication_worksheet.json", items)
    write_json(out_dir / "adjudication_key.json",
               {"salt": salt,
                "warning": "This file de-blinds the worksheet. It is excluded "
                           "from the packaged results bundle; do not open it "
                           "before adjudication is complete.",
                "rows": key})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=["stage_a", "stage_b", "stage_c"])
    ap.add_argument("--gold", default="data/gold_facts.json")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--fact-threshold", type=float, default=None,
                    help="OVERRIDE the rubric's frozen threshold. Re-scoring at a "
                         "threshold chosen after seeing results is a researcher "
                         "degree of freedom; the override is recorded in the summary.")
    ap.add_argument("--branch-threshold", type=float, default=0.5)
    ap.add_argument("--emit-adjudication", action="store_true")
    ap.add_argument("--condition", default=None,
                    help="score only this condition (default: all, reported per condition)")
    args = ap.parse_args()

    rd, stage_dir = resolve_stage_dir(results_dir(args.results_dir), args.stage)
    preds = read_json(stage_dir / "predictions.json", default=[]) or []
    gold_all = read_json(ROOT / args.gold)
    if not preds:
        raise SystemExit(f"no predictions at {stage_dir / 'predictions.json'}")
    if gold_all is None:
        raise SystemExit(f"missing rubric {args.gold}; run prepare_dataset.py")

    frozen_ft = float(gold_all.get("fact_match_threshold", 0.6))
    ft = args.fact_threshold if args.fact_threshold is not None else frozen_ft

    if args.condition:
        preds = [p for p in preds if p.get("condition") == args.condition]
        if not preds:
            raise SystemExit(f"no predictions with condition={args.condition!r}")

    # Chunk text for the content-level unsupported-claim measure, when the
    # frozen evidence is available.
    # fixed_evidence.json and provenance.json are precision-agnostic and live at
    # the real results root even for a quarantined harness run, so look one
    # level up rather than silently degrading to the citation-format fallback.
    chunk_texts: Dict[str, str] = {}
    fixed = read_json(rd / "fixed_evidence.json", default={}) or {}
    if not fixed and rd.name == "_harness":
        fixed = read_json(rd.parent / "fixed_evidence.json", default={}) or {}
    for entry in fixed.values():
        for hit in entry.get("hits") or []:
            if hit.get("chunk_id"):
                chunk_texts[hit["chunk_id"]] = hit.get("text", "")

    scored = []
    missing_gold = []
    for p in preds:
        g = gold_all["questions"].get(p["question_id"])
        if g is None:
            missing_gold.append(p["question_id"])
            continue
        scored.append(score_prediction(p, g, ft, args.branch_threshold, chunk_texts))

    # Only THIS stage's rubric entries determine whether the rubric is
    # reviewed. The old check spanned every question in the file, so a Stage A
    # report was stamped provisional because Stage B facts were unreviewed.
    stage_qids = {s["question_id"] for s in scored}
    unreviewed = sorted(qid for qid in stage_qids
                        if (gold_all["questions"].get(qid) or {}).get("needs_manual_review"))

    real = [s for s in scored
            if not s["is_fixture"] and not s["dry_run"] and not s["generation_error"]]
    excluded_errors = [s for s in scored if s["generation_error"]]
    by_prec: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in real:
        by_prec[s["precision"]].append(s)

    conditions = sorted({s["condition"] for s in scored if s.get("condition")})
    unscorable = [s["question_id"] for s in real if not s["answer_scorable"]]
    doc_gaps = sorted({d for s in scored for d in s["documents_without_gold_facts"]})

    summary: Dict[str, Any] = {
        "stage": args.stage,
        "conditions_present": conditions,
        "fact_match_threshold": ft,
        "fact_match_threshold_frozen": frozen_ft,
        "fact_match_threshold_overridden": args.fact_threshold is not None,
        "branch_success_threshold": args.branch_threshold,
        "n_scored": len(scored), "n_real": len(real),
        "n_fixture_or_dry_run": sum(1 for s in scored
                                    if s["is_fixture"] or s["dry_run"]),
        "n_excluded_generation_error": len(excluded_errors),
        "rubric_reviewed": not unreviewed,
        "unreviewed_question_ids": unreviewed,
        "missing_gold_for": missing_gold,
        "documents_without_gold_facts": doc_gaps,
        "questions_with_unscorable_gold_answer": sorted(set(unscorable)),
        "by_precision": {},
    }
    for prec, rows in sorted(by_prec.items()):
        summary["by_precision"][prec] = {
            "n": len(rows),
            "atomic_fact_recall": mean(r["atomic_fact_recall_final"] for r in rows),
            "atomic_fact_recall_agents": mean(r["atomic_fact_recall_agents"] for r in rows),
            "atomic_fact_precision": mean(r["atomic_fact_precision"] for r in rows),
            "answer_score": mean(r["answer_score"]["score"] for r in rows),
            "exact_match": mean(r["answer_score"]["exact_match"] for r in rows),
            "token_f1": mean(r["answer_score"]["token_f1"] for r in rows),
            "per_agent_recall": mean(b["recall"] for r in rows for b in r["branch"]),
            "required_document_coverage": mean(
                r["orchestration"]["required_document_coverage"] for r in rows),
            "required_document_correct_coverage": mean(
                r["orchestration"]["required_document_correct_coverage"] for r in rows),
            "branch_failure_probability": mean(
                (None if r["orchestration"]["any_required_branch_failed"] is None
                 else (1.0 if r["orchestration"]["any_required_branch_failed"] else 0.0))
                for r in rows),
            # None-valued synthesis_loss (agents recovered nothing) is EXCLUDED,
            # not averaged in as 0.0 -- the old behaviour reported perfect
            # synthesis for a pipeline whose branches had all failed.
            "synthesis_loss": mean(r["synthesis_loss"] for r in rows),
            "n_synthesis_loss_undefined": sum(
                1 for r in rows if r["synthesis_loss"] is None),
            "uncited_claim_rate": mean(r["uncited_claim_rate"] for r in rows),
            "unsupported_claim_rate": mean(r["unsupported_claim_rate"] for r in rows),
            "contradictions_introduced": mean(
                r["contradictions_introduced"] for r in rows),
            "mean_wall_s": mean(r["wall_s"] for r in rows
                                if not r["partially_resumed"]),
            "mean_document_agent_wall_s": mean(
                r["document_agent_wall_s"] for r in rows
                if not r["partially_resumed"]),
            "n_partially_resumed": sum(1 for r in rows if r["partially_resumed"]),
            "coordinator_fallback_rate": mean(
                1.0 if r["coordinator_fallback"] else 0.0 for r in rows),
            "forced_finalize_rate": mean(
                float(r["n_forced_finalize"]) for r in rows),
            "synthesis_parse_failure_rate": mean(
                0.0 if r["synthesis_parse_ok"] else 1.0 for r in rows),
            "branch_parse_failure_rate": mean(
                float(r["orchestration"].get("n_parse_failures") or 0) for r in rows),
        }
    f16 = summary["by_precision"].get("F16", {}).get("atomic_fact_recall")
    summary["calibration_status"] = calibration_status(f16)

    write_json(stage_dir / "scored.json", scored)
    write_json(stage_dir / "score_summary.json", summary)
    if args.emit_adjudication:
        emit_adjudication(scored, stage_dir)
        print(f"[score] blind adjudication worksheet -> {stage_dir}")
        print("[score] adjudication_key.json de-blinds it; it is excluded from "
              "the packaged bundle. Do not open it before adjudicating.")

    print(json.dumps(summary["by_precision"], indent=2))
    print(f"[score] calibration: {summary['calibration_status']}")
    if len(conditions) > 1:
        print(f"[score] WARNING scored rows mix conditions {conditions}; pass "
              "--condition to score them separately.")
    if summary["fact_match_threshold_overridden"]:
        print(f"[score] WARNING fact threshold overridden to {ft} (rubric froze "
              f"{frozen_ft}). Recorded in score_summary.json.")
    if unreviewed:
        print(f"[score] WARNING rubric_reviewed=false ({len(unreviewed)} questions "
              "still flagged needs_manual_review). Review data/gold_facts.json "
              "before quoting these numbers.")
    if doc_gaps:
        print(f"[score] WARNING {len(doc_gaps)} agent document id(s) have no gold "
              f"facts in the rubric: {doc_gaps[:5]}. Their branches scored recall "
              "None (excluded), not 0.")
    if unscorable:
        print(f"[score] WARNING {len(set(unscorable))} question(s) have an empty "
              "gold answer and were excluded from answer_score.")
    if summary["n_fixture_or_dry_run"]:
        print(f"[score] {summary['n_fixture_or_dry_run']} fixture/dry-run rows were "
              "excluded from every aggregate.")
    if excluded_errors:
        print(f"[score] {len(excluded_errors)} row(s) had a backend generation error "
              "and were excluded; infrastructure failure is not model quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
