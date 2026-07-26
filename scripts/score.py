#!/usr/bin/env python3
"""Score a stage's predictions. Label-blind, partial credit, no API judge.

The scoring functions in src/metrics.py never see a precision label. When
manual adjudication is needed, `--emit-adjudication` writes a shuffled,
de-labelled worksheet plus a separate key file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import metrics as M  # noqa: E402
from config import results_dir  # noqa: E402
from gates import calibration_status  # noqa: E402
from logging_utils import read_json, write_json  # noqa: E402


def score_prediction(pred: Dict[str, Any], gold: Dict[str, Any],
                     fact_threshold: float, branch_threshold: float
                     ) -> Dict[str, Any]:
    gold_facts = gold.get("facts") or []
    by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for gf in gold_facts:
        by_doc[gf["document_id"]].append(gf)

    agents = pred.get("agent_outputs") or []
    branch: List[Dict[str, Any]] = []
    recalls: List[float] = []
    for a in agents:
        r = M.per_agent_recall(a, by_doc.get(a.get("document_id"), []), fact_threshold)
        r["document_id"] = a.get("document_id")
        r["branch_success"] = M.branch_success(r["recall"], branch_threshold)
        r["insufficient_evidence"] = bool(a.get("insufficient_evidence"))
        r["parse_ok"] = bool(a.get("parse_ok"))
        r["hallucinated_citations"] = int(a.get("hallucinated_citations") or 0)
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
    contradictions = M.contradiction_count(
        final, agents, (pred.get("synthesis") or {}).get("unresolved_conflicts") or [])

    return {
        "question_id": pred["question_id"], "precision": pred["precision"],
        "condition": pred["condition"], "stage": pred.get("stage"),
        "width": pred.get("width"), "question_type": pred.get("question_type"),
        "is_fixture": bool(pred.get("is_fixture")), "dry_run": bool(pred.get("dry_run")),
        "final_answer": final, "gold_answer": gold.get("answer", ""),
        "answer_score": ans,
        "atomic_fact_recall_final": synth["atomic_fact_recall_final"],
        "atomic_fact_recall_agents": synth["atomic_fact_recall_agents"],
        "atomic_fact_precision": prec["precision"],
        "final_completeness": synth["final_completeness"],
        "synthesis_loss": synth["synthesis_loss"],
        "facts_available_before_synthesis": synth["facts_available_before_synthesis"],
        "facts_preserved_after_synthesis": synth["facts_preserved_after_synthesis"],
        "contradictions_introduced": contradictions,
        "unsupported_claim_rate": round(
            sum(b["unsupported_claim_rate"] for b in branch) / len(branch), 4)
        if branch else 0.0,
        "branch": branch, "orchestration": orch,
        "evidence_chunk_ids": pred.get("evidence_chunk_ids") or [],
        "wall_s": pred.get("wall_s"), "document_agent_wall_s": pred.get("document_agent_wall_s"),
        "coordinator_fallback": bool((pred.get("coordinator") or {}).get("coordinator_fallback")),
        "n_gold_facts": len(gold_facts),
    }


def emit_adjudication(scored: List[Dict[str, Any]], out_dir: Path) -> None:
    """De-labelled worksheet: the adjudicator cannot see which precision produced
    which answer."""
    items = []
    key = []
    for s in sorted(scored, key=lambda x: hashlib.sha256(
            f"{x['question_id']}|{x['precision']}".encode()).hexdigest()):
        sid = hashlib.sha256(
            f"adj|{x_id(s)}".encode()).hexdigest()[:12]
        items.append({"sample_id": sid, "question_id": s["question_id"],
                      "gold_answer": s["gold_answer"],
                      "system_answer": s["final_answer"],
                      "human_answer_correct": None,
                      "human_facts_recalled": None,
                      "notes": ""})
        key.append({"sample_id": sid, "question_id": s["question_id"],
                    "precision": s["precision"], "condition": s["condition"]})
    write_json(out_dir / "adjudication_worksheet.json", items)
    write_json(out_dir / "adjudication_key.json", key)


def x_id(s: Dict[str, Any]) -> str:
    return f"{s['question_id']}|{s['precision']}|{s['condition']}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=["stage_a", "stage_b", "stage_c"])
    ap.add_argument("--gold", default="data/gold_facts.json")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--fact-threshold", type=float, default=None)
    ap.add_argument("--branch-threshold", type=float, default=0.5)
    ap.add_argument("--emit-adjudication", action="store_true")
    args = ap.parse_args()

    rd = results_dir(args.results_dir)
    stage_dir = rd / args.stage
    preds = read_json(stage_dir / "predictions.json", default=[]) or []
    gold_all = read_json(ROOT / args.gold)
    if not preds:
        raise SystemExit(f"no predictions at {stage_dir / 'predictions.json'}")
    if gold_all is None:
        raise SystemExit(f"missing rubric {args.gold}; run prepare_dataset.py")

    ft = args.fact_threshold if args.fact_threshold is not None else float(
        gold_all.get("fact_match_threshold", 0.6))
    unreviewed = [qid for qid, g in gold_all["questions"].items()
                  if g.get("needs_manual_review")]

    scored = []
    missing_gold = []
    for p in preds:
        g = gold_all["questions"].get(p["question_id"])
        if g is None:
            missing_gold.append(p["question_id"])
            continue
        scored.append(score_prediction(p, g, ft, args.branch_threshold))

    real = [s for s in scored if not s["is_fixture"] and not s["dry_run"]]
    by_prec: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in real:
        by_prec[s["precision"]].append(s)

    def mean(vals: List[float]) -> float:
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    summary = {
        "stage": args.stage, "fact_match_threshold": ft,
        "branch_success_threshold": args.branch_threshold,
        "n_scored": len(scored), "n_real": len(real),
        "n_fixture_or_dry_run": len(scored) - len(real),
        "rubric_reviewed": not unreviewed,
        "unreviewed_question_ids": unreviewed,
        "missing_gold_for": missing_gold,
        "by_precision": {},
    }
    for prec, rows in sorted(by_prec.items()):
        summary["by_precision"][prec] = {
            "n": len(rows),
            "atomic_fact_recall": mean([r["atomic_fact_recall_final"] for r in rows]),
            "atomic_fact_recall_agents": mean([r["atomic_fact_recall_agents"] for r in rows]),
            "atomic_fact_precision": mean([r["atomic_fact_precision"] for r in rows]),
            "answer_score": mean([r["answer_score"]["score"] for r in rows]),
            "exact_match": mean([r["answer_score"]["exact_match"] for r in rows]),
            "token_f1": mean([r["answer_score"]["token_f1"] for r in rows]),
            "per_agent_recall": mean([b["recall"] for r in rows for b in r["branch"]]),
            "required_document_coverage": mean(
                [r["orchestration"]["required_document_coverage"] for r in rows]),
            "branch_failure_probability": mean(
                [1.0 if r["orchestration"]["any_required_branch_failed"] else 0.0
                 for r in rows]),
            "synthesis_loss": mean([r["synthesis_loss"] for r in rows]),
            "unsupported_claim_rate": mean([r["unsupported_claim_rate"] for r in rows]),
            "contradictions_introduced": mean(
                [r["contradictions_introduced"] for r in rows]),
            "mean_wall_s": mean([float(r["wall_s"] or 0) for r in rows]),
            "mean_document_agent_wall_s": mean(
                [float(r["document_agent_wall_s"] or 0) for r in rows]),
            "coordinator_fallback_rate": mean(
                [1.0 if r["coordinator_fallback"] else 0.0 for r in rows]),
        }
    f16 = summary["by_precision"].get("F16", {}).get("atomic_fact_recall")
    summary["calibration_status"] = (calibration_status(f16) if f16 is not None
                                     else "UNAVAILABLE — no real F16 results")

    write_json(stage_dir / "scored.json", scored)
    write_json(stage_dir / "score_summary.json", summary)
    if args.emit_adjudication:
        emit_adjudication(scored, stage_dir)
        print(f"[score] blind adjudication worksheet -> {stage_dir}")

    print(json.dumps(summary["by_precision"], indent=2))
    print(f"[score] calibration: {summary['calibration_status']}")
    if unreviewed:
        print(f"[score] WARNING rubric_reviewed=false ({len(unreviewed)} questions "
              "still flagged needs_manual_review). Review data/gold_facts.json "
              "before quoting these numbers.")
    if summary["n_fixture_or_dry_run"]:
        print(f"[score] {summary['n_fixture_or_dry_run']} fixture/dry-run rows were "
              "excluded from every aggregate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
