#!/usr/bin/env python3
"""Paired analysis, bootstrap CIs, gate evaluation and the go/no-go decision.

Deliberately conservative: the pilot is small, so this reports paired
per-question differences with percentile bootstrap intervals and a
predeclared non-inferiority margin, and refuses to call equivalence when the
interval is wide. Engineering status and scientific status are separate outputs.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import results_dir  # noqa: E402
from gates import (NOT_AVAILABLE, calibration_status, engineering_status,  # noqa: E402
                   evaluate_gates, scientific_outcome)
from logging_utils import read_json, write_json  # noqa: E402

BOOTSTRAP_SEED = 20260725


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def paired_bootstrap_ci(diffs: Sequence[float], resamples: int = 10000,
                        alpha: float = 0.05, seed: int = BOOTSTRAP_SEED
                        ) -> Dict[str, Any]:
    """Percentile bootstrap over PAIRED per-question differences."""
    n = len(diffs)
    if n == 0:
        return {"mean": 0.0, "lo": None, "hi": None, "n": 0,
                "note": "no paired observations"}
    if n < 3:
        return {"mean": round(mean(diffs), 4), "lo": None, "hi": None, "n": n,
                "note": "fewer than 3 pairs; interval not estimated"}
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        means.append(mean([diffs[rng.randrange(n)] for _ in range(n)]))
    means.sort()
    lo = means[int((alpha / 2) * resamples)]
    hi = means[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return {"mean": round(mean(diffs), 4), "lo": round(lo, 4), "hi": round(hi, 4),
            "n": n, "resamples": resamples, "alpha": alpha}


def paired(rows: List[Dict[str, Any]], metric: str, a: str, b: str,
           key=lambda r: r["question_id"]) -> List[float]:
    """a - b, per question, only where both precisions produced a result."""
    ma = {key(r): r for r in rows if r["precision"] == a}
    mb = {key(r): r for r in rows if r["precision"] == b}
    return [float(ma[k][metric]) - float(mb[k][metric])
            for k in sorted(set(ma) & set(mb))]


def get_metric(row: Dict[str, Any], metric: str) -> float:
    if metric == "answer_score":
        return float(row["answer_score"]["score"])
    return float(row.get(metric) or 0.0)


def flatten(rows: List[Dict[str, Any]], metric: str) -> List[Dict[str, Any]]:
    return [{"question_id": r["question_id"], "precision": r["precision"],
             "width": r.get("width"), metric: get_metric(r, metric)} for r in rows]


def ols(X: List[List[float]], y: List[float]) -> Optional[List[float]]:
    """Least squares via numpy; returns None if numpy is unavailable/singular."""
    try:
        import numpy as np  # noqa: WPS433
        A = np.asarray(X, dtype=float)
        b = np.asarray(y, dtype=float)
        if A.shape[0] <= A.shape[1]:
            return None
        beta, *_ = np.linalg.lstsq(A, b, rcond=None)
        return [round(float(v), 5) for v in beta]
    except Exception:
        return None


def exploratory_regression(rows: List[Dict[str, Any]], metric: str) -> Dict[str, Any]:
    """metric ~ 1 + is_q4 + is_q8 + width + is_q4*width + is_q8*width.

    Exploratory only. With 12-36 observations these coefficients are
    descriptive, not inferential.
    """
    X, y, used = [], [], 0
    for r in rows:
        w = r.get("width")
        if w is None:
            continue
        q4 = 1.0 if r["precision"] == "Q4_K_M" else 0.0
        q8 = 1.0 if r["precision"] == "Q8_0" else 0.0
        X.append([1.0, q4, q8, float(w), q4 * float(w), q8 * float(w)])
        y.append(get_metric(r, metric))
        used += 1
    names = ["intercept", "is_q4", "is_q8", "width", "is_q4_x_width", "is_q8_x_width"]
    beta = ols(X, y)
    return {"metric": metric, "n": used, "terms": names,
            "coefficients": dict(zip(names, beta)) if beta else None,
            "note": ("exploratory; the pilot is too small for inferential claims"
                     if beta else "not estimable at this sample size")}


def by_width_gaps(rows: List[Dict[str, Any]], metric: str, a: str, b: str,
                  resamples: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    widths = sorted({r.get("width") for r in rows if r.get("width") is not None})
    for w in widths:
        sub = [r for r in rows if r.get("width") == w]
        d = paired(sub, metric, a, b) if metric != "answer_score" else \
            paired_answer(sub, a, b)
        out[str(w)] = {"gap": round(mean(d), 4),
                       "ci": paired_bootstrap_ci(d, resamples), "n_pairs": len(d)}
    return out


def paired_answer(rows: List[Dict[str, Any]], a: str, b: str) -> List[float]:
    ma = {r["question_id"]: r for r in rows if r["precision"] == a}
    mb = {r["question_id"]: r for r in rows if r["precision"] == b}
    return [get_metric(ma[k], "answer_score") - get_metric(mb[k], "answer_score")
            for k in sorted(set(ma) & set(mb))]


def analyse(rows: List[Dict[str, Any]], metric: str, resamples: int,
            ni_margin: float) -> Dict[str, Any]:
    pf = (lambda sub, a, b: paired_answer(sub, a, b)) if metric == "answer_score" \
        else (lambda sub, a, b: paired(sub, metric, a, b))
    q4 = pf(rows, "F16", "Q4_K_M")
    q8 = pf(rows, "F16", "Q8_0")
    ci4 = paired_bootstrap_ci(q4, resamples)
    ci8 = paired_bootstrap_ci(q8, resamples)
    res = {
        "metric": metric,
        "f16_q4_gap_overall": round(mean(q4), 4), "f16_q4_gap_ci": ci4,
        "f16_q8_gap_overall": round(mean(q8), 4), "f16_q8_gap_ci": ci8,
        "f16_q4_gap_by_width": {k: v["gap"] for k, v in
                                by_width_gaps(rows, metric, "F16", "Q4_K_M", resamples).items()},
        "f16_q8_gap_by_width": {k: v["gap"] for k, v in
                                by_width_gaps(rows, metric, "F16", "Q8_0", resamples).items()},
        "by_width_detail": {
            "f16_minus_q4": by_width_gaps(rows, metric, "F16", "Q4_K_M", resamples),
            "f16_minus_q8": by_width_gaps(rows, metric, "F16", "Q8_0", resamples)},
        "non_inferiority_margin": ni_margin,
        "regression": exploratory_regression(rows, metric),
    }
    res["q4_verdict"] = verdict(ci4, ni_margin)
    res["q8_verdict"] = verdict(ci8, ni_margin)
    return res


def verdict(ci: Dict[str, Any], margin: float) -> Dict[str, Any]:
    """Distinguish 'not significant' from 'non-inferior within margin'."""
    lo, hi = ci.get("lo"), ci.get("hi")
    if lo is None or hi is None:
        return {"label": "INDETERMINATE", "reason": ci.get("note", "no interval")}
    width = hi - lo
    crosses_zero = lo <= 0.0 <= hi
    if hi <= margin and width <= 0.30:
        return {"label": "NON_INFERIOR_WITHIN_MARGIN",
                "reason": f"CI upper bound {hi:+.3f} <= margin {margin}, width {width:.3f}"}
    if hi <= margin:
        return {"label": "NOT_SIGNIFICANT_BUT_CI_TOO_WIDE",
                "reason": f"CI [{lo:+.3f}, {hi:+.3f}] width {width:.3f}; cannot claim equivalence"}
    if crosses_zero:
        return {"label": "NOT_STATISTICALLY_SIGNIFICANT",
                "reason": f"CI [{lo:+.3f}, {hi:+.3f}] includes 0 but exceeds the margin"}
    return {"label": "DEGRADATION_DETECTED",
            "reason": f"CI [{lo:+.3f}, {hi:+.3f}] excludes 0 and exceeds margin {margin}"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=["stage_a", "stage_b", "stage_c"])
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--metric", default="atomic_fact_recall_final")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--ni-margin", type=float, default=0.05,
                    help="predeclared non-inferiority margin in atomic-fact recall")
    ap.add_argument("--condition", default=None)
    args = ap.parse_args()

    rd = results_dir(args.results_dir)
    sd = rd / args.stage
    scored = read_json(sd / "scored.json", default=[]) or []
    summary = read_json(sd / "score_summary.json", default={}) or {}
    run_report = read_json(sd / "run_report.json", default={}) or {}
    provenance = read_json(rd / "provenance.json", default={}) or {}
    events = []
    ev_path = sd / "events.jsonl"
    if ev_path.exists():
        for line in ev_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not scored:
        raise SystemExit(f"no scored.json in {sd}; run scripts/score.py first")

    condition = args.condition or (scored[0].get("condition"))
    rows = [s for s in scored if s.get("condition") == condition]
    real = [s for s in rows if not s.get("is_fixture") and not s.get("dry_run")]

    required_q = {"stage_a": 6, "stage_b": 12, "stage_c": 6}[args.stage]
    gate_report = evaluate_gates(events, scored, provenance, required_q, condition)

    stats = analyse(real, args.metric, args.resamples, args.ni_margin) if real else {}
    answer_stats = analyse(real, "answer_score", args.resamples, args.ni_margin) if real else {}
    synth_stats = analyse(real, "synthesis_loss", args.resamples, args.ni_margin) if real else {}

    bp = summary.get("by_precision", {})
    outcome_input = {
        "f16_atomic_fact_recall": bp.get("F16", {}).get("atomic_fact_recall"),
        "q8_atomic_fact_recall": bp.get("Q8_0", {}).get("atomic_fact_recall"),
        "q4_atomic_fact_recall": bp.get("Q4_K_M", {}).get("atomic_fact_recall"),
        "f16_q4_gap_overall": stats.get("f16_q4_gap_overall"),
        "f16_q8_gap_overall": stats.get("f16_q8_gap_overall"),
        "f16_q4_gap_by_width": stats.get("f16_q4_gap_by_width"),
        "f16_q4_gap_ci": stats.get("f16_q4_gap_ci"),
    }
    sci = scientific_outcome(outcome_input, gate_report, ni_margin=args.ni_margin)

    harness_ok = bool(run_report) and not run_report.get("failures")
    real_attempted = bool(run_report) and not run_report.get("dry_run") and \
        run_report.get("backend_kind") not in ("fixture", "mock")
    eng = engineering_status(
        harness_ok=harness_ok, real_run_attempted=real_attempted,
        real_run_ok=bool(real) and gate_report.passed,
        detail=(f"{len(real)} real predictions, "
                f"{len(rows) - len(real)} fixture/dry-run, "
                f"{len(run_report.get('failures') or [])} question failures"))

    f16r = outcome_input["f16_atomic_fact_recall"]
    report = {
        "stage": args.stage, "condition": condition, "metric": args.metric,
        "non_inferiority_margin": args.ni_margin,
        **eng,
        **sci,
        "calibration_status": calibration_status(f16r) if f16r is not None
        else "UNAVAILABLE — no real F16 results",
        "gates": gate_report.to_dict(),
        "by_precision": bp,
        "primary_metric_analysis": stats,
        "answer_score_analysis": answer_stats,
        "synthesis_loss_analysis": synth_stats,
        "branch_failure_probability_by_width": branch_failure_by_width(real),
        "systems": systems_table(run_report, provenance, bp),
        "n_real_predictions": len(real),
        "n_excluded_fixture_or_dry_run": len(rows) - len(real),
        "rubric_reviewed": summary.get("rubric_reviewed"),
        "caveats": [
            "Screening pilot. Thresholds are decision aids, not significance tests.",
            "Bootstrap intervals over <=12 paired questions are wide by construction.",
            "Non-inferiority is only claimed when the CI upper bound is inside the "
            "predeclared margin AND the interval is narrow.",
            "The regression is exploratory and descriptive.",
        ],
    }
    if not summary.get("rubric_reviewed", True):
        report["caveats"].insert(
            0, "data/gold_facts.json still has needs_manual_review=true entries; "
               "these numbers are provisional.")

    proceed = (gate_report.passed and sci.get("scientific_outcome") != NOT_AVAILABLE
               and (f16r or 0.0) >= 0.50)
    write_json(sd / "analysis.json", report)
    write_json(sd / "gate.json", {
        "stage": args.stage, "proceed": bool(proceed),
        "engineering_status": eng["engineering_status"],
        "scientific_outcome": sci.get("scientific_outcome"),
        "calibration_status": report["calibration_status"],
        "reason": sci.get("reason", ""),
        "failed_gates": gate_report.failures(),
        "note": ("Set proceed=true manually only as an explicit, documented "
                 "override after human review."),
    })
    print(json.dumps({k: report[k] for k in
                      ("engineering_status", "scientific_outcome",
                       "calibration_status", "n_real_predictions",
                       "n_excluded_fixture_or_dry_run")}, indent=2))
    if gate_report.failures():
        print("[analyze] failed gates:")
        for f in gate_report.failures():
            print("  -", f)
    print(f"[analyze] -> {sd / 'analysis.json'}")
    return 0


def branch_failure_by_width(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for r in rows:
        w = str(r.get("width"))
        p = r["precision"]
        node = out.setdefault(w, {}).setdefault(p, {"n": 0, "any_branch_failed": 0,
                                                    "agent_recall_sum": 0.0,
                                                    "n_agents": 0})
        node["n"] += 1
        node["any_branch_failed"] += 1 if r["orchestration"]["any_required_branch_failed"] else 0
        for b in r["branch"]:
            node["agent_recall_sum"] += b["recall"]
            node["n_agents"] += 1
    for w, precs in out.items():
        for p, node in precs.items():
            node["branch_failure_probability"] = round(
                node["any_branch_failed"] / node["n"], 4) if node["n"] else 0.0
            node["mean_agent_recall"] = round(
                node["agent_recall_sum"] / node["n_agents"], 4) if node["n_agents"] else 0.0
            node.pop("agent_recall_sum")
    return out


def systems_table(run_report: Dict[str, Any], provenance: Dict[str, Any],
                  by_prec: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    models = provenance.get("models") or {}
    for prec, block in (run_report.get("blocks") or {}).items():
        m = models.get(prec) or {}
        size = int(m.get("size_bytes") or 0)
        quality = float((by_prec.get(prec) or {}).get("atomic_fact_recall") or 0.0)
        wall = float(block.get("wall_s") or 0.0)
        gb = size / (1024 ** 3) if size else 0.0
        out[prec] = {
            "model_size_bytes": size, "model_size_gb": round(gb, 4),
            "model_load_time_s": block.get("load_time_s"),
            "peak_cpu_rss_mib": (block.get("memory") or {}).get("peak_cpu_rss_mib"),
            "peak_gpu_used_mib": (block.get("memory") or {}).get("peak_gpu_used_mib"),
            "gpu_offload": block.get("gpu_offload"),
            "block_wall_s_serial": round(wall, 2),
            "runtime_basis": "actual serial wall time",
            "quality_per_second": round(quality / wall, 6) if wall else 0.0,
            "quality_per_gb": round(quality / gb, 6) if gb else 0.0,
            "mean_question_wall_s": (by_prec.get(prec) or {}).get("mean_wall_s"),
            "mean_document_agent_wall_s": (by_prec.get(prec) or {}).get(
                "mean_document_agent_wall_s"),
        }
    return out


if __name__ == "__main__":
    raise SystemExit(main())
