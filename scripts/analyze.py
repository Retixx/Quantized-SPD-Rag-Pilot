#!/usr/bin/env python3
"""Paired analysis, bootstrap CIs, gate evaluation and the go/no-go decision.

Deliberately conservative: the pilot is small, so this reports paired
per-question differences with percentile bootstrap intervals and a predeclared
non-inferiority margin, and refuses to call equivalence when the interval is
wide OR when too few pairs actually differ. Engineering status and scientific
status are separate outputs.

Design rules that the previous version violated:

  * **One primary contrast.** F16 - Q4_K_M on the primary metric, in the
    primary condition. Everything else is labelled exploratory and carries a
    Holm-adjusted companion, because ~24 uncorrected 95% intervals per stage
    followed by a first-match outcome selector is a garden of forking paths.
  * **A verdict needs information, not just a narrow interval.** With a coarse
    thresholded metric at n=6, many paired differences are exactly 0; the
    resulting zero-width bootstrap CI used to satisfy the "interval narrow"
    test perfectly, so no information produced the strongest possible
    equivalence claim.
  * **Missing is not zero.** `get_metric` used to coerce a missing value to
    0.0 while `paired` raised KeyError on the same field -- two different
    missing-data semantics for one metric name. Both now drop the pair.
  * **Loss metrics are not accuracy metrics.** `verdict` assumed higher is
    better, so on `synthesis_loss` it reported DEGRADATION_DETECTED when Q4
    lost *fewer* facts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import results_dir  # noqa: E402
from gates import (MIN_INFORMATIVE_PAIRS, NOT_AVAILABLE, calibration_status,  # noqa: E402
                   engineering_status, evaluate_gates, scientific_outcome)
from logging_utils import read_json, write_json  # noqa: E402
from metrics import system_efficiency  # noqa: E402

BOOTSTRAP_SEED = 20260725

# Metrics where a LOWER value is better. `verdict` flips the sign so
# "non-inferior" always means "Q4 is not meaningfully worse than F16".
LOWER_IS_BETTER = {"synthesis_loss", "omission_rate", "uncited_claim_rate",
                   "unsupported_claim_rate", "contradictions_introduced",
                   "branch_failure_probability"}

REQUIRED_QUESTIONS = {"stage_a": 6, "stage_b": 12, "stage_c": 6}


def mean(xs: Sequence[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _seed_for(*parts: str) -> int:
    """Distinct deterministic stream per contrast.

    Every interval used to share one RNG stream, which makes their Monte-Carlo
    errors correlated -- deterministic, but not independent replicates.
    """
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()[:8]
    return (BOOTSTRAP_SEED + int(h, 16)) % (2 ** 31)


def _percentile(sorted_xs: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile on the (R+1) convention."""
    if not sorted_xs:
        raise ValueError("empty sample")
    n = len(sorted_xs)
    pos = q * (n + 1) - 1
    lo = max(0, min(n - 1, int(pos)))
    hi = max(0, min(n - 1, lo + 1))
    frac = pos - lo
    return sorted_xs[lo] + frac * (sorted_xs[hi] - sorted_xs[lo])


def paired_bootstrap_ci(diffs: Sequence[float], resamples: int = 10000,
                        alpha: float = 0.05, seed: int = BOOTSTRAP_SEED
                        ) -> Dict[str, Any]:
    """Percentile bootstrap over PAIRED per-question differences.

    `n_informative` is the count of non-zero differences. A stage where every
    pair ties gives a zero-width interval that means "we learned nothing", not
    "the two are equivalent", and downstream verdicts must be able to tell.
    """
    n = len(diffs)
    n_informative = sum(1 for d in diffs if abs(d) > 1e-12)
    base = {"mean": round(mean(diffs), 4) if n else None, "n": n,
            "n_informative": n_informative}
    if n == 0:
        return {**base, "lo": None, "hi": None, "note": "no paired observations"}
    if n < 3:
        return {**base, "lo": None, "hi": None,
                "note": "fewer than 3 pairs; interval not estimated"}
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return {**base,
            "lo": round(_percentile(means, alpha / 2), 4),
            "hi": round(_percentile(means, 1 - alpha / 2), 4),
            "resamples": resamples, "alpha": alpha}


def get_metric(row: Dict[str, Any], metric: str) -> Optional[float]:
    """Value of `metric` on one scored row, or None if it is absent/unscorable."""
    if metric == "answer_score":
        v = (row.get("answer_score") or {}).get("score")
    elif metric in ("branch_failure_probability", "any_required_branch_failed"):
        v = (row.get("orchestration") or {}).get("any_required_branch_failed")
        v = None if v is None else (1.0 if v else 0.0)
    else:
        v = row.get(metric)
    if v is None or isinstance(v, bool) and metric != "any_required_branch_failed":
        return None if v is None else float(bool(v))
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def paired(rows: List[Dict[str, Any]], metric: str, a: str, b: str
           ) -> Tuple[List[float], List[str]]:
    """a - b per question, only where BOTH precisions produced a scorable value."""
    ma = {r["question_id"]: r for r in rows if r["precision"] == a}
    mb = {r["question_id"]: r for r in rows if r["precision"] == b}
    diffs, qids = [], []
    for k in sorted(set(ma) & set(mb)):
        va, vb = get_metric(ma[k], metric), get_metric(mb[k], metric)
        if va is None or vb is None:
            continue
        diffs.append(va - vb)
        qids.append(k)
    return diffs, qids


def ols(X: List[List[float]], y: List[float]) -> Optional[Dict[str, Any]]:
    """Least squares with an explicit RANK check.

    `lstsq(rcond=None)` returns a minimum-norm solution for a singular design
    rather than raising, so a rank-deficient fit (e.g. a stage where every
    question has the same width, which collapses width onto the intercept)
    used to return confident-looking coefficients for an unidentified model.
    """
    try:
        import numpy as np  # noqa: WPS433
        A = np.asarray(X, dtype=float)
        b = np.asarray(y, dtype=float)
        if A.shape[0] <= A.shape[1]:
            return {"error": f"{A.shape[0]} observations for {A.shape[1]} terms"}
        rank = int(np.linalg.matrix_rank(A))
        if rank < A.shape[1]:
            return {"error": f"design matrix is rank {rank} of {A.shape[1]}; "
                             "the model is not identified at this sample"}
        beta, *_ = np.linalg.lstsq(A, b, rcond=None)
        resid = b - A @ beta
        dof = A.shape[0] - A.shape[1]
        s2 = float(resid @ resid) / dof if dof > 0 else float("nan")
        try:
            cov = s2 * np.linalg.inv(A.T @ A)
            se = [float(np.sqrt(max(0.0, cov[i, i]))) for i in range(A.shape[1])]
        except np.linalg.LinAlgError:
            se = [float("nan")] * A.shape[1]
        return {"beta": [round(float(v), 5) for v in beta],
                "se_iid": [round(v, 5) for v in se], "dof": dof}
    except Exception as exc:  # pragma: no cover - numpy optional
        return {"error": f"{type(exc).__name__}: {exc}"}


def exploratory_regression(rows: List[Dict[str, Any]], metric: str) -> Dict[str, Any]:
    """metric ~ 1 + is_q4 + is_q8 + width + is_q4*width + is_q8*width.

    EXPLORATORY AND DESCRIPTIVE ONLY. Two things a reader must know, and which
    the previous version did not say:

      * The rows are repeated measures -- 12 questions x 3 precisions, not 36
        independent observations. The precision terms are within-question
        contrasts being estimated as if they were between-subject, so the
        reported iid standard errors are anticonservative. The paired bootstrap
        above is the inferential result; this is a shape description.
      * `width` enters as continuous over {2, 3, 4}, which forces linearity on
        the very curve the study exists to characterise.
    """
    X, y = [], []
    for r in rows:
        w = r.get("width")
        if w is None:
            continue
        v = get_metric(r, metric)
        if v is None:
            continue
        q4 = 1.0 if r["precision"] == "Q4_K_M" else 0.0
        q8 = 1.0 if r["precision"] == "Q8_0" else 0.0
        X.append([1.0, q4, q8, float(w), q4 * float(w), q8 * float(w)])
        y.append(v)
    names = ["intercept", "is_q4", "is_q8", "width", "is_q4_x_width", "is_q8_x_width"]
    fit = ols(X, y)
    n_q = len({r["question_id"] for r in rows})
    out: Dict[str, Any] = {
        "metric": metric, "n_observations": len(y), "n_questions": n_q,
        "terms": names,
        "design": "repeated measures (question x precision); NOT independent",
        "note": "exploratory and descriptive; the paired bootstrap is the "
                "inferential result. Standard errors shown are iid and "
                "anticonservative for this design.",
    }
    if not fit or "error" in (fit or {}):
        out["coefficients"] = None
        out["not_estimable"] = (fit or {}).get("error", "unavailable")
        return out
    out["coefficients"] = dict(zip(names, fit["beta"]))
    out["standard_errors_iid"] = dict(zip(names, fit["se_iid"]))
    out["residual_dof"] = fit["dof"]
    return out


def by_width_gaps(rows: List[Dict[str, Any]], metric: str, a: str, b: str,
                  resamples: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    widths = sorted({r.get("width") for r in rows if r.get("width") is not None})
    for w in widths:
        sub = [r for r in rows if r.get("width") == w]
        d, _ = paired(sub, metric, a, b)
        out[str(w)] = {
            "gap": round(mean(d), 4) if d else None,
            "ci": paired_bootstrap_ci(d, resamples,
                                      seed=_seed_for(metric, a, b, "w", str(w))),
            "n_pairs": len(d),
        }
    return out


def growth_ci(rows: List[Dict[str, Any]], metric: str, a: str, b: str,
              resamples: int) -> Dict[str, Any]:
    """CI on (gap at the widest cell) - (gap at the narrowest cell).

    This contrast IS the compounding hypothesis, and the previous code emitted
    GO_COMPOUNDING_DEGRADATION from a bare threshold on the difference of two
    point estimates while discarding the intervals it had already computed. At
    3-4 paired questions per cell, one flipped fact match moves a cell by
    0.25-1.0, i.e. 5-20x the 0.05 trigger.

    Resampling is stratified: questions are resampled with replacement inside
    each width cell, preserving the paired structure.
    """
    widths = sorted({r.get("width") for r in rows if r.get("width") is not None})
    if len(widths) < 2:
        return {"mean": None, "lo": None, "hi": None,
                "note": "fewer than two width cells"}
    lo_w, hi_w = widths[0], widths[-1]
    d_lo, _ = paired([r for r in rows if r.get("width") == lo_w], metric, a, b)
    d_hi, _ = paired([r for r in rows if r.get("width") == hi_w], metric, a, b)
    if not d_lo or not d_hi:
        return {"mean": None, "lo": None, "hi": None,
                "note": f"no paired data at width {lo_w} or {hi_w}"}
    point = mean(d_hi) - mean(d_lo)
    if len(d_lo) < 2 or len(d_hi) < 2:
        return {"mean": round(point, 4), "lo": None, "hi": None,
                "width_low": lo_w, "width_high": hi_w,
                "n_pairs_low": len(d_lo), "n_pairs_high": len(d_hi),
                "note": "cell too small to bootstrap"}
    rng = random.Random(_seed_for(metric, a, b, "growth"))
    draws = []
    for _ in range(resamples):
        rl = sum(d_lo[rng.randrange(len(d_lo))] for _ in range(len(d_lo))) / len(d_lo)
        rh = sum(d_hi[rng.randrange(len(d_hi))] for _ in range(len(d_hi))) / len(d_hi)
        draws.append(rh - rl)
    draws.sort()
    return {"mean": round(point, 4),
            "lo": round(_percentile(draws, 0.025), 4),
            "hi": round(_percentile(draws, 0.975), 4),
            "width_low": lo_w, "width_high": hi_w,
            "n_pairs_low": len(d_lo), "n_pairs_high": len(d_hi),
            "resamples": resamples}


def verdict(ci: Dict[str, Any], margin: float, higher_is_better: bool = True,
            max_ci_width: Optional[float] = None) -> Dict[str, Any]:
    """Distinguish 'not significant' from 'non-inferior within margin'.

    `ci` is over F16 - Q4. For an accuracy metric a positive difference means
    Q4 is worse; for a loss metric it is the reverse, so the interval is
    flipped before the tests. `max_ci_width` defaults to 2x the margin -- the
    old hardcoded 0.30 was 6x, so [-0.28, +0.02] counted as narrow enough to
    support an equivalence claim.
    """
    if max_ci_width is None:
        max_ci_width = 2.0 * margin
    lo, hi = ci.get("lo"), ci.get("hi")
    if lo is None or hi is None:
        return {"label": "INDETERMINATE", "reason": ci.get("note", "no interval")}
    if not higher_is_better:
        lo, hi = -float(hi), -float(lo)
    lo, hi = float(lo), float(hi)
    width = hi - lo
    n_informative = int(ci.get("n_informative") or 0)
    crosses_zero = lo <= 0.0 <= hi

    if n_informative < MIN_INFORMATIVE_PAIRS and hi <= margin:
        return {"label": "NOT_SIGNIFICANT_BUT_CI_TOO_WIDE",
                "reason": (f"only {n_informative} of {ci.get('n')} paired "
                           f"differences are non-zero; at least "
                           f"{MIN_INFORMATIVE_PAIRS} are needed before a "
                           "narrow interval means anything"),
                "degenerate": True}
    if hi <= margin and width <= max_ci_width:
        return {"label": "NON_INFERIOR_WITHIN_MARGIN",
                "reason": (f"CI upper bound {hi:+.3f} <= margin {margin}, "
                           f"width {width:.3f} <= {max_ci_width:.3f}, "
                           f"{n_informative} informative pairs")}
    if hi <= margin:
        return {"label": "NOT_SIGNIFICANT_BUT_CI_TOO_WIDE",
                "reason": (f"CI [{lo:+.3f}, {hi:+.3f}] width {width:.3f} > "
                           f"{max_ci_width:.3f}; cannot claim equivalence")}
    if crosses_zero:
        return {"label": "NOT_STATISTICALLY_SIGNIFICANT",
                "reason": f"CI [{lo:+.3f}, {hi:+.3f}] includes 0 but exceeds the margin"}
    return {"label": "DEGRADATION_DETECTED",
            "reason": f"CI [{lo:+.3f}, {hi:+.3f}] excludes 0 and exceeds margin {margin}"}


def analyse(rows: List[Dict[str, Any]], metric: str, resamples: int,
            ni_margin: float, primary: bool = False) -> Dict[str, Any]:
    higher_is_better = metric not in LOWER_IS_BETTER
    q4, q4_ids = paired(rows, metric, "F16", "Q4_K_M")
    q8, _ = paired(rows, metric, "F16", "Q8_0")
    ci4 = paired_bootstrap_ci(q4, resamples, seed=_seed_for(metric, "F16", "Q4_K_M"))
    ci8 = paired_bootstrap_ci(q8, resamples, seed=_seed_for(metric, "F16", "Q8_0"))
    bw4 = by_width_gaps(rows, metric, "F16", "Q4_K_M", resamples)
    bw8 = by_width_gaps(rows, metric, "F16", "Q8_0", resamples)
    res = {
        "metric": metric,
        "is_primary_contrast": primary,
        "higher_is_better": higher_is_better,
        "n_paired_questions": len(q4),
        "paired_question_ids": q4_ids,
        "f16_q4_gap_overall": round(mean(q4), 4) if q4 else None,
        "f16_q4_gap_ci": ci4,
        "f16_q8_gap_overall": round(mean(q8), 4) if q8 else None,
        "f16_q8_gap_ci": ci8,
        "f16_q4_gap_by_width": {k: v["gap"] for k, v in bw4.items()},
        "f16_q4_gap_by_width_n": {k: v["n_pairs"] for k, v in bw4.items()},
        "f16_q8_gap_by_width": {k: v["gap"] for k, v in bw8.items()},
        "f16_q4_gap_growth_ci": growth_ci(rows, metric, "F16", "Q4_K_M", resamples),
        "by_width_detail": {"f16_minus_q4": bw4, "f16_minus_q8": bw8},
        "non_inferiority_margin": ni_margin,
        "regression": exploratory_regression(rows, metric),
    }
    res["q4_verdict"] = verdict(ci4, ni_margin, higher_is_better)
    res["q8_verdict"] = verdict(ci8, ni_margin, higher_is_better)
    if not primary:
        res["multiplicity_note"] = (
            "Secondary/exploratory contrast. Only the primary metric's "
            "F16-Q4 comparison is confirmatory; read this alongside "
            "`multiplicity` in the report.")
    return res


def holm(pairs: Sequence[Tuple[str, float]]) -> Dict[str, Any]:
    """Holm-Bonferroni over a set of (name, p) pairs.

    We report intervals rather than p-values, so this operates on a bootstrap
    two-sided p-value proxy and exists to make the multiplicity burden visible
    rather than to be the primary inference.
    """
    m = len(pairs)
    ordered = sorted(pairs, key=lambda kv: kv[1])
    out, prev = {}, 0.0
    for i, (name, p) in enumerate(ordered):
        adj = min(1.0, max(prev, (m - i) * p))
        prev = adj
        out[name] = round(adj, 4)
    return out


def bootstrap_p(diffs: Sequence[float], resamples: int, seed: int) -> Optional[float]:
    """Two-sided bootstrap p-value proxy: P(resampled mean crosses 0)."""
    n = len(diffs)
    if n < 3:
        return None
    rng = random.Random(seed)
    m = mean(diffs) or 0.0
    centred = [d - m for d in diffs]
    hits = 0
    for _ in range(resamples):
        r = sum(centred[rng.randrange(n)] for _ in range(n)) / n
        if abs(r) >= abs(m):
            hits += 1
    return round((hits + 1) / (resamples + 1), 4)


def branch_failure_by_width(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for r in rows:
        w = str(r.get("width"))
        p = r["precision"]
        node = out.setdefault(w, {}).setdefault(
            p, {"n": 0, "n_scorable": 0, "any_branch_failed": 0,
                "agent_recall_sum": 0.0, "n_agents": 0})
        node["n"] += 1
        failed = (r.get("orchestration") or {}).get("any_required_branch_failed")
        if failed is not None:
            node["n_scorable"] += 1
            node["any_branch_failed"] += 1 if failed else 0
        for b in r.get("branch") or []:
            if b.get("recall") is None:
                continue
            node["agent_recall_sum"] += float(b["recall"])
            node["n_agents"] += 1
    for _w, precs in out.items():
        for _p, node in precs.items():
            node["branch_failure_probability"] = (
                round(node["any_branch_failed"] / node["n_scorable"], 4)
                if node["n_scorable"] else None)
            node["mean_agent_recall"] = (
                round(node["agent_recall_sum"] / node["n_agents"], 4)
                if node["n_agents"] else None)
            node.pop("agent_recall_sum")
    return out


def systems_table(run_report: Dict[str, Any], provenance: Dict[str, Any],
                  by_prec: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    models = provenance.get("models") or {}
    for prec, block in (run_report.get("blocks") or {}).items():
        m = models.get(prec) or {}
        size = int(m.get("size_bytes") or 0)
        quality = (by_prec.get(prec) or {}).get("atomic_fact_recall")
        # Wall time excluding model hashing and resume-skipped questions. The
        # old block_wall_s started before a full sha256 of the GGUF, charging
        # F16 ~6.2 GB of disk I/O against Q4's ~1.9 GB, then divided quality by
        # it; and on a resumed block it timed only the questions that re-ran.
        wall = block.get("inference_wall_s")
        if wall is None:
            wall = block.get("wall_s")
        resumed = int(block.get("n_resumed_questions") or 0)
        mem = block.get("memory") or {}
        eff = system_efficiency(quality, float(wall) if wall else None, size)
        out[prec] = {
            "model_size_bytes": size,
            "model_size_gb": eff["model_size_gb"],
            "model_load_time_s": block.get("load_time_s"),
            # Per-process where the driver reports it; device-total kept beside
            # it because the old single number silently included the embedder's
            # CUDA context, the previous block's residue and any co-tenant.
            "peak_model_rss_mib": mem.get("peak_model_rss_mib"),
            "peak_harness_rss_mib": mem.get("peak_harness_rss_mib"),
            "peak_gpu_used_mib": mem.get("peak_gpu_used_mib"),
            "peak_gpu_device_total_mib": mem.get("peak_gpu_device_total_mib"),
            "cpu_rss_source": mem.get("cpu_rss_source"),
            "gpu_used_source": mem.get("gpu_used_source"),
            "memory_is_final": block.get("memory_is_final"),
            "sampler_complete": mem.get("sampler_complete"),
            "gpu_offload": block.get("gpu_offload"),
            "block_wall_s_serial": round(float(wall), 2) if wall else None,
            "runtime_basis": ("actual serial wall time, excluding model hashing"
                              if not resumed else
                              "INVALID: block was resumed; wall time covers only "
                              f"the {block.get('n_questions_run', '?')} questions "
                              f"that re-ran ({resumed} were skipped)"),
            "n_resumed_questions": resumed,
            "quality_per_second": None if resumed else eff["quality_per_second"],
            "quality_per_gb": eff["quality_per_gb"],
            "mean_question_wall_s": (by_prec.get(prec) or {}).get("mean_wall_s"),
            "mean_document_agent_wall_s": (by_prec.get(prec) or {}).get(
                "mean_document_agent_wall_s"),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=["stage_a", "stage_b", "stage_c"])
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--metric", default="atomic_fact_recall_final",
                    help="primary (confirmatory) metric")
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

    conditions = {s.get("condition") for s in scored}
    if args.condition:
        condition = args.condition
    elif len(conditions) == 1:
        condition = conditions.pop()
    else:
        raise SystemExit(
            f"scored.json mixes conditions {sorted(c for c in conditions if c)}; "
            "pass --condition explicitly rather than letting the first row decide")

    rows = [s for s in scored if s.get("condition") == condition]
    # Real == not fixture, not dry-run, and not a row whose generation errored.
    # Infrastructure failure must never be aggregated as model quality.
    real = [s for s in rows
            if not s.get("is_fixture") and not s.get("dry_run")
            and not s.get("generation_error")]

    required_q = REQUIRED_QUESTIONS[args.stage]
    gate_report = evaluate_gates(events, scored, provenance, required_q, condition)

    stats = analyse(real, args.metric, args.resamples, args.ni_margin,
                    primary=True) if real else {}
    answer_stats = analyse(real, "answer_score", args.resamples,
                           args.ni_margin) if real else {}
    synth_stats = analyse(real, "synthesis_loss", args.resamples,
                          args.ni_margin) if real else {}

    # Make the multiplicity burden explicit instead of leaving ~24 uncorrected
    # intervals for an outcome selector to scan.
    mult_inputs: List[Tuple[str, float]] = []
    for name, st in (("primary_f16_q4", stats), ("answer_score_f16_q4", answer_stats),
                     ("synthesis_loss_f16_q4", synth_stats)):
        if not st:
            continue
        d, _ = paired(real, st["metric"], "F16", "Q4_K_M")
        p = bootstrap_p(d, args.resamples, _seed_for(st["metric"], "p"))
        if p is not None:
            mult_inputs.append((name, p))
    multiplicity = {
        "family": [n for n, _ in mult_inputs],
        "raw_bootstrap_p": {n: p for n, p in mult_inputs},
        "holm_adjusted_p": holm(mult_inputs) if mult_inputs else {},
        "note": ("Only `primary_f16_q4` is confirmatory. Adjusted p-values are "
                 "reported so the reader can see how much of the family is "
                 "being scanned; intervals remain the primary inference."),
    }

    bp = summary.get("by_precision", {})
    outcome_input = {
        "f16_atomic_fact_recall": bp.get("F16", {}).get("atomic_fact_recall"),
        "q8_atomic_fact_recall": bp.get("Q8_0", {}).get("atomic_fact_recall"),
        "q4_atomic_fact_recall": bp.get("Q4_K_M", {}).get("atomic_fact_recall"),
        "f16_q4_gap_overall": stats.get("f16_q4_gap_overall"),
        "f16_q8_gap_overall": stats.get("f16_q8_gap_overall"),
        "f16_q4_gap_by_width": stats.get("f16_q4_gap_by_width"),
        "f16_q4_gap_by_width_n": stats.get("f16_q4_gap_by_width_n"),
        "f16_q4_gap_growth_ci": stats.get("f16_q4_gap_growth_ci"),
        "f16_q4_gap_ci": stats.get("f16_q4_gap_ci"),
    }
    sci = scientific_outcome(outcome_input, gate_report, ni_margin=args.ni_margin)

    # Engineering status describes the HARNESS. A scientific provenance gate
    # failing says nothing about whether the run executed correctly, so
    # gate_report is deliberately not fed into real_run_ok here.
    harness_ok = bool(run_report) and not run_report.get("harness_failures")
    real_attempted = bool(run_report) and not run_report.get("dry_run") and \
        run_report.get("backend_kind") not in ("fixture", "mock")
    eng = engineering_status(
        harness_ok=harness_ok, real_run_attempted=real_attempted,
        real_run_ok=bool(real) and run_report.get("complete", False),
        detail=(f"{len(real)} real predictions, "
                f"{len(rows) - len(real)} excluded (fixture/dry-run/errored), "
                f"{len(run_report.get('failures') or [])} question failures"))

    f16r = outcome_input["f16_atomic_fact_recall"]
    report = {
        "stage": args.stage, "condition": condition, "metric": args.metric,
        "non_inferiority_margin": args.ni_margin,
        **eng,
        **sci,
        "calibration_status": calibration_status(f16r),
        "gates": gate_report.to_dict(),
        "by_precision": bp,
        "primary_metric_analysis": stats,
        "answer_score_analysis": answer_stats,
        "synthesis_loss_analysis": synth_stats,
        "multiplicity": multiplicity,
        "branch_failure_probability_by_width": branch_failure_by_width(real),
        "systems": systems_table(run_report, provenance, bp),
        "n_real_predictions": len(real),
        "n_excluded_fixture_or_dry_run": len(rows) - len(real),
        "rubric_reviewed": summary.get("rubric_reviewed"),
        "caveats": [
            "Screening pilot. Thresholds are decision aids, not significance tests.",
            "Bootstrap intervals over <=12 paired questions are wide by construction.",
            "Non-inferiority is only claimed when the CI upper bound is inside the "
            "predeclared margin, the interval is narrow relative to that margin, AND "
            f"at least {MIN_INFORMATIVE_PAIRS} paired differences are non-zero.",
            "The regression is exploratory: the rows are repeated measures, so its "
            "iid standard errors are anticonservative. The paired bootstrap is the "
            "inferential result.",
            "Precision blocks always run in the order F16 -> Q8_0 -> Q4_K_M, so any "
            "monotonic drift over the session (thermal, cache, co-tenancy) is "
            "confounded with precision in the timing metrics.",
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


if __name__ == "__main__":
    raise SystemExit(main())
