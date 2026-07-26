"""Hard separation between ENGINEERING status and SCIENTIFIC status.

Rule of the pilot: fixtures, mocks and dry runs may prove the harness works.
They may never produce a scientific recommendation. Every gate below must pass
before `scientific_outcome()` is even allowed to look at the numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

ENGINEERING_STATUSES = ("HARNESS_READY", "HARNESS_FAILED", "REAL_MODEL_RUN_BLOCKED")

SCIENTIFIC_OUTCOMES = (
    "GO_COMPOUNDING_DEGRADATION",
    "GO_Q8_OPERATING_POINT",
    "GO_DECOMPOSITION_ROBUSTNESS",
    "WEAK_GO_EXPAND",
    "NO_GO_NO_STRUCTURED_EFFECT",
    "NO_GO_MODEL_FLOOR",
    "NO_GO_EVALUATION_CEILING",
)

NOT_AVAILABLE = "SCIENTIFIC_RECOMMENDATION_NOT_AVAILABLE"

REQUIRED_PRECISIONS = ("F16", "Q8_0", "Q4_K_M")


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"gate": self.name, "passed": self.passed, "detail": self.detail}


@dataclass
class GateReport:
    gates: List[GateResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.gates)

    def failures(self) -> List[str]:
        return [f"{g.name}: {g.detail}" for g in self.gates if not g.passed]

    def to_dict(self) -> Dict[str, Any]:
        return {"all_gates_passed": self.passed,
                "gates": [g.to_dict() for g in self.gates],
                "failures": self.failures()}


def evaluate_gates(events: Sequence[Dict[str, Any]],
                   predictions: Sequence[Dict[str, Any]],
                   provenance: Dict[str, Any],
                   required_questions: int,
                   condition: Optional[str] = None) -> GateReport:
    """The eight hard scientific gates from the brief."""
    g: List[GateResult] = []
    preds = [p for p in predictions
             if condition is None or p.get("condition") == condition]

    # 1. all results came from real llama.cpp inference
    fixture_events = [e for e in events if e.get("is_fixture") or e.get("dry_run")]
    fixture_preds = [p for p in preds if p.get("is_fixture") or p.get("dry_run")]
    g.append(GateResult(
        "real_inference_only",
        not fixture_events and not fixture_preds,
        f"{len(fixture_events)} fixture/dry-run events, {len(fixture_preds)} "
        "fixture/dry-run predictions found" if (fixture_events or fixture_preds)
        else "no fixture or dry-run records present"))

    backends = {e.get("backend") for e in events if e.get("backend")}
    g.append(GateResult(
        "backend_is_llama_cpp",
        bool(backends) and backends <= {"llama-server", "llama-cli"},
        f"backends seen: {sorted(b for b in backends if b)}"))

    # 2/3. all three precisions completed with nonzero generations
    by_prec: Dict[str, List[Dict[str, Any]]] = {p: [] for p in REQUIRED_PRECISIONS}
    for p in preds:
        by_prec.setdefault(p.get("precision", "?"), []).append(p)
    missing = [pr for pr in REQUIRED_PRECISIONS if not by_prec.get(pr)]
    g.append(GateResult("all_three_precisions_present", not missing,
                        f"missing: {missing}" if missing
                        else "F16, Q8_0 and Q4_K_M all present"))

    zero_gen = []
    for pr in REQUIRED_PRECISIONS:
        tot = sum(int(e.get("generated_tokens") or 0)
                  for e in events if e.get("precision") == pr)
        if tot <= 0:
            zero_gen.append(pr)
    g.append(GateResult("nonzero_generations_per_precision", not zero_gen,
                        f"zero generated tokens for: {zero_gen}" if zero_gen
                        else "every precision generated tokens"))

    # 4. model hashes and provenance recorded
    models = (provenance or {}).get("models") or {}
    hashes = {pr: (models.get(pr) or {}).get("sha256") for pr in REQUIRED_PRECISIONS}
    bad_hash = [pr for pr, h in hashes.items()
                if not h or h in ("FIXTURE", "", None) or len(str(h)) != 64]
    distinct = len({h for h in hashes.values() if h}) == len(
        [h for h in hashes.values() if h])
    g.append(GateResult("model_hashes_recorded", not bad_hash and distinct,
                        f"bad or missing sha256 for {bad_hash}" if bad_hash
                        else ("duplicate sha256 across precisions" if not distinct
                              else "three distinct sha256 recorded")))
    common_src = (provenance or {}).get("common_source")
    g.append(GateResult(
        "common_source_checkpoint", bool(common_src),
        f"common source: {common_src}" if common_src
        else "no common source checkpoint/conversion path recorded"))

    # 5. fixed-evidence chunk ids identical across precisions
    if condition in (None, "fixed_verified_evidence"):
        chunk_sig: Dict[str, Dict[str, str]] = {}
        for p in preds:
            if p.get("condition") != "fixed_verified_evidence":
                continue
            chunk_sig.setdefault(p["question_id"], {})[p["precision"]] = \
                "|".join(sorted(p.get("evidence_chunk_ids") or []))
        mismatched = [q for q, m in chunk_sig.items() if len(set(m.values())) > 1]
        g.append(GateResult("fixed_evidence_chunk_parity", not mismatched,
                            f"chunk-id mismatch on {mismatched}" if mismatched
                            else f"{len(chunk_sig)} questions with identical chunk ids"))

        prompt_sig: Dict[str, Dict[str, str]] = {}
        for e in events:
            if e.get("condition") != "fixed_verified_evidence":
                continue
            if e.get("role") not in ("coordinator", "document_agent"):
                continue
            key = f"{e.get('question_id')}|{e.get('role')}|{e.get('document_id')}"
            prompt_sig.setdefault(key, {})[e.get("precision", "?")] = e.get("prompt_hash", "")
        bad_prompts = [k for k, m in prompt_sig.items() if len(set(m.values())) > 1]
        g.append(GateResult("prompt_hash_parity", not bad_prompts,
                            f"prompt-hash mismatch on {bad_prompts[:5]}" if bad_prompts
                            else f"{len(prompt_sig)} prompt slots hash-identical"))

    # 6. the stage's required question count completed at every precision
    short = {pr: len({p["question_id"] for p in by_prec.get(pr, [])})
             for pr in REQUIRED_PRECISIONS}
    incomplete = {pr: n for pr, n in short.items() if n < required_questions}
    g.append(GateResult("stage_question_count_complete", not incomplete,
                        f"need {required_questions} questions per precision, have "
                        f"{short}" if incomplete
                        else f"{required_questions} questions complete at each precision"))
    return GateReport(gates=g)


def engineering_status(harness_ok: bool, real_run_attempted: bool,
                       real_run_ok: bool, detail: str = "") -> Dict[str, Any]:
    if not harness_ok:
        status = "HARNESS_FAILED"
    elif real_run_attempted and not real_run_ok:
        status = "REAL_MODEL_RUN_BLOCKED"
    else:
        status = "HARNESS_READY"
    return {"engineering_status": status, "detail": detail}


def scientific_outcome(summary: Dict[str, Any], gate_report: GateReport,
                       calibration_floor: float = 0.50,
                       ceiling: float = 0.95,
                       ni_margin: float = 0.05,
                       min_gap_growth: float = 0.05,
                       wide_ci_width: float = 0.30) -> Dict[str, Any]:
    """Map the analysis summary onto one of the seven scientific outcomes.

    Returns NOT_AVAILABLE whenever a hard gate failed. Screening thresholds,
    not formal significance claims.
    """
    if not gate_report.passed:
        return {"scientific_outcome": NOT_AVAILABLE,
                "reason": "hard scientific gate(s) failed",
                "failed_gates": gate_report.failures()}

    f16 = summary.get("f16_atomic_fact_recall")
    if f16 is None:
        return {"scientific_outcome": NOT_AVAILABLE,
                "reason": "no F16 atomic-fact recall in summary"}
    if f16 < calibration_floor:
        return {"scientific_outcome": "NO_GO_MODEL_FLOOR",
                "reason": f"F16 atomic-fact recall {f16:.3f} < {calibration_floor}"}

    q4 = summary.get("q4_atomic_fact_recall")
    q8 = summary.get("q8_atomic_fact_recall")
    if q4 is not None and q8 is not None and min(f16, q8, q4) > ceiling:
        return {"scientific_outcome": "NO_GO_EVALUATION_CEILING",
                "reason": f"all precisions above {ceiling}; no stress condition separates them"}

    gap_by_width: Dict[str, float] = summary.get("f16_q4_gap_by_width") or {}
    ci = summary.get("f16_q4_gap_ci") or {}
    ci_width = (float(ci.get("hi", 1.0)) - float(ci.get("lo", -1.0))
                if ci else float("inf"))
    widths = sorted(gap_by_width, key=lambda w: int(w))
    growth = None
    if len(widths) >= 2:
        growth = gap_by_width[widths[-1]] - gap_by_width[widths[0]]

    if growth is not None and growth >= min_gap_growth:
        return {"scientific_outcome": "GO_COMPOUNDING_DEGRADATION",
                "reason": (f"F16-Q4 gap grows {gap_by_width[widths[0]]:.3f} -> "
                           f"{gap_by_width[widths[-1]]:.3f} from width {widths[0]} "
                           f"to {widths[-1]}"),
                "gap_growth": round(growth, 4)}

    q8_gap = summary.get("f16_q8_gap_overall")
    q4_gap = summary.get("f16_q4_gap_overall")
    if (q8_gap is not None and q4_gap is not None
            and abs(q8_gap) <= ni_margin and q4_gap > ni_margin):
        return {"scientific_outcome": "GO_Q8_OPERATING_POINT",
                "reason": (f"Q8 within margin ({q8_gap:+.3f}) while Q4 degrades "
                           f"({q4_gap:+.3f} > {ni_margin})")}

    if (q4_gap is not None and ci
            and float(ci.get("hi", 1.0)) <= ni_margin
            and ci_width <= wide_ci_width
            and f16 >= 0.70):
        return {"scientific_outcome": "GO_DECOMPOSITION_ROBUSTNESS",
                "reason": (f"Q4 non-inferior to F16: CI upper bound "
                           f"{ci.get('hi'):+.3f} <= margin {ni_margin}, "
                           f"CI width {ci_width:.3f}, F16 headroom {f16:.3f}")}

    if (growth is not None and growth > 0) or (q4_gap is not None and q4_gap > 0):
        return {"scientific_outcome": "WEAK_GO_EXPAND",
                "reason": ("directional pattern present but the interval is too wide "
                           f"to call (CI width {ci_width:.3f})")}

    return {"scientific_outcome": "NO_GO_NO_STRUCTURED_EFFECT",
            "reason": "differences show no structure by width or precision"}


def calibration_status(f16_atomic_fact_recall: float) -> str:
    r = f16_atomic_fact_recall
    if r < 0.50:
        return "BLOCKED — MODEL/BENCHMARK FLOOR"
    if r <= 0.70:
        return "MARGINAL — REVIEW PROMPTS AND FAILURES"
    if r <= 0.90:
        return "CALIBRATION PASS"
    if r <= 0.95:
        return "CALIBRATION PASS (upper band)"
    return "POSSIBLE CEILING — ADD HARDER ITEMS OR DISTRACTORS"
