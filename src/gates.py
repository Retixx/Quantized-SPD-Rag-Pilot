"""Hard separation between ENGINEERING status and SCIENTIFIC status.

Rule of the pilot: fixtures, mocks and dry runs may prove the harness works.
They may never produce a scientific recommendation. Every gate below must pass
before `scientific_outcome()` is even allowed to look at the numbers.

Two rules the gates now enforce that they previously did not:

  * **A gate may not pass vacuously.** The parity gates used to compare
    `len(set(values)) > 1` over whatever happened to be present, so a question
    that ran at only one precision had one value and "passed". Parity was
    therefore guaranteed exactly when data was missing. Every parity gate now
    requires all three precisions to be present for every slot it checks.
  * **A gate must check the artifact, not a claim about it.**
    `model_hashes_recorded` verified that three 64-character strings existed;
    it now re-hashes the files on disk when they are reachable, and
    `common_source_checkpoint` compares each variant's recorded parent against
    the F16 actually used instead of testing a note for truthiness.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# The precision that produces the frozen coordinator. Mirrors
# coordinator.FREEZE_PRECISION, duplicated here so gates.py stays importable
# without pulling in the pipeline.
FREEZE_PRECISION = "F16"

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

# An equivalence claim needs informative pairs, not just a narrow interval. A
# stage where every paired difference is exactly 0.0 produces a zero-width
# bootstrap CI, which used to satisfy "interval narrow" perfectly -- zero
# information yielding the strongest possible verdict.
MIN_INFORMATIVE_PAIRS = 4

# Minimum paired questions in a width cell before that cell may contribute to
# the compounding contrast. Stage A can otherwise leave a single question in a
# cell, where one flipped fact match moves it by 0.25-1.0.
MIN_PAIRS_PER_WIDTH = 3


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
        # An empty gate list is not a pass. `all([])` is True, which let a
        # caller construct GateReport(gates=[]) and short-circuit the entire
        # "no science without gates" property.
        return bool(self.gates) and all(g.passed for g in self.gates)

    def failures(self) -> List[str]:
        if not self.gates:
            return ["no gates were evaluated"]
        return [f"{g.name}: {g.detail}" for g in self.gates if not g.passed]

    def to_dict(self) -> Dict[str, Any]:
        return {"all_gates_passed": self.passed,
                "gates": [g.to_dict() for g in self.gates],
                "failures": self.failures()}


def _is_harness_record(rec: Dict[str, Any]) -> bool:
    """Fixture or dry-run provenance -- i.e. not real model output.

    A `question_failure` record is explicitly NOT harness contamination. It
    used to be written with `is_fixture: True`, so one transient server timeout
    permanently poisoned an append-only event log and forced
    SCIENTIFIC_RECOMMENDATION_NOT_AVAILABLE for that stage forever -- punishing
    the honest recording of a failure harder than crashing.
    """
    if rec.get("is_failure_record"):
        return False
    return bool(rec.get("is_fixture") or rec.get("dry_run"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def _parity(slots: Dict[str, Dict[str, str]], name: str,
            required: Sequence[str] = REQUIRED_PRECISIONS) -> GateResult:
    """All three precisions present for every slot, and all values equal."""
    if not slots:
        return GateResult(name, False, "no records to compare")
    incomplete = [k for k, m in slots.items()
                  if set(required) - set(m) or any(not v for v in m.values())]
    mismatched = [k for k, m in slots.items() if len(set(m.values())) > 1]
    if incomplete:
        return GateResult(name, False,
                          f"{len(incomplete)} slot(s) missing a precision or a value, "
                          f"e.g. {incomplete[:3]} -- parity cannot be claimed on "
                          "incomplete data")
    if mismatched:
        return GateResult(name, False, f"mismatch on {mismatched[:5]}")
    return GateResult(name, True, f"{len(slots)} slots complete and identical")


def evaluate_gates(events: Sequence[Dict[str, Any]],
                   predictions: Sequence[Dict[str, Any]],
                   provenance: Dict[str, Any],
                   required_questions: int,
                   condition: Optional[str] = None,
                   verify_model_files: bool = True) -> GateReport:
    """The hard scientific gates. Every one must pass before any outcome."""
    g: List[GateResult] = []
    prov = provenance or {}
    preds = [p for p in predictions
             if condition is None or p.get("condition") == condition]
    real_preds = [p for p in preds if not _is_harness_record(p)]

    # -- 1. every result came from real llama.cpp inference ------------------
    fixture_events = [e for e in events if _is_harness_record(e)]
    fixture_preds = [p for p in preds if _is_harness_record(p)]
    g.append(GateResult(
        "real_inference_only",
        not fixture_events and not fixture_preds,
        f"{len(fixture_events)} fixture/dry-run events, {len(fixture_preds)} "
        "fixture/dry-run predictions found" if (fixture_events or fixture_preds)
        else "no fixture or dry-run records present"))

    backends = {e.get("backend") for e in events
                if e.get("backend") and not _is_harness_record(e)}
    g.append(GateResult(
        "backend_is_llama_cpp",
        bool(backends) and backends <= {"llama-server", "llama-cli"},
        f"backends seen: {sorted(b for b in backends if b)}"))

    # -- 2. no generation errors were scored as model output -----------------
    errored = [e for e in events if e.get("generation_error")]
    g.append(GateResult(
        "no_generation_errors", not errored,
        f"{len(errored)} call(s) recorded a backend error; infrastructure "
        f"failure must not be scored as model quality "
        f"(first: {errored[0].get('generation_error')!r})" if errored
        else "no backend errors recorded"))

    # -- 3. all three precisions completed with nonzero generations ----------
    by_prec: Dict[str, List[Dict[str, Any]]] = {p: [] for p in REQUIRED_PRECISIONS}
    for p in real_preds:
        by_prec.setdefault(p.get("precision", "?"), []).append(p)
    missing = [pr for pr in REQUIRED_PRECISIONS if not by_prec.get(pr)]
    g.append(GateResult("all_three_precisions_present", not missing,
                        f"missing: {missing}" if missing
                        else "F16, Q8_0 and Q4_K_M all present"))

    zero_gen = []
    for pr in REQUIRED_PRECISIONS:
        tot = sum(int(e.get("generated_tokens") or 0) for e in events
                  if e.get("precision") == pr and not _is_harness_record(e))
        if tot <= 0:
            zero_gen.append(pr)
    g.append(GateResult("nonzero_generations_per_precision", not zero_gen,
                        f"zero generated tokens for: {zero_gen}" if zero_gen
                        else "every precision generated tokens"))

    # -- 4. model artifacts: recorded, distinct, and actually on disk --------
    models = prov.get("models") or {}
    hashes = {pr: (models.get(pr) or {}).get("sha256") for pr in REQUIRED_PRECISIONS}
    bad_hash = [pr for pr, h in hashes.items()
                if not h or h in ("FIXTURE", "", None) or len(str(h)) != 64]
    present = [h for h in hashes.values() if h]
    distinct = len(set(present)) == len(present)
    mismatched_files: List[str] = []
    if verify_model_files and not bad_hash:
        for pr in REQUIRED_PRECISIONS:
            path = (models.get(pr) or {}).get("path")
            if path and os.path.exists(path):
                actual = _sha256_file(path)
                if actual and actual != hashes[pr]:
                    mismatched_files.append(f"{pr} (recorded {hashes[pr][:12]}…, "
                                            f"on disk {actual[:12]}…)")
    ok_hashes = not bad_hash and distinct and not mismatched_files
    g.append(GateResult(
        "model_hashes_recorded", ok_hashes,
        f"bad or missing sha256 for {bad_hash}" if bad_hash
        else ("duplicate sha256 across precisions" if not distinct
              else (f"recorded hash does not match the file on disk: "
                    f"{mismatched_files}" if mismatched_files
                    else "three distinct sha256 recorded and verified on disk"))))

    # Each quantized variant must record the sha256 of the F16 GGUF it was
    # actually derived from, and it must be the F16 in use. The old gate was
    # `bool(provenance["common_source"])` -- any truthy value, including a note
    # written unconditionally whenever an F16 file happened to exist.
    common_src = prov.get("common_source") or {}
    # prepare_models.py records the authoritative F16 under common_source;
    # fall back to the per-model entry when only that is present.
    f16_sha = common_src.get("f16_gguf_sha256") or hashes.get("F16")
    derived_problems: List[str] = []
    if (common_src.get("f16_gguf_sha256") and hashes.get("F16")
            and common_src["f16_gguf_sha256"] != hashes["F16"]):
        derived_problems.append(
            f"common_source F16 {common_src['f16_gguf_sha256'][:12]}… does not "
            f"match the F16 model in use {hashes['F16'][:12]}…")
    for pr in ("Q8_0", "Q4_K_M"):
        parent = (models.get(pr) or {}).get("derived_from_f16_sha256")
        if not parent:
            derived_problems.append(f"{pr}: no derived_from_f16_sha256 recorded")
        elif f16_sha and parent != f16_sha:
            derived_problems.append(
                f"{pr}: derived from {parent[:12]}… but F16 in use is {f16_sha[:12]}…")
    if not f16_sha:
        derived_problems.append("no F16 sha256 to compare against")
    g.append(GateResult(
        "common_source_checkpoint", not derived_problems,
        "; ".join(derived_problems) if derived_problems
        else f"Q8_0 and Q4_K_M both derived from F16 {f16_sha[:12]}… "
             f"(source: {common_src.get('hf_repo', 'unrecorded')}@"
             f"{common_src.get('hf_revision', 'unrecorded')})"))

    # -- 5. retrieval was real ----------------------------------------------
    emb = prov.get("embedding") or {}
    is_real_embed = emb.get("embed_is_real_model")
    g.append(GateResult(
        "embedding_is_real_model", bool(is_real_embed),
        f"embedder: {emb.get('embed_model', 'unrecorded')}@"
        f"{emb.get('embed_revision', 'unrecorded')}" if is_real_embed
        else "retrieval ran on the hash-fallback embedder; evidence selection "
             "is not meaningful and no scientific claim may rest on it"))

    # -- 6. the three blocks differed ONLY in precision ----------------------
    # Both gates below are ALWAYS appended and both require all three
    # precisions to be represented. Building them only from the records that
    # happen to carry a value reproduces exactly the vacuous-pass bug that
    # `_parity` exists to prevent: drop `sampling_hash` from every Q4 event and
    # a "one sampling configuration across all precisions" pass falls out of
    # missing data. The offload gate was worse -- appended only `if off:`, so it
    # vanished from the report entirely (leaving `passed` True) on precisely the
    # CPU-only configuration where an offload asymmetry matters most.
    samp: Dict[str, set] = {}
    for e in events:
        # Only generation events carry a sampling config. A question_failure
        # or other diagnostic record legitimately has none, and counting it as
        # a blank would fail the gate for the honest recording of a failure.
        if _is_harness_record(e) or not e.get("precision"):
            continue
        if e.get("is_failure_record") or e.get("role") == "question_failure":
            continue
        samp.setdefault(e["precision"], set()).add(e.get("sampling_hash") or "")
    missing_samp = [pr for pr in REQUIRED_PRECISIONS if not samp.get(pr)]
    blank_samp = [pr for pr, hs in samp.items() if "" in hs]
    all_samp = {h for hs in samp.values() for h in hs}
    if missing_samp or blank_samp:
        g.append(GateResult(
            "sampling_identical_across_precisions", False,
            f"no sampling_hash recorded for {sorted(set(missing_samp) | set(blank_samp))}; "
            "sampling parity cannot be claimed on incomplete data"))
    else:
        g.append(GateResult(
            "sampling_identical_across_precisions", len(all_samp) == 1,
            f"sampling_hash values seen: {sorted(all_samp)}" if len(all_samp) != 1
            else f"one sampling configuration across all precisions "
                 f"({sorted(all_samp)[0][:12]}…)"))

    blocks = prov.get("blocks") or {}
    off = {pr: (blocks.get(pr) or {}).get("gpu_offload") for pr in REQUIRED_PRECISIONS}
    missing_off = [pr for pr, o in off.items() if not o]
    if missing_off:
        g.append(GateResult(
            "gpu_offload_identical_across_precisions", False,
            f"no GPU-offload record for {missing_off}; a block whose offload is "
            "unknown is not comparable to one whose offload is known, on quality "
            "OR throughput"))
    else:
        sig = {pr: (bool(o.get("offload_detected")), o.get("layers_offloaded"))
               for pr, o in off.items()}
        same = len(set(sig.values())) == 1
        g.append(GateResult(
            "gpu_offload_identical_across_precisions", same,
            f"offload differs per precision: {sig}; a block that fell back to "
            "partial CPU offload is not comparable on quality OR throughput"
            if not same else
            f"all precisions offloaded identically {next(iter(sig.values()))}"))

    # -- 7. parity ----------------------------------------------------------
    # The coordinator is produced ONCE at F16 and frozen, so there is exactly
    # one coordinator generation per question and a prompt-hash comparison
    # across precisions has nothing to compare -- checking it that way would
    # pass vacuously for the same reason the old parity gates did. What must
    # hold instead is that every precision consumed the SAME frozen record, and
    # that F16 produced it.
    coord_slots: Dict[str, Dict[str, str]] = {}
    produced_by: Dict[str, str] = {}
    for p in real_preds:
        c = p.get("coordinator") or {}
        payload = json.dumps(
            {"shared_tasks": c.get("shared_tasks") or [],
             "synthesis_directive": c.get("synthesis_directive") or ""},
            sort_keys=True)
        qid = str(p["question_id"])
        coord_slots.setdefault(qid, {})[p["precision"]] = _sha256_text(payload)
        src = (c.get("provenance") or {}).get("produced_by_precision")
        if src:
            produced_by[qid] = src
    g.append(_parity(coord_slots, "coordinator_frozen_and_shared"))
    if coord_slots:
        wrong_src = {q: s for q, s in produced_by.items() if s != FREEZE_PRECISION}
        unrecorded = sorted(set(coord_slots) - set(produced_by))
        g.append(GateResult(
            "coordinator_frozen_at_f16", not wrong_src and not unrecorded,
            (f"coordinator not produced at {FREEZE_PRECISION} for {wrong_src}"
             if wrong_src else
             f"no producing precision recorded for {unrecorded[:5]}")
            if (wrong_src or unrecorded)
            else f"all {len(produced_by)} coordinator records produced at "
                 f"{FREEZE_PRECISION} and reused unchanged"))

    if condition in (None, "fixed_verified_evidence"):
        chunk_slots: Dict[str, Dict[str, str]] = {}
        for p in real_preds:
            if p.get("condition") != "fixed_verified_evidence":
                continue
            chunk_slots.setdefault(str(p["question_id"]), {})[p["precision"]] = \
                "|".join(sorted(p.get("evidence_chunk_ids") or []))
        g.append(_parity(chunk_slots, "fixed_evidence_chunk_parity"))

        prompt_slots: Dict[str, Dict[str, str]] = {}
        for e in events:
            if _is_harness_record(e):
                continue
            if e.get("condition") != "fixed_verified_evidence":
                continue
            if e.get("role") != "document_agent":
                continue
            key = f"{e.get('question_id')}|{e.get('document_id')}"
            prompt_slots.setdefault(key, {})[e.get("precision", "?")] = \
                e.get("prompt_hash") or ""
        g.append(_parity(prompt_slots, "prompt_hash_parity"))

    # -- 8. the stage's required question count completed at every precision -
    short = {pr: len({p["question_id"] for p in by_prec.get(pr, [])})
             for pr in REQUIRED_PRECISIONS}
    incomplete = {pr: n for pr, n in short.items() if n < required_questions}
    g.append(GateResult("stage_question_count_complete", not incomplete,
                        f"need {required_questions} real questions per precision, "
                        f"have {short}" if incomplete
                        else f"{required_questions} questions complete at each precision"))
    return GateReport(gates=g)


def engineering_status(harness_ok: bool, real_run_attempted: bool,
                       real_run_ok: bool, detail: str = "") -> Dict[str, Any]:
    """Engineering status only. Callers must NOT feed a scientific gate result
    into `real_run_ok` -- a failed provenance gate says nothing about whether
    the harness executed correctly, and conflating them made a clean run report
    REAL_MODEL_RUN_BLOCKED.
    """
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
                       max_ci_width: Optional[float] = None) -> Dict[str, Any]:
    """Map the analysis summary onto one of the seven scientific outcomes.

    Returns NOT_AVAILABLE whenever a hard gate failed. Screening thresholds,
    not formal significance claims.

    `max_ci_width` defaults to `2 * ni_margin` rather than the old hardcoded
    0.30, which was six times the declared margin -- wide enough that an
    interval of [-0.28, +0.02] counted as "narrow enough" for an equivalence
    claim.
    """
    if not gate_report.passed:
        return {"scientific_outcome": NOT_AVAILABLE,
                "reason": "hard scientific gate(s) failed",
                "failed_gates": gate_report.failures()}

    if max_ci_width is None:
        max_ci_width = 2.0 * ni_margin

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

    ci = summary.get("f16_q4_gap_ci") or {}
    lo, hi = ci.get("lo"), ci.get("hi")
    ci_width = (float(hi) - float(lo)) if (lo is not None and hi is not None) \
        else float("inf")
    n_informative = int(ci.get("n_informative") or 0)
    q8_gap = summary.get("f16_q8_gap_overall")
    q4_gap = summary.get("f16_q4_gap_overall")

    # -- compounding: the headline claim, and it must carry an interval ------
    # The old code compared two point estimates and returned GO on a bare
    # threshold, ignoring the CI that was computed alongside them. At 4 paired
    # questions per width cell that is pure noise amplification.
    growth_ci = summary.get("f16_q4_gap_growth_ci") or {}
    growth = growth_ci.get("mean")
    growth_lo = growth_ci.get("lo")
    cells = summary.get("f16_q4_gap_by_width_n") or {}
    thin_cells = {w: n for w, n in cells.items() if int(n) < MIN_PAIRS_PER_WIDTH}

    if growth is not None and growth >= min_gap_growth:
        if thin_cells:
            return {"scientific_outcome": "WEAK_GO_EXPAND",
                    "reason": (f"gap growth {growth:+.3f} >= {min_gap_growth} but "
                               f"width cell(s) {thin_cells} have fewer than "
                               f"{MIN_PAIRS_PER_WIDTH} paired questions"),
                    "gap_growth": round(growth, 4)}
        if growth_lo is None or growth_lo <= 0:
            return {"scientific_outcome": "WEAK_GO_EXPAND",
                    "reason": (f"gap growth {growth:+.3f} >= {min_gap_growth} but its "
                               f"95% CI [{growth_lo}, {growth_ci.get('hi')}] "
                               "includes zero"),
                    "gap_growth": round(growth, 4)}
        return {"scientific_outcome": "GO_COMPOUNDING_DEGRADATION",
                "reason": (f"F16-Q4 gap grows {growth:+.3f} with document count, "
                           f"95% CI [{growth_lo:+.3f}, {growth_ci.get('hi'):+.3f}] "
                           "excludes zero"),
                "gap_growth": round(growth, 4)}

    if (q8_gap is not None and q4_gap is not None
            and abs(q8_gap) <= ni_margin and q4_gap > ni_margin):
        return {"scientific_outcome": "GO_Q8_OPERATING_POINT",
                "reason": (f"Q8 within margin ({q8_gap:+.3f}) while Q4 degrades "
                           f"({q4_gap:+.3f} > {ni_margin})")}

    # -- non-inferiority: needs informative pairs, not just a narrow interval
    if (q4_gap is not None and ci and hi is not None
            and float(hi) <= ni_margin
            and ci_width <= max_ci_width
            and n_informative >= MIN_INFORMATIVE_PAIRS
            and f16 >= 0.70):
        return {"scientific_outcome": "GO_DECOMPOSITION_ROBUSTNESS",
                "reason": (f"Q4 non-inferior to F16: CI upper bound {float(hi):+.3f} "
                           f"<= margin {ni_margin}, CI width {ci_width:.3f} <= "
                           f"{max_ci_width}, {n_informative} informative pairs, "
                           f"F16 headroom {f16:.3f}")}

    if (q4_gap is not None and ci and hi is not None and float(hi) <= ni_margin
            and n_informative < MIN_INFORMATIVE_PAIRS):
        return {"scientific_outcome": "WEAK_GO_EXPAND",
                "reason": (f"interval is inside the margin but only {n_informative} "
                           f"of the paired differences are non-zero; an equivalence "
                           f"claim needs at least {MIN_INFORMATIVE_PAIRS}")}

    if (growth is not None and growth > 0) or (q4_gap is not None and q4_gap > 0):
        return {"scientific_outcome": "WEAK_GO_EXPAND",
                "reason": ("directional pattern present but the interval is too wide "
                           f"to call (CI width {ci_width:.3f})")}

    return {"scientific_outcome": "NO_GO_NO_STRUCTURED_EFFECT",
            "reason": "differences show no structure by width or precision"}


# Half-open [min, max) bands, exhaustive over [0, 1], no gaps and no overlaps --
# mirroring configs/stage_a.yaml `decision_gate.bands`, which is the source of
# truth. The old table used `<=` boundaries and had an extra
# "CALIBRATION PASS (upper band)" row that README omitted entirely, so 0.70 fell
# in MARGINAL while `GO_DECOMPOSITION_ROBUSTNESS` simultaneously required
# `f16 >= 0.70` -- the harness could emit a GO and "MARGINAL, review your
# prompts" for the same number.
CALIBRATION_BANDS = (
    (0.50, "BLOCKED — MODEL/BENCHMARK FLOOR"),
    (0.70, "MARGINAL — REVIEW PROMPTS AND FAILURES"),
    (0.95, "CALIBRATION PASS"),
)
CALIBRATION_TOP = "POSSIBLE CEILING — ADD HARDER ITEMS OR DISTRACTORS"


def calibration_bands_from_config(stage_cfg: Optional[Dict[str, Any]] = None):
    """Bands from `configs/stage_a.yaml` when available, else the constants."""
    bands = ((stage_cfg or {}).get("decision_gate") or {}).get("bands")
    if not bands:
        return CALIBRATION_BANDS, CALIBRATION_TOP
    ordered = sorted(bands, key=lambda b: float(b["min"]))
    top = ordered[-1]["status"]
    return tuple((float(b["max"]), b["status"]) for b in ordered[:-1]), top


def calibration_status(f16_atomic_fact_recall: Optional[float],
                       stage_cfg: Optional[Dict[str, Any]] = None) -> str:
    if f16_atomic_fact_recall is None:
        return "UNAVAILABLE — no real F16 results"
    bands, top = calibration_bands_from_config(stage_cfg)
    r = float(f16_atomic_fact_recall)
    for upper, label in bands:
        if r < upper:
            return label
    return top
