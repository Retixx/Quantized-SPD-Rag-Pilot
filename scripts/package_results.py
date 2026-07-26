#!/usr/bin/env python3
"""Zip a run's artefacts and write the honest RUN_REPORT.md.

The report states plainly whether only the harness was exercised or real
llama.cpp inference completed at all three precisions.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import results_dir  # noqa: E402
from gates import NOT_AVAILABLE  # noqa: E402
from logging_utils import read_json  # noqa: E402


def render(stage: str, rd: Path) -> str:
    sd = rd / stage
    analysis = read_json(sd / "analysis.json", default={}) or {}
    summary = read_json(sd / "score_summary.json", default={}) or {}
    run = read_json(sd / "run_report.json", default={}) or {}
    prov = read_json(rd / "provenance.json", default={}) or {}
    gates = analysis.get("gates", {})

    real_n = analysis.get("n_real_predictions", 0)
    fake_n = analysis.get("n_excluded_fixture_or_dry_run", 0)
    kind = run.get("backend_kind", "unknown")
    dry = run.get("dry_run")
    if dry or kind in ("fixture", "mock") or real_n == 0:
        headline = ("ONLY THE HARNESS WAS TESTED. No real model inference entered "
                    "the analysis, so no scientific recommendation exists.")
    elif gates.get("all_gates_passed"):
        headline = (f"REAL MODEL INFERENCE COMPLETED. {real_n} real predictions passed "
                    "every hard scientific gate.")
    else:
        headline = (f"REAL MODEL INFERENCE RAN ({real_n} predictions) BUT AT LEAST ONE "
                    "HARD GATE FAILED. No scientific recommendation is issued.")

    lines: List[str] = [
        f"# Run report — {stage}", "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}", "",
        "## What kind of run was this?", "", headline, "",
        f"- backend: `{kind}`", f"- dry run: `{bool(dry)}`",
        f"- real predictions in analysis: **{real_n}**",
        f"- fixture / dry-run rows excluded: **{fake_n}**",
        f"- question failures recorded: **{len(run.get('failures') or [])}**", "",
        "## Status", "",
        f"- engineering status: **{analysis.get('engineering_status', 'UNKNOWN')}**",
        f"- scientific outcome: **{analysis.get('scientific_outcome', NOT_AVAILABLE)}**",
        f"- calibration status: **{analysis.get('calibration_status', 'UNAVAILABLE')}**",
        f"- reason: {analysis.get('reason', '')}", "",
        "## Hard scientific gates", "",
        "| gate | passed | detail |", "| --- | --- | --- |",
    ]
    for g in gates.get("gates", []):
        lines.append(f"| `{g['gate']}` | {'PASS' if g['passed'] else 'FAIL'} | "
                     f"{str(g['detail'])[:160]} |")
    lines += ["", "## Results by precision", ""]
    bp = summary.get("by_precision") or {}
    if bp:
        cols = ["atomic_fact_recall", "answer_score", "per_agent_recall",
                "synthesis_loss", "branch_failure_probability",
                "required_document_coverage", "mean_wall_s"]
        lines.append("| precision | n | " + " | ".join(cols) + " |")
        lines.append("| --- | --- | " + " | ".join("---" for _ in cols) + " |")
        for prec in ("F16", "Q8_0", "Q4_K_M"):
            row = bp.get(prec)
            if not row:
                continue
            lines.append(f"| {prec} | {row.get('n')} | " +
                         " | ".join(str(row.get(c)) for c in cols) + " |")
    else:
        lines.append("_no scored results_")

    stats = analysis.get("primary_metric_analysis") or {}
    if stats:
        lines += ["", "## Primary estimand: does the F16–Q4 gap grow with width?", "",
                  f"- metric: `{stats.get('metric')}`",
                  f"- overall F16−Q4 gap: {stats.get('f16_q4_gap_overall')} "
                  f"CI {stats.get('f16_q4_gap_ci', {}).get('lo')} to "
                  f"{stats.get('f16_q4_gap_ci', {}).get('hi')}",
                  f"- overall F16−Q8 gap: {stats.get('f16_q8_gap_overall')} "
                  f"CI {stats.get('f16_q8_gap_ci', {}).get('lo')} to "
                  f"{stats.get('f16_q8_gap_ci', {}).get('hi')}",
                  f"- gap by width (F16−Q4): {stats.get('f16_q4_gap_by_width')}",
                  f"- Q4 verdict: **{(stats.get('q4_verdict') or {}).get('label')}** "
                  f"— {(stats.get('q4_verdict') or {}).get('reason')}",
                  f"- Q8 verdict: **{(stats.get('q8_verdict') or {}).get('label')}** "
                  f"— {(stats.get('q8_verdict') or {}).get('reason')}"]

    lines += ["", "## Model provenance", ""]
    for prec, m in sorted((prov.get("models") or {}).items()):
        lines.append(f"- **{prec}**: `{m.get('sha256', 'n/a')}` "
                     f"({m.get('size_gb', '?')} GB) from `{m.get('derived_from', '?')}`")
    cs = prov.get("common_source")
    lines.append(f"- common source: `{json.dumps(cs)}`" if cs
                 else "- common source: **not recorded**")
    lc = prov.get("llama_cpp") or {}
    lines.append(f"- llama.cpp commit: `{lc.get('resolved_commit', 'unrecorded')}` "
                 f"(CUDA={lc.get('cuda')})")

    sysd = analysis.get("systems") or {}
    if sysd:
        lines += ["", "## Systems", "",
                  "| precision | size GB | load s | peak VRAM MiB | block wall s | "
                  "quality/s | quality/GB | GPU offload |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for prec, s in sysd.items():
            off = (s.get("gpu_offload") or {}).get("offload_detected")
            lines.append(f"| {prec} | {s.get('model_size_gb')} | "
                         f"{s.get('model_load_time_s')} | {s.get('peak_gpu_used_mib')} | "
                         f"{s.get('block_wall_s_serial')} | {s.get('quality_per_second')} | "
                         f"{s.get('quality_per_gb')} | {off} |")
        lines.append("")
        lines.append("Runtime is actual serial wall time. Document-agent calls are "
                     "executed sequentially against one model server, so any parallel "
                     "figure would be an idealised lower bound and is not reported here.")

    lines += ["", "## Caveats", ""]
    for c in analysis.get("caveats", []):
        lines.append(f"- {c}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=["stage_a", "stage_b", "stage_c"])
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rd = results_dir(args.results_dir)
    sd = rd / args.stage
    md = render(args.stage, rd)
    (sd / "RUN_REPORT.md").write_text(md, encoding="utf-8")

    out = Path(args.out) if args.out else rd / f"{args.stage}_results.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(sd.rglob("*")):
            if p.is_file() and p.suffix != ".zip":
                z.write(p, arcname=str(Path(args.stage) / p.relative_to(sd)))
        for extra in ("provenance.json", "index_provenance.json",
                      "isolation_report.json", "kaggle_preflight.json"):
            f = rd / extra
            if f.exists():
                z.write(f, arcname=extra)
        for f in sorted((rd / "server_logs").glob("*.log")) if (rd / "server_logs").exists() else []:
            z.write(f, arcname=f"server_logs/{f.name}")
        for f in ("data/pilot_manifest.json", "data/gold_facts.json"):
            p = ROOT / f
            if p.exists():
                z.write(p, arcname=f)
    print(md)
    print(f"[package] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
