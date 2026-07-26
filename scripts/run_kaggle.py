#!/usr/bin/env python3
"""Non-notebook Kaggle entry point: preflight -> models -> indexes -> stage ->
score -> analyze -> package, with hard early stopping between stages.

    python scripts/run_kaggle.py --stage stage_a
    python scripts/run_kaggle.py --stage stage_b            # blocked unless A passed
    python scripts/run_kaggle.py --stage stage_b --force    # documented override

Stops after 18 predictions (Stage A) or 54 (A+B) unless the gate says proceed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import kaggle_paths, results_dir  # noqa: E402
from logging_utils import read_json  # noqa: E402


def sh(cmd: list, check: bool = True) -> int:
    print("\n$", " ".join(str(c) for c in cmd), flush=True)
    rc = subprocess.run([str(c) for c in cmd], cwd=str(ROOT)).returncode
    if rc != 0 and check:
        raise SystemExit(f"step failed with exit code {rc}: {' '.join(map(str, cmd))}")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="stage_a",
                    choices=["stage_a", "stage_b", "stage_c"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--skip-build", action="store_true",
                    help="llama.cpp already built in this session")
    ap.add_argument("--skip-download", action="store_true",
                    help="HF checkpoint provided as a Kaggle input dataset")
    ap.add_argument("--hf-dir", default=None)
    ap.add_argument("--backend", default="llama-server")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ni-margin", type=float, default=0.05)
    args = ap.parse_args()

    py = sys.executable
    rd = results_dir(args.results_dir)
    rdarg = ["--results-dir", str(rd)]
    print(f"[kaggle] paths: {kaggle_paths()}")
    print(f"[kaggle] results -> {rd}")

    sh([py, "scripts/prepare_kaggle.py", *rdarg,
        *([] if not args.dry_run else ["--skip-preflight", "--allow-cpu"])])

    if not args.dry_run:
        model_cmd = [py, "scripts/prepare_models.py", *rdarg]
        if not args.skip_build:
            model_cmd.append("--build-llama-cpp")
        if args.skip_download:
            model_cmd.append("--skip-download")
        if args.hf_dir:
            model_cmd += ["--hf-dir", args.hf_dir]
        sh(model_cmd)

    if not (ROOT / "data" / "pilot_manifest.json").exists():
        sh([py, "scripts/prepare_dataset.py", "--data-dir", args.data_dir])

    index_stage = "stage_b" if args.stage in ("stage_b", "stage_c") else "stage_a"
    sh([py, "scripts/build_indexes.py", "--stage", index_stage, *rdarg])

    run_cmd = [py, "scripts/run_stage.py", "--stage", args.stage,
               "--backend", args.backend, *rdarg]
    if args.force:
        run_cmd.append("--force")
    if args.dry_run:
        run_cmd.append("--dry-run")
    sh(run_cmd)

    sh([py, "scripts/score.py", "--stage", args.stage, *rdarg, "--emit-adjudication"])
    sh([py, "scripts/analyze.py", "--stage", args.stage, *rdarg,
        "--ni-margin", str(args.ni_margin)])
    sh([py, "scripts/package_results.py", "--stage", args.stage, *rdarg])

    gate = read_json(rd / args.stage / "gate.json", default={}) or {}
    print(f"\n[kaggle] {args.stage} gate: {gate}")
    if not gate.get("proceed"):
        print(f"[kaggle] STOPPING after {args.stage}. The next stage is blocked until "
              "this gate passes or a documented --force override is used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
