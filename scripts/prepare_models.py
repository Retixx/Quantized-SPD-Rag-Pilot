#!/usr/bin/env python3
"""One checkpoint -> one F16 GGUF -> Q8_0 and Q4_K_M, with full provenance.

Chain (never deviate for the real pilot):
  1. download Qwen/Qwen2.5-3B-Instruct
  2. convert_hf_to_gguf.py  --outtype f16   -> f16.gguf
  3. llama-quantize f16.gguf q8_0.gguf Q8_0
  4. llama-quantize f16.gguf q4_k_m.gguf Q4_K_M

Q8_0 and Q4_K_M are derived from THAT EXACT f16.gguf, whose sha256 is recorded,
so all three variants provably share one source. Official Qwen GGUFs are fine
for a smoke test but fail the `common_source_checkpoint` gate.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_models_config, resolve, results_dir  # noqa: E402
from logging_utils import read_json, sha256_file, write_json  # noqa: E402


def run(cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None,
        check: bool = True) -> Dict[str, Any]:
    print("[cmd]", " ".join(str(c) for c in cmd))
    t0 = time.time()
    proc = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None,
                          env={**os.environ, **(env or {})},
                          capture_output=True, text=True)
    rec = {"command": [str(c) for c in cmd], "cwd": str(cwd or ""),
           "returncode": proc.returncode, "seconds": round(time.time() - t0, 2),
           "stdout_tail": (proc.stdout or "")[-4000:],
           "stderr_tail": (proc.stderr or "")[-4000:]}
    if check and proc.returncode != 0:
        print(rec["stderr_tail"])
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(map(str, cmd))}")
    return rec


def git_commit(path: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(path),
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def build_llama_cpp(cfg: Dict[str, Any], cuda: bool, jobs: int,
                    log: List[Dict[str, Any]]) -> Path:
    lc = cfg["llama_cpp"]
    build_dir = resolve(lc["build_dir"])
    if not (build_dir / "CMakeLists.txt").exists():
        build_dir.parent.mkdir(parents=True, exist_ok=True)
        log.append(run(["git", "clone", "--depth", "1", lc["repo"], str(build_dir)]))
    commit = lc.get("commit") or "master"
    if commit and commit != "master":
        log.append(run(["git", "fetch", "--depth", "1", "origin", commit], cwd=build_dir))
        log.append(run(["git", "checkout", commit], cwd=build_dir))
    flags = str(lc.get("cmake_flags") or "").split()
    if not cuda:
        flags = [f for f in flags if "GGML_CUDA" not in f]
    log.append(run(["cmake", "-B", "build", *flags], cwd=build_dir))
    log.append(run(["cmake", "--build", "build", "--config", "Release",
                    "-j", str(jobs), "--target", "llama-server", "llama-cli",
                    "llama-quantize"], cwd=build_dir))
    return build_dir


def download_checkpoint(cfg: Dict[str, Any], log: List[Dict[str, Any]]) -> Path:
    src = cfg["source"]
    local = resolve(src["local_dir"])
    if local.exists() and any(local.glob("*.safetensors")):
        print(f"[models] reusing checkpoint at {local}")
        return local
    local.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download  # noqa: WPS433
    t0 = time.time()
    path = snapshot_download(repo_id=src["hf_repo"], revision=src.get("hf_revision") or None,
                             local_dir=str(local),
                             allow_patterns=["*.json", "*.safetensors", "*.txt",
                                             "*.model", "*.py"])
    log.append({"command": ["snapshot_download", src["hf_repo"],
                            src.get("hf_revision") or "main"],
                "returncode": 0, "seconds": round(time.time() - t0, 2),
                "stdout_tail": str(path), "stderr_tail": ""})
    return Path(path)


def resolved_revision(local: Path) -> str:
    ref = local / ".cache" / "huggingface" / "download"
    for p in (local / "refs", ref):
        if p.exists():
            for f in p.rglob("*"):
                if f.is_file():
                    try:
                        txt = f.read_text().strip()
                        if len(txt) == 40:
                            return txt
                    except Exception:
                        pass
    return "unrecorded"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models-config", default="configs/models.yaml")
    ap.add_argument("--build-llama-cpp", action="store_true")
    ap.add_argument("--no-cuda", action="store_true")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--skip-download", action="store_true",
                    help="checkpoint already on disk (e.g. a Kaggle input dataset)")
    ap.add_argument("--hf-dir", default=None,
                    help="override the local HF checkpoint directory")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--only-hash", action="store_true",
                    help="just re-hash existing GGUFs and rewrite provenance")
    args = ap.parse_args()

    cfg = load_models_config(args.models_config)
    rd = results_dir(args.results_dir)
    log: List[Dict[str, Any]] = []
    prov: Dict[str, Any] = {"created_at": time.time(), "models": {}, "commands": log}

    lc = cfg["llama_cpp"]
    if args.build_llama_cpp and not args.only_hash:
        build_llama_cpp(cfg, cuda=not args.no_cuda, jobs=args.jobs, log=log)
    build_dir = resolve(lc["build_dir"])
    prov["llama_cpp"] = {"repo": lc["repo"], "configured_commit": lc.get("commit"),
                         "resolved_commit": git_commit(build_dir),
                         "cmake_flags": lc.get("cmake_flags"),
                         "cuda": not args.no_cuda}

    gguf_paths = {k: resolve(v["path"]) for k, v in cfg["models"].items()}
    for p in gguf_paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)

    if not args.only_hash:
        hf_dir = Path(args.hf_dir) if args.hf_dir else None
        if hf_dir is None:
            hf_dir = (resolve(cfg["source"]["local_dir"]) if args.skip_download
                      else download_checkpoint(cfg, log))
        prov["source"] = {"hf_repo": cfg["source"]["hf_repo"],
                          "configured_revision": cfg["source"].get("hf_revision"),
                          "resolved_revision": resolved_revision(hf_dir),
                          "local_dir": str(hf_dir)}

        f16 = gguf_paths["F16"]
        if not f16.exists():
            convert = resolve(lc["convert_script"])
            if not convert.exists():
                raise SystemExit(f"convert script not found: {convert} "
                                 "(run with --build-llama-cpp first)")
            log.append(run([sys.executable, str(convert), str(hf_dir),
                            "--outfile", str(f16), "--outtype", "f16"]))
        else:
            print(f"[models] reusing {f16}")

        quantize = resolve(lc["quantize_binary"])
        for name in ("Q8_0", "Q4_K_M"):
            spec = cfg["models"][name]
            if spec.get("derived_from") != "F16":
                raise SystemExit(f"{name} must be derived_from: F16")
            out = gguf_paths[name]
            if out.exists():
                print(f"[models] reusing {out}")
                continue
            if not quantize.exists():
                raise SystemExit(f"llama-quantize not found: {quantize}")
            log.append(run([str(quantize), str(f16), str(out), spec["quantize_type"]]))

    f16_sha = sha256_file(gguf_paths["F16"]) if gguf_paths["F16"].exists() else ""
    for name, path in gguf_paths.items():
        if not path.exists():
            prov["models"][name] = {"path": str(path), "present": False}
            continue
        prov["models"][name] = {
            "path": str(path), "present": True, "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "size_gb": round(path.stat().st_size / (1024 ** 3), 4),
            "quantize_type": cfg["models"][name]["quantize_type"],
            "derived_from": cfg["models"][name]["derived_from"],
        }
    prov["common_source"] = {
        "f16_gguf_sha256": f16_sha,
        "hf_repo": cfg["source"]["hf_repo"],
        "note": "Q8_0 and Q4_K_M were quantized from this exact F16 GGUF",
    } if f16_sha else None
    prov["sampling"] = cfg["sampling"]

    existing = read_json(rd / "provenance.json", default={}) or {}
    existing.update(prov)
    write_json(rd / "provenance.json", existing)
    print(json.dumps({k: v for k, v in prov["models"].items()}, indent=2))
    print(f"[models] provenance -> {rd / 'provenance.json'}")
    missing = [k for k, v in prov["models"].items() if not v.get("present")]
    if missing:
        print(f"[models] WARNING missing GGUFs: {missing}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
