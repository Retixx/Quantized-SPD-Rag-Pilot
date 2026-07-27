#!/usr/bin/env python3
"""Kaggle preflight: verify CUDA, discover /kaggle/input, prove GPU offload.

Run this first in the notebook. It fails loudly rather than letting a CPU-only
run masquerade as a GPU pilot.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import kaggle_paths, load_models_config, resolve, results_dir  # noqa: E402
from hardware import cuda_report, require_cuda  # noqa: E402
from hashcache import sha256_file_cached  # noqa: E402
from logging_utils import read_json, write_json  # noqa: E402


def discover_inputs() -> dict:
    kp = kaggle_paths()
    found = {"multihop_questions": None, "multihop_corpus": None,
             "hf_checkpoint": None, "gguf_dir": None, "llama_cpp": None}
    root = Path(kp["input_root"]) if kp["input_root"] else None
    if root and root.exists():
        for p in root.rglob("MultiHopRAG.json"):
            found["multihop_questions"] = str(p)
            break
        for p in root.rglob("corpus.json"):
            found["multihop_corpus"] = str(p)
            break
        for p in root.rglob("config.json"):
            if (p.parent / "tokenizer_config.json").exists() and \
                    any(p.parent.glob("*.safetensors")):
                found["hf_checkpoint"] = str(p.parent)
                break
        for p in root.rglob("*.gguf"):
            found["gguf_dir"] = str(p.parent)
            break
        for p in root.rglob("convert_hf_to_gguf.py"):
            found["llama_cpp"] = str(p.parent)
            break
    return {**kp, "discovered": found}


def preflight_offload(models_cfg: dict, precision: str, timeout: int) -> dict:
    """Load one GGUF through llama-server and confirm layers went to the GPU."""
    sys.path.insert(0, str(ROOT / "src"))
    from backends.llamacpp_backend import (LlamaServerBackend, ModelSpec,  # noqa: E402
                                           SamplingConfig)
    spec_cfg = models_cfg["models"][precision]
    path = resolve(spec_cfg["path"])
    if not path.exists():
        return {"ran": False, "reason": f"{path} not present"}
    binary = str(resolve(models_cfg["llama_cpp"]["server_binary"]))
    if shutil.which(binary) is None and not Path(binary).exists():
        return {"ran": False, "reason": f"llama-server not built at {binary}"}
    s = models_cfg["sampling"]
    budgets = s.get("max_tokens") or {}
    probe_tokens = int(budgets.get("preflight", 32)) if isinstance(budgets, dict) \
        else int(budgets)
    sampling = SamplingConfig(temperature=float(s["temperature"]), top_p=float(s["top_p"]),
                              top_k=int(s["top_k"]), min_p=float(s.get("min_p", 0.0)),
                              repeat_penalty=float(s.get("repeat_penalty", 1.0)),
                              seed=int(s["seed"]), max_tokens=probe_tokens,
                              n_ctx=int(s["n_ctx"]))
    # Use the shared hash cache: this GGUF is also hashed by prepare_models.py
    # and again per precision block in run_stage.py. ModelSpec.from_path() would
    # force a fresh full read of up to 6 GB every time.
    model = ModelSpec(precision=precision, path=str(path),
                      sha256=sha256_file_cached(path),
                      size_bytes=path.stat().st_size)
    backend = LlamaServerBackend(
        model=model, sampling=sampling, binary=binary,
        n_gpu_layers=int(models_cfg["backend"].get("n_gpu_layers", 99)),
        log_dir=str(results_dir() / "server_logs"), startup_timeout_s=timeout)
    report: dict = {"ran": False, "reason": "did not complete"}
    try:
        backend.start()
        gen = backend.chat([{"role": "user", "content": "Reply with the single word: ready"}],
                           max_tokens=min(16, probe_tokens))
        report = {"ran": True, "precision": precision,
                  "gpu_offload": backend.offload,
                  "load_time_s": round(backend.load_time_s, 2),
                  "generation_ok": bool(gen.text.strip()) and gen.error is None,
                  "generated_tokens": gen.generated_tokens,
                  "sample_text": gen.text.strip()[:200], "error": gen.error,
                  "model_sha256": model.sha256, "probe_max_tokens": probe_tokens}
    except Exception as exc:
        report = {"ran": False, "reason": f"{type(exc).__name__}: {exc}"}
    finally:
        backend.stop()
    # Peak-memory sampling is finalised inside stop(); reading backend.memory
    # from within the try block always returned {}. Read it AFTER stop().
    report["memory"] = backend.memory
    prov = backend.provenance() if hasattr(backend, "provenance") else {}
    report["backend_provenance"] = prov
    report.setdefault("gpu_offload", prov.get("gpu_offload"))
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models-config", default="configs/models.yaml")
    ap.add_argument("--require-cuda", action="store_true", default=True)
    ap.add_argument("--allow-cpu", dest="require_cuda", action="store_false")
    ap.add_argument("--preflight-precision", default="Q4_K_M")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()

    rd = results_dir(args.results_dir)
    report = {"cuda": require_cuda() if args.require_cuda else cuda_report(),
              "paths": discover_inputs()}
    models_cfg = load_models_config(args.models_config)
    if not args.skip_preflight:
        report["preflight"] = preflight_offload(models_cfg, args.preflight_precision,
                                                args.timeout)
        off = (report["preflight"].get("gpu_offload") or {})
        if report["preflight"].get("ran") and not off.get("offload_detected"):
            print("[preflight] WARNING: no GPU offload detected in the server log. "
                  "Rebuild llama.cpp with -DGGML_CUDA=ON.")
    else:
        report["preflight"] = {"ran": False, "reason": "skipped"}

    existing = read_json(rd / "provenance.json", default={}) or {}
    existing["kaggle_preflight"] = report
    write_json(rd / "provenance.json", existing)
    write_json(rd / "kaggle_preflight.json", report)
    print(json.dumps(report, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
