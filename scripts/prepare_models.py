#!/usr/bin/env python3
"""One checkpoint -> one F16 GGUF -> Q8_0 and Q4_K_M, with full provenance.

Chain (never deviate for the real pilot):
  1. download Qwen/Qwen2.5-3B-Instruct
  2. convert_hf_to_gguf.py  --outtype f16   -> f16.gguf
  3. llama-quantize f16.gguf q8_0.gguf Q8_0
  4. llama-imatrix -m f16.gguf -f <held-out calibration text> -o f16.imatrix
  5. llama-quantize --imatrix f16.imatrix f16.gguf q4_k_m.gguf Q4_K_M

Q8_0 and Q4_K_M are derived from THAT EXACT f16.gguf. The sha256 of the F16
each one was derived from is recorded PER VARIANT, at the moment it is derived,
as `derived_from_f16_sha256` -- so "they share one source" is a checkable fact
rather than an assertion in a note. A pre-existing quantized file whose recorded
parent does not match the current F16 is re-quantized (or, with
--no-requantize-stale, fails the run).

Q4_K_M is built WITH an importance matrix. A 4-bit k-quant without one is a
materially worse model than the same recipe with one, and comparing an
imatrix-less Q4_K_M against F16 measures the missing imatrix as much as it
measures quantization. The imatrix is calibrated on a deterministic HELD-OUT
slice of the corpus -- never on the evaluation questions or their supporting
documents. Use --no-imatrix to reproduce the old, weaker build.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_models_config, resolve, results_dir  # noqa: E402
from hashcache import invalidate as invalidate_hash  # noqa: E402
from hashcache import sha256_file_cached  # noqa: E402
from logging_utils import read_json, sha256_text, write_json  # noqa: E402

GB = 1024 ** 3

# Rough peak footprint of a from-scratch build, used only for the pre-flight
# disk check. Kaggle's /kaggle/working is capped at 20 GB.
DISK_ESTIMATE_GB = {
    "hf_safetensors": 6.2,
    "f16_gguf": 6.2,
    "q8_0_gguf": 3.3,
    "q4_k_m_gguf": 1.9,
    "llama_cpp_build_tree": 4.0,
    "imatrix_and_calibration_text": 0.1,
}


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


# -- disk ------------------------------------------------------------------


def nearest_existing(path: Path) -> Path:
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def disk_check(target: Path, min_free_gb: float, allow_low: bool,
               freeing_hf: bool) -> Dict[str, Any]:
    """Refuse to start a build that cannot fit. /kaggle/working is 20 GB and the
    weights alone are ~17.6 GB before the CUDA build tree."""
    usage = shutil.disk_usage(str(nearest_existing(target)))
    estimate = dict(DISK_ESTIMATE_GB)
    if freeing_hf:
        # the safetensors are deleted right after conversion, so they overlap
        # with the F16 only briefly and never with Q8_0/Q4_K_M
        estimate["hf_safetensors_note"] = ("freed after conversion "
                                           "(--free-hf-weights)")
        peak = (DISK_ESTIMATE_GB["hf_safetensors"] + DISK_ESTIMATE_GB["f16_gguf"]
                + DISK_ESTIMATE_GB["llama_cpp_build_tree"])
    else:
        peak = sum(v for k, v in DISK_ESTIMATE_GB.items())
    report = {
        "path": str(target),
        "free_gb": round(usage.free / GB, 2),
        "total_gb": round(usage.total / GB, 2),
        "estimated_peak_gb": round(peak, 2),
        "estimate_breakdown_gb": estimate,
        "min_free_gb": min_free_gb,
        "hf_weights_freed_after_conversion": freeing_hf,
    }
    print(f"[disk] {report['free_gb']} GB free of {report['total_gb']} GB at "
          f"{target} | estimated peak need {report['estimated_peak_gb']} GB "
          f"| threshold {min_free_gb} GB")
    if report["free_gb"] < min_free_gb:
        report["sufficient"] = False
        msg = (f"only {report['free_gb']} GB free at {target}, below "
               f"--min-free-gb {min_free_gb}. Estimated peak need is "
               f"{report['estimated_peak_gb']} GB "
               f"({json.dumps(DISK_ESTIMATE_GB)}). Free space, pass "
               "--free-hf-weights, or override with --allow-low-disk.")
        if not allow_low:
            raise SystemExit(f"[disk] {msg}")
        print(f"[disk] WARNING (--allow-low-disk): {msg}")
    else:
        report["sufficient"] = True
    return report


def free_hf_weights(hf_dir: Path) -> Dict[str, Any]:
    """Delete the HF safetensors once the F16 GGUF exists. ~6.2 GB back."""
    removed, freed = [], 0
    for p in sorted(hf_dir.rglob("*.safetensors")):
        try:
            freed += p.stat().st_size
            p.unlink()
            removed.append(str(p))
        except Exception as exc:
            print(f"[disk] could not remove {p}: {exc}")
    print(f"[disk] freed {round(freed / GB, 2)} GB of HF safetensors "
          f"({len(removed)} files)")
    return {"removed_files": removed, "freed_gb": round(freed / GB, 4),
            "note": ("HF safetensors deleted after the F16 GGUF was produced; "
                     "re-download to redo the conversion")}


# -- llama.cpp -------------------------------------------------------------


def build_llama_cpp(cfg: Dict[str, Any], cuda: bool, jobs: int,
                    log: List[Dict[str, Any]], with_imatrix: bool) -> Path:
    lc = cfg["llama_cpp"]
    build_dir = resolve(lc["build_dir"])
    if not (build_dir / "CMakeLists.txt").exists():
        build_dir.parent.mkdir(parents=True, exist_ok=True)
        log.append(run(["git", "clone", "--depth", "1", lc["repo"], str(build_dir)]))
    commit = lc.get("commit") or "master"
    if commit and commit != "master":
        log.append(run(["git", "fetch", "--depth", "1", "origin", commit], cwd=build_dir))
        log.append(run(["git", "checkout", commit], cwd=build_dir))
    else:
        print("[models] WARNING: llama_cpp.commit is unpinned ('master'). The "
              "resolved commit is recorded, but the build is not reproducible.")
    flags = str(lc.get("cmake_flags") or "").split()
    if not cuda:
        flags = [f for f in flags if "GGML_CUDA" not in f]
    targets = ["llama-server", "llama-cli", "llama-quantize"]
    if with_imatrix:
        targets.append("llama-imatrix")
    log.append(run(["cmake", "-B", "build", *flags], cwd=build_dir))
    log.append(run(["cmake", "--build", "build", "--config", "Release",
                    "-j", str(jobs), "--target", *targets], cwd=build_dir))
    return build_dir


# -- checkpoint ------------------------------------------------------------


def checkpoint_inventory(local: Path) -> Dict[str, Any]:
    files = sorted(p for p in local.rglob("*")
                   if p.is_file() and ".cache" not in p.parts)
    return {"n_files": len(files),
            "total_gb": round(sum(p.stat().st_size for p in files) / GB, 4),
            "safetensors": sorted(p.name for p in files
                                  if p.suffix == ".safetensors")}


def download_checkpoint(cfg: Dict[str, Any], log: List[Dict[str, Any]]) -> Path:
    src = cfg["source"]
    local = resolve(src["local_dir"])
    if local.exists() and any(local.glob("*.safetensors")):
        # Reuse is fine, silence is not: record WHAT was reused.
        inv = checkpoint_inventory(local)
        print(f"[models] reusing checkpoint at {local} "
              f"({inv['n_files']} files, {inv['total_gb']} GB)")
        log.append({"command": ["reuse-existing-checkpoint", str(local)],
                    "returncode": 0, "seconds": 0.0,
                    "stdout_tail": json.dumps(inv), "stderr_tail": ""})
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


def _hex40(token: str) -> Optional[str]:
    t = token.strip().lower()
    if len(t) == 40 and all(c in "0123456789abcdef" for c in t):
        return t
    return None


def _first_hex40_line(path: Path) -> Optional[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    for line in text.splitlines():
        got = _hex40(line)
        if got:
            return got
    return None


def resolved_revision(local: Path, repo: str,
                      configured: Optional[str]) -> Dict[str, Any]:
    """Resolve the commit the local checkpoint actually is.

    The old implementation looked for a file whose ENTIRE stripped content was
    exactly 40 characters. Modern huggingface_hub with `local_dir=` writes no
    `refs/` at all and records the commit as the FIRST LINE of multi-line
    `.cache/huggingface/download/**/*.metadata` files, so that scan returned
    "unrecorded" on every real download.
    """
    attempts: List[str] = []

    # 1. Authoritative, when there is a network.
    try:
        from huggingface_hub import HfApi  # noqa: WPS433
        info = HfApi().model_info(repo, revision=configured or None)
        sha = getattr(info, "sha", None)
        if sha:
            return {"revision": str(sha), "online": True,
                    "method": "huggingface_hub.HfApi().model_info(...).sha",
                    "attempts": attempts}
        attempts.append("HfApi.model_info returned no sha")
    except Exception as exc:
        attempts.append(f"HfApi.model_info failed: {type(exc).__name__}: {exc}")

    # 2. local_dir layout: .cache/huggingface/download/**/*.metadata,
    #    line 1 = commit hash, line 2 = etag, line 3 = timestamp.
    dl = local / ".cache" / "huggingface" / "download"
    if dl.exists():
        for f in sorted(dl.rglob("*.metadata")):
            got = _first_hex40_line(f)
            if got:
                return {"revision": got, "online": False,
                        "method": f"local_dir cache metadata ({f.name})",
                        "attempts": attempts}
        attempts.append(f"no commit hash in {dl}/**/*.metadata")
    else:
        attempts.append(f"no {dl}")

    # 3. Classic hub cache layout: models--org--name/refs/<branch>, snapshots/<sha>.
    for base in (local, *(p for p in local.glob("models--*") if p.is_dir())):
        refs = base / "refs"
        if refs.is_dir():
            for f in sorted(p for p in refs.rglob("*") if p.is_file()):
                got = _first_hex40_line(f)
                if got:
                    return {"revision": got, "online": False,
                            "method": f"hub cache refs/{f.name}",
                            "attempts": attempts}
        snaps = base / "snapshots"
        if snaps.is_dir():
            for d in sorted(p for p in snaps.iterdir() if p.is_dir()):
                got = _hex40(d.name)
                if got:
                    return {"revision": got, "online": False,
                            "method": "hub cache snapshots/<sha> directory name",
                            "attempts": attempts}
    attempts.append("no refs/ or snapshots/ layout found")

    # 4. The configured pin, if it is itself a commit sha. Unverified.
    pinned = _hex40(configured or "")
    if pinned:
        return {"revision": pinned, "online": False,
                "method": "configured pin, UNVERIFIED against the checkpoint",
                "attempts": attempts}

    print(f"[models] WARNING: could not resolve the checkpoint revision. "
          f"Attempts: {attempts}")
    return {"revision": "unrecorded", "online": False, "method": "none",
            "attempts": attempts}


# -- importance matrix -----------------------------------------------------


def build_calibration_text(cfg: Dict[str, Any], data_dir: str,
                           search: List[str]) -> Dict[str, Any]:
    """Write the imatrix calibration corpus and describe exactly what it is.

    Default: a deterministic HELD-OUT slice of the MultiHop-RAG corpus -- the
    articles sorted by document_id, minus every document any manifest question
    depends on. Calibrating on the evaluation documents would leak the test set
    into the quantization.
    """
    ic = cfg["imatrix"]
    text_path = resolve(ic["text_path"])
    text_path.parent.mkdir(parents=True, exist_ok=True)
    source = str(ic.get("calibration_source") or "held_out_corpus")

    if source == "file":
        cal = ic.get("calibration_file")
        if not cal:
            raise SystemExit("imatrix.calibration_source is 'file' but "
                             "imatrix.calibration_file is null")
        cal_path = resolve(cal)
        if not cal_path.exists():
            raise SystemExit(f"imatrix calibration file not found: {cal_path}")
        text = cal_path.read_text(encoding="utf-8", errors="replace")
        text_path.write_text(text, encoding="utf-8")
        return {
            "calibration_source": "file",
            "calibration_file": str(cal_path),
            "text_path": str(text_path),
            "text_sha256": sha256_text(text),
            "text_chars": len(text),
            "leakage_rule": ("operator-supplied calibration text; the operator is "
                             "responsible for confirming it is disjoint from the "
                             "evaluation set"),
        }

    # Sibling script: reuse its document-identity rules so "held out" is defined
    # by exactly the same document_id the manifest was built from.
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    import prepare_dataset as pds  # noqa: WPS433

    used = read_json(ROOT / "data" / "pilot_documents.json")
    if used is None:
        raise SystemExit(
            "imatrix calibration needs data/pilot_documents.json so it can hold "
            "the evaluation documents OUT of the calibration set. Run "
            "scripts/prepare_dataset.py first, or set imatrix.calibration_source "
            "to 'file', or pass --no-imatrix.")
    used_ids = set(used)

    c_path = pds.find_file([data_dir, *search, "/kaggle/input", "data", "."],
                           "corpus.json")
    if not c_path:
        raise SystemExit("corpus.json not found; pass --data-dir / --search, "
                         "or use imatrix.calibration_source: file")
    corpus = json.loads(Path(c_path).read_text(encoding="utf-8"))

    by_id: Dict[str, Dict[str, Any]] = {}
    for rec in corpus:
        ident = pds.doc_identity(rec)
        if ident:
            by_id[pds.doc_id(ident)] = rec
    held_out = [d for d in sorted(by_id) if d not in used_ids]
    n_docs = int(ic.get("calibration_documents", 128))
    chars = int(ic.get("calibration_chars_per_document", 6000))
    chosen = held_out[:n_docs]
    if not chosen:
        raise SystemExit("no held-out corpus documents left for imatrix "
                         "calibration; every document is used by the manifest")

    text = "\n\n".join((by_id[d].get("body") or "")[:chars] for d in chosen)
    text_path.write_text(text, encoding="utf-8")
    print(f"[imatrix] calibration text: {len(chosen)} held-out documents "
          f"({len(text)} chars) -> {text_path}")
    return {
        "calibration_source": "held_out_corpus",
        "calibration_corpus_file": str(c_path),
        "text_path": str(text_path),
        "text_sha256": sha256_text(text),
        "text_chars": len(text),
        "corpus_documents_total": len(by_id),
        "evaluation_documents_excluded": len(used_ids),
        "held_out_documents_available": len(held_out),
        "calibration_documents_used": len(chosen),
        "calibration_document_ids": chosen,
        "chars_per_document": chars,
        "selection_rule": ("corpus documents sorted by document_id, minus every "
                           "document_id in data/pilot_documents.json, take the "
                           "first calibration_documents"),
        "leakage_rule": ("NO evaluation question and NO supporting document of "
                         "any manifest question contributes to this text"),
    }


def ensure_imatrix(cfg: Dict[str, Any], f16: Path, f16_sha: str,
                   prev: Dict[str, Any], data_dir: str, search: List[str],
                   log: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Produce (or validate and reuse) the importance matrix for `f16`."""
    ic = cfg["imatrix"]
    lc = cfg["llama_cpp"]
    out = resolve(ic["output_path"])
    out.parent.mkdir(parents=True, exist_ok=True)

    cal = build_calibration_text(cfg, data_dir, search)

    prev_im = prev.get("imatrix") or {}
    if out.exists() and prev_im.get("derived_from_f16_sha256") == f16_sha \
            and prev_im.get("calibration", {}).get("text_sha256") == cal["text_sha256"] \
            and prev_im.get("sha256") == sha256_file_cached(out):
        print(f"[imatrix] reusing {out} (same F16 and same calibration text)")
        return dict(prev_im, calibration=cal, reused=True)

    binary = resolve(lc.get("imatrix_binary")
                     or "vendor/llama.cpp/build/bin/llama-imatrix")
    if shutil.which(str(binary)) is None and not binary.exists():
        raise SystemExit(
            f"llama-imatrix not found: {binary}. Build it with "
            "scripts/prepare_models.py --build-llama-cpp, or pass --no-imatrix "
            "to build Q4_K_M without an importance matrix (a materially worse "
            "model, recorded as imatrix_used=false).")
    if out.exists():
        out.unlink()
        invalidate_hash(out)
    log.append(run([str(binary), "-m", str(f16), "-f", cal["text_path"],
                    "-o", str(out),
                    "--output-format", str(ic.get("output_format") or "gguf"),
                    "--chunks", str(int(ic.get("chunks", 128))),
                    "-c", str(int(ic.get("n_ctx", 512))),
                    "-ngl", str(int(ic.get("n_gpu_layers", 99)))]))
    if not out.exists():
        raise SystemExit(f"llama-imatrix reported success but {out} is missing")
    sha = sha256_file_cached(out, force=True)
    print(f"[imatrix] {out} sha256={sha}")
    return {
        "path": str(out),
        "sha256": sha,
        "size_bytes": out.stat().st_size,
        "derived_from_f16_sha256": f16_sha,
        "chunks": int(ic.get("chunks", 128)),
        "n_ctx": int(ic.get("n_ctx", 512)),
        "output_format": str(ic.get("output_format") or "gguf"),
        "binary": str(binary),
        "calibration": cal,
        "reused": False,
        "created_at": time.time(),
    }


# -- quantization ----------------------------------------------------------


def stale_reason(rec: Dict[str, Any], out: Path, f16_sha: str,
                 want_imatrix: bool, imatrix_sha: str) -> Optional[str]:
    """Why a pre-existing quantized file may NOT be reused.

    The old code reused anything that merely existed, then wrote "quantized
    from this exact F16 GGUF" unconditionally.
    """
    parent = rec.get("derived_from_f16_sha256")
    if not parent:
        return ("no derived_from_f16_sha256 recorded for this file, so its "
                "source F16 is unknown")
    if parent != f16_sha:
        return f"recorded parent F16 {parent[:16]}... != current F16 {f16_sha[:16]}..."
    recorded = rec.get("sha256")
    if recorded and recorded != sha256_file_cached(out):
        return "file on disk no longer matches the sha256 in provenance.json"
    if bool(rec.get("imatrix_used")) != want_imatrix:
        return (f"imatrix policy changed (recorded imatrix_used="
                f"{bool(rec.get('imatrix_used'))}, requested {want_imatrix})")
    if want_imatrix and rec.get("imatrix_sha256") != imatrix_sha:
        return "the importance matrix changed since this file was quantized"
    return None


def quantize_variants(cfg: Dict[str, Any], gguf_paths: Dict[str, Path],
                      f16_sha: str, imatrix: Optional[Dict[str, Any]],
                      prev_models: Dict[str, Any], quantize: Path,
                      requantize_stale: bool,
                      log: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    derived: Dict[str, Dict[str, Any]] = {}
    for name in ("Q8_0", "Q4_K_M"):
        spec = cfg["models"][name]
        if spec.get("derived_from") != "F16":
            raise SystemExit(f"{name} must be derived_from: F16")
        out = gguf_paths[name]
        want_imatrix = bool(spec.get("use_imatrix")) and imatrix is not None
        imatrix_sha = (imatrix or {}).get("sha256", "")

        if out.exists():
            why = stale_reason(prev_models.get(name) or {}, out, f16_sha,
                               want_imatrix, imatrix_sha)
            if why is None:
                print(f"[models] reusing {out} (parent F16 verified)")
                rec = prev_models[name]
                derived[name] = {
                    "derived_from_f16_sha256": f16_sha,
                    "imatrix_used": bool(rec.get("imatrix_used")),
                    "imatrix_sha256": rec.get("imatrix_sha256", ""),
                    "quantized_at": rec.get("quantized_at"),
                    "reused": True,
                }
                continue
            if not requantize_stale:
                raise SystemExit(
                    f"[models] {out} cannot be trusted: {why}. Re-run without "
                    "--no-requantize-stale to rebuild it, or delete it.")
            print(f"[models] WARNING: re-quantizing {out.name}: {why}")
            out.unlink()
            invalidate_hash(out)

        if not quantize.exists():
            raise SystemExit(f"llama-quantize not found: {quantize}")
        cmd: List[str] = [str(quantize)]
        if want_imatrix:
            cmd += ["--imatrix", str(imatrix["path"])]
        cmd += [str(gguf_paths["F16"]), str(out), spec["quantize_type"]]
        log.append(run(cmd))
        derived[name] = {
            "derived_from_f16_sha256": f16_sha,
            "imatrix_used": want_imatrix,
            "imatrix_sha256": imatrix_sha if want_imatrix else "",
            "quantized_at": time.time(),
            "reused": False,
        }
        if not want_imatrix and spec.get("use_imatrix"):
            print(f"[models] NOTE: {name} is configured use_imatrix: true but "
                  "was built WITHOUT one (--no-imatrix). Recorded as "
                  "imatrix_used=false.")
    return derived


# -- main ------------------------------------------------------------------


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
    ap.add_argument("--data-dir", default="data",
                    help="where corpus.json lives, for imatrix calibration")
    ap.add_argument("--search", nargs="*", default=[],
                    help="extra directories to search for corpus.json")
    ap.add_argument("--no-imatrix", dest="imatrix", action="store_false",
                    default=True,
                    help="build Q4_K_M WITHOUT an importance matrix. Materially "
                         "worse at 4 bits; recorded as imatrix_used=false.")
    ap.add_argument("--no-requantize-stale", dest="requantize_stale",
                    action="store_false", default=True,
                    help="fail instead of rebuilding a quantized file whose "
                         "recorded parent F16 does not match the current one")
    ap.add_argument("--free-hf-weights", action="store_true",
                    help="delete the HF safetensors once the F16 GGUF exists "
                         "(~6.2 GB back; Kaggle /kaggle/working caps at 20 GB)")
    ap.add_argument("--min-free-gb", type=float, default=18.0,
                    help="refuse to start below this much free disk")
    ap.add_argument("--allow-low-disk", action="store_true",
                    help="downgrade the free-disk check to a warning")
    args = ap.parse_args()

    cfg = load_models_config(args.models_config)
    rd = results_dir(args.results_dir)
    log: List[Dict[str, Any]] = []
    prov: Dict[str, Any] = {"created_at": time.time(), "models": {}, "commands": log}
    previous = read_json(rd / "provenance.json", default={}) or {}
    prev_models: Dict[str, Any] = dict(previous.get("models") or {})

    gguf_paths = {k: resolve(v["path"]) for k, v in cfg["models"].items()}
    for p in gguf_paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)

    imatrix_wanted = bool(args.imatrix) and bool((cfg.get("imatrix") or {})
                                                 .get("enabled", True))
    if not args.only_hash:
        prov["disk"] = disk_check(gguf_paths["F16"].parent, args.min_free_gb,
                                  args.allow_low_disk, args.free_hf_weights)

    lc = cfg["llama_cpp"]
    if args.build_llama_cpp and not args.only_hash:
        build_llama_cpp(cfg, cuda=not args.no_cuda, jobs=args.jobs, log=log,
                        with_imatrix=imatrix_wanted)
    build_dir = resolve(lc["build_dir"])
    prov["llama_cpp"] = {"repo": lc["repo"], "configured_commit": lc.get("commit"),
                         "configured_tag": lc.get("commit_tag"),
                         "resolved_commit": git_commit(build_dir),
                         "cmake_flags": lc.get("cmake_flags"),
                         "cuda": not args.no_cuda}
    if lc.get("commit") and lc["commit"] != "master" and \
            prov["llama_cpp"]["resolved_commit"] not in ("unknown", lc["commit"]):
        print(f"[models] WARNING: configured llama.cpp commit {lc['commit']} but "
              f"the checkout is at {prov['llama_cpp']['resolved_commit']}")

    imatrix: Optional[Dict[str, Any]] = None
    if not args.only_hash:
        hf_dir = Path(args.hf_dir) if args.hf_dir else None
        if hf_dir is None:
            hf_dir = (resolve(cfg["source"]["local_dir"]) if args.skip_download
                      else download_checkpoint(cfg, log))
        rev = resolved_revision(hf_dir, cfg["source"]["hf_repo"],
                                cfg["source"].get("hf_revision"))
        prov["source"] = {"hf_repo": cfg["source"]["hf_repo"],
                          "configured_revision": cfg["source"].get("hf_revision"),
                          "resolved_revision": rev["revision"],
                          "revision_resolution": rev,
                          "local_dir": str(hf_dir)}
        cfg_rev = _hex40(cfg["source"].get("hf_revision") or "")
        if cfg_rev and _hex40(rev["revision"]) and cfg_rev != _hex40(rev["revision"]):
            print(f"[models] WARNING: configured hf_revision {cfg_rev} != resolved "
                  f"{rev['revision']}")

        f16 = gguf_paths["F16"]
        if not f16.exists():
            convert = resolve(lc["convert_script"])
            if not convert.exists():
                raise SystemExit(f"convert script not found: {convert} "
                                 "(run with --build-llama-cpp first)")
            log.append(run([sys.executable, str(convert), str(hf_dir),
                            "--outfile", str(f16), "--outtype", "f16"]))
            invalidate_hash(f16)
        else:
            print(f"[models] reusing {f16}")

        f16_sha = sha256_file_cached(f16)
        print(f"[models] F16 sha256={f16_sha}")

        if args.free_hf_weights:
            prov["hf_weights_freed"] = free_hf_weights(hf_dir)

        if imatrix_wanted and any(cfg["models"][n].get("use_imatrix")
                                  for n in ("Q8_0", "Q4_K_M")):
            imatrix = ensure_imatrix(cfg, f16, f16_sha, previous, args.data_dir,
                                     args.search, log)
        elif not imatrix_wanted:
            print("[models] --no-imatrix: Q4_K_M will be built WITHOUT an "
                  "importance matrix (recorded as imatrix_used=false)")

        quantize = resolve(lc["quantize_binary"])
        derived = quantize_variants(cfg, gguf_paths, f16_sha, imatrix,
                                    prev_models, quantize,
                                    args.requantize_stale, log)
        derived["F16"] = {"derived_from_f16_sha256": f16_sha,
                          "imatrix_used": False, "imatrix_sha256": "",
                          "reused": True}
    else:
        derived = {name: {k: (prev_models.get(name) or {}).get(k)
                          for k in ("derived_from_f16_sha256", "imatrix_used",
                                    "imatrix_sha256", "quantized_at")}
                   for name in gguf_paths}
        imatrix = previous.get("imatrix")

    f16_sha = sha256_file_cached(gguf_paths["F16"]) if gguf_paths["F16"].exists() else ""
    for name, path in gguf_paths.items():
        if not path.exists():
            prov["models"][name] = {"path": str(path), "present": False}
            continue
        prov["models"][name] = {
            "path": str(path), "present": True,
            "sha256": sha256_file_cached(path),
            "size_bytes": path.stat().st_size,
            "size_gb": round(path.stat().st_size / (1024 ** 3), 4),
            "quantize_type": cfg["models"][name]["quantize_type"],
            "derived_from": cfg["models"][name]["derived_from"],
            **{k: v for k, v in (derived.get(name) or {}).items() if k != "reused"},
        }

    if imatrix:
        prov["imatrix"] = imatrix
    prov["common_source"] = {
        "f16_gguf_sha256": f16_sha,
        "hf_repo": cfg["source"]["hf_repo"],
        "note": ("Every variant records the sha256 of the F16 GGUF it was "
                 "actually quantized from, as derived_from_f16_sha256, written "
                 "at the moment it was derived. Compare those against "
                 "f16_gguf_sha256 -- do not take this note on trust."),
        "variants_derived_from_f16_sha256": {
            n: (prov["models"].get(n) or {}).get("derived_from_f16_sha256")
            for n in gguf_paths},
        "all_variants_share_this_f16": bool(f16_sha) and all(
            (prov["models"].get(n) or {}).get("derived_from_f16_sha256") == f16_sha
            for n in gguf_paths if (prov["models"].get(n) or {}).get("present")),
    } if f16_sha else None
    prov["sampling"] = cfg["sampling"]

    existing = read_json(rd / "provenance.json", default={}) or {}
    existing.update(prov)
    write_json(rd / "provenance.json", existing)
    print(json.dumps({k: v for k, v in prov["models"].items()}, indent=2))
    print(f"[models] provenance -> {rd / 'provenance.json'}")
    cs = prov.get("common_source") or {}
    if cs and not cs.get("all_variants_share_this_f16"):
        print("[models] WARNING: at least one variant does not record the "
              "current F16 as its parent; common_source_checkpoint will fail.")
    missing = [k for k, v in prov["models"].items() if not v.get("present")]
    if missing:
        print(f"[models] WARNING missing GGUFs: {missing}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
