#!/usr/bin/env python3
"""Run one stage (A, B or C) at one or all precisions. Resumable. No silent skips.

Order of operations, per the brief:
  * one precision BLOCK at a time -- F16, then Q8_0, then Q4_K_M;
  * one physical model per block, shared by every logical agent in that block;
  * question order counterbalanced per precision by a deterministic rotation,
    and the realised order is recorded;
  * GPU memory released between blocks.

Every generation is appended to results/<stage>/events.jsonl before the next
call starts, so an interrupted Kaggle session resumes exactly where it stopped.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import coordinator as coord_mod  # noqa: E402
import document_agent as agent_mod  # noqa: E402
import prompts as prompt_mod  # noqa: E402
import synthesizer as synth_mod  # noqa: E402
from backends.llamacpp_backend import (ModelSpec, SamplingConfig,  # noqa: E402
                                       build_backend)
from config import load_models_config, load_stage_config, resolve, results_dir  # noqa: E402
from evidence import (FixedEvidenceCache, build_store, embed_signature,  # noqa: E402
                      get_fixed_evidence, make_embedder)
from hardware import cuda_report, free_gpu_memory  # noqa: E402
from llm_client import GuardedClient  # noqa: E402
from logging_utils import (EventLog, read_json, sha256_text, stable_id,  # noqa: E402
                           write_json)
from retrieval import Hit  # noqa: E402

PRECISION_ORDER = ["F16", "Q8_0", "Q4_K_M"]


# --------------------------------------------------------------------------


def counterbalanced(questions: List[Dict[str, Any]], precision: str
                    ) -> List[Dict[str, Any]]:
    """Deterministic rotation so precision blocks do not share question order."""
    if not questions:
        return []
    shift = PRECISION_ORDER.index(precision) if precision in PRECISION_ORDER else 0
    k = (shift * max(1, len(questions) // max(1, len(PRECISION_ORDER)))) % len(questions)
    return questions[k:] + questions[:k]


def select_stage_c_questions(stage_b_predictions: List[Dict[str, Any]],
                             manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Deterministic Stage-C selection from Stage-B FIXED-EVIDENCE results.

    2 low-width + 2 high-width + 2 largest F16/Q4 disagreement. Applied before
    any end-to-end output exists, let alone is read.
    """
    by_id = {q["question_id"]: q for q in manifest["stages"]["stage_b"]["questions"]}
    scored: Dict[str, Dict[str, float]] = {}
    for p in stage_b_predictions:
        if p.get("condition") != "fixed_verified_evidence":
            continue
        scored.setdefault(p["question_id"], {})[p["precision"]] = float(
            p.get("answer_score", {}).get("score", 0.0))
    recall: Dict[str, Dict[str, float]] = {}
    for p in stage_b_predictions:
        if p.get("condition") != "fixed_verified_evidence":
            continue
        recall.setdefault(p["question_id"], {})[p["precision"]] = float(
            p.get("atomic_fact_recall_final", 0.0))

    widths = {qid: by_id[qid]["width"] for qid in by_id}
    chosen: List[str] = []

    low = sorted([q for q, w in widths.items() if w == 2])
    chosen += low[:2]
    hi_w = max(widths.values()) if widths else 2
    high = sorted([q for q, w in widths.items() if w == hi_w and q not in chosen])
    chosen += high[:2]

    def disagreement(qid: str) -> tuple:
        s = scored.get(qid, {})
        r = recall.get(qid, {})
        return (-abs(s.get("F16", 0.0) - s.get("Q4_K_M", 0.0)),
                -abs(r.get("F16", 0.0) - r.get("Q4_K_M", 0.0)), qid)

    rest = sorted([q for q in by_id if q not in chosen], key=disagreement)
    chosen += rest[:2]
    return [by_id[q] for q in chosen[:6]]


# --------------------------------------------------------------------------


def run_question(client, store, cache, question: Dict[str, Any], condition: str,
                 rcfg: Dict[str, Any], sig: str, scfg: Dict[str, Any],
                 dry_run: bool) -> Dict[str, Any]:
    qid = question["question_id"]
    qtext = question["question"]
    doc_ids = [d["document_id"] for d in question["documents"]]
    t0 = time.time()

    coord = coord_mod.run_coordinator(client, qid, qtext, len(doc_ids))
    shared_tasks = coord.get("shared_tasks") or []

    agent_outputs: List[Dict[str, Any]] = []
    evidence_chunk_ids: List[str] = []
    agent_wall = 0.0
    for did in doc_ids:
        ta = time.time()
        if condition == "fixed_verified_evidence":
            hits: List[Hit] = get_fixed_evidence(
                store, cache, qid, qtext, did, int(rcfg["top_k"]),
                int(rcfg["chunk_tokens"]), int(rcfg["overlap_tokens"]), sig)
            evidence_chunk_ids += [h.chunk_id for h in hits]
            out = agent_mod.run_fixed_evidence_agent(
                client, qid, did, qtext, shared_tasks, hits)
        else:
            out = agent_mod.run_end_to_end_agent(
                client, store, qid, did, qtext, shared_tasks,
                top_k=int(rcfg["top_k"]),
                max_search_rounds=int(rcfg.get("max_search_rounds", 2)))
            evidence_chunk_ids += list(out.get("retrieved_chunk_ids") or [])
        agent_wall += time.time() - ta
        agent_outputs.append(out)

    synth = synth_mod.run_synthesizer(
        client, qid, qtext, coord.get("synthesis_directive", ""), agent_outputs)

    return {
        "question_id": qid, "question": qtext, "width": question["width"],
        "question_type": question.get("question_type", ""),
        "gold_answer": question.get("answer", ""),
        "required_document_ids": doc_ids,
        "stage": client.stage, "precision": client.precision, "condition": condition,
        "coordinator": coord,
        "agent_outputs": agent_outputs,
        "synthesis": synth,
        "final_answer": synth.get("final_answer", ""),
        "evidence_chunk_ids": sorted(set(evidence_chunk_ids)),
        "wall_s": round(time.time() - t0, 4),
        "document_agent_wall_s": round(agent_wall, 4),
        "is_fixture": bool(synth.get("is_fixture")
                           or any(a.get("is_fixture") for a in agent_outputs)),
        "dry_run": bool(dry_run),
    }


def make_sampling(models_cfg: Dict[str, Any]) -> SamplingConfig:
    s = models_cfg["sampling"]
    return SamplingConfig(
        temperature=float(s["temperature"]), top_p=float(s["top_p"]),
        top_k=int(s["top_k"]), min_p=float(s.get("min_p", 0.0)),
        repeat_penalty=float(s.get("repeat_penalty", 1.0)), seed=int(s["seed"]),
        max_tokens=int(s["max_tokens"]), n_ctx=int(s["n_ctx"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=["stage_a", "stage_b", "stage_c"])
    ap.add_argument("--precisions", nargs="*", default=None,
                    help="default: all three, in F16 -> Q8_0 -> Q4_K_M block order")
    ap.add_argument("--condition", default=None,
                    choices=["fixed_verified_evidence", "end_to_end"])
    ap.add_argument("--models-config", default="configs/models.yaml")
    ap.add_argument("--stage-config", default=None)
    ap.add_argument("--manifest", default="data/pilot_manifest.json")
    ap.add_argument("--documents", default="data/pilot_documents.json")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--backend", default=None,
                    help="llama-server | llama-cli | fixture (fixture is blocked "
                         "unless SPDQ_ALLOW_FIXTURE=1 and never yields science)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and hash every prompt, call no model")
    ap.add_argument("--limit", type=int, default=None, help="debug: fewer questions")
    ap.add_argument("--force", action="store_true",
                    help="run a later stage without its predecessor's gate")
    args = ap.parse_args()

    rd = results_dir(args.results_dir)
    stage_dir = rd / args.stage
    stage_dir.mkdir(parents=True, exist_ok=True)

    models_cfg = load_models_config(args.models_config)
    scfg = load_stage_config(args.stage, args.stage_config)
    manifest = read_json(ROOT / args.manifest)
    documents = read_json(ROOT / args.documents)
    if manifest is None or documents is None:
        raise SystemExit("run scripts/prepare_dataset.py first")

    condition = args.condition or (scfg.get("conditions") or ["fixed_verified_evidence"])[0]
    precisions = args.precisions or list(scfg.get("precisions") or PRECISION_ORDER)
    precisions = [p for p in PRECISION_ORDER if p in precisions] + \
                 [p for p in precisions if p not in PRECISION_ORDER]

    # -- stage gating ----------------------------------------------------
    if args.stage in ("stage_b", "stage_c") and not args.force:
        prev = "stage_a" if args.stage == "stage_b" else "stage_b"
        gate_file = rd / prev / "gate.json"
        gate = read_json(gate_file)
        if not gate or not gate.get("proceed"):
            raise SystemExit(
                f"{args.stage} requires a passing {prev} gate. Expected "
                f"{gate_file} with proceed=true. Run scripts/score.py --stage {prev} "
                f"and scripts/analyze.py --stage {prev} first, or pass --force to "
                "record an explicit manual approval.")

    # -- question set ----------------------------------------------------
    if args.stage == "stage_c":
        prev_preds = read_json(rd / "stage_b" / "predictions.json", default=[]) or []
        questions = select_stage_c_questions(prev_preds, manifest)
        manifest["stages"]["stage_c"]["questions"] = questions
        write_json(stage_dir / "stage_c_selection.json", {
            "selected_question_ids": [q["question_id"] for q in questions],
            "rule": scfg.get("selection", {}).get("rule", ""),
            "selected_before_reading_end_to_end_outputs": True})
    else:
        questions = manifest["stages"][args.stage]["questions"]
    if args.limit:
        questions = questions[:args.limit]
    expected = len(questions) * len(precisions)
    print(f"[run] {args.stage} | condition={condition} | precisions={precisions} | "
          f"{len(questions)} questions -> {expected} predictions")

    # -- retrieval -------------------------------------------------------
    rcfg = scfg.get("retrieval") or {}
    rcfg.setdefault("top_k", 3)
    rcfg.setdefault("chunk_tokens", 400)
    rcfg.setdefault("overlap_tokens", 50)
    rcfg.setdefault("max_search_rounds", 0 if condition == "fixed_verified_evidence" else 2)

    embedder = make_embedder(models_cfg, str(rd / "embed_cache"))
    doc_ids = sorted({d["document_id"] for q in questions for d in q["documents"]})
    store = build_store(documents, embedder, doc_ids, int(rcfg["chunk_tokens"]),
                        int(rcfg["overlap_tokens"]))
    iso = store.assert_isolation()
    write_json(stage_dir / "isolation_report.json", iso)
    if not iso["isolated"]:
        raise SystemExit(f"DOCUMENT ISOLATION VIOLATED: {iso['violations'][:3]}")
    cache = FixedEvidenceCache(rd / "fixed_evidence.json")
    sig = embed_signature(embedder)

    # freeze fixed evidence ONCE, before any model runs
    if condition == "fixed_verified_evidence":
        for q in questions:
            for d in q["documents"]:
                get_fixed_evidence(store, cache, q["question_id"], q["question"],
                                   d["document_id"], int(rcfg["top_k"]),
                                   int(rcfg["chunk_tokens"]),
                                   int(rcfg["overlap_tokens"]), sig)
        cache.flush()

    # -- run -------------------------------------------------------------
    log = EventLog(stage_dir / "events.jsonl")
    predictions: List[Dict[str, Any]] = read_json(stage_dir / "predictions.json",
                                                  default=[]) or []
    done = {(p["question_id"], p["precision"], p["condition"]) for p in predictions}
    backend_kind = args.backend or models_cfg["backend"]["kind"]
    run_report: Dict[str, Any] = {
        "stage": args.stage, "condition": condition, "precisions": precisions,
        "expected_predictions": expected, "dry_run": bool(args.dry_run),
        "backend_kind": backend_kind, "started_at": time.time(),
        "cuda": cuda_report(),
        "prompt_hashes": {k: sha256_text(v)
                          for k, v in prompt_mod.all_prompt_texts().items()},
        "retrieval": rcfg, "embedding": embedder.provenance(),
        "question_order": {}, "blocks": {}, "failures": [],
    }

    sampling = make_sampling(models_cfg)
    prov = read_json(rd / "provenance.json", default={}) or {}
    prov.setdefault("models", {})

    for precision in precisions:
        block_t0 = time.time()
        ordered = counterbalanced(list(questions), precision)
        run_report["question_order"][precision] = [q["question_id"] for q in ordered]

        model_spec: Optional[ModelSpec] = None
        if backend_kind not in ("fixture", "mock") and not args.dry_run:
            mpath = resolve(models_cfg["models"][precision]["path"])
            model_spec = ModelSpec.from_path(precision, str(mpath))
            prov["models"].setdefault(precision, {})
            prov["models"][precision].update(
                {"path": str(mpath), "sha256": model_spec.sha256,
                 "size_bytes": model_spec.size_bytes, "present": True})
        elif backend_kind in ("fixture", "mock"):
            model_spec = ModelSpec(precision=precision, path="FIXTURE",
                                   sha256="FIXTURE", size_bytes=0)

        backend = None
        if not args.dry_run:
            kw: Dict[str, Any] = {}
            if backend_kind in ("llama-server", "server"):
                kw = {"binary": str(resolve(models_cfg["llama_cpp"]["server_binary"])),
                      "n_gpu_layers": int(models_cfg["backend"].get("n_gpu_layers", 99)),
                      "log_dir": str(rd / "server_logs"),
                      "startup_timeout_s": int(models_cfg["backend"].get(
                          "startup_timeout_s", 900)),
                      "threads": models_cfg["backend"].get("threads")}
            elif backend_kind in ("llama-cli", "cli"):
                kw = {"binary": str(resolve(models_cfg["llama_cpp"]["cli_binary"])),
                      "n_gpu_layers": int(models_cfg["backend"].get("n_gpu_layers", 99)),
                      "log_dir": str(rd / "server_logs")}
            backend = build_backend(backend_kind, model_spec, sampling, **kw)
            backend.start()
        else:
            class _NullBackend:
                is_fixture = True

                def provenance(self):
                    return {"backend": "dry-run", "is_fixture": True,
                            "sampling_hash": sampling.hash()}

                def chat(self, *a, **k):
                    raise RuntimeError("dry run: no model calls allowed")
            backend = _NullBackend()

        client = GuardedClient(backend, log, args.stage, precision, condition,
                               dry_run=args.dry_run)
        try:
            for q in ordered:
                key = (q["question_id"], precision, condition)
                if key in done:
                    print(f"[run] resume-skip {key}")
                    continue
                try:
                    pred = run_question(client, store, cache, q, condition, rcfg,
                                        sig, scfg, args.dry_run)
                except Exception as exc:  # never a silent skip
                    run_report["failures"].append(
                        {"question_id": q["question_id"], "precision": precision,
                         "error": f"{type(exc).__name__}: {exc}"})
                    log.append({"call_id": stable_id("failure", args.stage, precision,
                                                     condition, q["question_id"]),
                                "stage": args.stage, "precision": precision,
                                "condition": condition, "role": "question_failure",
                                "question_id": q["question_id"], "is_fixture": True,
                                "error": f"{type(exc).__name__}: {exc}"})
                    print(f"[run] FAILED {q['question_id']} @ {precision}: {exc}")
                    continue
                if backend is not None and hasattr(backend, "provenance"):
                    pb = backend.provenance()
                    pred["backend"] = pb.get("backend")
                    pred["sampling_hash"] = pb.get("sampling_hash")
                    pred["model_sha256"] = (pb.get("model") or {}).get("sha256", "")
                predictions.append(pred)
                done.add(key)
                write_json(stage_dir / "predictions.json", predictions)
                print(f"[run] {precision} {q['question_id']} (w={q['width']}) "
                      f"{pred['wall_s']:.1f}s -> {pred['final_answer'][:80]!r}")
            cache.flush()
        finally:
            block_prov = backend.provenance() if hasattr(backend, "provenance") else {}
            run_report["blocks"][precision] = {
                "wall_s": round(time.time() - block_t0, 2),
                "provenance": block_prov,
                "gpu_offload": block_prov.get("gpu_offload"),
                "memory": block_prov.get("memory"),
                "load_time_s": block_prov.get("load_time_s"),
            }
            if hasattr(backend, "stop"):
                backend.stop()
            free_gpu_memory()

    run_report["finished_at"] = time.time()
    run_report["completed_predictions"] = len(predictions)
    run_report["complete"] = len(predictions) >= expected and not run_report["failures"]
    write_json(stage_dir / "run_report.json", run_report)
    write_json(stage_dir / "predictions.json", predictions)
    prov["common_source"] = prov.get("common_source")
    write_json(rd / "provenance.json", prov)

    print(f"[run] {len(predictions)}/{expected} predictions -> {stage_dir}")
    if run_report["failures"]:
        print(f"[run] {len(run_report['failures'])} question failures recorded "
              "(NOT skipped silently); see run_report.json")
    if args.dry_run or backend_kind in ("fixture", "mock"):
        print("[run] NOTE: this run is watermarked non-scientific "
              "(dry run or fixture backend). It can prove the harness only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
