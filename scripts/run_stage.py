#!/usr/bin/env python3
"""Run one stage (A, B or C) at one or all precisions. Resumable. No silent skips.

Order of operations:
  * the coordinator runs ONCE at F16 and its instruction set is FROZEN, exactly
    like the evidence -- see `--help` on `coordinator.freeze_coordinator`;
  * one precision BLOCK at a time -- F16, then Q8_0, then Q4_K_M;
  * one physical model per block, shared by every logical agent in that block;
  * one discarded warmup generation per block, so first-touch cost is not
    charged to whichever question happened to run first;
  * GPU memory released between blocks, and verified released.

Every generation is appended to `<stage>/events.jsonl` before the next call
starts, so an interrupted Kaggle session resumes exactly where it stopped.

**Harness runs are quarantined on disk.** `--dry-run` and `--backend fixture`
write to `<stage>__harness/` instead of `<stage>/`. Previously they shared the
event log and predictions file with real runs, and because `call_id` has no
run-mode component, a dry run followed by a real run in the same directory
resume-skipped all 18 questions, reported "18/18 predictions", and never called
the model once.
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
from hashcache import sha256_file_cached  # noqa: E402
from llm_client import BackendGenerationError, GuardedClient  # noqa: E402
from logging_utils import (EventLog, read_json, sha256_text, stable_id,  # noqa: E402
                           write_json)
from retrieval import Hit  # noqa: E402

PRECISION_ORDER = ["F16", "Q8_0", "Q4_K_M"]
FROZEN_COORDINATOR = "frozen_coordinator.json"


# --------------------------------------------------------------------------


def counterbalanced(questions: List[Dict[str, Any]], precision: str
                    ) -> List[Dict[str, Any]]:
    """Deterministic rotation so precision blocks do not share question order.

    Under `temperature=0, top_k=1, cache_prompt=false` every call is
    independent and greedy, so this rotation cannot change any output. It is
    kept only as a guard against future stateful backends, and it is now safe:
    its one real effect used to be attaching the un-discarded warmup cost to a
    different question in each block, which confounded exactly the paired
    per-question timings the study reports. A warmup call is now issued and
    discarded before the first question of every block.
    """
    if not questions:
        return []
    shift = PRECISION_ORDER.index(precision) if precision in PRECISION_ORDER else 0
    k = (shift * max(1, len(questions) // max(1, len(PRECISION_ORDER)))) % len(questions)
    return questions[k:] + questions[:k]


def select_stage_c_questions(stage_b_scored: List[Dict[str, Any]],
                             manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Deterministic Stage-C selection from Stage-B FIXED-EVIDENCE results.

    2 low-width + 2 high-width + 2 largest F16/Q4 disagreement. Applied before
    any end-to-end output exists, let alone is read.

    Reads `stage_b/scored.json`. The previous version read
    `stage_b/predictions.json` and pulled `answer_score` and
    `atomic_fact_recall_final` from it -- keys that file has never contained,
    because scoring writes them to `scored.json`. `disagreement()` therefore
    always returned (-0.0, -0.0, qid) and the third rule silently collapsed to
    a question_id sort, producing output identical to having no Stage-B data
    at all.
    """
    by_id = {q["question_id"]: q for q in manifest["stages"]["stage_b"]["questions"]}
    if not stage_b_scored:
        raise SystemExit(
            "Stage C selection needs scored Stage B results. Expected "
            "stage_b/scored.json with fixed_verified_evidence rows; run "
            "scripts/score.py --stage stage_b first.")

    scored: Dict[str, Dict[str, float]] = {}
    recall: Dict[str, Dict[str, float]] = {}
    for p in stage_b_scored:
        if p.get("condition") != "fixed_verified_evidence":
            continue
        if p.get("is_fixture") or p.get("dry_run"):
            continue
        qid, prec = p.get("question_id"), p.get("precision")
        s = (p.get("answer_score") or {}).get("score")
        if s is not None:
            scored.setdefault(qid, {})[prec] = float(s)
        r = p.get("atomic_fact_recall_final")
        if r is not None:
            recall.setdefault(qid, {})[prec] = float(r)
    if not scored and not recall:
        raise SystemExit(
            "stage_b/scored.json has no scorable fixed-evidence rows; the "
            "max-disagreement rule cannot be applied.")

    widths = {qid: by_id[qid]["width"] for qid in by_id}
    chosen: List[str] = []

    low = sorted([q for q, w in widths.items() if w == 2])
    chosen += low[:2]
    hi_w = max(widths.values()) if widths else 2
    high = sorted([q for q, w in widths.items() if w == hi_w and q not in chosen])
    chosen += high[:2]

    def disagreement(qid: str):
        s = scored.get(qid, {})
        r = recall.get(qid, {})
        have = ("F16" in s and "Q4_K_M" in s) or ("F16" in r and "Q4_K_M" in r)
        return (0 if have else 1,
                -abs(s.get("F16", 0.0) - s.get("Q4_K_M", 0.0)),
                -abs(r.get("F16", 0.0) - r.get("Q4_K_M", 0.0)), qid)

    rest = sorted([q for q in by_id if q not in chosen], key=disagreement)
    chosen += rest[:2]
    return [by_id[q] for q in chosen[:6]]


# --------------------------------------------------------------------------


def run_question(client, store, cache, coord: Dict[str, Any],
                 question: Dict[str, Any], condition: str,
                 rcfg: Dict[str, Any], sig: str, budgets: Dict[str, int],
                 dry_run: bool) -> Dict[str, Any]:
    qid = question["question_id"]
    qtext = question["question"]
    doc_ids = [d["document_id"] for d in question["documents"]]
    shared_tasks = coord.get("shared_tasks") or []
    t0 = time.time()

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
                client, qid, did, qtext, shared_tasks, hits,
                max_tokens=budgets.get("document_agent"))
        else:
            # A single PrivateIndex, not the whole store: the agent cannot
            # reach another document even if a caller passes the wrong id.
            out = agent_mod.run_end_to_end_agent(
                client, store.get(did), qid, did, qtext, shared_tasks,
                top_k=int(rcfg["top_k"]),
                max_search_rounds=int(rcfg.get("max_search_rounds", 2)),
                max_tokens=budgets.get("document_agent"))
            evidence_chunk_ids += list(out.get("retrieved_chunk_ids") or [])
        agent_wall += time.time() - ta
        agent_outputs.append(out)

    synth = synth_mod.run_synthesizer(
        client, qid, qtext, coord.get("synthesis_directive", ""), agent_outputs,
        max_tokens=budgets.get("synthesizer"))

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


def role_budgets(models_cfg: Dict[str, Any]) -> Dict[str, int]:
    """Per-role generation budgets from config, falling back to the defaults
    baked into src/prompts.py."""
    out = dict(prompt_mod.DEFAULT_MAX_TOKENS)
    cfg = (models_cfg.get("sampling") or {}).get("max_tokens")
    if isinstance(cfg, dict):
        out.update({k: int(v) for k, v in cfg.items()})
    elif cfg is not None:  # legacy scalar
        out = {k: int(cfg) for k in out}
    out.setdefault("document_agent_turn", out.get("document_agent", 512))
    return out


def make_sampling(models_cfg: Dict[str, Any], budgets: Dict[str, int]
                  ) -> SamplingConfig:
    s = models_cfg["sampling"]
    # `max_tokens` is per-role now. The scalar folded into `sampling_hash` is
    # the ceiling actually in force, so the hash describes a real constraint
    # instead of a config value no call site ever read.
    ceiling = max(int(v) for k, v in budgets.items() if k != "preflight")
    return SamplingConfig(
        temperature=float(s["temperature"]), top_p=float(s["top_p"]),
        top_k=int(s["top_k"]), min_p=float(s.get("min_p", 0.0)),
        repeat_penalty=float(s.get("repeat_penalty", 1.0)), seed=int(s["seed"]),
        max_tokens=ceiling, n_ctx=int(s["n_ctx"]))


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
                    help="build and hash every prompt, call no model; writes to "
                         "<stage>__harness/ so it can never poison a real run")
    ap.add_argument("--limit", type=int, default=None, help="debug: fewer questions")
    ap.add_argument("--force", action="store_true",
                    help="run a later stage without its predecessor's gate; the "
                         "override is recorded in run_report.json and surfaced "
                         "in every downstream report")
    ap.add_argument("--force-reason", default="",
                    help="why the gate was overridden; recorded alongside --force")
    args = ap.parse_args()

    rd = results_dir(args.results_dir)
    models_cfg = load_models_config(args.models_config)
    scfg = load_stage_config(args.stage, args.stage_config)
    backend_kind = args.backend or models_cfg["backend"]["kind"]
    harness_only = bool(args.dry_run or backend_kind in ("fixture", "mock"))

    # Quarantine harness output so it can never be resumed into a real run.
    run_root = (rd / "_harness") if harness_only else rd
    stage_dir = run_root / args.stage
    stage_dir.mkdir(parents=True, exist_ok=True)

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
        gate_file = run_root / prev / "gate.json"
        gate = read_json(gate_file)
        if not gate or not gate.get("proceed"):
            raise SystemExit(
                f"{args.stage} requires a passing {prev} gate. Expected "
                f"{gate_file} with proceed=true. Run scripts/score.py --stage {prev} "
                f"and scripts/analyze.py --stage {prev} first, or pass --force to "
                "record an explicit manual approval.")

    # -- question set ----------------------------------------------------
    if args.stage == "stage_c":
        prev_scored = read_json(run_root / "stage_b" / "scored.json", default=[]) or []
        questions = select_stage_c_questions(prev_scored, manifest)
        manifest["stages"]["stage_c"]["questions"] = questions
        write_json(stage_dir / "stage_c_selection.json", {
            "selected_question_ids": [q["question_id"] for q in questions],
            "rule": scfg.get("selection", {}).get("rule", ""),
            "source": str(run_root / "stage_b" / "scored.json"),
            "n_stage_b_scored_rows": len(prev_scored),
            "selected_before_reading_end_to_end_outputs": True})
    else:
        questions = manifest["stages"][args.stage]["questions"]
    if args.limit:
        questions = questions[:args.limit]
    expected = len(questions) * len(precisions)
    print(f"[run] {args.stage} | condition={condition} | precisions={precisions} | "
          f"{len(questions)} questions -> {expected} predictions")
    if harness_only:
        print(f"[run] HARNESS-ONLY run; output quarantined in {stage_dir}")

    # -- retrieval -------------------------------------------------------
    rcfg = dict(scfg.get("retrieval") or {})
    rcfg.setdefault("top_k", 3)
    rcfg.setdefault("chunk_tokens", 400)
    rcfg.setdefault("overlap_tokens", 50)
    rcfg.setdefault("max_search_rounds", 0 if condition == "fixed_verified_evidence" else 2)
    budgets = role_budgets(models_cfg)

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
    coord_path = run_root / FROZEN_COORDINATOR

    # run_report accumulates across invocations. It used to be rebuilt from
    # scratch and overwritten, so running precisions in separate invocations
    # left `blocks` holding only the last one while predictions.json correctly
    # accumulated -- collapsing the whole systems table to a single row.
    run_report: Dict[str, Any] = read_json(stage_dir / "run_report.json",
                                           default={}) or {}
    run_report.update({
        "stage": args.stage, "condition": condition,
        "expected_predictions": expected, "dry_run": bool(args.dry_run),
        "harness_only": harness_only,
        "backend_kind": backend_kind, "started_at": time.time(),
        "cuda": cuda_report(),
        "prompt_hashes": {k: sha256_text(v)
                          for k, v in prompt_mod.all_prompt_texts().items()},
        "retrieval": rcfg, "role_max_tokens": budgets,
        "embedding": embedder.provenance(),
        "gate_override_used": bool(args.force),
        "gate_override_reason": args.force_reason,
        "coordinator_frozen_at": str(coord_path),
    })
    run_report.setdefault("precisions", [])
    run_report["precisions"] = sorted(set(run_report["precisions"]) | set(precisions),
                                      key=lambda p: PRECISION_ORDER.index(p)
                                      if p in PRECISION_ORDER else 99)
    run_report.setdefault("question_order", {})
    run_report.setdefault("blocks", {})
    run_report.setdefault("failures", [])
    run_report.setdefault("harness_failures", [])
    if args.force and not args.force_reason:
        print("[run] WARNING --force used with no --force-reason; the override is "
              "recorded but undocumented.")

    sampling = make_sampling(models_cfg, budgets)
    prov = read_json(rd / "provenance.json", default={}) or {}
    prov.setdefault("models", {})
    prov["embedding"] = embedder.provenance()
    prov["role_max_tokens"] = budgets

    for precision in precisions:
        ordered = counterbalanced(list(questions), precision)
        run_report["question_order"][precision] = [q["question_id"] for q in ordered]

        model_spec: Optional[ModelSpec] = None
        if backend_kind not in ("fixture", "mock") and not args.dry_run:
            mpath = resolve(models_cfg["models"][precision]["path"])
            # Hashing 6.2 GB used to happen inside the block timer, charging
            # F16 tens of seconds of disk I/O that were then divided into
            # quality_per_second. It is cached and done before the timer now.
            digest = sha256_file_cached(mpath)
            model_spec = ModelSpec(precision=precision, path=str(mpath),
                                   sha256=digest,
                                   size_bytes=Path(mpath).stat().st_size)
            prov["models"].setdefault(precision, {})
            prov["models"][precision].update(
                {"path": str(mpath), "sha256": model_spec.sha256,
                 "size_bytes": model_spec.size_bytes, "present": True})
        elif backend_kind in ("fixture", "mock"):
            model_spec = ModelSpec(precision=precision, path="FIXTURE",
                                   sha256="FIXTURE", size_bytes=0)

        if args.dry_run:
            class _NullBackend:
                is_fixture = True
                name = "dry-run"

                def start(self):
                    return None

                def stop(self):
                    return None

                def warmup(self, max_tokens: int = 8):
                    return {"performed": False, "reason": "dry run"}

                def provenance(self):
                    return {"backend": "dry-run", "is_fixture": True,
                            "sampling_hash": sampling.hash()}

                def chat(self, *a, **k):
                    raise RuntimeError("dry run: no model calls allowed")
            backend = _NullBackend()
        else:
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

        client = GuardedClient(backend, log, args.stage, precision, condition,
                               dry_run=args.dry_run)
        n_run = n_resumed = 0
        block_t0 = None
        warm = {}
        try:
            # start() belongs INSIDE the try: it used to sit outside, so a
            # startup timeout left an already-Popen'ed server running with
            # stop() never called.
            # start() performs the discarded warmup itself (warmup_on_start).
            backend.start()
            warm = (backend.provenance() or {}).get("warmup") or {}
            block_t0 = time.time()  # after load, hashing and warmup

            # The coordinator is produced once, at F16, and frozen for every
            # precision. Running it per-precision made its output -- which is
            # precision-dependent by construction -- feed every document-agent
            # prompt, so `prompt_hash_parity` could never pass; and the JSON
            # fallback then rescued whichever precision was failing hardest.
            if precision == coord_mod.FREEZE_PRECISION and not args.dry_run:
                for q in questions:
                    coord_mod.freeze_coordinator(
                        coord_path, client, q["question_id"], q["question"],
                        max_tokens=budgets.get("coordinator"))
                print(f"[run] coordinator frozen at {coord_mod.FREEZE_PRECISION} "
                      f"for {len(questions)} questions -> {coord_path}")

            for q in ordered:
                key = (q["question_id"], precision, condition)
                if key in done:
                    n_resumed += 1
                    print(f"[run] resume-skip {key}")
                    continue
                try:
                    if args.dry_run:
                        coord = {"shared_tasks": [], "synthesis_directive": "",
                                 "dry_run": True}
                    else:
                        coord = coord_mod.load_frozen_coordinator(
                            coord_path, q["question_id"])
                    pred = run_question(client, store, cache, coord, q, condition,
                                        rcfg, sig, budgets, args.dry_run)
                except BackendGenerationError as exc:
                    # Infrastructure failure, not model quality. The failing
                    # call_id is deliberately not in the resume index, so a
                    # rerun genuinely retries it.
                    run_report["failures"].append(
                        {"question_id": q["question_id"], "precision": precision,
                         "kind": "backend_generation_error", "error": str(exc),
                         "call_id": getattr(exc, "call_id", None)})
                    log.append({"call_id": "", "stage": args.stage,
                                "precision": precision, "condition": condition,
                                "role": "question_failure",
                                "question_id": q["question_id"],
                                "is_failure_record": True,
                                "failure_kind": "backend_generation_error",
                                "error": str(exc)})
                    print(f"[run] BACKEND ERROR {q['question_id']} @ {precision}: {exc}")
                    continue
                except Exception as exc:  # never a silent skip
                    run_report["failures"].append(
                        {"question_id": q["question_id"], "precision": precision,
                         "kind": "question_failure",
                         "error": f"{type(exc).__name__}: {exc}"})
                    # `is_failure_record`, NOT `is_fixture`. Tagging a failure
                    # as fixture contamination made one transient timeout
                    # permanently void the stage in an append-only log.
                    log.append({"call_id": stable_id("failure", args.stage, precision,
                                                     condition, q["question_id"]),
                                "stage": args.stage, "precision": precision,
                                "condition": condition, "role": "question_failure",
                                "question_id": q["question_id"],
                                "is_failure_record": True,
                                "failure_kind": "question_failure",
                                "error": f"{type(exc).__name__}: {exc}"})
                    print(f"[run] FAILED {q['question_id']} @ {precision}: {exc}")
                    continue
                if hasattr(backend, "provenance"):
                    pb = backend.provenance()
                    pred["backend"] = pb.get("backend")
                    pred["sampling_hash"] = pb.get("sampling_hash")
                    pred["model_sha256"] = (pb.get("model") or {}).get("sha256", "")
                predictions.append(pred)
                done.add(key)
                n_run += 1
                write_json(stage_dir / "predictions.json", predictions)
                print(f"[run] {precision} {q['question_id']} (w={q['width']}) "
                      f"{pred['wall_s']:.1f}s -> {pred['final_answer'][:80]!r}")
            cache.flush()
        finally:
            inference_wall = round(time.time() - block_t0, 2) if block_t0 else None
            if hasattr(backend, "stop"):
                backend.stop()
            released = free_gpu_memory()
            # provenance AFTER stop: `memory` is only populated by stop(), so
            # reading it first meant peak RAM and peak VRAM were always empty.
            block_prov = backend.provenance() if hasattr(backend, "provenance") else {}
            run_report["blocks"][precision] = {
                "inference_wall_s": inference_wall,
                "n_questions_run": n_run,
                "n_resumed_questions": n_resumed,
                "warmup": warm,
                "provenance": block_prov,
                "gpu_offload": block_prov.get("gpu_offload"),
                "memory": block_prov.get("memory"),
                "memory_is_final": block_prov.get("memory_is_final"),
                "load_time_s": block_prov.get("load_time_s"),
                "gpu_release": released,
            }

    run_report["finished_at"] = time.time()
    run_report["completed_predictions"] = len(predictions)
    run_report["complete"] = (len(predictions) >= expected
                              and not run_report["failures"])
    run_report["corrupt_event_lines"] = getattr(log, "corrupt_line_count", 0)
    if run_report["corrupt_event_lines"]:
        run_report["harness_failures"].append(
            f"{run_report['corrupt_event_lines']} corrupt line(s) in events.jsonl")
    write_json(stage_dir / "run_report.json", run_report)
    write_json(stage_dir / "predictions.json", predictions)
    write_json(rd / "provenance.json", prov)

    print(f"[run] {len(predictions)}/{expected} predictions -> {stage_dir}")
    if run_report["failures"]:
        print(f"[run] {len(run_report['failures'])} question failures recorded "
              "(NOT skipped silently); see run_report.json")
    if args.force:
        print(f"[run] NOTE gate override recorded: {args.force_reason or '(no reason given)'}")
    if harness_only:
        print("[run] NOTE: this run is watermarked non-scientific "
              "(dry run or fixture backend). It can prove the harness only, and "
              f"its artifacts live in {stage_dir} so a real run cannot resume them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
