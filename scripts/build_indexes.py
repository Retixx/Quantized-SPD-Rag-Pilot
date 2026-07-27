#!/usr/bin/env python3
"""Build one private index per manifest document, freeze the fixed evidence,
and prove document isolation before any model is loaded.

Retrieval parameters come from configs/stage_*.yaml, NEVER from argparse
defaults. They are part of the fixed-evidence cache key, so if this script
froze evidence at top_k=3 while run_stage.py read top_k=5 from stage_b.yaml,
every lookup would miss and the evidence would be silently re-derived at
runtime inside the F16 precision block -- destroying the parity guarantee this
script exists to establish. The CLI flags remain, as explicit overrides only,
and an override is recorded in index_provenance.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_models_config, load_stage_config, results_dir  # noqa: E402
from evidence import (FixedEvidenceCache, build_store, embed_signature,  # noqa: E402
                      get_fixed_evidence, make_embedder)
from logging_utils import read_json, write_json  # noqa: E402

RETRIEVAL_KEYS = ("top_k", "chunk_tokens", "overlap_tokens")


def stage_retrieval(stage: str, overrides: Dict[str, Optional[int]]
                    ) -> Dict[str, Any]:
    """The retrieval parameters run_stage.py will use for `stage`, plus any
    explicit CLI override, with the provenance of each value."""
    rcfg = (load_stage_config(stage) or {}).get("retrieval") or {}
    missing = [k for k in RETRIEVAL_KEYS if rcfg.get(k) is None]
    if missing:
        raise SystemExit(
            f"configs/{stage}.yaml is missing retrieval keys {missing}. These "
            "must be declared in the stage config: they are part of the "
            "fixed-evidence cache key and run_stage.py reads them from there.")
    resolved = {k: int(rcfg[k]) for k in RETRIEVAL_KEYS}
    sources = {k: f"configs/{stage}.yaml" for k in RETRIEVAL_KEYS}
    for k, v in overrides.items():
        if v is not None:
            resolved[k] = int(v)
            sources[k] = "CLI override"
    return {"stage": stage, "resolved": resolved, "sources": sources,
            "stage_config": {k: int(rcfg[k]) for k in RETRIEVAL_KEYS}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="stage_a", choices=["stage_a", "stage_b", "all"])
    ap.add_argument("--models-config", default="configs/models.yaml")
    ap.add_argument("--manifest", default="data/pilot_manifest.json")
    ap.add_argument("--documents", default="data/pilot_documents.json")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--top-k", type=int, default=None,
                    help="override configs/stage_*.yaml retrieval.top_k "
                         "(default: whatever the stage config says)")
    ap.add_argument("--chunk-tokens", type=int, default=None,
                    help="override configs/stage_*.yaml retrieval.chunk_tokens")
    ap.add_argument("--overlap-tokens", type=int, default=None,
                    help="override configs/stage_*.yaml retrieval.overlap_tokens")
    args = ap.parse_args()

    rd = results_dir(args.results_dir)
    manifest = read_json(ROOT / args.manifest)
    documents = read_json(ROOT / args.documents)
    if manifest is None or documents is None:
        raise SystemExit("run scripts/prepare_dataset.py first")

    stages = ["stage_a", "stage_b"] if args.stage == "all" else [args.stage]
    overrides = {"top_k": args.top_k, "chunk_tokens": args.chunk_tokens,
                 "overlap_tokens": args.overlap_tokens}
    retrieval = {s: stage_retrieval(s, overrides) for s in stages}
    for s, r in retrieval.items():
        overridden = [k for k, v in r["sources"].items() if v == "CLI override"]
        print(f"[index] {s} retrieval: {r['resolved']}"
              + (f" (CLI override: {overridden})" if overridden else ""))
    distinct = {tuple(sorted(r["resolved"].items())) for r in retrieval.values()}
    if len(distinct) > 1:
        print("[index] NOTE: stages declare different retrieval parameters; a "
              "separate index and a separate frozen-evidence entry is built per "
              "stage. This is supported, but the stages are no longer comparable "
              "to each other on retrieval.")

    models_cfg = load_models_config(args.models_config)
    embedder = make_embedder(models_cfg, str(rd / "embed_cache"))
    print(f"[index] embedder: {embedder.provenance()}")
    cache = FixedEvidenceCache(rd / "fixed_evidence.json")
    sig = embed_signature(embedder)

    all_doc_ids: List[str] = []
    isolation: Dict[str, Any] = {}
    per_stage: Dict[str, Any] = {}
    n_total = 0
    for stage in stages:
        rparams = retrieval[stage]["resolved"]
        questions = manifest["stages"][stage]["questions"]
        doc_ids = sorted({d["document_id"] for q in questions
                          for d in q["documents"]})
        store = build_store(documents, embedder, doc_ids,
                            rparams["chunk_tokens"], rparams["overlap_tokens"])
        print(f"[index] {stage}: built {len(doc_ids)} private indexes, "
              f"{sum(len(store.get(d).chunks) for d in doc_ids)} chunks")

        iso = store.assert_isolation()
        print(f"[index] {stage} isolation: {iso['isolated']} "
              f"({iso['documents']} documents)")
        if not iso["isolated"]:
            write_json(rd / "isolation_report.json", iso)
            raise SystemExit(f"DOCUMENT ISOLATION VIOLATED: {iso['violations'][:3]}")
        isolation = iso if not isolation else {
            "isolated": isolation["isolated"] and iso["isolated"],
            "documents": isolation["documents"] + iso["documents"],
            "violations": (isolation.get("violations") or [])
            + (iso.get("violations") or []),
        }

        n = 0
        for q in questions:
            for d in q["documents"]:
                get_fixed_evidence(store, cache, q["question_id"], q["question"],
                                   d["document_id"], rparams["top_k"],
                                   rparams["chunk_tokens"],
                                   rparams["overlap_tokens"], sig)
                n += 1
        n_total += n
        all_doc_ids += doc_ids
        per_stage[stage] = {"documents": len(doc_ids), "fixed_evidence_entries": n,
                            **retrieval[stage]}

    cache.flush()
    write_json(rd / "isolation_report.json", isolation)
    # Flat keys only when every stage agrees, so a reader can never mistake one
    # stage's parameters for the whole run's.
    flat = dict(next(iter(retrieval.values()))["resolved"]) if len(distinct) == 1 else {}
    write_json(rd / "index_provenance.json", {
        "stages": stages, "documents": len(set(all_doc_ids)),
        "fixed_evidence_entries": n_total,
        **flat,
        "retrieval_by_stage": per_stage,
        "retrieval_parameter_source": (
            "configs/stage_*.yaml, the same file run_stage.py reads; CLI flags "
            "are explicit overrides and are labelled as such above"),
        "embed_signature": sig, **embedder.provenance()})
    print(f"[index] froze {n_total} fixed-evidence entries -> "
          f"{rd / 'fixed_evidence.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
