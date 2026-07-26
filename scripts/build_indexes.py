#!/usr/bin/env python3
"""Build one private index per manifest document, freeze the fixed evidence,
and prove document isolation before any model is loaded."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_models_config, results_dir  # noqa: E402
from evidence import (FixedEvidenceCache, build_store, embed_signature,  # noqa: E402
                      get_fixed_evidence, make_embedder)
from logging_utils import read_json, write_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="stage_a", choices=["stage_a", "stage_b", "all"])
    ap.add_argument("--models-config", default="configs/models.yaml")
    ap.add_argument("--manifest", default="data/pilot_manifest.json")
    ap.add_argument("--documents", default="data/pilot_documents.json")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--chunk-tokens", type=int, default=400)
    ap.add_argument("--overlap-tokens", type=int, default=50)
    args = ap.parse_args()

    rd = results_dir(args.results_dir)
    manifest = read_json(ROOT / args.manifest)
    documents = read_json(ROOT / args.documents)
    if manifest is None or documents is None:
        raise SystemExit("run scripts/prepare_dataset.py first")

    stages = ["stage_a", "stage_b"] if args.stage == "all" else [args.stage]
    questions = [q for s in stages for q in manifest["stages"][s]["questions"]]
    doc_ids = sorted({d["document_id"] for q in questions for d in q["documents"]})

    models_cfg = load_models_config(args.models_config)
    embedder = make_embedder(models_cfg, str(rd / "embed_cache"))
    print(f"[index] embedder: {embedder.provenance()}")
    store = build_store(documents, embedder, doc_ids, args.chunk_tokens,
                        args.overlap_tokens)
    print(f"[index] built {len(doc_ids)} private indexes, "
          f"{sum(len(store.get(d).chunks) for d in doc_ids)} chunks")

    iso = store.assert_isolation()
    print(f"[index] isolation: {iso['isolated']} ({iso['documents']} documents)")
    if not iso["isolated"]:
        write_json(rd / "isolation_report.json", iso)
        raise SystemExit(f"DOCUMENT ISOLATION VIOLATED: {iso['violations'][:3]}")

    cache = FixedEvidenceCache(rd / "fixed_evidence.json")
    sig = embed_signature(embedder)
    n = 0
    for q in questions:
        for d in q["documents"]:
            get_fixed_evidence(store, cache, q["question_id"], q["question"],
                               d["document_id"], args.top_k, args.chunk_tokens,
                               args.overlap_tokens, sig)
            n += 1
    cache.flush()
    write_json(rd / "isolation_report.json", iso)
    write_json(rd / "index_provenance.json", {
        "stages": stages, "documents": len(doc_ids), "fixed_evidence_entries": n,
        "top_k": args.top_k, "chunk_tokens": args.chunk_tokens,
        "overlap_tokens": args.overlap_tokens, "embed_signature": sig,
        **embedder.provenance()})
    print(f"[index] froze {n} fixed-evidence entries -> {rd / 'fixed_evidence.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
