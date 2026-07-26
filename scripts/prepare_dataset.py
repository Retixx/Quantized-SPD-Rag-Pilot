#!/usr/bin/env python3
"""Build the deterministic pilot manifest and the seed gold-fact rubric.

Benchmark: MultiHop-RAG (yixuantt/MultiHopRAG, 2,556 questions, 609 corpus docs).
  MultiHopRAG.json rows: {query, answer, question_type, evidence_list:[{title,url,
                          source,author,category,published_at,fact}]}
  corpus.json rows:      {title, body, url, source, author, category, published_at}

Document identity: the article URL (falling back to the title). "Width" = the
number of DISTINCT evidence documents for a question, derived from those unique
identifiers; the derivation is logged into the manifest.

Sampling is a deterministic hash order over question text, so the same manifest
is produced on any machine. Stage A and Stage B pools are disjoint: calibration
items never re-enter the main precision-by-width test.

The question set is frozen the moment this file is written. Never regenerate it
after looking at model outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from logging_utils import sha256_text, write_json  # noqa: E402

SALT = "quantized-spd-rag-pilot/v1"
EXCLUDED_TYPES = {"null_query"}   # unanswerable by construction


def doc_identity(rec: Dict[str, Any]) -> str:
    return (rec.get("url") or rec.get("title") or "").strip()


def doc_id(identity: str) -> str:
    return "doc_" + hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]


def find_file(candidates: List[str], name: str) -> Optional[Path]:
    for c in candidates:
        p = Path(c)
        if p.is_file() and p.name.lower() == name.lower():
            return p
        if p.is_dir():
            for hit in sorted(p.rglob(name)):
                return hit
    return None


def load_dataset(data_dir: str, extra_search: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    search = [data_dir, *extra_search, "/kaggle/input", "data", "."]
    q_path = find_file(search, "MultiHopRAG.json")
    c_path = find_file(search, "corpus.json")
    if not q_path or not c_path:
        raise SystemExit(
            "Could not locate MultiHopRAG.json and corpus.json.\n"
            "Download them from https://huggingface.co/datasets/yixuantt/MultiHopRAG\n"
            "(or https://github.com/yixuantt/MultiHop-RAG) and pass --data-dir, or "
            "attach them as a Kaggle dataset under /kaggle/input.\n"
            f"Searched: {search}")
    with open(q_path, "r", encoding="utf-8") as fh:
        questions = json.load(fh)
    with open(c_path, "r", encoding="utf-8") as fh:
        corpus = json.load(fh)
    print(f"[dataset] questions={len(questions)} from {q_path}")
    print(f"[dataset] corpus={len(corpus)} from {c_path}")
    return {"questions": questions, "corpus": corpus, "_paths":
            [str(q_path), str(c_path)]}


def build_corpus_index(corpus: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_identity: Dict[str, Dict[str, Any]] = {}
    for rec in corpus:
        ident = doc_identity(rec)
        if not ident:
            continue
        by_identity[ident] = {
            "document_id": doc_id(ident), "identity": ident,
            "title": rec.get("title", ""), "url": rec.get("url", ""),
            "source": rec.get("source", ""), "author": rec.get("author", ""),
            "category": rec.get("category", ""),
            "published_at": rec.get("published_at", ""),
            "body": rec.get("body", ""),
            "body_sha256": sha256_text(rec.get("body", "")),
        }
    return by_identity


def derive_width(evidence_list: List[Dict[str, Any]],
                 corpus_by_identity: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Width = number of unique evidence document identifiers, with the
    derivation recorded so the count is auditable."""
    idents: List[str] = []
    facts_by_ident: Dict[str, List[str]] = defaultdict(list)
    for ev in evidence_list or []:
        ident = doc_identity(ev)
        if not ident:
            continue
        if ident not in idents:
            idents.append(ident)
        fact = (ev.get("fact") or "").strip()
        if fact:
            facts_by_ident[ident].append(fact)
    resolved = [i for i in idents if i in corpus_by_identity]
    missing = [i for i in idents if i not in corpus_by_identity]
    return {
        "width": len(resolved),
        "evidence_identities": idents,
        "resolved_identities": resolved,
        "unresolved_identities": missing,
        "facts_by_identity": {k: v for k, v in facts_by_ident.items()},
        "derivation": ("width = count of unique evidence 'url' (fallback 'title') "
                       "values in evidence_list that resolve to a corpus document"),
    }


def order_key(question_text: str) -> str:
    return hashlib.sha256((SALT + "|" + question_text).encode("utf-8")).hexdigest()


def build_candidates(questions, corpus_by_identity) -> List[Dict[str, Any]]:
    cands: List[Dict[str, Any]] = []
    type_counts = Counter()
    for row in questions:
        q = (row.get("query") or "").strip()
        if not q:
            continue
        qtype = row.get("question_type") or ""
        type_counts[qtype] += 1
        if qtype in EXCLUDED_TYPES:
            continue
        d = derive_width(row.get("evidence_list") or [], corpus_by_identity)
        if d["unresolved_identities"] or d["width"] < 2 or d["width"] > 4:
            continue
        qid = "q_" + hashlib.sha1(q.encode("utf-8")).hexdigest()[:12]
        docs = []
        for ident in d["resolved_identities"]:
            meta = corpus_by_identity[ident]
            docs.append({"document_id": meta["document_id"], "identity": ident,
                         "title": meta["title"], "url": meta["url"],
                         "source": meta["source"],
                         "body_sha256": meta["body_sha256"],
                         "gold_facts": d["facts_by_identity"].get(ident, [])})
        cands.append({
            "question_id": qid, "question": q,
            "answer": (row.get("answer") or "").strip(),
            "question_type": qtype, "width": d["width"], "documents": docs,
            "width_derivation": {k: d[k] for k in
                                 ("derivation", "evidence_identities",
                                  "resolved_identities", "unresolved_identities")},
            "order_key": order_key(q),
        })
    cands.sort(key=lambda c: (c["order_key"], c["question_id"]))
    print("[dataset] question_type counts:", dict(type_counts))
    print("[dataset] usable candidates by width:",
          dict(Counter(c["width"] for c in cands)))
    return cands


def take(pool: List[Dict[str, Any]], n: int, used: set) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in pool:
        if len(out) >= n:
            break
        if c["question_id"] in used:
            continue
        out.append(c)
        used.add(c["question_id"])
    if len(out) < n:
        raise SystemExit(f"not enough candidates: needed {n}, got {len(out)}")
    return out


def build_manifest(cands: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_width: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for c in cands:
        by_width[c["width"]].append(c)
    used: set = set()

    # Stage A: 3 x width-2, 3 x width-(3 or 4)  -> prefer a 2/1 split of 3 and 4
    stage_a = take(by_width[2], 3, used)
    high = take(by_width[3], 2, used) + take(by_width[4], 1, used)
    stage_a += high

    # Stage B: 4 x width-2, 4 x width-3, 4 x width-4, disjoint from Stage A
    stage_b = (take(by_width[2], 4, used) + take(by_width[3], 4, used)
               + take(by_width[4], 4, used))

    def strip(c: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in c.items() if k != "order_key"}

    return {
        "manifest_version": 1,
        "salt": SALT,
        "benchmark": {
            "name": "MultiHop-RAG",
            "dataset": "https://huggingface.co/datasets/yixuantt/MultiHopRAG",
            "repository": "https://github.com/yixuantt/MultiHop-RAG",
            "excluded_question_types": sorted(EXCLUDED_TYPES),
            "width_range": [2, 4],
        },
        "sampling_rule": (
            "Exclude null_query. Keep questions whose evidence resolves to 2-4 "
            "distinct corpus documents. Order by sha256(SALT|question) ascending, "
            "tie-break question_id. Fill Stage A first, then Stage B from the "
            "remaining pool (disjoint). Deterministic on any machine."),
        "pool_size": len(cands),
        "pool_by_width": {str(w): len(v) for w, v in sorted(by_width.items())},
        "stages": {
            "stage_a": {"expected_questions": 6,
                        "questions": [strip(c) for c in stage_a]},
            "stage_b": {"expected_questions": 12,
                        "questions": [strip(c) for c in stage_b]},
            "stage_c": {"expected_questions": 6,
                        "selection": "deterministic, from Stage B results; see configs/stage_c.yaml",
                        "questions": []},
        },
    }


def build_gold_facts(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Seed the rubric from the dataset's own evidence facts.

    EVERY entry is marked needs_manual_review: true. Review the rubric before
    quoting any number from a run as a scientific result.
    """
    out: Dict[str, Any] = {
        "_README": (
            "Seeded from MultiHop-RAG evidence_list[].fact. Each entry is a gold "
            "atomic fact attributed to one supporting document. Review and edit "
            "the text/aliases, then set needs_manual_review to false. Scoring "
            "still runs while the flag is true, but score.py stamps every report "
            "with rubric_reviewed=false."),
        "fact_match_threshold": 0.6,
        "questions": {},
    }
    for stage in ("stage_a", "stage_b"):
        for q in manifest["stages"][stage]["questions"]:
            facts: List[Dict[str, Any]] = []
            for di, doc in enumerate(q["documents"], 1):
                for fi, fact in enumerate(doc["gold_facts"], 1):
                    facts.append({
                        "fact_id": f"{doc['document_id']}#f{di}_{fi}",
                        "document_id": doc["document_id"],
                        "text": fact,
                        "aliases": [],
                        "required": True,
                        "needs_manual_review": True,
                    })
            out["questions"][q["question_id"]] = {
                "question": q["question"],
                "stage": stage,
                "width": q["width"],
                "question_type": q["question_type"],
                "answer": q["answer"],
                "answer_aliases": [],
                "facts": facts,
                "needs_manual_review": True,
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data",
                    help="directory holding MultiHopRAG.json and corpus.json")
    ap.add_argument("--search", nargs="*", default=[],
                    help="extra directories to search")
    ap.add_argument("--out-manifest", default="data/pilot_manifest.json")
    ap.add_argument("--out-gold", default="data/gold_facts.json")
    ap.add_argument("--out-corpus", default="data/pilot_documents.json",
                    help="the document bodies actually needed by the manifest")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing manifest (refuses by default)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    man_path = root / args.out_manifest
    if man_path.exists() and not args.force:
        raise SystemExit(
            f"{man_path} already exists. The question set is frozen once written; "
            "pass --force only if no model output has been generated yet.")

    ds = load_dataset(args.data_dir, args.search)
    corpus_by_identity = build_corpus_index(ds["corpus"])
    cands = build_candidates(ds["questions"], corpus_by_identity)
    manifest = build_manifest(cands)
    manifest["source_files"] = ds["_paths"]

    needed = {d["identity"] for st in ("stage_a", "stage_b")
              for q in manifest["stages"][st]["questions"] for d in q["documents"]}
    documents = {corpus_by_identity[i]["document_id"]: corpus_by_identity[i]
                 for i in sorted(needed)}

    write_json(man_path, manifest)
    write_json(root / args.out_corpus, documents)
    write_json(root / args.out_gold, build_gold_facts(manifest))

    n_facts = sum(len(v["facts"]) for v in
                  json.loads((root / args.out_gold).read_text())["questions"].values())
    print(f"[dataset] wrote {man_path}")
    print(f"[dataset] wrote {root / args.out_corpus} ({len(documents)} documents)")
    print(f"[dataset] wrote {root / args.out_gold} ({n_facts} seed gold facts, "
          "ALL flagged needs_manual_review=true)")
    for st in ("stage_a", "stage_b"):
        widths = Counter(q["width"] for q in manifest["stages"][st]["questions"])
        print(f"[dataset] {st}: {len(manifest['stages'][st]['questions'])} questions, "
              f"widths {dict(sorted(widths.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
