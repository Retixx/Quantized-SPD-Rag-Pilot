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

Two exclusions, both recorded in the manifest:
  * question_type in EXCLUDED_TYPES  -- null_query is unanswerable by construction;
  * normalised gold answer in EXCLUDED_GOLD_ANSWERS -- ~60% of MultiHop-RAG's
    answerable questions answer literally "yes" or "no", which collapses answer
    diversity to a handful of strings and makes exact-match/F1 meaningless at
    pilot sample sizes. Pass --include-yes-no to keep them.

The question set is frozen the moment this file is written. Never regenerate it
after looking at model outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from logging_utils import sha256_text, write_json  # noqa: E402

SALT = "quantized-spd-rag-pilot/v1"
EXCLUDED_TYPES = {"null_query"}          # unanswerable by construction
EXCLUDED_GOLD_ANSWERS = {"yes", "no"}    # degenerate answer space, see module docstring

_PUNCT = re.compile(r"[^a-z0-9 ]+")
_SPACE = re.compile(r"\s+")


def normalize_question_type(raw: Any) -> str:
    return _SPACE.sub(" ", str(raw or "").strip().lower())


def normalize_answer(raw: Any) -> str:
    """Case-folded, punctuation-stripped answer used only for the exclusion and
    diversity statistics. The manifest still carries the verbatim answer."""
    return _SPACE.sub(" ", _PUNCT.sub(" ", str(raw or "").strip().lower())).strip()


def doc_identity(rec: Dict[str, Any]) -> str:
    """URL first, title as fallback.

    Strip BEFORE the fallback: a whitespace-only url is truthy, so
    `(rec.get("url") or rec.get("title") or "").strip()` would return "" and
    silently drop the evidence item, under-deriving the question's width.
    """
    return (rec.get("url") or "").strip() or (rec.get("title") or "").strip()


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


def build_corpus_index(corpus: List[Dict[str, Any]]
                       ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Index the corpus by document identity and REPORT the collisions.

    Repeated identities resolve last-record-wins (unchanged behaviour), but the
    count, and in particular the subset whose bodies actually differ, is now
    counted, printed and recorded in the manifest instead of vanishing.
    """
    by_identity: Dict[str, Dict[str, Any]] = {}
    seen = Counter()
    body_conflicts: List[Dict[str, str]] = []
    n_no_identity = 0
    for rec in corpus:
        ident = doc_identity(rec)
        if not ident:
            n_no_identity += 1
            continue
        seen[ident] += 1
        body_sha = sha256_text(rec.get("body", ""))
        prev = by_identity.get(ident)
        if prev is not None and prev["body_sha256"] != body_sha:
            body_conflicts.append({"identity": ident,
                                   "previous_body_sha256": prev["body_sha256"],
                                   "replacement_body_sha256": body_sha})
        by_identity[ident] = {
            "document_id": doc_id(ident), "identity": ident,
            "title": rec.get("title", ""), "url": rec.get("url", ""),
            "source": rec.get("source", ""), "author": rec.get("author", ""),
            "category": rec.get("category", ""),
            "published_at": rec.get("published_at", ""),
            "body": rec.get("body", ""),
            "body_sha256": body_sha,
        }
    duplicated = {ident: n for ident, n in seen.items() if n > 1}
    report = {
        "corpus_records": len(corpus),
        "records_without_identity": n_no_identity,
        "unique_identities": len(by_identity),
        "duplicate_identities": len(duplicated),
        "duplicate_records_dropped": sum(n - 1 for n in duplicated.values()),
        "duplicate_identities_with_differing_body": len(body_conflicts),
        "resolution": "last record wins",
        "examples": [{"identity": i, "records": duplicated[i]}
                     for i in sorted(duplicated)[:10]],
        "body_conflict_examples": body_conflicts[:10],
    }
    print(f"[dataset] corpus identities: {report['unique_identities']} unique, "
          f"{report['duplicate_identities']} duplicated "
          f"({report['duplicate_records_dropped']} records dropped, last wins), "
          f"{report['records_without_identity']} without any identity")
    if report["duplicate_identities_with_differing_body"]:
        print("[dataset] WARNING: "
              f"{report['duplicate_identities_with_differing_body']} duplicated "
              "identities carry DIFFERENT bodies; the last record won. "
              f"First: {body_conflicts[0]['identity']!r}")
    return by_identity, report


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


def answer_diversity(questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Unique normalised gold answers / number of questions, plus the worst
    offenders. Printed and recorded so a collapse can never go unnoticed."""
    norm = [normalize_answer(q.get("answer")) for q in questions]
    counts = Counter(norm)
    n = len(norm)
    top = counts.most_common(5)
    return {
        "n_questions": n,
        "n_unique_answers": len(counts),
        "unique_answer_ratio": round(len(counts) / n, 4) if n else 0.0,
        "max_answer_share": round(top[0][1] / n, 4) if n else 0.0,
        "most_common_answers": [{"answer": a, "count": c} for a, c in top],
    }


def build_candidates(questions, corpus_by_identity, excluded_answers: set
                     ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cands: List[Dict[str, Any]] = []
    type_counts = Counter()
    answer_counts = Counter()
    n_excluded_type = 0
    n_excluded_answer = 0
    n_no_question = 0
    for row in questions:
        q = (row.get("query") or "").strip()
        if not q:
            n_no_question += 1
            continue
        qtype = normalize_question_type(row.get("question_type"))
        type_counts[qtype] += 1
        if qtype in EXCLUDED_TYPES:
            n_excluded_type += 1
            continue
        norm_answer = normalize_answer(row.get("answer"))
        answer_counts[norm_answer] += 1
        if norm_answer in excluded_answers:
            n_excluded_answer += 1
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

    # A schema change (renamed/re-cased question_type) must not turn the
    # null_query filter into a silent no-op.
    if n_excluded_type == 0:
        raise SystemExit(
            "no question was excluded by question_type, but "
            f"{sorted(EXCLUDED_TYPES)} must match something in this dataset. "
            f"Observed normalised question_type values: {sorted(type_counts)}. "
            "The schema changed; fix EXCLUDED_TYPES before generating a manifest.")
    if excluded_answers and n_excluded_answer == 0:
        print(f"[dataset] WARNING: gold-answer exclusion {sorted(excluded_answers)} "
              "matched nothing. Either the dataset changed or normalisation broke; "
              f"most common normalised answers: {answer_counts.most_common(5)}")

    exclusions = {
        "rows_without_question_text": n_no_question,
        "excluded_question_types": sorted(EXCLUDED_TYPES),
        "excluded_by_question_type": n_excluded_type,
        "excluded_gold_answers": sorted(excluded_answers),
        "excluded_by_gold_answer": n_excluded_answer,
        "question_type_counts": dict(sorted(type_counts.items())),
        "candidate_pool_answer_diversity": answer_diversity(cands),
    }
    print("[dataset] question_type counts:", exclusions["question_type_counts"])
    print(f"[dataset] excluded {n_excluded_type} by question_type "
          f"{sorted(EXCLUDED_TYPES)}, {n_excluded_answer} by gold answer "
          f"{sorted(excluded_answers) or '(none)'}")
    print("[dataset] usable candidates by width:",
          dict(Counter(c["width"] for c in cands)))
    div = exclusions["candidate_pool_answer_diversity"]
    print(f"[dataset] candidate pool answer diversity: {div['n_unique_answers']}"
          f"/{div['n_questions']} unique ({div['unique_answer_ratio']})")
    return cands, exclusions


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


def build_manifest(cands: List[Dict[str, Any]], exclusions: Dict[str, Any],
                   corpus_report: Dict[str, Any]) -> Dict[str, Any]:
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

    # order_key is KEPT on every question: the ordering is deterministic, and it
    # must also be auditable straight from the artefact without re-running this
    # script against the raw dataset.
    excluded_answers = exclusions["excluded_gold_answers"]
    return {
        "manifest_version": 2,
        "salt": SALT,
        "benchmark": {
            "name": "MultiHop-RAG",
            "dataset": "https://huggingface.co/datasets/yixuantt/MultiHopRAG",
            "repository": "https://github.com/yixuantt/MultiHop-RAG",
            "excluded_question_types": sorted(EXCLUDED_TYPES),
            "excluded_gold_answers": excluded_answers,
            "excluded_gold_answers_rationale": (
                "~60% of MultiHop-RAG's answerable questions have the literal gold "
                "answer 'yes' or 'no'. At pilot sample sizes that collapses the "
                "answer space to a handful of strings, makes exact match and token "
                "F1 uninformative, and lets a degenerate model score well by "
                "guessing. Disable with --include-yes-no."),
            "width_range": [2, 4],
        },
        "exclusions": exclusions,
        "corpus_index_report": corpus_report,
        "sampling_rule": (
            "Exclude question_type in " + str(sorted(EXCLUDED_TYPES)) +
            " (normalised: stripped and case-folded) and gold answers normalising "
            "to " + str(excluded_answers) + ". Keep questions whose evidence "
            "resolves to 2-4 distinct corpus documents. Order by "
            "sha256(SALT|question) ascending, tie-break question_id. Fill Stage A "
            "first, then Stage B from the remaining pool (disjoint). Deterministic "
            "on any machine; every question carries its order_key."),
        "pool_size": len(cands),
        "pool_by_width": {str(w): len(v) for w, v in sorted(by_width.items())},
        "answer_diversity": {
            "stage_a": answer_diversity(stage_a),
            "stage_b": answer_diversity(stage_b),
            "stage_a_and_b": answer_diversity(stage_a + stage_b),
        },
        "stages": {
            "stage_a": {"expected_questions": 6, "questions": stage_a},
            "stage_b": {"expected_questions": 12, "questions": stage_b},
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


def gold_reviewed(path: Path) -> bool:
    """True if any entry in an existing rubric has had needs_manual_review cleared."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    for q in (data.get("questions") or {}).values():
        if q.get("needs_manual_review") is False:
            return True
        for f in q.get("facts") or []:
            if f.get("needs_manual_review") is False:
                return True
    return False


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
                    help="overwrite an existing manifest OR gold-fact rubric "
                         "(refuses by default)")
    ap.add_argument("--force-reviewed-gold", action="store_true",
                    help="also overwrite a gold_facts.json that has already been "
                         "hand-reviewed. Destroys manual review work.")
    ap.add_argument("--include-yes-no", action="store_true",
                    help="keep questions whose gold answer normalises to yes/no. "
                         "They are EXCLUDED by default: ~60%% of MultiHop-RAG's "
                         "answerable questions are yes/no, which collapses the "
                         "answer space at pilot sample sizes.")
    ap.add_argument("--exclude-answer", action="append", default=None,
                    metavar="ANSWER",
                    help="exclude questions whose normalised gold answer equals "
                         "this. Repeatable. Replaces the default set "
                         f"{sorted(EXCLUDED_GOLD_ANSWERS)} entirely.")
    ap.add_argument("--min-answer-diversity", type=float, default=0.5,
                    help="warn if unique normalised answers / questions falls "
                         "below this in the final Stage A+B set (default 0.5). "
                         "For reference: MultiHop-RAG has only ~100 distinct gold "
                         "answers across 2,255 answerable questions, so perfect "
                         "diversity is not attainable; this catches a COLLAPSE.")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    man_path = root / args.out_manifest
    gold_path = root / args.out_gold

    # --force guards BOTH frozen artefacts. A missing manifest next to a
    # hand-reviewed gold_facts.json must not silently clobber the rubric.
    existing = [p for p in (man_path, gold_path) if p.exists()]
    if existing and not args.force:
        raise SystemExit(
            f"refusing to overwrite: {', '.join(str(p) for p in existing)}. "
            "The question set is frozen once written and the gold-fact rubric may "
            "already be hand-reviewed; pass --force only if no model output has "
            "been generated yet.")
    if gold_path.exists() and gold_reviewed(gold_path) and not args.force_reviewed_gold:
        raise SystemExit(
            f"{gold_path} contains hand-reviewed entries (needs_manual_review "
            "cleared). Regenerating would replace them with fresh unreviewed "
            "seeds. Back it up, then pass --force-reviewed-gold to proceed.")

    if args.include_yes_no and args.exclude_answer:
        raise SystemExit("--include-yes-no and --exclude-answer are mutually exclusive")
    if args.include_yes_no:
        excluded_answers: set = set()
        print("[dataset] WARNING: --include-yes-no is set; yes/no gold answers are "
              "kept and answer diversity will be poor.")
    elif args.exclude_answer:
        excluded_answers = {normalize_answer(a) for a in args.exclude_answer} - {""}
    else:
        excluded_answers = set(EXCLUDED_GOLD_ANSWERS)

    ds = load_dataset(args.data_dir, args.search)
    corpus_by_identity, corpus_report = build_corpus_index(ds["corpus"])
    cands, exclusions = build_candidates(ds["questions"], corpus_by_identity,
                                         excluded_answers)
    manifest = build_manifest(cands, exclusions, corpus_report)
    manifest["source_files"] = ds["_paths"]
    manifest["answer_diversity"]["min_answer_diversity_threshold"] = \
        args.min_answer_diversity

    needed = {d["identity"] for st in ("stage_a", "stage_b")
              for q in manifest["stages"][st]["questions"] for d in q["documents"]}
    documents = {corpus_by_identity[i]["document_id"]: corpus_by_identity[i]
                 for i in sorted(needed)}

    write_json(man_path, manifest)
    write_json(root / args.out_corpus, documents)
    write_json(gold_path, build_gold_facts(manifest))

    n_facts = sum(len(v["facts"]) for v in
                  json.loads(gold_path.read_text())["questions"].values())
    print(f"[dataset] wrote {man_path}")
    print(f"[dataset] wrote {root / args.out_corpus} ({len(documents)} documents)")
    print(f"[dataset] wrote {gold_path} ({n_facts} seed gold facts, "
          "ALL flagged needs_manual_review=true)")
    for st in ("stage_a", "stage_b"):
        widths = Counter(q["width"] for q in manifest["stages"][st]["questions"])
        print(f"[dataset] {st}: {len(manifest['stages'][st]['questions'])} questions, "
              f"widths {dict(sorted(widths.items()))}")
    print("[dataset] answer diversity (unique normalised gold answers / questions):")
    for scope in ("stage_a", "stage_b", "stage_a_and_b"):
        div = manifest["answer_diversity"][scope]
        print(f"[dataset]   {scope}: {div['n_unique_answers']}/{div['n_questions']} "
              f"= {div['unique_answer_ratio']} (max single-answer share "
              f"{div['max_answer_share']}) | most common "
              f"{[(d['answer'][:40], d['count']) for d in div['most_common_answers']]}")
    overall = manifest["answer_diversity"]["stage_a_and_b"]
    if overall["n_questions"] and \
            overall["unique_answer_ratio"] < args.min_answer_diversity:
        print(f"[dataset] WARNING: answer diversity "
              f"{overall['unique_answer_ratio']} is below "
              f"--min-answer-diversity {args.min_answer_diversity}. The answer "
              "space is collapsing; widen --exclude-answer before running a model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
