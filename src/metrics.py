"""Partial-credit scoring and the branch/orchestration/synthesis metric set.

Everything here is deterministic and label-blind: no function takes a precision
argument, so a scorer cannot behave differently for F16 than for Q4_K_M.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Dict, List, Sequence, Set

_ARTICLES = {"a", "an", "the"}
_STOP = _ARTICLES | {
    "of", "in", "on", "at", "to", "for", "and", "or", "is", "are", "was", "were",
    "be", "been", "by", "with", "that", "this", "it", "its", "as", "from", "has",
    "have", "had", "but", "not", "which", "their", "there", "his", "her", "they",
}
_PUNCT = str.maketrans("", "", string.punctuation)
_NUM = re.compile(r"\b\d[\d,\.]*\b")


def normalize_answer(text: str) -> str:
    """SQuAD-style normalisation: lowercase, strip punctuation/articles/space."""
    s = (text or "").lower().translate(_PUNCT)
    return " ".join(w for w in s.split() if w not in _ARTICLES)


def exact_match(pred: str, gold: str, aliases: Sequence[str] = ()) -> float:
    p = normalize_answer(pred)
    golds = [gold, *aliases]
    return 1.0 if any(p == normalize_answer(g) for g in golds) else 0.0


def _tokens(text: str) -> List[str]:
    return normalize_answer(text).split()


def token_f1(pred: str, gold: str) -> float:
    p, g = _tokens(pred), _tokens(gold)
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    prec = overlap / len(p)
    rec = overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def char_f1(pred: str, gold: str, n: int = 3) -> float:
    def grams(s: str) -> Counter:
        s = normalize_answer(s).replace(" ", "")
        return Counter(s[i:i + n] for i in range(max(0, len(s) - n + 1)))
    p, g = grams(pred), grams(gold)
    if not p or not g:
        return float(p == g)
    overlap = sum((p & g).values())
    if overlap == 0:
        return 0.0
    prec = overlap / sum(p.values())
    rec = overlap / sum(g.values())
    return 2 * prec * rec / (prec + rec)


def contains_answer(pred: str, gold: str, aliases: Sequence[str] = ()) -> float:
    p = normalize_answer(pred)
    for g in [gold, *aliases]:
        gn = normalize_answer(g)
        if gn and gn in p:
            return 1.0
    return 0.0


def answer_score(pred: str, gold: str, aliases: Sequence[str] = ()) -> Dict[str, float]:
    """Partial-credit answer score. `score` is the headline number."""
    em = exact_match(pred, gold, aliases)
    cont = contains_answer(pred, gold, aliases)
    f1 = max([token_f1(pred, g) for g in [gold, *aliases]] or [0.0])
    cf1 = max([char_f1(pred, g) for g in [gold, *aliases]] or [0.0])
    score = max(em, cont, f1)
    return {"exact_match": em, "contains": cont, "token_f1": round(f1, 4),
            "char_f1": round(cf1, 4), "score": round(score, 4)}


# --------------------------------------------------------------------------
# atomic facts
# --------------------------------------------------------------------------


def content_tokens(text: str) -> List[str]:
    return [t for t in _tokens(text) if t not in _STOP]


def numbers_in(text: str) -> Set[str]:
    return {m.group(0).rstrip(".").replace(",", "") for m in _NUM.finditer(text or "")}


def fact_match_score(gold_fact: str, candidate: str,
                     require_numbers: bool = True) -> float:
    """How well `candidate` covers `gold_fact`, in [0, 1].

    Content-token recall of the gold fact, gated on numeric agreement: if the
    gold fact carries numbers or dates and none of them survive, the score is
    halved. Coarse by design -- every rubric entry ships with
    needs_manual_review=true until a human confirms it.
    """
    gold_toks = content_tokens(gold_fact)
    if not gold_toks:
        return 0.0
    cand = set(content_tokens(candidate))
    recall = sum(1 for t in gold_toks if t in cand) / len(gold_toks)
    if require_numbers:
        gnums = numbers_in(gold_fact)
        if gnums and not (gnums & numbers_in(candidate)):
            recall *= 0.5
    return round(recall, 4)


def fact_supported(gold_fact: str, candidate: str, threshold: float = 0.6) -> bool:
    return fact_match_score(gold_fact, candidate) >= threshold


def atomic_fact_recall(gold_facts: Sequence[Dict[str, Any]], text: str,
                       threshold: float = 0.6) -> Dict[str, Any]:
    matched: List[str] = []
    scores: Dict[str, float] = {}
    for gf in gold_facts:
        s = fact_match_score(gf["text"], text)
        scores[gf["fact_id"]] = s
        if s >= threshold:
            matched.append(gf["fact_id"])
    n = len(gold_facts)
    return {"recall": round(len(matched) / n, 4) if n else 0.0,
            "matched_fact_ids": matched, "n_gold": n, "per_fact_score": scores}


def atomic_fact_precision(pred_facts: Sequence[str],
                          gold_facts: Sequence[Dict[str, Any]],
                          threshold: float = 0.6) -> Dict[str, Any]:
    if not pred_facts:
        return {"precision": 0.0, "n_pred": 0, "n_supported": 0}
    gold_texts = [gf["text"] for gf in gold_facts]
    supported = 0
    for pf in pred_facts:
        if any(fact_match_score(gt, pf) >= threshold
               or fact_match_score(pf, gt) >= threshold for gt in gold_texts):
            supported += 1
    return {"precision": round(supported / len(pred_facts), 4),
            "n_pred": len(pred_facts), "n_supported": supported}


def unsupported_claim_rate(pred_facts: Sequence[Dict[str, Any]]) -> float:
    """Fraction of emitted facts with no valid in-document citation."""
    if not pred_facts:
        return 0.0
    bad = sum(1 for f in pred_facts if not f.get("citation_valid"))
    return round(bad / len(pred_facts), 4)


# --------------------------------------------------------------------------
# branch / orchestration / synthesis
# --------------------------------------------------------------------------


def per_agent_recall(agent_output: Dict[str, Any], gold_facts_for_doc: Sequence[Dict[str, Any]],
                     threshold: float = 0.6) -> Dict[str, Any]:
    text = " ".join(f.get("fact", "") for f in agent_output.get("supported_facts") or [])
    res = atomic_fact_recall(gold_facts_for_doc, text, threshold)
    res["omission_rate"] = round(1.0 - res["recall"], 4)
    res["unsupported_claim_rate"] = unsupported_claim_rate(
        agent_output.get("supported_facts") or [])
    res["retrieval_success"] = bool(agent_output.get("retrieval_success"))
    res["retrieval_rounds"] = int(agent_output.get("search_rounds") or 0)
    res["generated_tokens"] = int(agent_output.get("generated_tokens") or 0)
    res["latency_s"] = float(agent_output.get("latency_s") or 0.0)
    return res


def branch_success(agent_recall: float, threshold: float = 0.5) -> bool:
    return agent_recall >= threshold


def orchestration_metrics(agent_outputs: Sequence[Dict[str, Any]],
                          required_document_ids: Sequence[str],
                          agent_recalls: Sequence[float],
                          branch_threshold: float = 0.5) -> Dict[str, Any]:
    covered = {a.get("document_id") for a in agent_outputs
               if (a.get("supported_facts") and not a.get("insufficient_evidence"))}
    required = set(required_document_ids)
    n_ok = sum(1 for r in agent_recalls if branch_success(r, branch_threshold))
    n = len(agent_recalls) or 1
    all_chunks = [c for a in agent_outputs for c in (a.get("retrieved_chunk_ids") or [])]
    return {
        "n_required_agents": len(required),
        "n_agents_succeeded": n_ok,
        "fraction_agents_succeeded": round(n_ok / n, 4),
        "any_required_branch_failed": bool(n_ok < len(agent_recalls)),
        "required_document_coverage": round(
            len(covered & required) / len(required), 4) if required else 0.0,
        "duplicate_evidence_chunks": len(all_chunks) - len(set(all_chunks)),
        "total_retrieved_chunks": len(all_chunks),
        "total_generated_tokens": sum(int(a.get("generated_tokens") or 0)
                                      for a in agent_outputs),
        "total_prompt_tokens": sum(int(a.get("prompt_tokens") or 0)
                                   for a in agent_outputs),
    }


def synthesis_metrics(agent_outputs: Sequence[Dict[str, Any]],
                      final_answer: str, gold_facts: Sequence[Dict[str, Any]],
                      threshold: float = 0.6) -> Dict[str, Any]:
    """synthesis_loss = correct facts held by agents but missing from the answer
                        / correct facts held by agents."""
    agent_text = " ".join(f.get("fact", "")
                          for a in agent_outputs
                          for f in (a.get("supported_facts") or []))
    before = atomic_fact_recall(gold_facts, agent_text, threshold)
    after = atomic_fact_recall(gold_facts, final_answer or "", threshold)
    available = set(before["matched_fact_ids"])
    preserved = available & set(after["matched_fact_ids"])
    lost = available - preserved
    return {
        "facts_available_before_synthesis": len(available),
        "facts_preserved_after_synthesis": len(preserved),
        "lost_fact_ids": sorted(lost),
        "synthesis_loss": round(len(lost) / len(available), 4) if available else 0.0,
        "atomic_fact_recall_agents": before["recall"],
        "atomic_fact_recall_final": after["recall"],
        "final_completeness": after["recall"],
    }


def contradiction_count(final_answer: str, agent_outputs: Sequence[Dict[str, Any]],
                        unresolved_conflicts: Sequence[str]) -> int:
    """Conservative proxy: numbers asserted in the answer that no agent stated."""
    agent_nums: Set[str] = set()
    for a in agent_outputs:
        for f in a.get("supported_facts") or []:
            agent_nums |= numbers_in(f.get("fact", ""))
    novel = numbers_in(final_answer or "") - agent_nums
    return len(novel) + len(unresolved_conflicts or [])


def system_efficiency(quality: float, wall_s: float, model_bytes: int) -> Dict[str, float]:
    gb = model_bytes / (1024 ** 3) if model_bytes else 0.0
    return {"quality_per_second": round(quality / wall_s, 6) if wall_s else 0.0,
            "quality_per_gb": round(quality / gb, 6) if gb else 0.0,
            "model_size_gb": round(gb, 4)}
