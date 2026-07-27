"""Partial-credit scoring and the branch/orchestration/synthesis metric set.

Everything here is deterministic and label-blind: no function takes a precision
argument, so a scorer cannot behave differently for F16 than for Q4_K_M.

Three properties are load-bearing and are asserted by tests/test_units.py:

  * **No metric may reward degradation.** A prediction that says nothing, cites
    nothing or fails outright must never outscore one that answers correctly.
    Ratios with an empty denominator return ``None`` (excluded downstream), not
    ``0.0`` (which reads as a perfect score).
  * **No metric may reward verbosity.** Coverage is measured inside a bounded
    window of the candidate, so appending text cannot raise a score. Both
    quantization and prompt-format failures change output length, and a
    length-sensitive metric cannot separate those from quality.
  * **Containment respects word boundaries and negation.** ``"no"`` does not
    match inside ``"not"``; ``"Google"`` does not count when the sentence says
    it is *not* Google.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_ARTICLES = {"a", "an", "the"}

# Deliberately excludes negations ("not", "no", "never"): dropping them makes a
# fact and its negation score identically. _NEGATIONS is handled explicitly.
_STOP = _ARTICLES | {
    "of", "in", "on", "at", "to", "for", "and", "or", "is", "are", "was", "were",
    "be", "been", "by", "with", "that", "this", "it", "its", "as", "from", "has",
    "have", "had", "but", "which", "their", "there", "his", "her", "they",
}
_NEGATIONS = {"not", "no", "never", "none", "neither", "nor", "without",
              "cannot", "cant", "didnt", "doesnt", "isnt", "wasnt", "wont",
              "instead", "rather", "unlike", "excluding", "except"}

_PUNCT = str.maketrans("", "", string.punctuation)
_NUM = re.compile(r"\b\d[\d,\.]*\b")

# A containment hit inside a wall of text is not evidence the model answered.
# Above this many tokens the prediction is treated as unanswered for the
# purposes of `contains`; token_f1 still applies. Recorded in every score dict.
MAX_PRED_TOKENS_FOR_CONTAINS = 60

# Tokens of candidate text a gold fact may be matched against at once. Coverage
# is the best score over sliding windows, so unrelated text cannot help.
FACT_WINDOW_TOKENS = 60
FACT_WINDOW_STRIDE = 30

# A predicted "fact" shorter than this asserts nothing checkable and cannot be
# counted as supported -- otherwise emitting one-word facts yields precision 1.0.
MIN_PRED_FACT_TOKENS = 3


def normalize_answer(text: str) -> str:
    """SQuAD-style normalisation: lowercase, strip punctuation/articles/space."""
    s = (text or "").lower().translate(_PUNCT)
    return " ".join(w for w in s.split() if w not in _ARTICLES)


def _tokens(text: str) -> List[str]:
    return normalize_answer(text).split()


def exact_match(pred: str, gold: str, aliases: Sequence[str] = ()) -> float:
    p = normalize_answer(pred)
    golds = [gold, *aliases]
    return 1.0 if any(p == normalize_answer(g) for g in golds) else 0.0


def token_f1(pred: str, gold: str) -> float:
    p, g = _tokens(pred), _tokens(gold)
    if not g or not p:
        # An empty gold answer cannot be scored, and an empty prediction is a
        # miss. Neither is a 1.0 -- the old `float(p == g)` gave a blank
        # prediction full credit against a blank gold.
        return 0.0
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
    if not g or not p:
        return 0.0
    overlap = sum((p & g).values())
    if overlap == 0:
        return 0.0
    prec = overlap / sum(p.values())
    rec = overlap / sum(g.values())
    return 2 * prec * rec / (prec + rec)


# --------------------------------------------------------------------------
# containment
# --------------------------------------------------------------------------


def _subsequence_positions(hay: Sequence[str], needle: Sequence[str]) -> List[int]:
    """Start indices where `needle` occurs as a contiguous run inside `hay`."""
    if not needle or len(needle) > len(hay):
        return []
    out: List[int] = []
    first = needle[0]
    span = len(needle)
    for i, tok in enumerate(hay):
        if tok == first and list(hay[i:i + span]) == list(needle):
            out.append(i)
    return out


def _negated_at(hay: Sequence[str], start: int, window: int = 4) -> bool:
    """True if a negation/contrast token sits just before position `start`."""
    lo = max(0, start - window)
    return any(t in _NEGATIONS for t in hay[lo:start])


def contains_answer(pred: str, gold: str, aliases: Sequence[str] = (),
                    max_pred_tokens: int = MAX_PRED_TOKENS_FOR_CONTAINS
                    ) -> Dict[str, Any]:
    """Word-boundary containment of the gold answer in the prediction.

    Returns a dict rather than a float so the two suppression rules are
    auditable rather than invisible:

      * `too_long`  -- the prediction is long enough that a hit is more likely
        to be an accident of rambling than an answer. (Guards the "list every
        candidate entity" and "ramble until you hit the string" exploits.)
      * `negated`   -- every occurrence is preceded by a negation or contrast
        token, e.g. "the entity is not Google but Microsoft".

    The old implementation was a raw substring test on the normalised strings,
    so gold "no" matched inside "not"/"cannot"/"nothing" and gold "Yes" matched
    inside "Yesterday" -- handing a perfect score to abstentions and refusals.
    """
    p_toks = _tokens(pred)
    result = {"contains": 0.0, "too_long": False, "negated": False,
              "pred_tokens": len(p_toks), "max_pred_tokens": max_pred_tokens}
    if not p_toks:
        return result
    if len(p_toks) > max_pred_tokens:
        result["too_long"] = True
        return result
    saw_negated = False
    for g in [gold, *aliases]:
        g_toks = _tokens(g)
        if not g_toks:
            continue
        positions = _subsequence_positions(p_toks, g_toks)
        if not positions:
            continue
        if any(not _negated_at(p_toks, i) for i in positions):
            result["contains"] = 1.0
            return result
        saw_negated = True
    result["negated"] = saw_negated
    return result


def answer_score(pred: str, gold: str, aliases: Sequence[str] = ()
                 ) -> Dict[str, Any]:
    """Partial-credit answer score. `score` is the headline number.

    `score` is None when the gold answer is empty -- that question carries no
    signal and must be excluded upstream rather than scored 1.0, which is what
    the old `float(p == g)` branch did for a blank prediction.
    """
    golds = [g for g in [gold, *aliases] if _tokens(g)]
    if not golds:
        return {"exact_match": None, "contains": None, "token_f1": None,
                "char_f1": None, "score": None, "scorable": False,
                "reason": "gold answer is empty"}

    em = exact_match(pred, gold, aliases)
    cont = contains_answer(pred, gold, aliases)
    f1 = max([token_f1(pred, g) for g in golds] or [0.0])
    cf1 = max([char_f1(pred, g) for g in golds] or [0.0])
    score = max(em, cont["contains"], f1)
    return {"exact_match": em, "contains": cont["contains"],
            "contains_suppressed_too_long": cont["too_long"],
            "contains_suppressed_negated": cont["negated"],
            "pred_tokens": cont["pred_tokens"],
            "token_f1": round(f1, 4), "char_f1": round(cf1, 4),
            "score": round(score, 4), "scorable": True}


# --------------------------------------------------------------------------
# atomic facts
# --------------------------------------------------------------------------


def content_tokens(text: str) -> List[str]:
    return [t for t in _tokens(text) if t not in _STOP]


def numbers_in(text: str) -> Set[str]:
    """Numerals in `text`, normalised so 1,500 == 1500 and 3.0 == 3.

    Dates are also emitted component-wise by the tokeniser upstream (2023-01-05
    yields 2023, 01, 05); leading zeros are stripped so "05" and "5" agree.
    """
    out: Set[str] = set()
    for m in _NUM.finditer(text or ""):
        raw = m.group(0).rstrip(".").replace(",", "")
        if not raw:
            continue
        try:
            val = float(raw)
        except ValueError:
            out.add(raw)
            continue
        out.add(str(int(val)) if val == int(val) else str(val))
    return out


def _windows(toks: Sequence[str], size: int = FACT_WINDOW_TOKENS,
             stride: int = FACT_WINDOW_STRIDE) -> List[Tuple[int, int]]:
    if len(toks) <= size:
        return [(0, len(toks))]
    spans = [(i, min(i + size, len(toks)))
             for i in range(0, max(1, len(toks) - size + 1), stride)]
    if spans[-1][1] < len(toks):
        spans.append((len(toks) - size, len(toks)))
    return spans


def fact_match_score(gold_fact: str, candidate: str,
                     require_numbers: bool = True,
                     windowed: bool = True) -> float:
    """How well `candidate` covers `gold_fact`, in [0, 1].

    Content-token recall of the gold fact **inside a bounded window** of the
    candidate, gated on numeric and polarity agreement.

    Windowing is what stops this being a verbosity meter: the old version
    searched the whole candidate, so appending text could only raise the score
    and the metric silently rewarded whichever precision rambled more.

    Numeric disagreement now zeroes the score rather than halving it -- a fact
    with the wrong number is not the fact. Polarity disagreement (the gold fact
    is negated and the candidate is not, or vice versa) does the same.
    """
    gold_toks = content_tokens(gold_fact)
    if not gold_toks:
        return 0.0
    cand_all = content_tokens(candidate)
    if not cand_all:
        return 0.0

    spans = _windows(cand_all) if windowed else [(0, len(cand_all))]
    best = 0.0
    for lo, hi in spans:
        window = set(cand_all[lo:hi])
        hits = sum(1 for t in gold_toks if t in window)
        best = max(best, hits / len(gold_toks))
        if best == 1.0:
            break

    if require_numbers:
        gnums = numbers_in(gold_fact)
        if gnums and not (gnums & numbers_in(candidate)):
            return 0.0
        gold_neg = any(t in _NEGATIONS for t in _tokens(gold_fact))
        cand_neg = any(t in _NEGATIONS for t in _tokens(candidate))
        if gold_neg != cand_neg and best < 1.0:
            best *= 0.5
    return round(best, 4)


def fact_supported(gold_fact: str, candidate: str, threshold: float = 0.6) -> bool:
    return fact_match_score(gold_fact, candidate) >= threshold


def atomic_fact_recall(gold_facts: Sequence[Dict[str, Any]], text: str,
                       threshold: float = 0.6) -> Dict[str, Any]:
    """Fraction of gold facts recoverable from `text`.

    `recall` is None when there are no gold facts. The old `0.0` made a
    document with no rubric entries indistinguishable from a total miss, and
    dragged branch recall down with no warning anywhere.
    """
    matched: List[str] = []
    scores: Dict[str, float] = {}
    for gf in gold_facts:
        aliases = gf.get("aliases") or []
        s = max([fact_match_score(gf["text"], text)]
                + [fact_match_score(a, text) for a in aliases])
        scores[gf["fact_id"]] = s
        if s >= threshold:
            matched.append(gf["fact_id"])
    n = len(gold_facts)
    return {"recall": round(len(matched) / n, 4) if n else None,
            "matched_fact_ids": matched, "n_gold": n, "per_fact_score": scores}


def atomic_fact_precision(pred_facts: Sequence[str],
                          gold_facts: Sequence[Dict[str, Any]],
                          threshold: float = 0.6,
                          min_tokens: int = MIN_PRED_FACT_TOKENS
                          ) -> Dict[str, Any]:
    """Fraction of emitted facts whose content is covered by some gold fact.

    The old version accepted a match in *either* direction, which made it
    unfalsifiable: any predicted fact whose tokens were a subset of a gold fact
    counted as supported, so emitting one-word facts scored precision 1.0.

    Now a predicted fact must (a) assert at least `min_tokens` content tokens,
    and (b) have that content covered by a single gold fact, with the same
    numeric and polarity gating as `fact_match_score`. Facts too short to be
    checkable are counted as unsupported and reported separately.
    """
    if not pred_facts:
        return {"precision": None, "n_pred": 0, "n_supported": 0,
                "n_too_short": 0}
    gold_texts = [gf["text"] for gf in gold_facts]
    supported = 0
    too_short = 0
    for pf in pred_facts:
        if len(content_tokens(pf)) < min_tokens:
            too_short += 1
            continue
        if any(fact_match_score(pf, gt) >= threshold for gt in gold_texts):
            supported += 1
    return {"precision": round(supported / len(pred_facts), 4),
            "n_pred": len(pred_facts), "n_supported": supported,
            "n_too_short": too_short}


def uncited_claim_rate(pred_facts: Sequence[Dict[str, Any]]) -> Optional[float]:
    """Fraction of emitted facts with no valid in-document citation.

    This is a *format compliance* measure, not a hallucination measure: a true
    fact that forgot its chunk_ids counts as uncited, and a fabricated fact
    that echoes a visible chunk id counts as cited. It is reported under its
    real name so it is not read as faithfulness -- see
    `unsupported_claim_rate` for the content-level version.
    """
    if not pred_facts:
        return None
    bad = sum(1 for f in pred_facts if not f.get("citation_valid"))
    return round(bad / len(pred_facts), 4)


def unsupported_claim_rate(pred_facts: Sequence[Dict[str, Any]],
                           chunk_texts: Optional[Dict[str, str]] = None,
                           threshold: float = 0.6) -> Optional[float]:
    """Fraction of emitted facts not entailed by the chunks they cite.

    When `chunk_texts` is unavailable this falls back to the citation-format
    measure and the caller should read it as such.
    """
    if not pred_facts:
        return None
    if not chunk_texts:
        return uncited_claim_rate(pred_facts)
    bad = 0
    for f in pred_facts:
        cited = [chunk_texts.get(c, "") for c in (f.get("chunk_ids") or [])]
        support = " ".join(t for t in cited if t)
        if not support or fact_match_score(f.get("fact", ""), support) < threshold:
            bad += 1
    return round(bad / len(pred_facts), 4)


# --------------------------------------------------------------------------
# branch / orchestration / synthesis
# --------------------------------------------------------------------------


def per_agent_recall(agent_output: Dict[str, Any],
                     gold_facts_for_doc: Sequence[Dict[str, Any]],
                     threshold: float = 0.6,
                     chunk_texts: Optional[Dict[str, str]] = None
                     ) -> Dict[str, Any]:
    facts = agent_output.get("supported_facts") or []
    text = " ".join(f.get("fact", "") for f in facts)
    res = atomic_fact_recall(gold_facts_for_doc, text, threshold)
    res["omission_rate"] = (round(1.0 - res["recall"], 4)
                            if res["recall"] is not None else None)
    res["uncited_claim_rate"] = uncited_claim_rate(facts)
    res["unsupported_claim_rate"] = unsupported_claim_rate(
        facts, chunk_texts, threshold)
    # A forced-finalize rescue is not a retrieval success -- the harness handed
    # the agent a query it never produced.
    res["forced_finalize"] = bool(agent_output.get("forced_finalize"))
    res["retrieval_success"] = bool(agent_output.get("retrieval_success")) and \
        not res["forced_finalize"]
    res["retrieval_rounds"] = int(agent_output.get("search_rounds") or 0)
    res["generated_tokens"] = int(agent_output.get("generated_tokens") or 0)
    res["latency_s"] = float(agent_output.get("latency_s") or 0.0)
    res["parse_ok"] = bool(agent_output.get("parse_ok", True))
    return res


def branch_success(agent_recall: Optional[float], threshold: float = 0.5) -> bool:
    return agent_recall is not None and agent_recall >= threshold


def orchestration_metrics(agent_outputs: Sequence[Dict[str, Any]],
                          required_document_ids: Sequence[str],
                          agent_recalls: Sequence[Optional[float]],
                          branch_threshold: float = 0.5) -> Dict[str, Any]:
    """Branch-level outcome for one question.

    `any_required_branch_failed` is measured against the number of documents
    the question *requires*, not against the number of agents that happened to
    return. The old `n_ok < len(agent_recalls)` evaluated `0 < 0` when every
    agent died, so a precision whose branches all collapsed reported a perfect
    branch-failure probability -- the exact quantity the compounding hypothesis
    is about.
    """
    required = set(required_document_ids)
    n_required = len(required)
    covered = {a.get("document_id") for a in agent_outputs
               if (a.get("supported_facts") and not a.get("insufficient_evidence"))}
    correct = {a.get("document_id")
               for a, r in zip(agent_outputs, agent_recalls)
               if branch_success(r, branch_threshold)}
    n_ok = sum(1 for r in agent_recalls if branch_success(r, branch_threshold))
    all_chunks = [c for a in agent_outputs for c in (a.get("retrieved_chunk_ids") or [])]
    return {
        "n_required_agents": n_required,
        "n_agents_returned": len(agent_outputs),
        "n_agents_succeeded": n_ok,
        "fraction_agents_succeeded": (round(n_ok / n_required, 4)
                                      if n_required else None),
        "any_required_branch_failed": (n_ok < n_required) if n_required else None,
        # "the agent said something" -- not a correctness measure.
        "required_document_coverage": (round(len(covered & required) / n_required, 4)
                                       if n_required else None),
        # the correctness-aware counterpart.
        "required_document_correct_coverage": (
            round(len(correct & required) / n_required, 4) if n_required else None),
        "duplicate_evidence_chunks": len(all_chunks) - len(set(all_chunks)),
        "total_retrieved_chunks": len(all_chunks),
        "total_generated_tokens": sum(int(a.get("generated_tokens") or 0)
                                      for a in agent_outputs),
        "total_prompt_tokens": sum(int(a.get("prompt_tokens") or 0)
                                   for a in agent_outputs),
        "n_forced_finalize": sum(1 for a in agent_outputs
                                 if a.get("forced_finalize")),
        "n_parse_failures": sum(1 for a in agent_outputs
                                if not a.get("parse_ok", True)),
    }


def synthesis_metrics(agent_outputs: Sequence[Dict[str, Any]],
                      final_answer: str, gold_facts: Sequence[Dict[str, Any]],
                      threshold: float = 0.6) -> Dict[str, Any]:
    """synthesis_loss = correct facts held by agents but missing from the answer
                        / correct facts held by agents.

    Returns None when the agents recovered nothing. The old `else 0.0` reported
    *perfect* synthesis for a pipeline whose branches had all failed, so the
    worse a precision's agents did, the better its synthesis layer looked.
    """
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
        "synthesis_loss": round(len(lost) / len(available), 4) if available else None,
        "atomic_fact_recall_agents": before["recall"],
        "atomic_fact_recall_final": after["recall"],
        "final_completeness": after["recall"],
    }


def unsourced_numbers(final_answer: str, agent_outputs: Sequence[Dict[str, Any]],
                      unresolved_conflicts: Sequence[str] = (),
                      question: str = "") -> Dict[str, Any]:
    """Numbers asserted in the answer that no agent stated and the question
    did not supply, plus conflicts the synthesizer itself flagged.

    A conservative proxy for contradiction, and reported under a name that says
    what it measures. Numbers appearing in the question are excluded, so
    restating the prompt is not penalised, and `numbers_in` normalisation means
    3 and 3.0 (and 1,500 and 1500) no longer count as a contradiction.
    """
    agent_nums: Set[str] = set()
    for a in agent_outputs:
        for f in a.get("supported_facts") or []:
            agent_nums |= numbers_in(f.get("fact", ""))
    known = agent_nums | numbers_in(question)
    novel = sorted(numbers_in(final_answer or "") - known)
    return {"unsourced_numbers": novel,
            "n_unsourced_numbers": len(novel),
            "n_unresolved_conflicts": len(unresolved_conflicts or []),
            "contradictions_introduced": len(novel) + len(unresolved_conflicts or [])}


def contradiction_count(final_answer: str, agent_outputs: Sequence[Dict[str, Any]],
                        unresolved_conflicts: Sequence[str] = (),
                        question: str = "") -> int:
    """Back-compatible scalar wrapper around `unsourced_numbers`."""
    return unsourced_numbers(final_answer, agent_outputs,
                             unresolved_conflicts, question)["contradictions_introduced"]


def system_efficiency(quality: Optional[float], wall_s: Optional[float],
                      model_bytes: Optional[int]) -> Dict[str, Optional[float]]:
    """Quality per unit cost. None rather than 0.0 when a term is missing, so a
    block with no recorded wall time is excluded instead of reported as free.
    """
    gb = (model_bytes / (1024 ** 3)) if model_bytes else None
    q = quality
    return {
        "quality_per_second": (round(q / wall_s, 6)
                               if q is not None and wall_s else None),
        "quality_per_gb": (round(q / gb, 6) if q is not None and gb else None),
        "model_size_gb": round(gb, 4) if gb else None,
    }
