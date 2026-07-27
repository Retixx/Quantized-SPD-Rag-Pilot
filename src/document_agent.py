"""Per-document retrieval layer. One logical agent per supporting document.

Both conditions live here:
  * fixed_verified_evidence -- chunks are precomputed once and frozen, so all
    three precisions receive identical chunk ids and identical prompts;
  * end_to_end -- the agent writes its own queries against its PRIVATE index,
    at most `max_search_rounds` (default 2) rounds.

An agent is handed ONE `PrivateIndex` and nothing else -- not the store. It has
no handle on any other document, so isolation is structural rather than a
filter that has to be passed the right variable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prompts  # noqa: E402
from coordinator import format_instructions  # noqa: E402
from logging_utils import call_id as make_call_id  # noqa: E402
from retrieval import (DocumentIsolationError, Hit,  # noqa: E402
                       PrivateIndex, format_chunks_for_prompt)
from schemas import AgentAction, DocumentAgentOutput  # noqa: E402

# stop_reason values recorded on every end-to-end agent output
STOP_FINALIZE = "finalize"
STOP_BUDGET_EXHAUSTED = "budget_exhausted"
STOP_PARSE_FAILURE = "parse_failure"
STOP_SEARCH_WITHOUT_QUERY = "search_without_query"
STOP_UNRECOGNISED_ACTION = "unrecognised_action"


def _hits_to_ids(hits: List[Hit]) -> List[str]:
    return [h.chunk_id for h in hits]


def _search(index: PrivateIndex, query: str, top_k: int) -> List[Hit]:
    """Text search inside ONE private index. The index owns its embedder."""
    search_text = getattr(index, "search_text", None)
    if search_text is None:
        raise AttributeError(
            "PrivateIndex.search_text(query, top_k) is required by "
            "run_end_to_end_agent; the agent is never handed the store or an "
            "embedder, so it cannot embed the query itself.")
    return list(search_text(query, top_k=top_k))


def build_fixed_messages(question_id: str, document_id: str, question: str,
                         instructions: str, hits: List[Hit]) -> List[Dict[str, str]]:
    chunk_ids = ", ".join(f'"{c}"' for c in _hits_to_ids(hits))
    system = prompts.fill(prompts.DOC_AGENT_FIXED_SYSTEM, question_id=question_id,
                          document_id=document_id, chunk_id_list=chunk_ids)
    user = prompts.fill(prompts.DOC_AGENT_FIXED_USER, question=question,
                        instructions=instructions, document_id=document_id,
                        passages=format_chunks_for_prompt(hits))
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def run_fixed_evidence_agent(client, question_id: str, document_id: str,
                             question: str, shared_tasks: List[Dict[str, Any]],
                             hits: List[Hit], max_tokens: Optional[int] = None,
                             role: str = "document_agent",
                             attempt: int = 0) -> Dict[str, Any]:
    if max_tokens is None:
        max_tokens = prompts.DEFAULT_MAX_TOKENS["document_agent"]
    instructions = format_instructions(shared_tasks)
    msgs = build_fixed_messages(question_id, document_id, question, instructions, hits)
    cid = make_call_id(client.stage, client.precision, client.condition,
                       question_id, role, document_id, attempt)
    parsed, rec = client.call(
        call_id=cid, role=role, question_id=question_id,
        document_id=document_id, messages=msgs, model_cls=DocumentAgentOutput,
        schema=prompts.DOC_AGENT_SCHEMA, max_tokens=max_tokens,
        extra={"evidence_chunk_ids": _hits_to_ids(hits), "search_rounds": 0})
    out = DocumentAgentOutput(**(parsed or {})).model_dump()
    out["question_id"] = question_id
    out["document_id"] = document_id
    out["retrieved_chunk_ids"] = _hits_to_ids(hits)
    out["search_queries"] = []
    out["search_rounds"] = 0
    out["parse_ok"] = bool(rec.get("parse_ok"))
    out["turn_parse_flags"] = [bool(rec.get("parse_ok"))]
    out["turn_parse_failures"] = 0 if rec.get("parse_ok") else 1
    out["repair_attempted"] = bool(rec.get("repair_attempted"))
    out["repair_used"] = bool(rec.get("repair_used"))
    out["truncated"] = bool(rec.get("truncated"))
    out["forced_finalize"] = False
    out["call_id"] = cid
    out["latency_s"] = rec.get("latency_s", 0.0)
    out["generated_tokens"] = rec.get("generated_tokens", 0)
    out["prompt_tokens"] = rec.get("prompt_tokens", 0)
    out["retrieval_success"] = bool(hits)
    out["is_fixture"] = bool(rec.get("is_fixture"))
    out = _drop_uncited(out)
    return out


def run_end_to_end_agent(client, index: PrivateIndex, question_id: str,
                         document_id: str, question: str,
                         shared_tasks: List[Dict[str, Any]], top_k: int = 3,
                         max_search_rounds: int = 2,
                         max_tokens: Optional[int] = None) -> Dict[str, Any]:
    """Agentic retrieval against ONE private index.

    `index` is a single `PrivateIndex` (obtain it with `store.get(document_id)`).
    The agent never receives the store, so it cannot reach another document even
    if a filter argument is forgotten.
    """
    if max_tokens is None:
        max_tokens = prompts.DEFAULT_MAX_TOKENS["document_agent_turn"]
    if getattr(index, "document_id", None) != document_id:
        raise DocumentIsolationError(
            f"agent for {document_id!r} was handed the index for "
            f"{getattr(index, 'document_id', None)!r}")
    instructions = format_instructions(shared_tasks)
    system = prompts.fill(prompts.DOC_AGENT_LOOP_SYSTEM, document_id=document_id,
                          max_rounds=max_search_rounds)
    queries: List[str] = []
    seen_ids: List[str] = []
    hits_all: List[Hit] = []
    parse_flags: List[bool] = []
    repair_attempted = False
    repair_used = False
    truncated = False
    degenerate_turns = 0
    stop_reason = STOP_BUDGET_EXHAUSTED
    latency = 0.0
    gen_tokens = 0
    prompt_tokens = 0
    is_fixture = False
    final: Optional[Dict[str, Any]] = None

    for turn in range(max_search_rounds + 1):
        user = prompts.fill(
            prompts.DOC_AGENT_LOOP_USER, question=question, instructions=instructions,
            rounds_used=len(queries), max_rounds=max_search_rounds,
            passages=format_chunks_for_prompt(hits_all))
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        cid = make_call_id(client.stage, client.precision, client.condition,
                           question_id, "document_agent_turn", document_id, turn)
        parsed, rec = client.call(
            call_id=cid, role="document_agent_turn", question_id=question_id,
            document_id=document_id, messages=msgs, model_cls=AgentAction,
            schema=prompts.DOC_AGENT_ACTION_SCHEMA, max_tokens=max_tokens,
            extra={"turn": turn, "queries_so_far": list(queries)})
        parse_flags.append(bool(rec.get("parse_ok")))
        repair_attempted = repair_attempted or bool(rec.get("repair_attempted"))
        repair_used = repair_used or bool(rec.get("repair_used"))
        truncated = truncated or bool(rec.get("truncated"))
        latency += float(rec.get("latency_s") or 0.0)
        gen_tokens += int(rec.get("generated_tokens") or 0)
        prompt_tokens += int(rec.get("prompt_tokens") or 0)
        is_fixture = is_fixture or bool(rec.get("is_fixture"))
        action = str((parsed or {}).get("action") or "").strip().lower()
        query = str((parsed or {}).get("query") or "").strip()
        budget_left = len(queries) < max_search_rounds

        if not parsed:
            stop_reason = STOP_PARSE_FAILURE
            final = None
            break
        if action == "finalize":
            stop_reason = STOP_FINALIZE
            final = parsed
            break
        if action == "search" and budget_left and query:
            q = query[:512]
            queries.append(q)
            for h in _search(index, q, top_k):
                if h.chunk_id not in seen_ids:
                    seen_ids.append(h.chunk_id)
                    hits_all.append(h)
            continue
        if not budget_left:
            stop_reason = STOP_BUDGET_EXHAUSTED
            final = parsed
            break
        # Budget remains but this turn neither searched nor finalized. Under
        # greedy decoding the next prompt would be byte-identical, so looping
        # only burns generations. Stop and record it.
        degenerate_turns += 1
        stop_reason = (STOP_SEARCH_WITHOUT_QUERY if action == "search"
                       else STOP_UNRECOGNISED_ACTION)
        final = parsed
        break

    common = {
        "turn_parse_flags": list(parse_flags),
        "turn_parse_failures": sum(1 for f in parse_flags if not f),
        "turns_used": len(parse_flags),
        "degenerate_turns": degenerate_turns,
        "stop_reason": stop_reason,
        "repair_attempted": repair_attempted,
        "repair_used": repair_used,
        "truncated": truncated,
    }

    if final is None or not (final.get("supported_facts") or final.get("insufficient_evidence")):
        # No clean finalize: force one final extraction. This is a rescue that
        # prevents total data loss, NOT a retrieval success -- it is a free
        # retry handed to whichever precision failed, so it is flagged for
        # exclusion/stratification and never counted as retrieval_success.
        if not queries:
            queries.append(question[:512])
            for h in _search(index, question, top_k):
                if h.chunk_id not in seen_ids:
                    seen_ids.append(h.chunk_id)
                    hits_all.append(h)
        forced = run_fixed_evidence_agent(
            client, question_id, document_id, question, shared_tasks, hits_all,
            max_tokens=max_tokens, role="document_agent_forced_finalize")
        forced_parse_ok = bool(forced.get("parse_ok"))
        forced.update(common)
        forced.update(
            search_queries=queries, retrieved_chunk_ids=seen_ids,
            search_rounds=len(queries), forced_finalize=True,
            forced_finalize_parse_ok=forced_parse_ok,
            forced_finalize_retrieved=bool(hits_all),
            parse_ok=all(parse_flags) and forced_parse_ok,
            turn_parse_flags=list(parse_flags) + [forced_parse_ok],
            turn_parse_failures=sum(1 for f in parse_flags if not f)
            + (0 if forced_parse_ok else 1),
            repair_attempted=repair_attempted or bool(forced.get("repair_attempted")),
            repair_used=repair_used or bool(forced.get("repair_used")),
            truncated=truncated or bool(forced.get("truncated")),
            latency_s=round(latency + float(forced.get("latency_s") or 0.0), 4),
            generated_tokens=gen_tokens + int(forced.get("generated_tokens") or 0),
            prompt_tokens=prompt_tokens + int(forced.get("prompt_tokens") or 0),
            retrieval_success=False,
            is_fixture=is_fixture or bool(forced.get("is_fixture")))
        # retrieved_chunk_ids just changed, so re-audit citations against the
        # final id set. _drop_uncited is idempotent; this recomputes from the
        # facts it already normalised rather than reclassifying them.
        return _drop_uncited(forced, force=True)

    out = DocumentAgentOutput(
        question_id=question_id, document_id=document_id, search_queries=queries,
        retrieved_chunk_ids=seen_ids,
        supported_facts=final.get("supported_facts") or [],
        insufficient_evidence=bool(final.get("insufficient_evidence"))).model_dump()
    out.update(common)
    out.update(search_rounds=len(queries), forced_finalize=False,
               parse_ok=all(parse_flags), latency_s=round(latency, 4),
               generated_tokens=gen_tokens, prompt_tokens=prompt_tokens,
               retrieval_success=bool(hits_all), is_fixture=is_fixture)
    return _drop_uncited(out)


def _drop_uncited(out: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
    """Keep facts whose citations point inside this agent's retrieved chunks.

    Facts with no citation are kept but flagged, so unsupported-claim rate stays
    measurable rather than being silently cleaned up.

    IDEMPOTENT. The first pass rewrites `chunk_ids` to the surviving ids, so a
    second naive pass would see an empty list and reclassify a hallucinated
    citation as merely uncited -- which zeroed `hallucinated_citations` on the
    whole forced-finalize path. Once audited, an output is left alone unless
    `force=True`, and a forced re-audit reuses the counts from the first pass
    (only membership against the new id set can change).
    """
    audited = bool(out.get("citation_audit_done"))
    if audited and not force:
        return out
    allowed = set(out.get("retrieved_chunk_ids") or [])
    kept: List[Dict[str, Any]] = []
    bad_citations = 0
    uncited = 0
    for fact in out.get("supported_facts") or []:
        cids = [c for c in (fact.get("chunk_ids") or []) if isinstance(c, str)]
        good = [c for c in cids if c in allowed]
        if audited:
            # already normalised: trust the first pass's classification
            was_hallucinated = bool(fact.get("citation_hallucinated"))
            was_uncited = bool(fact.get("citation_absent"))
        else:
            was_hallucinated = bool(cids) and not good
            was_uncited = not cids
        bad_citations += 1 if was_hallucinated else 0
        uncited += 1 if was_uncited else 0
        kept.append({"fact": fact.get("fact", ""), "chunk_ids": good,
                     "citation_valid": bool(good),
                     "citation_hallucinated": was_hallucinated,
                     "citation_absent": was_uncited})
    out["supported_facts"] = kept
    out["hallucinated_citations"] = bad_citations
    out["uncited_facts"] = uncited
    out["citation_audit_done"] = True
    return out
