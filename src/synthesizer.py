"""Synthesis layer: merge all document-agent findings into one final answer."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prompts  # noqa: E402
from logging_utils import call_id as make_call_id  # noqa: E402
from schemas import SynthesizerOutput  # noqa: E402


def index_facts(agent_outputs: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """Render findings with stable fact ids (d1.f2) and return the id map."""
    lines: List[str] = []
    fact_map: Dict[str, Dict[str, Any]] = {}
    for di, agent in enumerate(sorted(agent_outputs, key=lambda a: a.get("document_id", "")), 1):
        did = agent.get("document_id", f"doc{di}")
        lines.append(f"Document {did} (agent d{di}):")
        facts = agent.get("supported_facts") or []
        if not facts:
            lines.append("  (no supported facts; insufficient evidence)")
        for fi, fact in enumerate(facts, 1):
            fid = f"d{di}.f{fi}"
            fact_map[fid] = {"fact_id": fid, "document_id": did,
                             "fact": fact.get("fact", ""),
                             "chunk_ids": fact.get("chunk_ids") or []}
            lines.append(f"  [{fid}] {fact.get('fact','')}")
        lines.append("")
    return "\n".join(lines).strip(), fact_map


def build_messages(question: str, synthesis_directive: str,
                   findings: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": prompts.SYNTHESIZER_SYSTEM},
        {"role": "user", "content": prompts.fill(
            prompts.SYNTHESIZER_USER, question=question,
            synthesis_directive=synthesis_directive, findings=findings)},
    ]


def validate_citations(out: Dict[str, Any], fact_map: Dict[str, Dict[str, Any]],
                       agent_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check the synthesizer's own citations against what the agents returned.

    `used_fact_ids` / `used_document_ids` are demanded by the prompt and cost
    tokens in the most context-pressured call, so they are used: they are a
    cheap faithfulness signal. Nothing is dropped or rewritten -- the raw lists
    stay as the model wrote them and the validation lands beside them.
    """
    known_facts = set(fact_map)
    known_docs = {a.get("document_id", "") for a in agent_outputs if a.get("document_id")}
    used_facts = [str(f) for f in (out.get("used_fact_ids") or [])]
    used_docs = [str(d) for d in (out.get("used_document_ids") or [])]
    known_used_facts = [f for f in used_facts if f in known_facts]
    unknown_facts = [f for f in used_facts if f not in known_facts]
    known_used_docs = [d for d in used_docs if d in known_docs]
    unknown_docs = [d for d in used_docs if d not in known_docs]
    return {
        "used_fact_ids_known": known_used_facts,
        "used_fact_ids_unknown": unknown_facts,
        "used_document_ids_known": known_used_docs,
        "used_document_ids_unknown": unknown_docs,
        "fact_citation_precision": round(len(known_used_facts) / len(used_facts), 4)
        if used_facts else 0.0,
        "fact_citation_coverage": round(len(set(known_used_facts)) / len(known_facts), 4)
        if known_facts else 0.0,
        "document_citation_precision": round(len(known_used_docs) / len(used_docs), 4)
        if used_docs else 0.0,
        "hallucinated_fact_ids": len(unknown_facts),
        "hallucinated_document_ids": len(unknown_docs),
    }


def run_synthesizer(client, question_id: str, question: str,
                    synthesis_directive: str, agent_outputs: List[Dict[str, Any]],
                    max_tokens: Optional[int] = None) -> Dict[str, Any]:
    if max_tokens is None:
        max_tokens = prompts.DEFAULT_MAX_TOKENS["synthesizer"]
    findings, fact_map = index_facts(agent_outputs)
    msgs = build_messages(question, synthesis_directive, findings)
    cid = make_call_id(client.stage, client.precision, client.condition,
                       question_id, "synthesizer")
    parsed, rec = client.call(
        call_id=cid, role="synthesizer", question_id=question_id, messages=msgs,
        model_cls=SynthesizerOutput, schema=prompts.SYNTHESIZER_SCHEMA,
        max_tokens=max_tokens, extra={"n_agent_facts": len(fact_map)})
    out = SynthesizerOutput(**(parsed or {})).model_dump()
    # A synthesizer that failed both attempts returns final_answer="" and would
    # otherwise be scored 0 with no diagnostic. Surface the parse state.
    out.update(call_id=cid,
               parse_ok=bool(rec.get("parse_ok")),
               parse_error=rec.get("parse_error"),
               repair_attempted=bool(rec.get("repair_attempted")),
               repair_used=bool(rec.get("repair_used")),
               truncated=bool(rec.get("truncated")),
               empty_generation=bool(rec.get("empty_generation")),
               latency_s=rec.get("latency_s", 0.0),
               generated_tokens=rec.get("generated_tokens", 0),
               prompt_tokens=rec.get("prompt_tokens", 0),
               is_fixture=bool(rec.get("is_fixture")),
               facts_available=len(fact_map),
               fact_map=fact_map)
    out.update(validate_citations(out, fact_map, agent_outputs))
    return out
