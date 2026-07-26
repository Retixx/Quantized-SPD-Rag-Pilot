"""Synthesis layer: merge all document-agent findings into one final answer."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def run_synthesizer(client, question_id: str, question: str,
                    synthesis_directive: str, agent_outputs: List[Dict[str, Any]],
                    max_tokens: int = 384) -> Dict[str, Any]:
    findings, fact_map = index_facts(agent_outputs)
    msgs = build_messages(question, synthesis_directive, findings)
    cid = make_call_id(client.stage, client.precision, client.condition,
                       question_id, "synthesizer")
    parsed, rec = client.call(
        call_id=cid, role="synthesizer", question_id=question_id, messages=msgs,
        model_cls=SynthesizerOutput, schema=prompts.SYNTHESIZER_SCHEMA,
        max_tokens=max_tokens, extra={"n_agent_facts": len(fact_map)})
    out = SynthesizerOutput(**(parsed or {})).model_dump()
    out.update(call_id=cid, parse_ok=bool(rec.get("parse_ok")),
               latency_s=rec.get("latency_s", 0.0),
               generated_tokens=rec.get("generated_tokens", 0),
               prompt_tokens=rec.get("prompt_tokens", 0),
               is_fixture=bool(rec.get("is_fixture")),
               facts_available=len(fact_map),
               fact_map=fact_map)
    return out
