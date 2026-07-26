"""Coordination layer: question -> shared atomic instructions + directive.

Per the brief the coordinator does NOT choose which documents exist; the
benchmark manifest fixes the supporting document set for the controlled study.
It only sees the question and how many documents are in the set.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prompts  # noqa: E402
from logging_utils import call_id as make_call_id  # noqa: E402
from schemas import CoordinatorOutput  # noqa: E402


def build_messages(question: str, n_documents: int) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": prompts.COORDINATOR_SYSTEM},
        {"role": "user", "content": prompts.fill(
            prompts.COORDINATOR_USER, question=question, n_documents=n_documents)},
    ]


def fallback_tasks(question: str) -> Dict[str, Any]:
    """Deterministic degradation when the model cannot emit valid JSON.

    Recorded as coordinator_fallback=True; it is a harness-level failure mode,
    not a silent skip, and it is counted in the run report.
    """
    return CoordinatorOutput(shared_tasks=[
        {"task_id": "t1",
         "instruction": ("Extract every statement in this document that is directly "
                         f"relevant to the question: {question}"),
         "required_fields": ["entities", "numbers", "dates"]},
        {"task_id": "t2",
         "instruction": ("Extract any explicit comparison, ranking, cause, effect or "
                         "date needed to answer that question."),
         "required_fields": ["claim"]},
    ], synthesis_directive=(
        "Answer the question directly and briefly using only the sub-agent findings. "
        "Preserve exact names, numbers and dates.")).model_dump()


def run_coordinator(client, question_id: str, question: str, n_documents: int,
                    max_tokens: int = 384) -> Dict[str, Any]:
    msgs = build_messages(question, n_documents)
    cid = make_call_id(client.stage, client.precision, client.condition,
                       question_id, "coordinator")
    parsed, rec = client.call(
        call_id=cid, role="coordinator", question_id=question_id, messages=msgs,
        model_cls=CoordinatorOutput, schema=prompts.COORDINATOR_SCHEMA,
        max_tokens=max_tokens)
    used_fallback = False
    if not parsed or not parsed.get("shared_tasks"):
        parsed = fallback_tasks(question)
        used_fallback = True
    parsed["coordinator_fallback"] = used_fallback
    parsed["call_id"] = cid
    return parsed


def format_instructions(shared_tasks: List[Dict[str, Any]]) -> str:
    lines = []
    for t in shared_tasks:
        fields = ", ".join(t.get("required_fields") or [])
        suffix = f" (required fields: {fields})" if fields else ""
        lines.append(f"- [{t.get('task_id','t?')}] {t.get('instruction','')}{suffix}")
    return "\n".join(lines) if lines else "- Extract facts relevant to the question."
