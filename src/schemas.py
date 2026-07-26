"""Pydantic schemas for the three SPD-RAG layers, plus JSON extraction/repair.

Mirrors backend/core/state.py in NebulAICompany/SPD-RAG, restated as the
strict JSON contracts required by the pilot brief.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

# --------------------------------------------------------------------------
# Layer 1: coordination
# --------------------------------------------------------------------------


class SharedTask(BaseModel):
    task_id: str = Field(description="Stable id, e.g. 't1'.")
    instruction: str = Field(description="Atomic extraction instruction.")
    required_fields: List[str] = Field(default_factory=list)


class CoordinatorOutput(BaseModel):
    shared_tasks: List[SharedTask] = Field(default_factory=list)
    synthesis_directive: str = ""


# --------------------------------------------------------------------------
# Layer 2: per-document retrieval
# --------------------------------------------------------------------------


class SupportedFact(BaseModel):
    fact: str
    chunk_ids: List[str] = Field(default_factory=list)


class DocumentAgentOutput(BaseModel):
    question_id: str = ""
    document_id: str = ""
    search_queries: List[str] = Field(default_factory=list)
    retrieved_chunk_ids: List[str] = Field(default_factory=list)
    supported_facts: List[SupportedFact] = Field(default_factory=list)
    insufficient_evidence: bool = False


class AgentAction(BaseModel):
    """One turn of the end-to-end retrieval loop (SPD-RAG AgentAction)."""

    action: str = Field(description="'search' or 'finalize'.")
    query: Optional[str] = None
    reasoning: str = ""
    supported_facts: List[SupportedFact] = Field(default_factory=list)
    insufficient_evidence: bool = False


# --------------------------------------------------------------------------
# Layer 3: synthesis
# --------------------------------------------------------------------------


class SynthesizerOutput(BaseModel):
    final_answer: str = ""
    used_document_ids: List[str] = Field(default_factory=list)
    used_fact_ids: List[str] = Field(default_factory=list)
    unresolved_conflicts: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# JSON extraction / one-shot repair
# --------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort recovery of a single JSON object from raw model text.

    Deterministic. Tries, in order: whole string, fenced block, first balanced
    brace span. Returns None when nothing parses.
    """
    if not text:
        return None
    candidates: List[str] = [text.strip()]
    m = _FENCE.search(text)
    if m:
        candidates.append(m.group(1).strip())
    start = text.find("{")
    if start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


class ParseResult(BaseModel):
    ok: bool
    model_obj: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    repair_used: bool = False


def parse_into(model_cls, text: str) -> ParseResult:
    obj = extract_json_object(text)
    if obj is None:
        return ParseResult(ok=False, error="no_json_object_found")
    try:
        parsed = model_cls(**obj)
    except ValidationError as exc:
        return ParseResult(ok=False, error=f"validation_error: {exc.errors()[:3]}")
    return ParseResult(ok=True, model_obj=parsed.model_dump())


REPAIR_INSTRUCTION = (
    "Your previous reply was not valid JSON matching the required schema.\n"
    "Reply again with ONE JSON object only. No prose, no markdown fences.\n"
    "Required schema:\n{schema}\n\nYour previous reply was:\n{previous}\n"
)
