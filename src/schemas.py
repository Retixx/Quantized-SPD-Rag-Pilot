"""Pydantic schemas for the three SPD-RAG layers, plus JSON extraction/repair.

Mirrors backend/core/state.py in NebulAICompany/SPD-RAG, restated as the
strict JSON contracts required by the pilot brief.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

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
    # Citation audit, set by document_agent._drop_uncited and never by the model.
    # Declared here so the audit survives a round trip through this schema:
    # metrics reads citation_valid, and a re-validated output used to lose it,
    # silently making every fact unsupported (rate 1.0).
    citation_valid: bool = False
    citation_hallucinated: bool = False   # cited chunks, none inside this document
    citation_absent: bool = False         # no citation at all


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


def _balanced_spans(text: str) -> List[Tuple[int, int, int]]:
    """Every balanced `{...}` span as (start, end, depth). One pass, no break.

    A stray brace in prose no longer aborts the scan, and nested objects are
    recovered too, so a reply that opens an object it never closes still yields
    its inner candidates.
    """
    spans: List[Tuple[int, int, int]] = []
    stack: List[int] = []
    in_str = False
    esc = False
    for i, ch in enumerate(text):
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
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            spans.append((start, i + 1, len(stack)))
    return sorted(spans)


def _candidate_strings(text: str) -> List[Tuple[int, int, str]]:
    """All JSON candidates as (depth, offset, raw). Deterministic order."""
    out: List[Tuple[int, int, str]] = []
    whole = text.strip()
    if whole:
        out.append((0, 0, whole))
    for m in _FENCE.finditer(text):  # every fenced block, not just the first
        body = m.group(1).strip()
        if body:
            out.append((0, m.start(1), body))
    for start, end, depth in _balanced_spans(text):
        out.append((depth, start, text[start:end]))
    return out


def _schema_field_names(model_cls) -> Set[str]:
    fields = getattr(model_cls, "model_fields", None) or {}
    names: Set[str] = set(fields)
    for name, info in fields.items():
        alias = getattr(info, "alias", None)
        if alias:
            names.add(str(alias))
    return names


def _fits_schema(model_cls, obj: Dict[str, Any]) -> bool:
    if model_cls is None:
        return True
    try:
        model_cls(**obj)
    except Exception:
        return False
    return True


def extract_json_object(text: str, model_cls=None) -> Optional[Dict[str, Any]]:
    """Best-effort recovery of a single JSON object from raw model text.

    Deterministic, and schema-aware when `model_cls` is supplied. Every system
    prompt ends with a literal JSON example, so a reply that echoes the schema
    before answering used to hand back the example. Candidates are therefore
    ranked, not tried in positional order:

      1. objects that validate against the target schema beat ones that do not;
      2. more target-schema keys present beats fewer (an outer object beats one
         of its own nested fragments);
      3. shallower beats deeper;
      4. LAST occurrence beats first (the schema echo precedes the answer).

    Returns None when nothing parses.
    """
    if not text:
        return None
    known = _schema_field_names(model_cls) if model_cls is not None else set()
    best: Optional[Dict[str, Any]] = None
    best_rank: Optional[Tuple[int, int, int, int]] = None
    for depth, offset, cand in _candidate_strings(text):
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        matched = len(known & {str(k) for k in obj}) if known else 0
        rank = (0 if _fits_schema(model_cls, obj) else 1, -matched, depth, -offset)
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best = obj
    return best


class ParseResult(BaseModel):
    ok: bool
    model_obj: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    repair_used: bool = False


def parse_into(model_cls, text: str) -> ParseResult:
    """Always returns a ParseResult. Never raises on adversarial model output."""
    obj = extract_json_object(text, model_cls)
    if obj is None:
        return ParseResult(ok=False, error="no_json_object_found")
    try:
        parsed = model_cls(**obj)
    except ValidationError as exc:
        return ParseResult(ok=False, error=f"validation_error: {str(exc.errors()[:3])[:500]}")
    except Exception as exc:
        # e.g. non-string keys make model_cls(**obj) raise TypeError. A malformed
        # generation must never abort the question.
        return ParseResult(ok=False, error=f"{type(exc).__name__}: {exc}"[:500])
    return ParseResult(ok=True, model_obj=parsed.model_dump())


REPAIR_INSTRUCTION = (
    "Your previous reply was not valid JSON matching the required schema.\n"
    "Reply again with ONE JSON object only. No prose, no markdown fences.\n"
    "Required schema:\n{schema}\n\nYour previous reply was:\n{previous}\n"
)
