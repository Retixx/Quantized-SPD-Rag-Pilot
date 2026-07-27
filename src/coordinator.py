"""Coordination layer: question -> shared atomic instructions + directive.

Per the brief the coordinator does NOT choose which documents exist; the
benchmark manifest fixes the supporting document set for the controlled study.
It sees the question and nothing else -- not even how many documents are in the
set, which is gold evidence width and the study's independent variable.

The coordinator output is FROZEN, exactly like the fixed evidence: it is
produced once at `FREEZE_PRECISION` (F16), written to disk with its provenance,
and reused verbatim by every precision. Two reasons:

  * its output feeds every document-agent prompt, so a precision-dependent
    coordinator makes the `prompt_hash_parity` gate unsatisfiable;
  * `fallback_tasks()` would otherwise rescue exactly the precision that is
    failing -- malformed JSON is Q4_K_M's dominant failure mode -- attenuating
    the very gap the pilot measures.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prompts  # noqa: E402
from logging_utils import call_id as make_call_id, read_json, write_json  # noqa: E402
from schemas import CoordinatorOutput  # noqa: E402

# The one precision allowed to produce the shared coordinator output.
FREEZE_PRECISION = "F16"

# Why the deterministic fallback fired. "parsed but degenerate" is a different
# failure from "unparseable" and the two are counted separately.
FALLBACK_NONE = "none"
FALLBACK_PARSE_FAILED = "parse_failed"
FALLBACK_EMPTY_TASKS = "empty_shared_tasks"


def build_messages(question: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": prompts.COORDINATOR_SYSTEM},
        {"role": "user", "content": prompts.fill(
            prompts.COORDINATOR_USER, question=question)},
    ]


def fallback_tasks(question: str) -> Dict[str, Any]:
    """Deterministic degradation when the model cannot emit usable tasks.

    Recorded as coordinator_fallback=True with a reason; it is a harness-level
    failure mode, not a silent skip, and it is counted in the run report.
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


def run_coordinator(client, question_id: str, question: str,
                    max_tokens: Optional[int] = None) -> Dict[str, Any]:
    """One coordinator call. Use freeze_coordinator() for the pilot proper."""
    if max_tokens is None:
        max_tokens = prompts.DEFAULT_MAX_TOKENS["coordinator"]
    msgs = build_messages(question)
    cid = make_call_id(client.stage, client.precision, client.condition,
                       question_id, "coordinator")
    parsed, rec = client.call(
        call_id=cid, role="coordinator", question_id=question_id, messages=msgs,
        model_cls=CoordinatorOutput, schema=prompts.COORDINATOR_SCHEMA,
        max_tokens=max_tokens)
    if not parsed:
        reason = FALLBACK_PARSE_FAILED
    elif not parsed.get("shared_tasks"):
        reason = FALLBACK_EMPTY_TASKS
    else:
        reason = FALLBACK_NONE
    if reason != FALLBACK_NONE:
        parsed = fallback_tasks(question)
    parsed["coordinator_fallback"] = reason != FALLBACK_NONE
    parsed["coordinator_fallback_reason"] = reason
    parsed["parse_ok"] = bool(rec.get("parse_ok"))
    parsed["repair_used"] = bool(rec.get("repair_used"))
    parsed["repair_attempted"] = bool(rec.get("repair_attempted"))
    parsed["truncated"] = bool(rec.get("truncated"))
    parsed["call_id"] = cid
    parsed["prompt_hash"] = rec.get("prompt_hash", "")
    parsed["precision"] = client.precision
    return parsed


def format_instructions(shared_tasks: List[Dict[str, Any]]) -> str:
    lines = []
    for t in shared_tasks:
        fields = ", ".join(t.get("required_fields") or [])
        suffix = f" (required fields: {fields})" if fields else ""
        lines.append(f"- [{t.get('task_id','t?')}] {t.get('instruction','')}{suffix}")
    return "\n".join(lines) if lines else "- Extract facts relevant to the question."


# --------------------------------------------------------------------------
# frozen coordinator output (mirrors evidence.FixedEvidenceCache)
# --------------------------------------------------------------------------


class FrozenCoordinatorCache:
    """Disk-backed, precision-agnostic. Written once, read by all precisions."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: Dict[str, Any] = read_json(self.path, default={}) or {}
        self._dirty = False

    def get(self, question_id: str) -> Optional[Dict[str, Any]]:
        return self.data.get(question_id)

    def put(self, question_id: str, record: Dict[str, Any]) -> None:
        self.data[question_id] = record
        self._dirty = True

    def question_ids(self) -> List[str]:
        return sorted(self.data)

    def flush(self) -> None:
        if self._dirty:
            write_json(self.path, self.data)
            self._dirty = False


def freeze_coordinator(path: str | Path, client, question_id: str, question: str,
                       max_tokens: Optional[int] = None, force: bool = False,
                       allow_precision: Optional[str] = FREEZE_PRECISION
                       ) -> Dict[str, Any]:
    """Run the coordinator ONCE and freeze the result for every precision.

    Idempotent: an existing frozen record is returned untouched unless `force`.
    `allow_precision=None` disables the producer-precision check (tests only).
    """
    if allow_precision is not None and client.precision != allow_precision:
        raise ValueError(
            f"the coordinator may only be frozen at {allow_precision}; "
            f"client is at {client.precision}. Other precisions must call "
            "load_frozen_coordinator().")
    cache = FrozenCoordinatorCache(path)
    existing = cache.get(question_id)
    if existing is not None and not force:
        return existing
    out = run_coordinator(client, question_id, question, max_tokens=max_tokens)
    record = {
        "question_id": question_id,
        "shared_tasks": out.get("shared_tasks") or [],
        "synthesis_directive": out.get("synthesis_directive", ""),
        "frozen": True,
        "provenance": {
            "produced_by_precision": out.get("precision", client.precision),
            "stage": client.stage,
            "condition": client.condition,
            "call_id": out.get("call_id", ""),
            "prompt_hash": out.get("prompt_hash", ""),
            "coordinator_fallback": bool(out.get("coordinator_fallback")),
            "coordinator_fallback_reason": out.get(
                "coordinator_fallback_reason", FALLBACK_NONE),
            "parse_ok": bool(out.get("parse_ok")),
            "repair_attempted": bool(out.get("repair_attempted")),
            "repair_used": bool(out.get("repair_used")),
            "truncated": bool(out.get("truncated")),
        },
    }
    cache.put(question_id, record)
    cache.flush()
    return record


def load_frozen_coordinator(path: str | Path, question_id: str,
                            required: bool = True) -> Optional[Dict[str, Any]]:
    """Read the frozen coordinator output. Every precision reads the same bytes.

    Returns a dict shaped like run_coordinator()'s output (shared_tasks,
    synthesis_directive, coordinator_fallback, coordinator_fallback_reason,
    plus a `provenance` block naming the precision that produced it).
    """
    record = FrozenCoordinatorCache(path).get(question_id)
    if record is None:
        if required:
            raise KeyError(
                f"no frozen coordinator output for {question_id!r} in {path}. "
                f"Run freeze_coordinator() at {FREEZE_PRECISION} first.")
        return None
    prov = record.get("provenance") or {}
    return {
        "shared_tasks": record.get("shared_tasks") or [],
        "synthesis_directive": record.get("synthesis_directive", ""),
        "frozen": True,
        "coordinator_fallback": bool(prov.get("coordinator_fallback")),
        "coordinator_fallback_reason": prov.get("coordinator_fallback_reason",
                                                FALLBACK_NONE),
        "call_id": prov.get("call_id", ""),
        "prompt_hash": prov.get("prompt_hash", ""),
        "produced_by_precision": prov.get("produced_by_precision", ""),
        "provenance": prov,
    }
