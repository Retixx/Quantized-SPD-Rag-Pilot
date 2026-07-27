"""One guarded, logged, resumable model call: the only path to the model.

Responsibilities:
  * deterministic call_id -> resume without duplicating a prediction;
  * prompt hash recorded for every call (fixed-evidence parity check);
  * at most ONE structured-output repair attempt, with both the original and
    the repaired text written to the event log;
  * infrastructure failures (502, timeout, context overflow) raise
    `BackendGenerationError` instead of being scored as model quality;
  * every record carries the same JSON-compliance fields (parse_ok,
    repair_attempted, repair_used, truncated, generation_error) so
    per-precision compliance rates aggregate without special cases;
  * every record carries is_fixture, so fixture data can be quarantined.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from logging_utils import EventLog, sha256_text  # noqa: E402
from prompts import fill  # noqa: E402
from schemas import REPAIR_INSTRUCTION, parse_into  # noqa: E402

# The repair prompt is strictly longer than the original (it appends the failed
# reply and the schema), so reusing the same budget makes a truncation-induced
# failure near-certain to fail repair too.
REPAIR_TOKEN_FACTOR = 1.5
REPAIR_TOKEN_FLOOR = 256

# Fields written on EVERY record, whatever path produced it.
_COMPLIANCE_DEFAULTS: Dict[str, Any] = {
    "parse_ok": False,
    "parse_error": None,
    "repair_attempted": False,
    "repair_used": False,
    "truncated": False,
    "empty_generation": False,
    "generation_error": None,
    "finish_reason": "",
    "repair_finish_reason": "",
    "finish_reasons": [],
}


class BackendGenerationError(RuntimeError):
    """The backend failed to generate: HTTP error, timeout, context overflow.

    Infrastructure, not model quality. run_stage should log a question_failure
    and may retry; the call_id is deliberately NOT marked completed, so a
    resume re-runs the call instead of inheriting an empty answer.
    """

    def __init__(self, message: str, *, call_id: str = "", role: str = "",
                 question_id: str = "", document_id: str = "",
                 precision: str = "", backend: str = ""):
        super().__init__(message)
        self.call_id = call_id
        self.role = role
        self.question_id = question_id
        self.document_id = document_id
        self.precision = precision
        self.backend = backend


def messages_hash(messages: List[Dict[str, str]]) -> str:
    joined = "\n␟\n".join(f"{m['role']}:{m['content']}" for m in messages)
    return sha256_text(joined)


def repair_budget(max_tokens: Optional[int]) -> Optional[int]:
    """A strictly larger budget for the repair call."""
    if not max_tokens:
        return None
    return int(max(max_tokens * REPAIR_TOKEN_FACTOR, max_tokens + REPAIR_TOKEN_FLOOR))


class GuardedClient:
    def __init__(self, backend, event_log: EventLog, stage: str, precision: str,
                 condition: str, dry_run: bool = False):
        self.backend = backend
        self.log = event_log
        self.stage = stage
        self.precision = precision
        self.condition = condition
        self.dry_run = dry_run

    def call(self, *, call_id: str, role: str, question_id: str,
             messages: List[Dict[str, str]], model_cls, schema: str,
             document_id: str = "-", max_tokens: Optional[int] = None,
             repair_max_tokens: Optional[int] = None,
             extra: Optional[Dict[str, Any]] = None
             ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """Returns (parsed_dict_or_None, event_record). Resumes from the log.

        Raises BackendGenerationError if the backend reports an error.
        """
        cached = self.log.get(call_id) if self.log.completed(call_id) else None
        if cached is not None:
            return cached.get("parsed"), cached

        p_hash = messages_hash(messages)
        if self.dry_run:
            rec = dict(_COMPLIANCE_DEFAULTS)
            rec.update({
                "call_id": call_id, "stage": self.stage, "precision": self.precision,
                "condition": self.condition, "role": role, "question_id": question_id,
                "document_id": document_id, "prompt_hash": p_hash, "dry_run": True,
                "is_fixture": True, "parsed": None,
                "prompt_chars": sum(len(m["content"]) for m in messages),
            })
            if extra:
                rec.update(extra)
            self.log.append(rec)
            return None, rec

        t0 = time.time()
        gen = self.backend.chat(messages, max_tokens=max_tokens)

        if gen.error:
            # Infrastructure failure. Repairing against the same broken server is
            # pointless, and scoring it as a parse failure would charge model
            # quality for a 502. Record a diagnostic event with NO call_id (so
            # the resume index stays clean and this call is retried), then raise.
            err_rec = dict(_COMPLIANCE_DEFAULTS)
            err_rec.update({
                "call_id": "", "failed_call_id": call_id, "event": "generation_error",
                "stage": self.stage, "precision": self.precision,
                "condition": self.condition, "role": role, "question_id": question_id,
                "document_id": document_id, "prompt_hash": p_hash, "dry_run": False,
                "is_fixture": bool(gen.is_fixture
                                   or getattr(self.backend, "is_fixture", False)),
                "backend": gen.backend, "model_sha256": gen.model_sha256,
                "generation_error": gen.error,
                "finish_reason": gen.finish_reason,
                "finish_reasons": [gen.finish_reason],
                "raw_text": gen.text,
                "latency_s": round(gen.latency_s, 4),
                "wall_s": round(time.time() - t0, 4),
                "parsed": None,
            })
            if extra:
                err_rec.update(extra)
            self.log.append(err_rec)
            raise BackendGenerationError(
                f"{role} generation failed at {self.precision}: {gen.error}",
                call_id=call_id, role=role, question_id=question_id,
                document_id=document_id, precision=self.precision,
                backend=gen.backend)

        empty = not (gen.text or "").strip()
        parse = parse_into(model_cls, gen.text)
        repair_text = None
        repair_gen = None
        # Exactly one repair attempt, never two -- and none at all for an empty
        # generation, which has nothing to repair.
        if not parse.ok and not empty:
            repair_msgs = messages + [
                {"role": "assistant", "content": gen.text},
                {"role": "user", "content": fill(
                    REPAIR_INSTRUCTION, schema=schema,
                    previous=gen.text[:2000])},
            ]
            budget = repair_max_tokens if repair_max_tokens else repair_budget(max_tokens)
            repair_gen = self.backend.chat(repair_msgs, max_tokens=budget)
            if repair_gen.error:
                err_rec = dict(_COMPLIANCE_DEFAULTS)
                err_rec.update({
                    "call_id": "", "failed_call_id": call_id,
                    "event": "generation_error", "repair_attempted": True,
                    "stage": self.stage, "precision": self.precision,
                    "condition": self.condition, "role": role,
                    "question_id": question_id, "document_id": document_id,
                    "prompt_hash": p_hash, "dry_run": False,
                    "is_fixture": bool(gen.is_fixture or repair_gen.is_fixture
                                       or getattr(self.backend, "is_fixture", False)),
                    "backend": gen.backend, "model_sha256": gen.model_sha256,
                    "generation_error": repair_gen.error,
                    "raw_text": gen.text, "repair_text": repair_gen.text,
                    "finish_reason": gen.finish_reason,
                    "repair_finish_reason": repair_gen.finish_reason,
                    "finish_reasons": [gen.finish_reason, repair_gen.finish_reason],
                    "latency_s": round(gen.latency_s + repair_gen.latency_s, 4),
                    "wall_s": round(time.time() - t0, 4),
                    "parsed": None,
                })
                if extra:
                    err_rec.update(extra)
                self.log.append(err_rec)
                raise BackendGenerationError(
                    f"{role} repair generation failed at {self.precision}: "
                    f"{repair_gen.error}",
                    call_id=call_id, role=role, question_id=question_id,
                    document_id=document_id, precision=self.precision,
                    backend=gen.backend)
            repair_text = repair_gen.text
            parse = parse_into(model_cls, repair_text)
            if parse.ok:
                parse.repair_used = True

        total_gen_tokens = gen.generated_tokens + (
            repair_gen.generated_tokens if repair_gen else 0)
        total_prompt_tokens = gen.prompt_tokens + (
            repair_gen.prompt_tokens if repair_gen else 0)
        total_latency = gen.latency_s + (repair_gen.latency_s if repair_gen else 0.0)
        finish_reasons = [gen.finish_reason] + (
            [repair_gen.finish_reason] if repair_gen else [])
        rec: Dict[str, Any] = dict(_COMPLIANCE_DEFAULTS)
        rec.update({
            "call_id": call_id, "stage": self.stage, "precision": self.precision,
            "condition": self.condition, "role": role, "question_id": question_id,
            "document_id": document_id, "prompt_hash": p_hash,
            "sampling_hash": self.backend.provenance().get("sampling_hash", ""),
            "model_sha256": gen.model_sha256, "backend": gen.backend,
            "is_fixture": bool(gen.is_fixture
                               or (repair_gen.is_fixture if repair_gen else False)
                               or getattr(self.backend, "is_fixture", False)),
            "dry_run": False,
            "raw_text": gen.text,
            "empty_generation": empty,
            "repair_attempted": repair_gen is not None,
            "repair_text": repair_text,
            "repair_used": bool(parse.repair_used),
            "parse_ok": parse.ok, "parse_error": parse.error,
            "parsed": parse.model_obj,
            "prompt_tokens": total_prompt_tokens,
            "generated_tokens": total_gen_tokens,
            "latency_s": round(total_latency, 4),
            "wall_s": round(time.time() - t0, 4),
            # recomputed over BOTH calls: summing tokens while keeping the first
            # call's rate made generated_tokens / latency_s disagree with
            # tokens_per_second, and the divergence tracked precision because
            # weaker quantizations repair more often.
            "tokens_per_second": round(total_gen_tokens / total_latency, 6)
            if total_latency > 0 else 0.0,
            "finish_reason": gen.finish_reason,
            "repair_finish_reason": repair_gen.finish_reason if repair_gen else "",
            "finish_reasons": finish_reasons,
            # "length" truncation is otherwise indistinguishable from malformed
            # output once the text is thrown at the parser.
            "truncated": any(fr == "length" for fr in finish_reasons),
            "generation_error": None,
        })
        if extra:
            rec.update(extra)
        self.log.append(rec)
        return parse.model_obj, rec
