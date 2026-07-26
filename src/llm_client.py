"""One guarded, logged, resumable model call: the only path to the model.

Responsibilities:
  * deterministic call_id -> resume without duplicating a prediction;
  * prompt hash recorded for every call (fixed-evidence parity check);
  * at most ONE structured-output repair attempt, with both the original and
    the repaired text written to the event log;
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


def messages_hash(messages: List[Dict[str, str]]) -> str:
    joined = "\n␟\n".join(f"{m['role']}:{m['content']}" for m in messages)
    return sha256_text(joined)


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
             extra: Optional[Dict[str, Any]] = None
             ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """Returns (parsed_dict_or_None, event_record). Resumes from the log."""
        cached = self.log.get(call_id) if self.log.completed(call_id) else None
        if cached is not None:
            return cached.get("parsed"), cached

        p_hash = messages_hash(messages)
        if self.dry_run:
            rec = {
                "call_id": call_id, "stage": self.stage, "precision": self.precision,
                "condition": self.condition, "role": role, "question_id": question_id,
                "document_id": document_id, "prompt_hash": p_hash, "dry_run": True,
                "is_fixture": True, "parsed": None,
                "prompt_chars": sum(len(m["content"]) for m in messages),
            }
            if extra:
                rec.update(extra)
            self.log.append(rec)
            return None, rec

        t0 = time.time()
        gen = self.backend.chat(messages, max_tokens=max_tokens)
        parse = parse_into(model_cls, gen.text)
        repair_text = None
        repair_gen = None
        if not parse.ok:
            repair_msgs = messages + [
                {"role": "assistant", "content": gen.text},
                {"role": "user", "content": fill(
                    REPAIR_INSTRUCTION, schema=schema,
                    previous=gen.text[:2000])},
            ]
            repair_gen = self.backend.chat(repair_msgs, max_tokens=max_tokens)
            repair_text = repair_gen.text
            parse = parse_into(model_cls, repair_text)
            if parse.ok:
                parse.repair_used = True

        total_gen_tokens = gen.generated_tokens + (
            repair_gen.generated_tokens if repair_gen else 0)
        total_prompt_tokens = gen.prompt_tokens + (
            repair_gen.prompt_tokens if repair_gen else 0)
        rec: Dict[str, Any] = {
            "call_id": call_id, "stage": self.stage, "precision": self.precision,
            "condition": self.condition, "role": role, "question_id": question_id,
            "document_id": document_id, "prompt_hash": p_hash,
            "sampling_hash": self.backend.provenance().get("sampling_hash", ""),
            "model_sha256": gen.model_sha256, "backend": gen.backend,
            "is_fixture": bool(gen.is_fixture or getattr(self.backend, "is_fixture", False)),
            "dry_run": False,
            "raw_text": gen.text,
            "repair_attempted": repair_gen is not None,
            "repair_text": repair_text,
            "repair_used": bool(parse.repair_used),
            "parse_ok": parse.ok, "parse_error": parse.error,
            "parsed": parse.model_obj,
            "prompt_tokens": total_prompt_tokens,
            "generated_tokens": total_gen_tokens,
            "latency_s": round(gen.latency_s + (repair_gen.latency_s if repair_gen else 0.0), 4),
            "wall_s": round(time.time() - t0, 4),
            "tokens_per_second": gen.tokens_per_second,
            "finish_reason": gen.finish_reason,
            "generation_error": gen.error,
        }
        if extra:
            rec.update(extra)
        self.log.append(rec)
        return parse.model_obj, rec
