"""Append-only JSONL event log, deterministic IDs, and resume support.

Every generation is flushed to disk immediately (write + flush + fsync) so an
interrupted Kaggle session loses at most the in-flight call.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Set

_LOCK = threading.Lock()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def stable_id(*parts: Any) -> str:
    """Deterministic short id from its parts. Never depends on wall clock."""
    joined = "␟".join(str(p) for p in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


def call_id(stage: str, precision: str, condition: str, question_id: str,
            role: str, document_id: str = "-", attempt: int = 0) -> str:
    return stable_id(stage, precision, condition, question_id, role, document_id, attempt)


class EventLog:
    """JSONL writer with fsync-per-record and a resume index."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seen: Set[str] = set()
        if self.path.exists():
            for rec in self.read():
                cid = rec.get("call_id")
                if cid:
                    self._seen.add(cid)

    # -- reading ---------------------------------------------------------
    def read(self) -> Iterator[Dict[str, Any]]:
        if not self.path.exists():
            return iter(())

        def _gen():
            with open(self.path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        # A torn final line from a hard kill. Skip, never crash.
                        continue

        return _gen()

    def completed(self, cid: str) -> bool:
        return cid in self._seen

    def get(self, cid: str) -> Optional[Dict[str, Any]]:
        for rec in self.read():
            if rec.get("call_id") == cid:
                return rec
        return None

    # -- writing ---------------------------------------------------------
    def append(self, record: Dict[str, Any]) -> Dict[str, Any]:
        record.setdefault("logged_at", time.time())
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with _LOCK:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        cid = record.get("call_id")
        if cid:
            self._seen.add(cid)
        return record


def write_json(path: str | os.PathLike, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)


def read_json(path: str | os.PathLike, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)
