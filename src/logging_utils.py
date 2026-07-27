"""Append-only JSONL event log, deterministic IDs, and resume support.

Every generation is flushed to disk immediately (write + flush + fsync) so an
interrupted Kaggle session loses at most the in-flight call.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

_LOCK = threading.Lock()
_LOGGER = logging.getLogger(__name__)


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


def stable_id(*parts: Any, variant: str = "") -> str:
    """Deterministic short id from its parts. Never depends on wall clock.

    `variant` is appended as a trailing component only when it is non-empty, so
    ids built without it stay byte-identical to every id already on disk. Use
    it to separate run modes that must not share a resume slot.
    """
    joined = "␟".join(str(p) for p in parts)
    if variant:
        joined = f"{joined}␟{variant}"
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


def call_id(stage: str, precision: str, condition: str, question_id: str,
            role: str, document_id: str = "-", attempt: int = 0,
            variant: str = "") -> str:
    """Deterministic id for one model call; the resume key.

    `attempt` is a real component of the id, not decoration: bump it and the
    call gets a distinct id, so a retry (or an agent's second turn, which
    passes its turn index here) is logged as its own record instead of being
    swallowed by the resume cache. Leave it at 0 for the first attempt.

    `variant` distinguishes run modes that would otherwise collide -- an empty
    variant reproduces the historical id exactly.
    """
    return stable_id(stage, precision, condition, question_id, role, document_id,
                     attempt, variant=variant)


class CorruptEventLogError(RuntimeError):
    """A JSONL line that is not the torn final line failed to decode."""


class EventLog:
    """JSONL writer with fsync-per-record and a resume index.

    The file is parsed once and kept in memory; `read()` and `get()` serve that
    cache, which `append()` extends. The cache is re-read only if the file
    changes underneath us (size or mtime), so a resumed stage costs one scan
    rather than one scan per lookup.

    A line that fails to decode is tolerated silently ONLY if it is the last
    line in the file -- that is the torn write a hard kill leaves behind.
    Anything corrupt further up is real damage: it is counted, warned about and
    exposed via `corrupt_line_count` / `corrupt_line_numbers`.
    """

    def __init__(self, path: str | os.PathLike, strict: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.strict = strict
        self._seen: Set[str] = set()
        self._records: Optional[List[Dict[str, Any]]] = None
        self._corrupt_lines: List[int] = []
        self._torn_final_line = False
        self._file_sig: Optional[Tuple[int, int]] = None
        self._load()

    # -- reading ---------------------------------------------------------
    def _signature(self) -> Optional[Tuple[int, int]]:
        try:
            st = self.path.stat()
        except OSError:
            return None
        return (st.st_size, st.st_mtime_ns)

    def _load(self, force: bool = False) -> List[Dict[str, Any]]:
        sig = self._signature()
        if not force and self._records is not None and sig == self._file_sig:
            return self._records
        records: List[Dict[str, Any]] = []
        corrupt: List[int] = []
        torn = False
        if sig is not None:
            raw: List[Tuple[int, str]] = []
            with open(self.path, "r", encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, start=1):
                    line = line.strip()
                    if line:
                        raw.append((lineno, line))
            last_lineno = raw[-1][0] if raw else -1
            for lineno, line in raw:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    if lineno == last_lineno:
                        # A torn final line from a hard kill. Skip, never crash.
                        torn = True
                        continue
                    corrupt.append(lineno)
        if corrupt:
            msg = (f"{self.path}: {len(corrupt)} corrupt mid-file JSONL line(s) "
                   f"at {corrupt[:10]} were skipped -- this is NOT a torn final "
                   f"line and means records were lost")
            if self.strict:
                raise CorruptEventLogError(msg)
            _LOGGER.warning(msg)
        self._records = records
        self._corrupt_lines = corrupt
        self._torn_final_line = torn
        self._file_sig = sig
        self._seen = {r["call_id"] for r in records if r.get("call_id")}
        return records

    @property
    def corrupt_line_count(self) -> int:
        """Corrupt lines that were NOT the torn final line (i.e. lost records)."""
        self._load()
        return len(self._corrupt_lines)

    @property
    def corrupt_line_numbers(self) -> List[int]:
        """1-based line numbers of the lost records."""
        self._load()
        return list(self._corrupt_lines)

    @property
    def torn_final_line(self) -> bool:
        """True if the last line was unparseable -- the tolerated hard-kill case."""
        self._load()
        return self._torn_final_line

    def read(self) -> Iterator[Dict[str, Any]]:
        return iter(list(self._load()))

    def completed(self, cid: str) -> bool:
        self._load()
        return cid in self._seen

    def get(self, cid: str) -> Optional[Dict[str, Any]]:
        for rec in self._load():
            if rec.get("call_id") == cid:
                return rec
        return None

    # -- writing ---------------------------------------------------------
    def _needs_newline(self) -> bool:
        """True if the file ends mid-line (a torn write) and needs closing off."""
        try:
            if not self.path.stat().st_size:
                return False
            with open(self.path, "rb") as fh:
                fh.seek(-1, os.SEEK_END)
                return fh.read(1) != b"\n"
        except OSError:
            return False

    def append(self, record: Dict[str, Any]) -> Dict[str, Any]:
        record.setdefault("logged_at", time.time())
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with _LOCK:
            self._load()  # pick up anyone else's writes before extending the cache
            torn = self._torn_final_line
            with open(self.path, "a", encoding="utf-8") as fh:
                # a torn final line has no newline: keep it from swallowing this record
                fh.write(("\n" if self._needs_newline() else "") + line + "\n")
                fh.flush()
                os.fsync(fh.fileno())  # durable before we call it logged
            if torn:
                # that torn line is now mid-file damage; recount it on next read
                self._records = None
                self._file_sig = None
            else:
                if self._records is not None:
                    self._records.append(record)
                self._file_sig = self._signature()
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
