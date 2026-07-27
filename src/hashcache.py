"""On-disk sha256 cache keyed by (absolute path, size, mtime_ns).

The three GGUFs total ~11.5 GB and used to be hashed from scratch by
prepare_models.py, again per precision block in run_stage.py, and again in
prepare_kaggle.py -- three or four full passes over the same bytes on a Kaggle
session that is capped at 9 hours.

The cache is deliberately conservative: any change to size or mtime_ns
invalidates the entry, so a re-quantized file is always re-hashed. Deleting the
cache file only costs time, never correctness. Use `force=True` when a hash is
being recorded as evidence and you want to pay for a fresh read.

    from hashcache import sha256_file_cached
    digest = sha256_file_cached(path)
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from logging_utils import sha256_file, write_json

CACHE_FILENAME = "hash_cache.json"
_LOCK = threading.Lock()
_MEM: Dict[str, Dict[str, Any]] = {}


def default_cache_path() -> Path:
    """results/hash_cache.json, honouring the Kaggle results dir when present."""
    try:
        from config import results_dir  # noqa: WPS433 - optional, keeps this standalone
        return results_dir() / CACHE_FILENAME
    except Exception:
        return Path("results") / CACHE_FILENAME


def _key(path: Path) -> str:
    st = path.stat()
    return f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}"


def _load(cache_path: Path) -> Dict[str, Any]:
    cached = _MEM.get(str(cache_path))
    if cached is not None:
        return cached
    data: Dict[str, Any] = {}
    if cache_path.exists():
        try:
            loaded = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except Exception:
            data = {}          # a corrupt cache is a performance bug, never a fatal one
    _MEM[str(cache_path)] = data
    return data


def sha256_file_cached(path: str | os.PathLike,
                       cache_path: Optional[str | os.PathLike] = None,
                       force: bool = False) -> str:
    """sha256 of `path`, reusing a previous digest when size and mtime match."""
    p = Path(path)
    cp = Path(cache_path) if cache_path is not None else default_cache_path()
    key = _key(p)
    with _LOCK:
        data = _load(cp)
        if not force:
            hit = data.get(key)
            if isinstance(hit, str) and len(hit) == 64:
                return hit
    digest = sha256_file(p)
    with _LOCK:
        data = _load(cp)
        data[key] = digest
        try:
            write_json(cp, data)
        except Exception:
            pass               # read-only results dir: still correct, just uncached
    return digest


def invalidate(path: str | os.PathLike,
               cache_path: Optional[str | os.PathLike] = None) -> None:
    """Forget every entry for `path` (any size/mtime). Call after overwriting."""
    p = Path(path).resolve()
    cp = Path(cache_path) if cache_path is not None else default_cache_path()
    with _LOCK:
        data = _load(cp)
        for key in [k for k in data if k.split("|")[0] == str(p)]:
            data.pop(key, None)
        try:
            write_json(cp, data)
        except Exception:
            pass
