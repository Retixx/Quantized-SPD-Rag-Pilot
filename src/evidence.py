"""Index construction and the frozen fixed-evidence cache.

The fixed-evidence chunk selection is computed ONCE per (question, document)
and cached to disk. Every precision then reads the same cache, which is what
makes the chunk-id and prompt-hash parity gates satisfiable by construction
rather than by hope.

The cache key includes a hash of the indexed document body, so a changed
corpus produces a cache miss instead of silently reusing stale passages, and
every frozen record carries the provenance needed to audit it later.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from logging_utils import read_json, sha256_text, write_json  # noqa: E402
from retrieval import (DEFAULT_EMBED_MODEL, DEFAULT_EMBED_REVISION,  # noqa: E402
                       DocumentIndexStore, Embedder, Hit)


def make_embedder(models_cfg: Dict[str, Any], cache_dir: str) -> Embedder:
    emb = (models_cfg or {}).get("embedding") or {}
    return Embedder(model_name=emb.get("model") or DEFAULT_EMBED_MODEL,
                    revision=emb.get("revision") or DEFAULT_EMBED_REVISION,
                    cache_dir=cache_dir,
                    device=emb.get("device"),
                    hash_fallback=bool(emb.get("hash_fallback", False)))


def build_store(documents: Dict[str, Dict[str, Any]], embedder: Embedder,
                document_ids: Optional[Sequence[str]] = None,
                chunk_tokens: int = 400, overlap_tokens: int = 50
                ) -> DocumentIndexStore:
    store = DocumentIndexStore(embedder, chunk_tokens=chunk_tokens,
                               overlap_tokens=overlap_tokens)
    wanted = sorted(document_ids) if document_ids else sorted(documents)
    for did in wanted:
        rec = documents.get(did)
        if rec is None:
            raise KeyError(f"document {did!r} missing from pilot_documents.json")
        store.add_document(did, rec.get("body", ""))
    return store


def document_body_sha256(store: DocumentIndexStore, document_id: str) -> str:
    """Content hash of what is actually indexed for `document_id`.

    Ordered (chunk_id, chunk_text) pairs, so it moves if the corpus body, the
    chunker or the chunk parameters move.
    """
    return store.get(document_id).content_hash


def fixed_evidence_key(question_id: str, document_id: str, top_k: int,
                       chunk_tokens: int, overlap_tokens: int,
                       embed_sig: str, body_sha256: str = "") -> str:
    """Cache key for one frozen (question, document) evidence selection.

    `body_sha256` is part of the key: without it a changed corpus.json would be
    served stale text from the cache while the chunk-id parity gate still
    passed, because the ids are stable even when the bodies behind them are not.
    """
    return sha256_text("|".join([question_id, document_id, str(top_k),
                                 str(chunk_tokens), str(overlap_tokens), embed_sig,
                                 body_sha256]))


class FixedEvidenceCache:
    """Disk-backed, precision-agnostic. Written once, read by all precisions."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: Dict[str, Any] = read_json(self.path, default={}) or {}
        self._dirty = False

    def get(self, key: str) -> Optional[List[Dict[str, Any]]]:
        entry = self.data.get(key)
        return entry.get("hits") if entry else None

    def get_record(self, key: str) -> Optional[Dict[str, Any]]:
        """The whole frozen record, including its provenance block."""
        return self.data.get(key)

    def put(self, key: str, question_id: str, document_id: str,
            hits: Sequence[Hit], *, body_sha256: str = "",
            retrieval_params: Optional[Dict[str, Any]] = None,
            embed_signature: str = "",
            embedder_provenance: Optional[Dict[str, Any]] = None) -> None:
        """Freeze one selection. The provenance fields make it auditable later:
        which embedder produced it, with which retrieval parameters, over which
        document body."""
        self.data[key] = {
            "question_id": question_id, "document_id": document_id,
            "hits": [{"chunk_id": h.chunk_id, "document_id": h.document_id,
                      "score": h.score, "text": h.text} for h in hits],
            "chunk_ids": [h.chunk_id for h in hits],
            "key": key,
            "body_sha256": body_sha256,
            "retrieval_params": dict(retrieval_params or {}),
            "embed_signature": embed_signature,
            "embedder_provenance": dict(embedder_provenance or {}),
        }
        self._dirty = True

    def flush(self) -> None:
        if self._dirty:
            write_json(self.path, self.data)
            self._dirty = False


def hits_from_cache(records: Sequence[Dict[str, Any]]) -> List[Hit]:
    return [Hit(chunk_id=r["chunk_id"], document_id=r["document_id"],
                score=float(r.get("score", 0.0)), text=r["text"]) for r in records]


def get_fixed_evidence(store: DocumentIndexStore, cache: FixedEvidenceCache,
                       question_id: str, question: str, document_id: str,
                       top_k: int, chunk_tokens: int, overlap_tokens: int,
                       embed_sig: str) -> List[Hit]:
    body_sha = document_body_sha256(store, document_id)
    key = fixed_evidence_key(question_id, document_id, top_k, chunk_tokens,
                             overlap_tokens, embed_sig, body_sha)
    cached = cache.get(key)
    if cached is not None:
        return hits_from_cache(cached)
    hits = store.fixed_evidence(document_id, question, top_k=top_k)
    cache.put(key, question_id, document_id, hits, body_sha256=body_sha,
              retrieval_params={"top_k": top_k, "chunk_tokens": chunk_tokens,
                                "overlap_tokens": overlap_tokens},
              embed_signature=embed_sig,
              embedder_provenance=store.embedder.provenance())
    return hits


def embed_signature(embedder: Embedder) -> str:
    p = embedder.provenance()
    return f"{p['embed_model']}@{p['embed_revision']}"
