"""Index construction and the frozen fixed-evidence cache.

The fixed-evidence chunk selection is computed ONCE per (question, document)
and cached to disk. Every precision then reads the same cache, which is what
makes the chunk-id and prompt-hash parity gates satisfiable by construction
rather than by hope.
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


def fixed_evidence_key(question_id: str, document_id: str, top_k: int,
                       chunk_tokens: int, overlap_tokens: int,
                       embed_sig: str) -> str:
    return sha256_text("|".join([question_id, document_id, str(top_k),
                                 str(chunk_tokens), str(overlap_tokens), embed_sig]))


class FixedEvidenceCache:
    """Disk-backed, precision-agnostic. Written once, read by all precisions."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: Dict[str, Any] = read_json(self.path, default={}) or {}
        self._dirty = False

    def get(self, key: str) -> Optional[List[Dict[str, Any]]]:
        entry = self.data.get(key)
        return entry.get("hits") if entry else None

    def put(self, key: str, question_id: str, document_id: str,
            hits: Sequence[Hit]) -> None:
        self.data[key] = {
            "question_id": question_id, "document_id": document_id,
            "hits": [{"chunk_id": h.chunk_id, "document_id": h.document_id,
                      "score": h.score, "text": h.text} for h in hits],
            "chunk_ids": [h.chunk_id for h in hits],
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
    key = fixed_evidence_key(question_id, document_id, top_k, chunk_tokens,
                             overlap_tokens, embed_sig)
    cached = cache.get(key)
    if cached is not None:
        return hits_from_cache(cached)
    hits = store.fixed_evidence(document_id, question, top_k=top_k)
    cache.put(key, question_id, document_id, hits)
    return hits


def embed_signature(embedder: Embedder) -> str:
    p = embedder.provenance()
    return f"{p['embed_model']}@{p['embed_revision']}"
