"""Per-document private indexes with hard isolation, plus embedding cache.

Design choice: one PrivateIndex object per document, each holding only its own
chunks. Isolation is structural, not a filter that can be forgotten. A shared
store is still exposed (DocumentIndexStore) but every lookup goes through
`get(document_id)` and a `search` can never see another document's vectors.
`assert_isolation()` proves this at runtime and in tests.

Chunking and embedding are deterministic and cached by content hash, so the
fixed-evidence condition hands byte-identical chunk IDs to F16, Q8_0 and Q4_K_M.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from logging_utils import sha256_text  # noqa: E402

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Pinned revision -- change it only with a matching entry in results/provenance.
DEFAULT_EMBED_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------

_WS = re.compile(r"\s+")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def approx_tokens(text: str) -> int:
    """Whitespace-word count scaled to a rough token count (deterministic)."""
    return max(1, int(len(_WS.split(text.strip())) * 1.3))


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    ordinal: int
    text: str
    n_tokens: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def chunk_document(document_id: str, text: str, chunk_tokens: int = 400,
                   overlap_tokens: int = 50) -> List[Chunk]:
    """Sentence-aligned, fixed-size, fully deterministic chunking."""
    text = (text or "").strip()
    if not text:
        return []
    sentences = [s.strip() for s in _SENT.split(text) if s.strip()]
    if not sentences:
        sentences = [text]
    chunks: List[Chunk] = []
    i = 0
    ordinal = 0
    while i < len(sentences):
        buf: List[str] = []
        tok = 0
        j = i
        while j < len(sentences) and tok < chunk_tokens:
            buf.append(sentences[j])
            tok += approx_tokens(sentences[j])
            j += 1
        body = " ".join(buf)
        chunks.append(Chunk(
            chunk_id=f"{document_id}::c{ordinal:04d}",
            document_id=document_id, ordinal=ordinal, text=body,
            n_tokens=approx_tokens(body)))
        ordinal += 1
        if j >= len(sentences):
            break
        # step back far enough to cover `overlap_tokens` of context
        back = 0
        k = j - 1
        while k > i and back < overlap_tokens:
            back += approx_tokens(sentences[k])
            k -= 1
        i = max(i + 1, k + 1)
    return chunks


# --------------------------------------------------------------------------
# embeddings
# --------------------------------------------------------------------------


class Embedder:
    """SentenceTransformer wrapper with a pinned revision and a disk cache.

    `hash_fallback=True` swaps in a deterministic hashing embedder so the harness
    (and its tests) run with no model download. Real runs must use the real model;
    `is_real` is recorded in provenance.
    """

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL,
                 revision: str = DEFAULT_EMBED_REVISION,
                 cache_dir: str = "results/embed_cache", device: Optional[str] = None,
                 hash_fallback: bool = False, dim: int = 384):
        self.model_name = model_name
        self.revision = revision
        self.dim = dim
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hash_fallback = hash_fallback
        self._model = None
        self._device = device
        self._mem: Dict[str, np.ndarray] = {}

    @property
    def is_real(self) -> bool:
        return not self.hash_fallback

    def provenance(self) -> Dict[str, object]:
        return {"embed_model": self.model_name if self.is_real else "hash-fallback",
                "embed_revision": self.revision if self.is_real else "n/a",
                "embed_dim": self.dim, "embed_is_real_model": self.is_real}

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: WPS433
            self._model = SentenceTransformer(
                self.model_name, revision=self.revision, device=self._device)
            self.dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def _hash_embed(self, text: str) -> np.ndarray:
        """Deterministic bag-of-words hashing embedding (harness fallback)."""
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in _WS.split(text.lower()):
            tok = tok.strip(".,;:!?\"'()[]")
            if not tok:
                continue
            h = int(sha256_text(tok)[:16], 16)
            vec[h % self.dim] += 1.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.npy"

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        keys = [sha256_text(f"{self.model_name}|{self.revision}|{self.hash_fallback}|{t}")
                for t in texts]
        out: List[Optional[np.ndarray]] = [None] * len(texts)
        missing: List[int] = []
        for idx, key in enumerate(keys):
            if key in self._mem:
                out[idx] = self._mem[key]
                continue
            p = self._cache_path(key)
            if p.exists():
                try:
                    arr = np.load(p)
                    self._mem[key] = arr
                    out[idx] = arr
                    continue
                except Exception:
                    pass
            missing.append(idx)
        if missing:
            if self.hash_fallback:
                fresh = np.stack([self._hash_embed(texts[i]) for i in missing])
            else:
                model = self._load()
                fresh = np.asarray(model.encode(
                    [texts[i] for i in missing], batch_size=batch_size,
                    normalize_embeddings=True, show_progress_bar=False),
                    dtype=np.float32)
            for n, idx in enumerate(missing):
                arr = np.asarray(fresh[n], dtype=np.float32)
                self._mem[keys[idx]] = arr
                p = self._cache_path(keys[idx])
                p.parent.mkdir(parents=True, exist_ok=True)
                np.save(p, arr)
                out[idx] = arr
        return np.stack([o for o in out if o is not None])


# --------------------------------------------------------------------------
# indexes
# --------------------------------------------------------------------------


class DocumentIsolationError(RuntimeError):
    """Raised when an agent reaches for a document it was not assigned."""


@dataclass
class Hit:
    chunk_id: str
    document_id: str
    score: float
    text: str


class PrivateIndex:
    """A single document's retrieval universe. Holds only that document."""

    def __init__(self, document_id: str, chunks: List[Chunk], vectors: np.ndarray):
        if any(c.document_id != document_id for c in chunks):
            raise DocumentIsolationError(
                f"PrivateIndex({document_id}) was handed foreign chunks")
        self.document_id = document_id
        self.chunks = chunks
        self.vectors = np.asarray(vectors, dtype=np.float32)
        if len(chunks) != self.vectors.shape[0]:
            raise ValueError("chunk/vector count mismatch")

    @property
    def chunk_ids(self) -> List[str]:
        return [c.chunk_id for c in self.chunks]

    def search(self, query_vec: np.ndarray, top_k: int = 3) -> List[Hit]:
        if not self.chunks:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        qn = float(np.linalg.norm(q))
        if qn > 0:
            q = q / qn
        mat = self.vectors
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        sims = (mat / norms) @ q
        # deterministic ordering: score desc, then ordinal asc
        order = sorted(range(len(sims)), key=lambda i: (-float(sims[i]), i))
        return [Hit(chunk_id=self.chunks[i].chunk_id, document_id=self.document_id,
                    score=round(float(sims[i]), 6), text=self.chunks[i].text)
                for i in order[:top_k]]

    def get_chunk(self, chunk_id: str) -> Chunk:
        for c in self.chunks:
            if c.chunk_id == chunk_id:
                return c
        raise DocumentIsolationError(
            f"chunk {chunk_id!r} is not inside document {self.document_id!r}")


class DocumentIndexStore:
    """Owns one PrivateIndex per document id. The only handle agents get."""

    def __init__(self, embedder: Embedder, chunk_tokens: int = 400,
                 overlap_tokens: int = 50):
        self.embedder = embedder
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self._indexes: Dict[str, PrivateIndex] = {}

    def add_document(self, document_id: str, text: str) -> PrivateIndex:
        chunks = chunk_document(document_id, text, self.chunk_tokens,
                                self.overlap_tokens)
        vecs = (self.embedder.encode([c.text for c in chunks]) if chunks
                else np.zeros((0, self.embedder.dim), dtype=np.float32))
        idx = PrivateIndex(document_id, chunks, vecs)
        self._indexes[document_id] = idx
        return idx

    def get(self, document_id: str) -> PrivateIndex:
        if document_id not in self._indexes:
            raise DocumentIsolationError(f"no index for document {document_id!r}")
        return self._indexes[document_id]

    def document_ids(self) -> List[str]:
        return sorted(self._indexes)

    def search(self, document_id: str, query: str, top_k: int = 3) -> List[Hit]:
        qv = self.embedder.encode([query])[0]
        hits = self.get(document_id).search(qv, top_k=top_k)
        for h in hits:  # belt and braces
            if h.document_id != document_id:
                raise DocumentIsolationError(
                    f"cross-document leak: {h.chunk_id} returned for {document_id}")
        return hits

    # -- fixed evidence ---------------------------------------------------
    def fixed_evidence(self, document_id: str, question: str,
                       top_k: int = 3) -> List[Hit]:
        """Frozen chunk selection, identical for every precision."""
        return self.search(document_id, question, top_k=top_k)

    def assert_isolation(self) -> Dict[str, object]:
        """Prove every index can only ever return its own chunks."""
        report: Dict[str, object] = {"documents": len(self._indexes), "violations": []}
        all_ids = {did: set(ix.chunk_ids) for did, ix in self._indexes.items()}
        for did, ix in self._indexes.items():
            mine = all_ids[did]
            foreign = set().union(*[v for k, v in all_ids.items() if k != did]) \
                if len(all_ids) > 1 else set()
            if mine & foreign:
                report["violations"].append(
                    {"document_id": did, "shared_chunk_ids": sorted(mine & foreign)})
            if ix.chunks:
                hits = ix.search(ix.vectors[0], top_k=len(ix.chunks) + 5)
                for h in hits:
                    if h.chunk_id not in mine:
                        report["violations"].append(
                            {"document_id": did, "leaked": h.chunk_id})
            for other in self._indexes:
                if other == did:
                    continue
                other_chunk = self._indexes[other].chunk_ids[:1]
                if other_chunk:
                    try:
                        ix.get_chunk(other_chunk[0])
                        report["violations"].append(
                            {"document_id": did, "reachable_foreign_chunk": other_chunk[0]})
                    except DocumentIsolationError:
                        pass
        report["isolated"] = not report["violations"]
        return report


def format_chunks_for_prompt(hits: Iterable[Hit]) -> str:
    lines = []
    for h in hits:
        lines.append(f"[{h.chunk_id}]\n{h.text.strip()}")
    return "\n\n".join(lines) if lines else "(no passages retrieved)"
