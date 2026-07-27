"""Per-document private indexes with hard isolation, plus embedding cache.

Design choice: one PrivateIndex object per document, each holding only its own
chunks. Isolation is structural, not a filter that can be forgotten. A shared
store is still exposed (DocumentIndexStore) but every lookup goes through
`get(document_id)` and a `search` can never see another document's vectors.
A PrivateIndex carries its own embedder, so an agent can run `search_text()`
holding nothing but its index -- it never needs, or gets, the store.
`assert_isolation()` proves this at runtime and in tests.

Chunking and embedding are deterministic and cached by content hash, so the
fixed-evidence condition hands byte-identical chunk IDs to F16, Q8_0 and Q4_K_M.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

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
# The token immediately left of a candidate boundary, minus its final period.
_TRAILING_TOKEN = re.compile(r"([A-Za-z][A-Za-z.]*)\.$")
# "U.S", "e.g", "J.R" -- a dotted initialism is never a sentence end.
_INITIALISM = re.compile(r"^(?:[A-Za-z]\.)+[A-Za-z]?$")

# Abbreviations that end in a period mid-sentence. News corpora are full of
# them ("Jan. 5", "U.S. Treasury", "Mr. Smith"), and splitting there produces
# fragments that retrieve badly. Deliberately excludes tokens that are also
# ordinary words ("no.", "etc.", weekday abbreviations): wrongly refusing a
# split costs more than wrongly taking one, since a real sentence end after
# "etc." is common while "No. 5" is rare.
_ABBREVIATIONS = frozenset("""
mr mrs ms mx dr prof sr jr st rev hon gov sen rep gen adm lt col sgt capt supt
jan feb mar apr jun jul aug sep sept oct nov dec
inc ltd co corp llc llp plc dept div fig figs vol vols pp
al eg ie vs approx univ assn bros ave blvd rd mt
a.m p.m u.s u.k u.n e.u e.g i.e
""".split())

# words -> tokens. A crude heuristic, NOT a tokenizer count; see
# `token_estimate_error()` to compare it against a server's `prompt_tokens`.
TOKEN_ESTIMATE_SCALE = 1.3


def estimate_tokens(text: str) -> int:
    """ESTIMATE a token count from whitespace words (deterministic, no model).

    This is a heuristic (`words * TOKEN_ESTIMATE_SCALE`), never a real
    tokenization. It is used for chunk sizing only; anything reported as a
    token count in results must come from the server's own accounting.
    """
    return max(1, int(len(_WS.split(text.strip())) * TOKEN_ESTIMATE_SCALE))


# Backwards-compatible alias: the older, less explicit name.
approx_tokens = estimate_tokens


def token_estimate_error(text: str, actual_tokens: int) -> Dict[str, float]:
    """Compare the heuristic against a real `prompt_tokens` from the server.

    Callers that have both a prompt string and the server-reported count can
    record this so the 1.3 scale is validated rather than assumed.
    """
    est = estimate_tokens(text)
    actual = int(actual_tokens)
    return {"estimated_tokens": float(est), "actual_tokens": float(actual),
            "abs_error": float(est - actual),
            "ratio": round(est / actual, 6) if actual > 0 else 0.0,
            "scale": TOKEN_ESTIMATE_SCALE}


# -- chunk ids: one place that builds them, one place that parses them ------

CHUNK_ID_SEP = "::c"


def format_chunk_id(document_id: str, ordinal: int) -> str:
    return f"{document_id}{CHUNK_ID_SEP}{ordinal:04d}"


def parse_chunk_id(chunk_id: str) -> Tuple[str, int]:
    """Inverse of `format_chunk_id`. Raises ValueError on a malformed id."""
    document_id, sep, ordinal = str(chunk_id).rpartition(CHUNK_ID_SEP)
    if not sep or not document_id or not ordinal.isdigit():
        raise ValueError(f"malformed chunk_id {chunk_id!r}")
    return document_id, int(ordinal)


def _ends_with_abbreviation(left: str) -> bool:
    """True if `left` ends in an abbreviation rather than a real sentence."""
    stripped = left.rstrip()
    if not stripped.endswith("."):  # '!' and '?' never abbreviate
        return False
    m = _TRAILING_TOKEN.search(stripped)
    if not m:
        return False
    token = m.group(1)
    if token.lower() in _ABBREVIATIONS:
        return True
    # a lone initial ("J. Smith") or any dotted initialism ("U.S. Treasury")
    return len(token) == 1 or _INITIALISM.match(token) is not None


def split_sentences(text: str) -> List[str]:
    """Split into sentences, refusing boundaries that follow an abbreviation."""
    text = (text or "").strip()
    if not text:
        return []
    out: List[str] = []
    start = 0
    for m in _SENT.finditer(text):
        left = text[start:m.start()]
        if _ends_with_abbreviation(left):
            continue
        piece = left.strip()
        if piece:
            out.append(piece)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def _words(text: str) -> List[str]:
    return [w for w in _WS.split(text.strip()) if w]


def _budget_words(chunk_tokens: int) -> int:
    """Word budget whose token ESTIMATE stays inside `chunk_tokens`."""
    return max(1, int(max(1, chunk_tokens) / TOKEN_ESTIMATE_SCALE))


def split_on_token_budget(text: str, budget_tokens: int) -> List[str]:
    """Hard fallback: cut `text` on word boundaries into <= budget pieces.

    Used when the sentence splitter yields a segment larger than one chunk --
    including the degenerate case of text with no detectable sentence ends,
    where the whole article would otherwise become a single chunk.
    """
    words = _words(text)
    if not words:
        return []
    per = _budget_words(budget_tokens)
    return [" ".join(words[i:i + per]) for i in range(0, len(words), per)]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    ordinal: int
    text: str
    n_tokens: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _chunk_units(text: str, chunk_tokens: int) -> List[str]:
    """Sentences, with any over-long sentence cut down to the chunk budget."""
    sentences = split_sentences(text) or [text]
    max_words = _budget_words(chunk_tokens)
    units: List[str] = []
    for s in sentences:
        if len(_words(s)) > max_words:
            units.extend(split_on_token_budget(s, chunk_tokens))
        else:
            units.append(s)
    return [u for u in units if u.strip()]


def chunk_document(document_id: str, text: str, chunk_tokens: int = 400,
                   overlap_tokens: int = 50) -> List[Chunk]:
    """Sentence-aligned, fixed-size, fully deterministic chunking.

    No chunk exceeds `chunk_tokens`: a sentence that would overshoot starts the
    next chunk instead, and a sentence that is itself larger than the budget is
    split on a word budget first. Sizing is done in words, because the token
    estimate of a joined chunk is not the sum of its parts' estimates.
    """
    text = (text or "").strip()
    if not text:
        return []
    sentences = _chunk_units(text, chunk_tokens)
    if not sentences:
        return []
    max_words = _budget_words(chunk_tokens)
    chunks: List[Chunk] = []
    i = 0
    ordinal = 0
    while i < len(sentences):
        buf: List[str] = []
        used = 0
        j = i
        while j < len(sentences):
            w = len(_words(sentences[j]))
            if buf and used + w > max_words:
                break  # bound the overshoot: this sentence starts the next chunk
            buf.append(sentences[j])
            used += w
            j += 1
        body = " ".join(buf)
        chunks.append(Chunk(
            chunk_id=format_chunk_id(document_id, ordinal),
            document_id=document_id, ordinal=ordinal, text=body,
            n_tokens=estimate_tokens(body)))
        ordinal += 1
        if j >= len(sentences):
            break
        # step back far enough to cover `overlap_tokens` of context
        back = 0
        k = j - 1
        while k > i and back < overlap_tokens:
            back += estimate_tokens(sentences[k])
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
        if not out:
            return np.zeros((0, self.dim), dtype=np.float32)
        empty = [i for i, o in enumerate(out) if o is None]
        if empty:
            # Dropping a slot would silently return fewer rows than inputs and
            # misalign every chunk with its vector. Fail loudly instead.
            raise RuntimeError(
                f"Embedder.encode produced no vector for {len(empty)} of "
                f"{len(texts)} inputs (first missing index {empty[0]}); "
                "refusing to return misaligned rows")
        return np.stack(out)

    def close(self) -> None:
        """Drop the model so its CUDA context can be reclaimed between blocks.

        Also clears the in-memory vector cache; the disk cache is untouched, so
        a later `encode()` still returns byte-identical vectors (it just reloads).
        Safe to call more than once, and safe when no model was ever loaded.
        The torch cache is only emptied if torch is already imported AND
        already holds a CUDA context -- this never creates one.
        """
        self._model = None
        self._mem.clear()
        torch = sys.modules.get("torch")
        if torch is None:
            return
        try:
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.empty_cache()
        except Exception:  # a broken/partial torch must never fail a run
            pass

    # `release` reads better at a stage boundary; same operation.
    release = close


# --------------------------------------------------------------------------
# indexes
# --------------------------------------------------------------------------


# A generic news-shaped query used by assert_isolation() to probe every index
# with a REAL retrieval call. Content-free on purpose: it must match something
# in any document rather than favour one corpus.
ISOLATION_PROBE_QUERY = "who said what happened, when, where and how much"


class DocumentIsolationError(RuntimeError):
    """Raised when an agent reaches for a document it was not assigned."""


@dataclass
class Hit:
    chunk_id: str
    document_id: str
    score: float
    text: str


class PrivateIndex:
    """A single document's retrieval universe. Holds only that document.

    It also holds the embedder, so an agent handed nothing but a PrivateIndex
    can run a full text query (`search_text`) without ever seeing the store --
    isolation stops depending on the caller passing the right document id.
    """

    def __init__(self, document_id: str, chunks: List[Chunk], vectors: np.ndarray,
                 embedder: Optional[Embedder] = None):
        if any(c.document_id != document_id for c in chunks):
            raise DocumentIsolationError(
                f"PrivateIndex({document_id}) was handed foreign chunks")
        self.document_id = document_id
        self.chunks = chunks
        self.vectors = np.asarray(vectors, dtype=np.float32)
        if len(chunks) != self.vectors.shape[0]:
            raise ValueError("chunk/vector count mismatch")
        self._embedder = embedder

    @property
    def chunk_ids(self) -> List[str]:
        return [c.chunk_id for c in self.chunks]

    @property
    def content_hash(self) -> str:
        """sha256 over this document's ordered (chunk_id, chunk_text) pairs.

        Identifies the indexed content itself, so a cache keyed on it cannot
        serve text from a corpus that has since changed.
        """
        return sha256_text("\n".join(f"{c.chunk_id}\x1f{c.text}" for c in self.chunks))

    def search_text(self, query: str, top_k: int = 3) -> List[Hit]:
        """Embed `query` and search THIS document. No store handle required."""
        if self._embedder is None:
            raise DocumentIsolationError(
                f"PrivateIndex({self.document_id!r}) has no embedder; build it "
                "through DocumentIndexStore.add_document to use search_text")
        if not self.chunks:
            return []
        hits = self.search(self._embedder.encode([query])[0], top_k=top_k)
        for h in hits:  # belt and braces
            if h.document_id != self.document_id:
                raise DocumentIsolationError(
                    f"cross-document leak: {h.chunk_id} returned for "
                    f"{self.document_id}")
        return hits

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
        idx = PrivateIndex(document_id, chunks, vecs, embedder=self.embedder)
        self._indexes[document_id] = idx
        return idx

    def get(self, document_id: str) -> PrivateIndex:
        if document_id not in self._indexes:
            raise DocumentIsolationError(f"no index for document {document_id!r}")
        return self._indexes[document_id]

    def document_ids(self) -> List[str]:
        return sorted(self._indexes)

    def search(self, document_id: str, query: str, top_k: int = 3) -> List[Hit]:
        return self.get(document_id).search_text(query, top_k=top_k)

    # -- fixed evidence ---------------------------------------------------
    def fixed_evidence(self, document_id: str, question: str,
                       top_k: int = 3) -> List[Hit]:
        """Frozen chunk selection, identical for every precision."""
        return self.search(document_id, question, top_k=top_k)

    def assert_isolation(self, probe_query: str = ISOLATION_PROBE_QUERY,
                         max_pairs: int = 400, probe_top_k: int = 5
                         ) -> Dict[str, object]:
        """Prove every index can only ever return its own chunks.

        Five independent checks, all reported in `violations`:
          * `constructor`        -- a search cannot surface a foreign chunk id;
          * `chunk_id_disjoint`  -- no two indexes share a chunk id;
          * `document_id_matches_index` -- every chunk names its own document;
          * `chunk_id_round_trip`-- ids parse back to (document_id, ordinal);
          * `query_probe`        -- a REAL text query on index A never returns a
            chunk belonging to index B, checked over every pair (or a
            deterministic sample of `max_pairs` pairs on a large store).

        The report keeps its original keys (`documents`, `violations`,
        `isolated`) and adds counters; violation entries gain a `check` field.
        """
        violations: List[Dict[str, object]] = []
        report: Dict[str, object] = {"documents": len(self._indexes),
                                     "violations": violations}
        all_ids = {did: set(ix.chunk_ids) for did, ix in self._indexes.items()}
        for did, ix in self._indexes.items():
            mine = all_ids[did]
            foreign = set().union(*[v for k, v in all_ids.items() if k != did]) \
                if len(all_ids) > 1 else set()
            if mine & foreign:
                violations.append({"check": "chunk_id_disjoint", "document_id": did,
                                   "shared_chunk_ids": sorted(mine & foreign)})
            if len(mine) != len(ix.chunks):
                violations.append({"check": "chunk_id_disjoint", "document_id": did,
                                   "duplicate_chunk_ids_within_document": True})
            for c in ix.chunks:
                if c.document_id != did:
                    violations.append({"check": "document_id_matches_index",
                                       "document_id": did, "chunk_id": c.chunk_id,
                                       "chunk_document_id": c.document_id})
                try:
                    parsed_did, parsed_ord = parse_chunk_id(c.chunk_id)
                except ValueError as exc:
                    violations.append({"check": "chunk_id_round_trip",
                                       "document_id": did, "chunk_id": c.chunk_id,
                                       "error": str(exc)})
                    continue
                if (parsed_did != did or parsed_ord != c.ordinal
                        or format_chunk_id(parsed_did, parsed_ord) != c.chunk_id):
                    violations.append({"check": "chunk_id_round_trip",
                                       "document_id": did, "chunk_id": c.chunk_id,
                                       "parsed": [parsed_did, parsed_ord]})
            if ix.chunks:
                hits = ix.search(ix.vectors[0], top_k=len(ix.chunks) + 5)
                for h in hits:
                    if h.chunk_id not in mine:
                        violations.append({"check": "constructor",
                                           "document_id": did, "leaked": h.chunk_id})
            for other in self._indexes:
                if other == did:
                    continue
                other_chunk = self._indexes[other].chunk_ids[:1]
                if other_chunk:
                    try:
                        ix.get_chunk(other_chunk[0])
                        violations.append({"check": "constructor", "document_id": did,
                                           "reachable_foreign_chunk": other_chunk[0]})
                    except DocumentIsolationError:
                        pass

        # -- real-query probe, pairwise ------------------------------------
        dids = sorted(self._indexes)
        pairs = [(a, b) for n, a in enumerate(dids) for b in dids[n + 1:]]
        sampled = pairs
        if max_pairs and len(pairs) > max_pairs:
            stride = len(pairs) / float(max_pairs)
            sampled = [pairs[int(n * stride)] for n in range(max_pairs)]
        probe: Dict[str, Set[str]] = {}
        probe_skipped: List[str] = []
        for did in {d for pair in sampled for d in pair}:
            ix = self._indexes[did]
            try:
                probe[did] = {h.chunk_id
                              for h in ix.search_text(probe_query, top_k=probe_top_k)}
            except DocumentIsolationError:
                # no embedder on a hand-built index: record it, never pass silently
                probe_skipped.append(did)
        for a, b in sampled:
            for src, dst in ((a, b), (b, a)):
                leaked = probe.get(src, set()) & all_ids.get(dst, set())
                if leaked:
                    violations.append({"check": "query_probe", "document_id": src,
                                       "other_document_id": dst,
                                       "leaked_chunk_ids": sorted(leaked)})

        report["chunks"] = sum(len(ix.chunks) for ix in self._indexes.values())
        report["pairs_total"] = len(pairs)
        report["pairs_probed"] = len(sampled)
        report["pairs_sampled"] = len(sampled) < len(pairs)
        report["probe_query"] = probe_query
        report["probe_top_k"] = probe_top_k
        report["probe_skipped_documents"] = sorted(probe_skipped)
        report["checks"] = ["constructor", "chunk_id_disjoint",
                            "document_id_matches_index", "chunk_id_round_trip",
                            "query_probe"]
        report["isolated"] = not violations and not probe_skipped
        return report


def format_chunks_for_prompt(hits: Iterable[Hit]) -> str:
    lines = []
    for h in hits:
        lines.append(f"[{h.chunk_id}]\n{h.text.strip()}")
    return "\n\n".join(lines) if lines else "(no passages retrieved)"
