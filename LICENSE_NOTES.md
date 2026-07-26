# License and attribution notes

This repository is a research harness. It reimplements an architecture and uses
third-party models and datasets, each under its own terms. Check each one before
publishing results or redistributing artefacts.

## Architecture provenance

**SPD-RAG — Sub-Agent Per Document Retrieval-Augmented Generation**
- Paper: <https://arxiv.org/abs/2603.08329>
- Code: <https://github.com/NebulAICompany/SPD-RAG> — **MIT License**
- Citation: Akay, Y. C., Kartal, M. Y., Alparslan, E., Ortakoyluoglu, F., Akpinar, A.
  *SPD-RAG: Sub-Agent Per Document Retrieval-Augmented Generation*, 2026.

This pilot **reimplements** the three-layer structure (coordination →
per-document sub-agents → synthesis) locally. It does not vendor SPD-RAG source.
`src/prompts.py` is adapted from that repository's `backend/core/prompts.py`
(`LEAD_RESEARCHER_PROMPT`, `RESEARCH_SYSTEM_PROMPT`, `SYNTHESIS_PROMPT`) under
the MIT License; the adaptations are documented in that file's docstring.

Deliberate divergences from the original, all because this pilot targets a local
3B model rather than the Gemini 2.5 / Cohere / Qdrant stack:

| SPD-RAG (original) | This pilot |
| --- | --- |
| Gemini 2.5 Pro coordinator + Flash sub-agents | one local GGUF for all three layers |
| Cohere `embed-v4.0` + Qdrant + `rerank-v4.0-fast` | local sentence-transformers, per-document numpy index, no reranker |
| up to 5 searches per sub-agent | at most 2 search rounds |
| recursive UPGMA-clustered synthesis, 750k token budget | single-pass synthesis (pilot widths are 2–4 documents) |
| free-text findings | strict JSON schemas with one repair attempt |
| Loong benchmark, GPT-5 judge | MultiHop-RAG, local partial-credit scoring |
| LangGraph `Send` fan-out | plain sequential Python (LangGraph optional, not required) |

## Benchmark

**MultiHop-RAG**
- Code: <https://github.com/yixuantt/MultiHop-RAG>
- Data: <https://huggingface.co/datasets/yixuantt/MultiHopRAG>
- 2,556 questions; 609 news articles; evidence spans 2–4 documents.
- The corpus is news text from third-party outlets. Check the dataset card's
  license before redistributing any derived corpus. This repo does **not** ship
  the dataset; `scripts/prepare_dataset.py` expects you to supply it.
- `data/pilot_documents.json` (generated, not shipped) contains article bodies.
  Treat it as derived data under the dataset's terms.

## Model

**Qwen2.5-3B-Instruct** — <https://huggingface.co/Qwen/Qwen2.5-3B-Instruct>,
Qwen Research License. Review it before any commercial use. GGUF conversion and
quantization are performed locally by `scripts/prepare_models.py`; no model
weights are shipped here.

## Inference and embedding

- **llama.cpp** — <https://github.com/ggml-org/llama.cpp>, MIT License. Cloned
  and built at run time; the resolved commit is recorded in
  `results/provenance.json`.
- **sentence-transformers / all-MiniLM-L6-v2** — Apache-2.0. The revision is
  pinned in `configs/models.yaml`.

## This repository

MIT, matching the SPD-RAG code it adapts. Nothing here is affiliated with or
endorsed by the SPD-RAG, MultiHop-RAG, Qwen or llama.cpp authors.
