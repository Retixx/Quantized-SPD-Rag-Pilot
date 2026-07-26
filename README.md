# Quantized SPD-RAG pilot

A go/no-go research pilot for a possible NeurIPS 2026 workshop paper.

> **Does hierarchical multi-agent RAG remain robust under aggressive weight
> quantization because each document agent solves a small focused problem, or do
> small branch-level failures compound as the number of independently required
> RAG agents increases?**

Three precisions, one source checkpoint: **F16**, **Q8_0**, **Q4_K_M**. Uniform
system precision — the whole pipeline runs at one precision per block. Stage-specific
quantization is out of scope until this characterization study finds a clear failure stage.

**Nothing here has been run against a real model yet.** The harness is built and
tested; the acceptance tests pass on fixtures. Launch Stage A on Kaggle to produce
scientific results. See [Engineering vs. scientific status](#engineering-vs-scientific-status).

---

## Architecture

Reimplemented from [SPD-RAG](https://github.com/NebulAICompany/SPD-RAG)
([paper](https://arxiv.org/abs/2603.08329)), factored along the **document axis**,
not the task axis. Each document is an isolated retrieval universe.

```
                       question
                          │
              ┌───────────▼────────────┐
              │   Coordination layer   │  atomic extraction instructions
              │   (src/coordinator.py) │  + synthesis directive
              └───────────┬────────────┘
                          │  same instruction set to every branch
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
  ┌───────────┐     ┌───────────┐     ┌───────────┐
  │  Agent 1  │     │  Agent 2  │ ... │  Agent N  │   one per supporting document
  │  Doc A    │     │  Doc B    │     │  Doc N    │   (src/document_agent.py)
  │ ┌───────┐ │     │ ┌───────┐ │     │ ┌───────┐ │
  │ │private│ │     │ │private│ │     │ │private│ │   PrivateIndex — structurally
  │ │ index │ │     │ │ index │ │     │ │ index │ │   cannot see another document
  │ └───────┘ │     │ └───────┘ │     │ └───────┘ │
  └─────┬─────┘     └─────┬─────┘     └─────┬─────┘
        └─────────────────┼─────────────────┘
                          ▼
              ┌────────────────────────┐
              │    Synthesis layer     │  merges findings -> final answer
              │  (src/synthesizer.py)  │
              └───────────┬────────────┘
                          ▼
                    final answer
```

All logical agents call **one** model server per precision block. One physical model
per block, never one per agent.

**Isolation is structural, not a filter.** Each agent receives only
`store.get(document_id)`, a `PrivateIndex` holding that document's chunks and nothing
else. `assert_isolation()` runs before any model loads and fails the run on violation.

---

## Benchmark

[MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG) —
[dataset](https://huggingface.co/datasets/yixuantt/MultiHopRAG). 2,556 questions,
609 news articles, evidence spread across 2–4 documents. Each supporting document
becomes one document-agent branch.

Replaces the cancelled Loong pilot, where the 3B model sat near the task floor
(3/12 F16, 4/12 Q8, 2/12 Q4). MultiHop-RAG plus partial-credit scoring should give
the F16 baseline enough headroom to expose quantization effects.

**Sampling.** Deterministic. Drop `null_query` (unanswerable by construction), keep
questions whose evidence resolves to 2–4 distinct corpus documents, order by
`sha256(SALT | question)`, fill Stage A first, then Stage B from the remaining pool.
Same manifest on any machine. Width is derived from unique evidence document
identifiers (`url`, falling back to `title`) and the derivation is logged per question.

Stage A and Stage B are **disjoint** — calibration items never re-enter the main test.

Once `data/pilot_manifest.json` exists, `prepare_dataset.py` refuses to overwrite it
without `--force`. Never alter the question set after viewing model outputs.

---

## Staged pilot with hard early stopping

| Stage | Questions | Widths | Condition | Predictions | Runs only if |
| --- | --- | --- | --- | --- | --- |
| **A** — capability calibration | 6 | 3×w2, 3×w3/w4 | fixed evidence | **18** | — |
| **B** — precision × width | 12 | 4×w2, 4×w3, 4×w4 | fixed evidence | **36** | A passes or `--force` |
| **C** — end-to-end retrieval | 6 | 2 low, 2 high, 2 max-disagreement | agent-generated queries, ≤2 rounds | **18** | B is informative |

Maximum **72** predictions. The system stops cleanly after 18 or 54.

**Stage A decision gate**, on F16 atomic-fact recall:

| F16 recall | Status |
| --- | --- |
| < 0.50 | `BLOCKED — MODEL/BENCHMARK FLOOR` |
| 0.50–0.70 | `MARGINAL — REVIEW PROMPTS AND FAILURES` |
| 0.70–0.90 | `CALIBRATION PASS` |
| > 0.95 | `POSSIBLE CEILING — ADD HARDER ITEMS OR DISTRACTORS` |

Screening thresholds. Not claims of formal statistical significance.

**Stage C selection** is deterministic and applied to Stage B *fixed-evidence*
results before any end-to-end output exists: 2 width-2 questions by ascending
question_id, 2 max-width questions the same way, then the 2 remaining questions with
the largest |F16 − Q4| answer-score disagreement (ties broken by recall, then id).

---

## Two evidence conditions

**`fixed_verified_evidence`** — chunks are selected once, frozen to
`results/fixed_evidence.json`, and reused by every precision. This is what makes the
chunk-id and prompt-hash parity gates satisfiable *by construction*.

Called `fixed_verified_evidence` rather than *oracle* deliberately: until you have
manually confirmed each passage, it means **frozen**, not **verified**.

**`end_to_end`** (Stage C) — each agent gets the question, the shared instruction set
and its assigned document id; writes a focused query; searches only its private index;
inspects top-k; may issue one more query; returns structured facts with chunk citations.

---

## Setup

```bash
git clone <this-repo> && cd quantized-spd-rag-pilot
pip install -r requirements.txt
python -m pytest tests -q          # 25 tests, no GPU, no model, no network
```

Fetch `MultiHopRAG.json` and `corpus.json` into `data/` (see `data/README.md`), then:

```bash
python scripts/prepare_models.py --build-llama-cpp   # one F16 -> Q8_0 + Q4_K_M
python scripts/prepare_dataset.py --data-dir data    # frozen manifest + seed rubric
python scripts/build_indexes.py --stage all          # private indexes + frozen evidence
```

### Exact Kaggle steps

1. New Notebook → Settings → **Accelerator: GPU** (T4 x2 or P100), **Internet: On**.
2. Add data: the MultiHop-RAG files as an input dataset. Optionally add
   `Qwen2.5-3B-Instruct` and a llama.cpp checkout too, if you must run without internet.
3. Upload this repository (as an input dataset or a zip) and open
   `notebooks/kaggle_run.ipynb`.
4. Edit the `SRC` path in cell 1 to point at your copy. The notebook copies it into
   `/kaggle/working` so it is writable.
5. Run cells top to bottom. Stage A takes roughly 30–60 min after the ~30 min
   one-time llama.cpp build and quantization.

Or, without the notebook:

```bash
python scripts/run_kaggle.py --stage stage_a
python scripts/run_kaggle.py --stage stage_b     # blocked unless Stage A passed
python scripts/run_kaggle.py --stage stage_b --force   # documented manual override
python scripts/run_kaggle.py --stage stage_c
```

### Expected input layout

```
/kaggle/input/
  multihop-rag/            MultiHopRAG.json, corpus.json
  qwen2.5-3b-instruct/     config.json, *.safetensors, tokenizer*   (optional)
  quantized-spd-rag-pilot/ this repository                          (optional)
/kaggle/working/
  quantized-spd-rag-pilot/ writable copy
    models/gguf/           f16, q8_0, q4_k_m
    vendor/llama.cpp/      CUDA build
    results/               everything below
```

`prepare_kaggle.py` discovers all of this and records what it found. The notebook
assumes POSIX paths throughout — no Windows paths anywhere.

### One-command stage execution

```bash
python scripts/run_stage.py --stage stage_a --backend llama-server
python scripts/score.py     --stage stage_a --emit-adjudication
python scripts/analyze.py   --stage stage_a --ni-margin 0.05
python scripts/package_results.py --stage stage_a
```

`--backend llama-cli` swaps in the subprocess fallback. `--dry-run` builds and hashes
every prompt without calling a model.

---

## Scoring

No binary-only success, no external API judge. `data/gold_facts.json` holds atomic
facts per question, seeded from the dataset's own `evidence_list[].fact` and flagged
`needs_manual_review: true` until you review them. Reports carry
`rubric_reviewed: false` until the flags are cleared.

Metrics: normalized exact match, token/char F1, atomic-fact recall and precision,
required-document coverage, unsupported-claim rate, per-agent supported-fact recall,
final completeness, synthesis loss.

```
                 correct atomic facts in document-agent outputs but absent from final answer
synthesis_loss = ───────────────────────────────────────────────────────────────────────────
                        correct atomic facts in document-agent outputs
```

**The scorer is blind to precision labels.** No function in `src/metrics.py` takes a
precision argument. `score.py --emit-adjudication` writes a shuffled, de-labelled
worksheet plus a separate key, so manual adjudication stays blind.

### Metrics collected

- **Branch** — retrieval success, supported-fact recall, omission rate, unsupported
  facts, retrieval rounds, generated tokens, latency
- **Orchestration** — agents succeeding, P(≥1 required branch fails), required-document
  coverage, duplicate evidence, retrieved/generated tokens
- **Synthesis** — facts before, facts preserved, synthesis loss, contradictions introduced
- **Systems** — model size, load time, peak CPU RAM, peak GPU VRAM, prompt/generated
  tokens, end-to-end wall time, document-agent wall time, tokens/s, quality/s, quality/GB

Runtime is **actual serial wall time**. Document-agent calls run sequentially against
one server, so no idealized parallel figure is reported; if you add one, label it a
lower bound.

---

## Statistical analysis

Small pilot, conservative treatment: paired per-question differences, percentile
bootstrap 95% CIs (deterministic seed), results split by width, precision and
condition, plus an exploratory regression with precision × document-count interaction.

Central estimand: **does the F16–Q4 gap grow from 2 to 4 required agents?**

`not statistically significant` and `non-inferior within a predeclared margin` are
reported as distinct verdicts:

| Verdict | Meaning |
| --- | --- |
| `NON_INFERIOR_WITHIN_MARGIN` | CI upper bound ≤ margin **and** interval narrow |
| `NOT_SIGNIFICANT_BUT_CI_TOO_WIDE` | inside the margin but the interval cannot support an equivalence claim |
| `NOT_STATISTICALLY_SIGNIFICANT` | CI includes 0 but exceeds the margin |
| `DEGRADATION_DETECTED` | CI excludes 0 and exceeds the margin |

Default non-inferiority margin: **0.05** atomic-fact recall (`--ni-margin`).

---

## Engineering vs. scientific status

Kept strictly separate. Fixtures and mocks may prove the harness works. They may
never produce a scientific recommendation.

**Engineering:** `HARNESS_READY` · `HARNESS_FAILED` · `REAL_MODEL_RUN_BLOCKED`

**Scientific:** `GO_COMPOUNDING_DEGRADATION` · `GO_Q8_OPERATING_POINT` ·
`GO_DECOMPOSITION_ROBUSTNESS` · `WEAK_GO_EXPAND` · `NO_GO_NO_STRUCTURED_EFFECT` ·
`NO_GO_MODEL_FLOOR` · `NO_GO_EVALUATION_CEILING`

### Hard scientific gates

No scientific outcome is emitted unless every one passes:

| Gate | Checks |
| --- | --- |
| `real_inference_only` | zero fixture or dry-run records anywhere |
| `backend_is_llama_cpp` | only `llama-server` / `llama-cli` produced output |
| `all_three_precisions_present` | F16, Q8_0, Q4_K_M all completed |
| `nonzero_generations_per_precision` | each generated tokens |
| `model_hashes_recorded` | three distinct 64-char sha256 |
| `common_source_checkpoint` | one recorded F16 source for all variants |
| `fixed_evidence_chunk_parity` | identical chunk ids across precisions |
| `prompt_hash_parity` | identical prompt hashes across precisions |
| `stage_question_count_complete` | the stage's full question count at every precision |

Any failure → **`SCIENTIFIC_RECOMMENDATION_NOT_AVAILABLE`**.

The fixture backend refuses to start unless `SPDQ_ALLOW_FIXTURE=1`, watermarks every
output with `FIXTURE_NOT_REAL_MODEL_OUTPUT`, and flags every record `is_fixture: true`.
Those records are excluded from all aggregates and trip `real_inference_only`.

---

## Operational guarantees

- **Resumable.** One JSONL event written and `fsync`ed immediately after every
  generation; deterministic `call_id`s; a torn final line from a hard kill is skipped,
  not fatal. Rerun and it continues.
- **No silent skips.** A failed question is logged as a `question_failure` event and
  listed in `run_report.json`.
- **Deterministic ids** from content, never wall clock.
- **Caching** for embeddings and frozen evidence.
- **All prompts, hashes and server logs recorded.**
- **Clean model lifecycle** — server started and stopped per precision block, GPU
  memory released between blocks.
- **`--dry-run`** builds and hashes every prompt without calling a model.
- **One structured-output repair attempt**, never two; original and repaired text both logged.

---

## Repository layout

```
quantized-spd-rag-pilot/
  configs/       models.yaml, stage_a/b/c.yaml
  data/          manifest, gold_facts, README (dataset files are yours to supply)
  notebooks/     kaggle_run.ipynb
  scripts/       prepare_kaggle, prepare_models, prepare_dataset, build_indexes,
                 run_stage, run_kaggle, score, analyze, package_results
  src/           backends/llamacpp_backend.py, coordinator, document_agent,
                 synthesizer, retrieval, evidence, schemas, metrics, gates,
                 llm_client, prompts, config, logging_utils, hardware
  tests/         25 tests incl. the 10 required acceptance tests
  results/       events.jsonl, predictions, scored, analysis, gate, RUN_REPORT.md
```

## Acceptance tests

```bash
python -m pytest tests -q
```

1. Each document agent can access only its assigned document
2. Fixed-evidence chunk ids identical across precisions
3. Prompt hashes match across precisions
4. Three model files with recorded hashes and common provenance
5. Fixture data cannot enter scientific analysis
6. Interrupted runs resume without duplicating predictions
7. Malformed JSON logged and repaired at most once
8. GPU offload detected and recorded
9. F16, Q8, Q4 all produce a real generation
10. Stage-A recommendation unavailable until all 18 real predictions complete

---

## Attribution

Architecture from SPD-RAG (MIT), benchmark from MultiHop-RAG, model Qwen2.5-3B-Instruct
(Qwen Research License), inference via llama.cpp (MIT). Full detail and the list of
deliberate divergences from the original SPD-RAG stack: [`LICENSE_NOTES.md`](LICENSE_NOTES.md).
