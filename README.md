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

> **Audit remediation.** A full pre-run audit found that the harness could not emit a
> scientific outcome as written, and that six independent defects in the scorer all
> handed lower precisions free credit — so an early Q4-beats-F16 result was an
> artifact, not a finding. Those are fixed. Read
> [`docs/AUDIT_FIXES.md`](docs/AUDIT_FIXES.md) before interpreting anything, and
> [`docs/PROPOSAL_REVIEW.md`](docs/PROPOSAL_REVIEW.md) for the experiment-design
> review. Known remaining limitations are listed at the bottom of this file.

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

**The coordination layer is frozen, not re-run per precision.** The coordinator runs
once at F16; its `shared_tasks` and synthesis directive are written to
`results/frozen_coordinator.json` and reused byte-identically by Q8_0 and Q4_K_M,
exactly as the evidence is frozen. Two reasons:

1. Its output is precision-dependent by construction, and it is formatted into every
   document-agent prompt — so running it per precision made `prompt_hash_parity`
   unsatisfiable and the harness structurally incapable of emitting a scientific
   outcome.
2. When the coordinator's JSON fails to parse, the harness substitutes a hand-written
   fallback instruction set. Malformed JSON is the dominant Q4 failure mode, so the
   fallback rescued precisely the precision that was failing, attenuating the gap
   being measured.

The measured estimand is therefore **quantization of the document agents and the
synthesizer, under a fixed decomposition**. Quantizing the coordinator is a separate
question and is not in scope here — see the limitations section.

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

**Sampling.** Deterministic. Drop `null_query` (unanswerable by construction), **drop
questions whose gold answer is yes/no**, keep questions whose evidence resolves to 2–4
distinct corpus documents, order by `sha256(SALT | question)`, fill Stage A first, then
Stage B from the remaining pool. Same manifest on any machine. Width is derived from
unique evidence document identifiers (`url`, falling back to `title`) and the
derivation is logged per question.

**Why yes/no questions are excluded.** 59.6% of MultiHop-RAG's 2,255 answerable
questions have gold answer `yes` or `no`, and the whole benchmark has only 105 unique
answers (the top four cover 81%). The previous manifest ended up with **5 unique gold
answers across 18 questions**. On a binary item a degraded model scores 50% by
guessing, which compresses the F16–Q4 gap toward zero — in the direction of the
hypothesis. The manifest now records `excluded_answers` alongside
`excluded_question_types`, and `prepare_dataset.py` prints answer diversity so this
cannot silently regress.

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

**Stage A decision gate**, on F16 atomic-fact recall. Bands are half-open `[min, max)`,
exhaustive over `[0, 1]`, with `configs/stage_a.yaml` as the single source of truth:

| F16 recall | Status | Proceed |
| --- | --- | --- |
| `[0.00, 0.50)` | `BLOCKED — MODEL/BENCHMARK FLOOR` | no |
| `[0.50, 0.70)` | `MARGINAL — REVIEW PROMPTS AND FAILURES` | no |
| `[0.70, 0.95)` | `CALIBRATION PASS` | **yes** |
| `[0.95, 1.00]` | `POSSIBLE CEILING — ADD HARDER ITEMS OR DISTRACTORS` | no |

The previous table left `0.90–0.95` undefined and used `<=` boundaries, so a recall of
exactly 0.70 was reported `MARGINAL — REVIEW PROMPTS AND FAILURES` while the outcome
map simultaneously emitted a GO verdict for it.

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
python scripts/prepare_dataset.py --data-dir data    # frozen manifest + seed rubric
python scripts/prepare_models.py --build-llama-cpp --free-hf-weights
python scripts/build_indexes.py --stage all          # private indexes + frozen evidence
```

**Dataset preparation must run first.** The Q4_K_M importance matrix is calibrated on
a held-out corpus slice — every corpus document *minus* every document the pilot
evaluates on — so `prepare_models.py` needs `data/pilot_documents.json` to know what to
exclude. Calibrating on the evaluation documents would be leakage.

`--free-hf-weights` deletes the safetensors once the F16 GGUF exists. Without it the
build peaks at **21.7 GB**, over Kaggle's 20 GB `/kaggle/working` cap; with it, 16.4 GB.
`prepare_models.py` checks free space before starting and refuses rather than dying
halfway through a quantization.

### Exact Kaggle steps

1. New Notebook → Settings → **Accelerator: GPU** (T4 x2 or P100), **Internet: On**.
2. Add data: the MultiHop-RAG files as an input dataset. Optionally add
   `Qwen2.5-3B-Instruct` and a llama.cpp checkout too, if you must run without internet.
3. Upload this repository (as an input dataset or a zip) and open
   `notebooks/kaggle_run.ipynb`.
4. Edit the `SRC` path in cell 1 to point at your copy. The notebook copies it into
   `/kaggle/working` so it is writable.
5. Run cells top to bottom. Stage A takes roughly 30–60 min after the one-time
   llama.cpp build, imatrix generation and quantization (~40–60 min).

**Kaggle caps GPU notebook execution at 9 hours, not 12** — the 12 h figure is the
CPU-only limit. The notebook carries a per-step budget table and explicit stop points.
`/kaggle/working` is capped at 20 GB; see `--free-hf-weights` above.

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
required-document coverage, uncited- and unsupported-claim rate, per-agent
supported-fact recall, final completeness, synthesis loss.

**Three rules every metric obeys** (asserted by `tests/test_units.py`):

- **No metric may reward degradation.** Ratios with an empty denominator return
  `None` and are excluded from aggregates, never `0.0`. Previously
  `synthesis_loss` was `0.0` — the best possible value — when the agents recovered
  nothing, and `any_required_branch_failed` evaluated `0 < 0 == False` when every
  agent died, so total branch collapse scored perfectly on both.
- **No metric may reward verbosity.** Fact coverage is measured inside a bounded
  sliding window of the candidate, so appending text cannot raise a score. Quantization
  changes output length, and a length-sensitive metric cannot separate that from
  quality.
- **Containment respects word boundaries, length and negation.** `contains_answer`
  is a token-subsequence test, so gold `"no"` no longer matches inside `"not"` /
  `"cannot"` / `"nothing"` and gold `"Yes"` no longer matches inside `"Yesterday"`.
  A hit is suppressed when the prediction exceeds 60 tokens (a hit buried in a wall of
  text is not an answer) or when every occurrence is preceded by a negation. Both
  suppressions are recorded on the score, not applied silently.

`answer_score` returns `score: None, scorable: false` for a question with an empty
gold answer instead of awarding a blank prediction 1.0. Numeric disagreement zeroes a
fact match rather than halving it. `atomic_fact_precision` no longer accepts a match in
either direction — that made it unfalsifiable, scoring 1.0 for one-word "facts".

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
| `NON_INFERIOR_WITHIN_MARGIN` | CI upper bound ≤ margin, interval width ≤ 2× margin, **and** ≥ 4 paired differences are non-zero |
| `NOT_SIGNIFICANT_BUT_CI_TOO_WIDE` | inside the margin but the interval — or the number of informative pairs — cannot support an equivalence claim |
| `NOT_STATISTICALLY_SIGNIFICANT` | CI includes 0 but exceeds the margin |
| `DEGRADATION_DETECTED` | CI excludes 0 and exceeds the margin |

Default non-inferiority margin: **0.05** atomic-fact recall (`--ni-margin`).

**A verdict needs information, not just a narrow interval.** With a coarse thresholded
metric at n=6, many paired differences are exactly 0, and the resulting *zero-width*
bootstrap CI used to satisfy the "interval narrow" test perfectly — so no information
produced the strongest possible equivalence claim. Non-inferiority now additionally
requires at least four non-zero pairs, and the width ceiling is tied to the margin
(2×) rather than the previous hardcoded 0.30, which was six times the declared margin.

**The compounding contrast carries an interval.** `GO_COMPOUNDING_DEGRADATION` is the
paper's headline claim; it was previously emitted from a bare threshold on the
difference of two point estimates while the CI computed alongside them was discarded.
It now requires a stratified bootstrap CI on (gap at widest width − gap at narrowest)
that excludes zero, **and** at least 3 paired questions in every width cell.

**Loss metrics are not accuracy metrics.** `verdict` assumed higher-is-better, so on
`synthesis_loss` it reported `DEGRADATION_DETECTED` when Q4 lost *fewer* facts. The
direction is now explicit per metric.

**Multiplicity is reported.** One contrast is confirmatory — F16 − Q4 on the primary
metric, in the primary condition. Everything else is labelled exploratory and carries
a Holm-adjusted companion p-value, so the reader can see how much of the family the
outcome selector is scanning.

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
| `no_generation_errors` | no call recorded a backend error — infrastructure failure is never scored as model quality |
| `all_three_precisions_present` | F16, Q8_0, Q4_K_M all completed |
| `nonzero_generations_per_precision` | each generated tokens |
| `model_hashes_recorded` | three distinct 64-char sha256, **re-verified against the files on disk** |
| `common_source_checkpoint` | each quantized variant records the sha256 of the F16 it was **actually derived from**, and it matches the F16 in use |
| `embedding_is_real_model` | retrieval did not run on the hash-fallback embedder |
| `sampling_identical_across_precisions` | one `sampling_hash` across all blocks |
| `gpu_offload_identical_across_precisions` | no block silently fell back to partial CPU offload |
| `coordinator_prompt_parity` | the frozen coordinator prompt is identical everywhere |
| `fixed_evidence_chunk_parity` | identical chunk ids across precisions |
| `prompt_hash_parity` | identical document-agent prompt hashes across precisions |
| `stage_question_count_complete` | the stage's full question count at every precision |

Any failure → **`SCIENTIFIC_RECOMMENDATION_NOT_AVAILABLE`**.

Two properties the gates did not previously have:

- **A gate cannot pass vacuously.** The parity gates compared
  `len(set(values)) > 1` over whatever happened to be present, so a question that
  ran at only one precision had one value and "passed" — parity was guaranteed
  exactly when data was missing. Every parity gate now requires all three
  precisions to be present, with a non-empty value, for every slot it checks.
- **A gate checks the artifact, not a claim about it.** `model_hashes_recorded`
  verified that three 64-character strings existed; it now re-hashes the GGUFs.
  `common_source_checkpoint` was `bool(provenance["common_source"])` — any truthy
  value — while `prepare_models.py` wrote that note unconditionally whenever an F16
  file happened to exist, so a stale Q4 from a different checkpoint passed green.

A `question_failure` record is tagged `is_failure_record`, **not** `is_fixture`.
Previously, honestly recording one transient server timeout permanently voided the
stage in an append-only event log — punishing correct behaviour harder than crashing.

The fixture backend refuses to start unless `SPDQ_ALLOW_FIXTURE=1`, watermarks every
output with `FIXTURE_NOT_REAL_MODEL_OUTPUT`, and flags every record `is_fixture: true`.
Those records are excluded from all aggregates and trip `real_inference_only`.

---

## Operational guarantees

- **Harness runs are quarantined on disk.** `--dry-run` and `--backend fixture` write
  to `results/_harness/<stage>/`. They previously shared `events.jsonl` and
  `predictions.json` with real runs and, because `call_id` has no run-mode component,
  a dry run followed by a real run in the same directory resume-skipped every
  question, printed `18/18 predictions`, and never called the model once.
- **Resumable.** One JSONL event written and `fsync`ed immediately after every
  generation; deterministic `call_id`s; a torn *final* line from a hard kill is
  skipped, not fatal. A corrupt line anywhere else is counted, warned about and
  surfaced in `run_report.json` rather than silently dropping a completed call.
  Rerun and it continues.
- **A failed generation is retried, not cached.** A backend error raises
  `BackendGenerationError`; the failing `call_id` is deliberately kept out of the
  resume index so a rerun genuinely re-runs it, and the row is excluded from every
  aggregate instead of being scored as an empty answer.
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
2. Fixed-evidence chunk ids identical across precisions, and the cache key changes when the corpus body changes
3. The frozen coordinator makes document-agent prompt hashes match across precisions
4. Three model files with recorded hashes, verified on disk, and a *matching* derived-from provenance chain
5. Fixture data cannot enter scientific analysis
6. Interrupted runs resume without duplicating predictions
7. Malformed JSON logged and repaired at most once
8. GPU offload detected and recorded, and a CPU-only build does **not** report offload
9. F16, Q8, Q4 all produce a real generation
10. Stage-A recommendation unavailable until all 18 real predictions complete

Plus one end-to-end integration test that drives
`run_stage.py → score.py → analyze.py` through `subprocess` on a two-question manifest
and asserts on the artifacts on disk. Four of the seven blocking bugs the audit found
lived in the wiring *between* these scripts — each script wrote fields the next one
never read — and every unit test passed throughout.

Several of the original acceptance tests proved nothing and were rewritten. The
clearest case: `test_3_prompt_hashes_match_across_precisions` never used its
`precision` loop variable — it called two pure functions with identical arguments
three times and asserted the outputs were equal. It could not fail, which is exactly
why the unsatisfiable `prompt_hash_parity` gate shipped.

---

## Known limitations

Read these before quoting any number.

**Design**

- **Precision blocks always run F16 → Q8_0 → Q4_K_M.** Anything that drifts
  monotonically over a session (thermal throttling, host page cache, co-tenant GPU
  pressure) is confounded with precision in the *timing* metrics. The order is
  recorded and flagged in every analysis; it is not counterbalanced. Question-order
  rotation within a block does not address this, and under greedy decoding with
  `cache_prompt: false` it cannot change any output.
- **The coordinator is frozen at F16, so the coordination layer is not under test.**
  Per-role precision assignment — "F16 orchestrator + Q4 workers" and its converse —
  is not supported. It needs either two servers resident simultaneously or a re-load
  between roles, which would destroy the latency measurement.
- **Width is capped at 4.** MultiHop-RAG evidence spans 2–4 documents
  (`{2: 1169, 3: 774, 4: 312}` usable candidates). Any claim about 8 or 16 branches
  needs a different corpus or explicit distractor branches, which measure a different
  quantity.
- **Q4_K_M is built with an importance matrix** from a held-out corpus slice, never
  from the evaluation questions. A non-imatrix K-quant inflates the F16→Q4 gap by an
  unknown amount; `imatrix_used` and the imatrix sha256 are recorded in
  `provenance.json`.

**Measurement**

- `contains_answer` remains vulnerable to *enumeration*: a prediction that lists
  several candidate entities can contain the right one. Word boundaries, the 60-token
  cap, the negation guard and dropping yes/no items remove the practical exploits, and
  `token_f1` penalises it, but the primary metric (windowed atomic-fact recall) is the
  one to trust. Human adjudication is the backstop.
- `uncited_claim_rate` measures citation *format compliance*, not faithfulness — a
  true fact that forgot its `chunk_ids` counts as uncited. `unsupported_claim_rate`
  is the content-level measure and needs the chunk texts to be available. Format
  compliance degrades sharply with quantization, so do not read the former as
  hallucination.
- The exploratory regression is fit on repeated measures (questions × precisions).
  Its iid standard errors are anticonservative; the paired bootstrap is the
  inferential result.
- `assert_isolation()` now runs real cross-index query probes, but on a sampled subset
  of index pairs when the store is large. The sample is deterministic and its size is
  recorded.

**Operational**

- Kaggle caps **GPU** notebook execution at **9 hours**, not 12.
- `/kaggle/working` is capped at 20 GB. The HF checkpoint plus three GGUFs plus a CUDA
  build tree exceeds that; `prepare_models.py` frees the safetensors after conversion
  and checks free space first.
- `llama.cpp` and the HF revision must be pinned. An unpinned `master` will eventually
  rename a CLI flag and kill the session at model-server startup.

## Attribution

Architecture from SPD-RAG (MIT), benchmark from MultiHop-RAG, model Qwen2.5-3B-Instruct
(Qwen Research License), inference via llama.cpp (MIT). Full detail and the list of
deliberate divergences from the original SPD-RAG stack: [`LICENSE_NOTES.md`](LICENSE_NOTES.md).
