# Audit remediation

Every finding from the pre-run audit, and what was done about it. Findings marked
**[reproduced]** were demonstrated by executing the code against the real MultiHop-RAG
corpus (2,556 questions / 609 articles) before the fix.

---

## A. Why the harness reported Q4 beating F16

Six independent defects, all pointing the same way: toward "quantization is robust",
which is one of the two answers the study exists to give.

| # | Defect | Fix |
| --- | --- | --- |
| 1 **[reproduced]** | `contains_answer` was a raw substring test and `score = max(em, contains, f1)`, so gold `"no"` matched inside `"not"`/`"cannot"`/`"nothing"` and gold `"Yes"` matched inside `"Yesterday"`. `answer_score("I do not know.", "no")` returned **1.00**. Abstention and hedging — what quantized models do more of — scored perfectly. | Word-boundary token-subsequence match, a negation/contrast guard, and a 60-token length cap on the prediction. Both suppressions are recorded on the score rather than applied silently. `metrics.py` |
| 2 **[reproduced]** | `synthesis_loss` returned `0.0` — the best possible value — when the agents recovered nothing, then went into a plain mean. The worse a precision's branches failed, the better its synthesis looked. | Returns `None`; excluded from aggregates. `metrics.py`, `score.py` |
| 3 **[reproduced]** | `any_required_branch_failed` was `n_ok < len(agent_recalls)`, i.e. `0 < 0 == False` when every agent died — a perfect branch-failure probability for total collapse, which is the exact quantity the compounding hypothesis is about. | Measured against `len(required_document_ids)`. Added `required_document_correct_coverage` beside the "did the agent say anything" measure. `metrics.py` |
| 4 | `fallback_tasks()` substituted a hand-written coordinator instruction set whenever the model's JSON failed to parse — the dominant Q4 failure mode — so the harness propped up whichever precision was failing. The Stage C forced-finalize did the same at the branch layer and then reported `retrieval_success: True`. | The coordinator is frozen at F16 and reused, so the fallback can no longer differ by precision. The branch rescue is kept (it prevents data loss) but sets `forced_finalize: True`, reports `retrieval_success: False`, and is surfaced per precision. `coordinator.py`, `document_agent.py` |
| 5 **[reproduced]** | `verdict(paired_bootstrap_ci([0.0]*6), 0.05)` returned `NON_INFERIOR_WITHIN_MARGIN` with width 0.000 — zero information yielding the strongest possible equivalence claim. The width ceiling was a hardcoded 0.30, six times the declared margin. | Requires ≥ 4 non-zero paired differences; width ceiling tied to `2 × ni_margin`. `analyze.py`, `gates.py` |
| 6 **[reproduced]** | `block_t0` started before a full sha256 of the GGUF — ~6.2 GB for F16 vs ~1.9 GB for Q4 — and the result was divided into `quality_per_second`. On a resumed block it timed only the questions that re-ran. | Hashes are cached (`hashcache.py`) and computed before the timer; `inference_wall_s` starts after load and warmup; resumed blocks report `quality_per_second: null` with an explicit `runtime_basis` explaining why. `run_stage.py`, `analyze.py` |

---

## B. Blocking — the pilot could not have produced a result

| # | Defect | Fix |
| --- | --- | --- |
| B1 | **`prompt_hash_parity` was unsatisfiable.** The coordinator ran inside the per-precision loop and its output was formatted into every document-agent prompt, while the gate required those hashes identical across precisions. It could only pass if all three precisions emitted byte-identical coordinator JSON. | Coordinator frozen once at F16 (`freeze_coordinator` / `load_frozen_coordinator`, mirroring the evidence cache) and reused. `coordinator.py`, `run_stage.py` |
| B2 **[reproduced]** | **`--dry-run` silently disabled the next real run.** Dry-run events shared `events.jsonl` and `predictions.json`, and `call_id` had no run-mode component. Verified: dry run, then real run → 18 resume-skips, `18/18 predictions`, all `dry_run: True`, empty answers, model never called. | Harness runs write to `results/_harness/<stage>/`. `logging_utils.stable_id` also gained an optional `variant` component (ids unchanged when empty). `run_stage.py`, `logging_utils.py` |
| B3 **[reproduced]** | **The JSON extractor returned the echoed schema example.** Every system prompt ends with a literal JSON example; the scanner took the first balanced object. `extract_json_object('…like {"final_answer": "..."}. Answer: {"final_answer": "Sam Bankman-Fried"}')` returned `{'final_answer': '...'}` with `parse_ok=True`. A stray brace in prose (`'{see below}: {"final_answer":"x"}'`) returned `None`. | Collects **all** balanced spans and **all** fenced blocks, then ranks candidates: validates-against-target-schema first, then key overlap, then shallowest, then last occurrence. `schemas.py` |
| B4 | **Generation errors were scored as model quality.** `gen.error` was written to the event and read by nothing; a 502 produced `text=""`, triggered a repair against the same dead server, and scored 0. The `call_id` entered the resume index, so a rerun never retried it. | `BackendGenerationError` is raised; no repair on errored/empty generations; the failing `call_id` is kept out of the resume index; `score.py` excludes errored rows; a `no_generation_errors` gate blocks any run containing one. `llm_client.py`, `score.py`, `gates.py`, `run_stage.py` |
| B5 | **Recording a failure honestly voided the stage.** The `question_failure` event was tagged `is_fixture: True`, and `real_inference_only` counted any such event, in an append-only log. | Tagged `is_failure_record: True`; the gate ignores it. `run_stage.py`, `gates.py` |
| B6 | **Peak RAM and VRAM were always empty.** `backend.provenance()` was called before `backend.stop()`, and `stop()` is the only place `self.memory` was populated. | `provenance()` after `stop()`; memory survives on the backend; same fix in `prepare_kaggle.py`. `run_stage.py`, `llamacpp_backend.py` |
| B7 | **The llama-cli fallback could return the prompt as the model's answer.** `if rendered in text: text = text.split(rendered, 1)[1]` — on a failed exact match the whole prompt, containing the evidence and therefore the gold facts, was returned and would have scored near-perfect recall. | Exact match → whitespace-normalised match → **hard error** with empty text and `error` set if any prompt slab survives. Token counts and a memory sampler added; `--min-p` now passed; the hand-rolled ChatML template is recorded in provenance as such. `llamacpp_backend.py` |
| B8 **[reproduced]** | **Stage C's max-disagreement selection was dead code.** It read `answer_score` and `atomic_fact_recall_final` from `predictions.json`; those keys only exist in `scored.json`. `disagreement()` always returned `(-0.0, -0.0, qid)`, collapsing rule 3 to a question_id sort — output identical to having no Stage B data at all. It then wrote `selected_before_reading_end_to_end_outputs: true` even when the file did not exist. | Reads `stage_b/scored.json`, excludes fixture rows, and raises `SystemExit` when there is nothing to select on. `run_stage.py` |
| B9 **[reproduced]** | **The Kaggle notebook crashed at the Stage A gate.** Cell 2 chdirs into the repo copy while `config.results_dir()` returns `/kaggle/working/results`; every relative `open('results/…')` raised `FileNotFoundError`. | All notebook reads go through `config.results_dir_abs()`. `kaggle_run.ipynb`, `config.py` |

---

## C. Model provenance

| # | Defect | Fix |
| --- | --- | --- |
| C1 | **Q4_K_M was quantized with no importance matrix**, disclosed nowhere (`grep -rni imatrix` found nothing). A non-imatrix K-quant inflates the F16→Q4 gap by an unknown amount — the first thing a reviewer asks. | `llama-imatrix` run on a held-out corpus slice (never the evaluation questions), `--imatrix` passed to the Q4_K_M quantize step, `imatrix_used` and the imatrix sha256 recorded. `prepare_models.py` |
| C2 | **`common_source_checkpoint` was asserted, never verified.** Files were reused on a bare `out.exists()` check and the note "quantized from this exact F16 GGUF" was written unconditionally. A stale Q4 from a different checkpoint passed the gate green — the exact confound the design exists to prevent. | Each variant records `derived_from_f16_sha256` at derivation time; the gate compares it to the F16 in use; a mismatched parent triggers re-quantization. `prepare_models.py`, `gates.py` |
| C3 | **The source revision was neither pinned nor recorded.** `resolved_revision()` looked for a file whose entire content is 40 chars, which modern `huggingface_hub` does not write, so it returned `"unrecorded"`. `prepare_models.py` also skipped `git checkout` when the commit was `master`. | Real revision resolution online and offline; `hf_revision` and the llama.cpp commit pinned in `configs/models.yaml`. |
| C4 | **Precision order was a fixed, unrecorded confound** and `--precisions` was re-sorted back into it. `counterbalanced()` rotated *question* order, which under greedy decoding changes nothing — its only real effect was moving the un-discarded warmup cost to a different question per block. | A discarded warmup generation now runs at the start of every block; the fixed precision order is recorded and appears as a caveat in every analysis and in the README limitations. |

---

## D. Measurement validity

- `atomic_fact_recall` / `fact_match_score` were recall over a bag of words with no
  length penalty — a verbosity meter. Now scored inside a bounded sliding window, so
  appending text cannot raise a score. Numeric disagreement zeroes rather than halves.
  Polarity disagreement is penalised (`"not"` was previously a stopword, so a fact and
  its negation scored identically).
- `atomic_fact_precision` accepted a match in **either** direction, making it
  unfalsifiable — one-word "facts" scored 1.0 **[reproduced]**. Now one direction, with
  a minimum content-token requirement and the same numeric gating.
- `unsupported_claim_rate` measured citation *formatting*, which degrades sharply with
  quantization and would have shown a large, publishable-looking effect that was
  entirely an artifact. Renamed `uncited_claim_rate`; a real content-level
  `unsupported_claim_rate` now checks the fact against its cited chunk text. Both are
  weighted by fact count instead of macro-averaged per branch.
- `required_document_coverage` measured "the agent said something". Kept, renamed in
  the docs for what it is, and joined by `required_document_correct_coverage`.
- `contradiction_count` counted any novel numeral, so restating a figure from the
  question, or `3` vs `3.0`, or `1,500` vs `1500`, counted as a contradiction. Numbers
  are normalised and question numbers excluded; reported as `unsourced_numbers`.
- `answer_score("", "")` returned **1.00** **[reproduced]** — full credit to a blank
  prediction against a blank gold. Now `score: None, scorable: false`, excluded.
- `tokens_per_second` came from the first call while `latency_s` and
  `generated_tokens` summed original + repair. They diverged exactly when repair
  fired, which is more often at Q4, so Q4's real cost was under-reported. Recomputed
  from summed tokens over summed latency.
- `finish_reason` was recorded and never read, so `"length"` truncation was
  indistinguishable from malformed output. Surfaced as `truncated`. The repair call
  also reused the same budget on a strictly longer prompt; it now gets a larger one.
- Per-role `max_tokens` were hardcoded at the call sites while `models.yaml` carried a
  scalar that no call site read — yet that scalar was hashed into `sampling_hash`.
  Config now holds explicit per-role budgets; the hash carries the real ceiling.
- Systems metrics measured the wrong things: `peak_cpu_rss_mib` sampled the harness
  process rather than the llama-server child, and `peak_gpu_used_mib` was device-total
  across all processes. Both now have per-process variants with the device-total kept
  beside them, and the sampler join no longer races `nvidia-smi`.

## E. Statistics

- The compounding contrast was a bare threshold on a difference of two point
  estimates, discarding the CIs computed alongside them. Now a stratified bootstrap CI
  that must exclude zero, plus a minimum of 3 paired questions per width cell.
- The interaction regression is repeated measures fit by plain OLS with no rank check;
  `lstsq(rcond=None)` returned a minimum-norm solution for a singular design instead of
  raising, so a stage where every question had the same width returned confident-looking
  coefficients for an unidentified model **[reproduced]**. Rank is now checked, standard
  errors are reported and labelled anticonservative, and the design is stated.
- ~24 uncorrected 95% intervals per stage were scanned by a first-match outcome
  selector. One contrast is now designated confirmatory; the rest carry Holm-adjusted
  companions.
- Bootstrap percentiles were asymmetric by one and used neither the `(R+1)` convention
  nor independent streams. Both fixed.
- `GateReport(gates=[]).passed` was `True` because `all([]) is True` — and a unit test
  used exactly that to bypass the "no science without gates" property. Now `False`.

## F. Infrastructure

- `backend.start()` sat outside the `try`, so a startup timeout left an orphaned
  server; `stop()` killed only the group leader, never waited again, and cleared
  `_proc` unconditionally. Now `start_new_session=True`, start inside the `try`,
  SIGTERM→SIGKILL escalation to the process group with a second wait, and `_proc`
  cleared only once reaped. `free_gpu_memory()` polls until VRAM actually drops and
  returns what happened.
- `detect_gpu_offload` reported `offload_detected: True` on a CPU-only build, because
  llama.cpp echoes `n_gpu_layers = 99` regardless. Weak evidence now sets
  `n_gpu_layers_requested` only. A new gate requires offload to be identical across
  precisions — a block that fell back to partial CPU is comparable on neither quality
  nor throughput.
- Stage C isolation depended on passing the right variable to `store.search(...)`.
  `run_end_to_end_agent` now takes a single `PrivateIndex` and `assert_isolation()`
  runs five real checks including cross-index query probes, instead of one that could
  only fail on a constructor bug.
- `hash_fallback` retrieval passed all eight gates. New `embedding_is_real_model` gate.
- The frozen-evidence cache key omitted the document body, so a changed corpus served
  stale text while parity still passed. Body sha256 added to the key.
- The blind adjudication worksheet was recoverable by brute force in three hashes
  (`sha256("adj|{qid}|{precision}|{condition}")[:12]` over a three-element space with
  the qid in plaintext) and `package_results.py` bundled the key beside it. Now a
  random per-run salt held only in the key file, which is excluded from the bundle.
- `EventLog.get` was O(file) per call; a corrupt line anywhere was silently dropped
  though the README promised only a torn *final* line. Cached, and mid-file corruption
  is counted and warned.
- `run_report` was rebuilt and overwritten each invocation, collapsing the systems
  table to one row when precisions ran separately. Now merged.
- `--force` left no trace in any artifact. Now recorded with `--force-reason` and
  surfaced downstream.
- `build_indexes.py` hardcoded retrieval defaults while `run_stage.py` read them from
  the stage config; a divergence would have silently re-derived evidence at runtime
  inside the F16 block. Both read the config now.
- `prepare_dataset.py`'s `doc_identity` returned `""` for a whitespace-only URL because
  the `or` chain resolved before `.strip()`, silently dropping evidence and
  under-deriving width **[reproduced]**.
- `config.load_yaml` silently fell back to a mini-parser that cannot represent the
  repo's own config. Missing PyYAML is now a hard error.
- Redundant full-disk sha256 passes over 11.5 GB (three to four of them) replaced by a
  cache keyed on path/size/mtime.

---

## Not fixed, deliberately

**Per-role precision assignment** — "F16 orchestrator + Q4 workers" and its converse.
This is a feature, not a defect: the README has always scoped the pilot to uniform
precision. It is also the proposal's headline contribution, so it should be the next
piece of work. It needs either two model servers resident simultaneously (~8 GB
combined for a 3B — tight but feasible on a T4) or a re-load between roles, which would
destroy the latency measurement. See `docs/PROPOSAL_REVIEW.md` §11.
