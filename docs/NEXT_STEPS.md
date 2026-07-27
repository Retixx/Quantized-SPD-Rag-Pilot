# Next steps

Ordered. Each step says what "done" looks like, and steps 1–3 are all cheaper than
the GPU time they protect. Do not skip to step 5.

---

## 0. Land the branch — 5 minutes

- Close **PR #1** ("Make the no-PyYAML config fallback agree with PyYAML"). Its commit
  is already in `fix/audit-remediation`; merging both would conflict.
- Merge `fix/audit-remediation` → `main`.

**Done when:** `main` has the audit fixes and `pytest tests -q` is 182 passing.

---

## 1. Review the gold rubric — 1–2 hours, human, blocks everything

**This is the highest-leverage hour in the project and no code can do it.**

`data/gold_facts.json` holds 55 atomic facts across 18 questions, seeded automatically
from MultiHop-RAG's own `evidence_list[].fact`. **Every one is flagged
`needs_manual_review: true`**, and until they are cleared every report is stamped
`rubric_reviewed: false` and every number is provisional.

Atomic-fact recall is the pilot's primary metric. It is measured by string-matching
against exactly these sentences. If a fact is not actually atomic, not actually
required to answer the question, or phrased so no correct answer would match it, then
the headline number is measuring the rubric rather than the model.

For each question in `data/gold_facts.json`:

1. Check each `text` is a single, atomic, checkable claim. Split compound ones.
2. Add `aliases` for legitimate paraphrases — `"1.8 billion"` / `"$1.8B"` /
   `"1,800,000,000"`. The matcher is token-overlap with a numeric gate, so a fact
   whose number is written differently in a correct answer will score 0.
3. Add `answer_aliases` for the short answer.
4. **Delete facts that are not required to answer the question.** Seeded evidence often
   includes context that a correct answer would not restate. Every one of these is a
   permanent, uniform drag on recall.
5. Set `needs_manual_review: false` on the entry, and on the question once all its
   facts are done.

**Done when:** `python scripts/score.py --stage stage_a` no longer prints the
`rubric_reviewed=false` warning.

> Do this **before** any model output exists. Editing the rubric after seeing which
> facts a model missed is the single easiest way to fabricate a result without meaning
> to.

---

## 2. Settle three experiment-design questions — before spending compute

From [`PROPOSAL_REVIEW.md`](PROPOSAL_REVIEW.md). All three change what you run.

| # | Question | Why it blocks |
| --- | --- | --- |
| §1 | How is novelty framed? | The "never been done" claim is refuted by arXiv 2606.05658 and by DMR, which the proposal cites as reference #10. Needs narrowing to "first systematic study of per-agent precision in document-parallel RAG" before anyone writes an abstract. |
| §2 | What gets cut from the grid? | As specified it is ~444k model calls ≈ 493 T4-hours ≈ 20 GPU-days, against a 30 GPU-hour/week Kaggle quota. §2 has a cut to ~42 hours that keeps the compounding figure. |
| §3 | Where does width > 4 come from? | MultiHop-RAG evidence spans 2–4 documents. Widths 8 and 16 do not exist in this corpus, and width is the primary independent variable. Options: distractor branches (measures something different — say so), a different corpus, or cap the claim at 4. |

**Done when:** each has a written answer in the proposal, not a placeholder.

---

## 3. Null calibration run — ~1 GPU-hour, do this first

Run **F16 against itself**, twice, under two different question orders, scored as if
the two runs were different precisions.

Every gap should be exactly 0.000 and every verdict should be `NON_INFERIOR` *for the
right reason* (enough informative pairs, not a degenerate zero-width interval).
Anything that moves is the harness measuring question order, warmup, KV-cache reuse or
scorer noise — not precision. Six such effects were found and fixed in the audit; this
is how you find the seventh before it becomes a finding.

**Small code task, not yet done:** `run_stage.py` de-duplicates `--precisions`, so it
cannot currently run the same precision twice. It needs a `--replicate-label` (or
equivalent) so two F16 blocks are recorded as distinct arms. Perhaps an hour of work.

**Done when:** the null run reports a gap of 0.000 on every metric, and the systems
table shows the two blocks within noise of each other on tokens/s and peak VRAM.

---

## 4. Stage A — ~1 hour GPU, 18 predictions

```bash
python scripts/prepare_dataset.py --data-dir data     # FIRST: imatrix needs the holdout
python scripts/prepare_models.py --build-llama-cpp --free-hf-weights
python scripts/build_indexes.py --stage all
python scripts/run_stage.py --stage stage_a --backend llama-server
python scripts/score.py     --stage stage_a --emit-adjudication
python scripts/analyze.py   --stage stage_a --ni-margin 0.05
python scripts/package_results.py --stage stage_a
```

Then read `results/stage_a/gate.json`. **Do not read anything else first** — the
per-question outputs are exactly what you should not be looking at while deciding
whether to continue.

| F16 recall | Meaning | Do |
| --- | --- | --- |
| `< 0.50` | model or benchmark floor | **Stop.** A 3B at F16 that cannot do the task cannot show a quantization effect. Change model or benchmark; do not `--force`. |
| `0.50–0.70` | marginal | Read the failures. Usually prompts or rubric, not the model. |
| `0.70–0.95` | pass | Go to Stage B. |
| `≥ 0.95` | ceiling | No headroom to detect degradation. Add distractors or harder items. |

**Done when:** `gate.json` has `proceed: true`, or you have written down why you are
stopping.

---

## 5. Stage B — ~2 hours GPU, 36 predictions

This is the actual result: precision × width, and the F16−Q4 gap growth with document
count. Runs only if Stage A's gate passed.

Read, in this order: `scientific_outcome`, then `primary_metric_analysis.f16_q4_gap_ci`,
then `f16_q4_gap_growth_ci`. A `GO_COMPOUNDING_DEGRADATION` now requires the growth CI
to exclude zero and ≥ 3 paired questions per width cell — if you get
`WEAK_GO_EXPAND` instead, the reason string says which of those failed.

**Done when:** you can state the outcome and its interval in one sentence without
looking anything up.

---

## 6. Stage C — optional, ~1 hour, 18 predictions

End-to-end retrieval on 6 Stage-B questions selected deterministically. Only worth it
if Stage B is informative.

---

## 7. Per-role precision — the headline contribution, not yet built

`F16 orchestrator + Q4 workers` and its converse. The harness is uniform-precision by
design; the README has always said so. This is the gap between this repo and the
experiment the proposal describes.

Needs either two model servers resident simultaneously (≈8 GB combined for a 3B —
feasible on a T4, tight) or a re-load between roles, which would destroy the latency
measurement. Budget a day.

---

## Things that will quietly ruin the result

- **Do not regenerate `data/pilot_manifest.json` after any model output exists.**
  `prepare_dataset.py` refuses without `--force` for this reason.
- **Do not quote numbers while `rubric_reviewed: false`.**
- **Do not open `results/*/adjudication_key.json`** before adjudicating. It de-blinds
  the worksheet and is excluded from the packaged bundle on purpose.
- **Do not pass `--force` without `--force-reason`.** It is recorded and surfaced
  downstream; an undocumented override is indistinguishable from a legitimate run six
  months later.
- **Do not change `--fact-threshold` after seeing results.** It is recorded, and
  re-scoring at a threshold picked afterwards is a researcher degree of freedom.
- **Watch for `NOT_SIGNIFICANT_BUT_CI_TOO_WIDE` with `degenerate: true`.** It means
  the paired differences were nearly all exactly zero. That is "we learned nothing",
  not "they are equivalent".

## Where things are

| | |
| --- | --- |
| What was broken and why | [`AUDIT_FIXES.md`](AUDIT_FIXES.md) |
| Experiment-design review | [`PROPOSAL_REVIEW.md`](PROPOSAL_REVIEW.md) |
| Known limitations | README, bottom section |
| Harness-only smoke test | `SPDQ_ALLOW_FIXTURE=1 python scripts/run_stage.py --stage stage_a --backend fixture` → writes to `results/_harness/`, always ends in `SCIENTIFIC_RECOMMENDATION_NOT_AVAILABLE` |
