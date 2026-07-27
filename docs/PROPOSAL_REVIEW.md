# Proposal review — *Hierarchical Multi-Agent RAG Under Aggressive Quantization*

Target venue: NeurIPS 2026 workshop, **Efficient and On-Device AI Agents**.

Reviewed against the codebase in this repository at the audit commit. Ordered by
what would sink the submission first.

---

## 0. The thing you already noticed

> *"there were cases where 4Q was just consistently better than 8Q and 16F in terms
> of accuracy; which makes little to no sense."*

It wasn't the model. Six separate mechanisms in the harness handed Q4 free credit,
and every one pointed the same way. All six are fixed on this branch; the two that
almost certainly produced your Qwen result:

- **`contains_answer` was a raw substring test.** Gold `"no"` matched inside
  `"not"`, `"cannot"`, `"nothing"`; gold `"Yes"` matched inside `"Yesterday"`. And
  `score = max(em, contains, f1)`, so one accidental substring pinned the headline
  at 1.0. `answer_score("I do not know.", gold="no")` returned **1.00**. Abstention
  and hedging are exactly what a smaller/quantized model does more of, and ~60% of
  MultiHop-RAG's answerable questions have gold answer yes/no.
- **`synthesis_loss` returned `0.0` — perfect — when the agents recovered nothing**,
  and `any_required_branch_failed` evaluated `0 < 0 == False` when every agent died.
  Total branch collapse scored as flawless orchestration.

Two further notes on that preliminary run:

- **There is no Qwen 2B.** Qwen2.5 ships 0.5B / 1.5B / 3B / 7B / 14B / 32B / 72B.
  Whatever you ran, name it exactly in the paper — a reviewer will check.
- **~50 questions of a ~2000-question benchmark is not a null result, and it is not
  a positive one either.** With paired per-question differences at n=50 on a coarse
  metric, the interval on a 3-point accuracy gap is roughly ±10 points. The run
  could not have distinguished the hypotheses even with a correct scorer.

Do not put the preliminary result in the paper as a finding. It is a debugging
story, and it is a good one — a short "why we rebuilt the harness" paragraph in
Methods is worth more than a table.

---

## 1. The novelty claim is false, and it is falsified by a paper you cite yourself

> *"I can almost say with 100% confidence that multiagent RAG orch with quantized
> agents has never been done before"*

| Paper | What it already does |
| --- | --- |
| [arXiv 2606.05658](https://arxiv.org/abs/2606.05658) — *Agent-Orchestrated Adaptive RAG* (Maharjan, Jun 2026) | A multi-agent RAG system — Query Classifier, Query Decomposer, Answer Evaluator, central Orchestrator — running entirely on Llama-3.1-8B-Instruct **quantized to 4-bit GGUF via llama.cpp**, evaluated on MuSiQue. This *is* quantized multi-agent RAG orchestration. |
| [arXiv 2602.02711](https://arxiv.org/abs/2602.02711) — *Dynamic Mixed-Precision Routing* (Li, Feb 2026) | **You cite this as reference #10.** Adaptively routes between high- and low-precision models at each step of multi-step agentic interaction. This is "mixed-precision agent routing" almost verbatim. |
| [arXiv 2505.19433](https://arxiv.org/abs/2505.19433) — *ACBench* (Dong, May 2025) | First benchmark of compression's effect on agentic ability: GPTQ/AWQ + pruning across workflow generation, tool use, long-context retrieval, 15 models. Pre-empts "we characterize quantization degradation in agents." |
| [arXiv 2605.20315](https://arxiv.org/abs/2605.20315) — *Mix-Quant* (Lu, May 2026) | Phase-aware selective precision for agentic LLMs (FP4 prefill, BF16 decode). Already establishes that uniform quantization degrades agentic workflows and that *selective* precision is the fix. |
| [arXiv 2512.17914](https://arxiv.org/abs/2512.17914) — *Q-KVComm* (Kriuk, Nov 2025) | Multi-agent communication with adaptive layer-wise variable bit-width allocation by sensitivity profiling. Pre-empts "heterogeneous quantization across cooperating agents." |

Asserting 100% novelty in one section while citing the nearest prior art in another
reads as either carelessness or overclaiming. That is a worse failure than a bad
citation, and it is the kind of thing a workshop reviewer flags in one line.

**What is still defensible, if you narrow hard.** Per-*agent* precision assignment
across concurrently-running, role-specialized RAG agents — as opposed to DMR's
per-*step*, Mix-Quant's per-*phase*, and Q-KVComm's per-*layer* — evaluated with
RAG-specific faithfulness and attribution metrics rather than task success. That
is a real gap. Frame it as **"the first systematic study of X under Y"**, never as
"never been done," and cite all five papers above in Related Work.

---

## 2. The experiment as specified is roughly 20 GPU-days, not a workshop pilot

Counting the grid in Experimental Setup §2:

```
monolithic          3 precision x 3 complexity x 2 corpora            =  18 cells
flat + hierarchical 2 arch x 3 prec x 4 width x 3 cplx x 2 corpora    = 144 cells
role-precision      5 arms x 4 width x 3 cplx x 2 corpora             = 120 cells
                                                                        --------
                                                                        282 cells
x 175 items per cell                                              = 49,350 runs
x ~9 model calls per hierarchical run (1 coordinator + w agents + 1 synth)
                                                                  ~444,000 calls
x ~4 s/call serial on a T4                                     ~493 GPU-hours
```

That is **~20 days of continuous GPU**, before the 3-seed replication, and Kaggle
gives you 30 GPU-hours a week. The harness in this repo is sized for **72
predictions**. The gap is four orders of magnitude.

This is the single biggest problem with the proposal after the novelty claim, and
it is fixable by choosing. Suggested cut that keeps the paper's spine:

- **Drop the corpus axis** (×2) to a limitation. One corpus, stated as such.
- **Drop `flat` architecture** (×~0.5) or make it a single spot-check at one width.
- **Drop the 3-seed replication.** You are at `T=0` with `top_k=1`; spend the
  budget on items instead, and verify determinism with a same-prompt-twice
  assertion (this branch adds one).
- **Cut complexity to 2 levels** (multi-hop vs. aggregation) — single-hop does not
  need decomposition and will show nothing.
- **Keep width as the primary axis.** It is the compounding test and it is the
  paper.

That lands near 3 prec × 4 width × 2 complexity × 175 = 4,200 runs ≈ 38k calls ≈
**~42 GPU-hours**. Feasible, and still the strongest figure in the paper.

---

## 3. Width {2, 4, 8, 16} does not exist in your benchmark

MultiHop-RAG evidence spans **2–4 documents**, maximum. I checked the corpus: usable
candidates are `{2: 1169, 3: 774, 4: 312}`. Widths 8 and 16 are not available, and
width is your primary independent variable.

Three honest options:

1. **Add distractor branches.** Run 8 or 16 document agents where only 2–4 are
   required and the rest are distractors. This measures something real — *does
   orchestration degrade as irrelevant branches multiply* — but it is not the same
   quantity as "more required evidence," and the paper must not conflate them.
2. **Switch or add a benchmark that actually has high fan-out.** HotpotQA is also
   2-hop. Consider Loong (which you already piloted), FanOutQA, or a long-document
   set chunked so that required evidence genuinely spans 8+ retrieval units.
3. **Reframe width as "number of active agents"** rather than "required documents,"
   and be explicit that beyond 4 the extra agents are distractor branches.

Whichever you pick, say it in one sentence in Methods. Right now the proposal
implies MultiHop-RAG supports width 16, and it does not.

---

## 4. The compounding model is stated wrong

> *`P(success) = p(agent success)^# of agents`*

Two problems:

- **Branches are not independent.** They share a coordinator, share one model, and
  their difficulty is correlated through the question. Independence is the
  assumption that makes the exponent meaningful, and it is false here.
- **The final answer does not require every branch to succeed.** You score partial
  credit (atomic-fact recall), so the outcome is graded, not Bernoulli.

Also, if you *did* assume independence, the model predicts that width dominates
precision: at p=0.9, width 16 gives 0.19 regardless of quantization. Everything
interesting is in whether `p` itself falls faster for Q4 as width grows — which is
exactly the **precision × width interaction** in RQ2. RQ2 is right; the Key Ideas
section contradicts it.

Fix: state the estimand as the interaction coefficient with a CI, drop the
independence product, and keep the exponential model only as an illustrative
motivation clearly labelled as such.

---

## 5. Primary metric contradicts itself, and contradicts the harness

Experimental Setup §6 puts an **LLM judge** in the primary metric. The harness in
this repo says, in the README: *"No binary-only success, no external API judge."*
Limitations §5 then concedes the judge introduces noise and that a quantized judge
would contaminate attribution.

Pick one. My recommendation, and what this branch is built for:

- **Primary:** atomic-fact recall against a human-reviewed rubric — deterministic,
  reproducible, no API dependency, and it is what the gates and CIs are wired to.
- **Secondary:** exact match and token F1 (both already computed).
- **Validation, not scoring:** a blind human adjudication pass on a sample. This
  branch fixes the blinding, which was recoverable by brute force in three hashes.

If you keep an LLM judge, it must be a *fixed, full-precision, named* model, and
you must report judge–human agreement on a sample before using it for anything.

---

## 6. "Equal-compute comparison" is undefined, and it decides the headline

Multi-agent RAG trivially wins if it may spend unlimited calls, and your own
contribution list says the matched-baseline comparison is "almost certain to FAIL
for basic RAG." So this control is load-bearing, and "equal compute" currently has
no definition. Equal what?

- equal **wall-clock** — favors quantized multi-agent (parallelizable)
- equal **generated tokens** — favors monolithic
- equal **prefill+decode FLOPs** — the most defensible, and the hardest to measure
- equal **peak memory** — the one that actually matters for the on-device framing

Pick one as primary, report at least one other, and say why. For this venue I would
make **peak memory** primary and report tokens as the secondary.

---

## 7. For *this* venue, nothing is measured on-device

The workshop is **Efficient and On-Device AI Agents**. The motivation section is
entirely about edge deployment. But every number in the proposal comes from a T4 in
a Kaggle notebook, which is not an edge device and has a memory and bandwidth
profile nothing like one.

This is the cheapest big win available to you. One additional measurement column
from a real constrained device — a Jetson Orin Nano, an M-series laptop on CPU, or
even a Raspberry Pi 5 for the Q4 arm only — changes the paper from "quantization
study that mentions edge" to "on-device agent study." You do not need the full grid
there; the *Pareto* figure at one width, on real hardware, is enough.

If you cannot get a device, drop the edge framing to a Future Work paragraph and
motivate on memory cost instead. Do not claim on-device relevance from T4 numbers.

---

## 8. Falsifiability: no number is actually committed

> *"We know we're wrong if accuracy is not maintained within some certain percentile
> and latency does not improve over some percentile."*

> *"State a threshold in advance … Commit to the number before running so the result
> can disappoint us."*

The instinct is right and the number is missing. Commit to something like:

> **H2 is refuted if** the F16→Q4 gap in atomic-fact recall at maximum width exceeds
> **0.05** with the 95% CI upper bound above that margin, **or** if the precision ×
> width interaction coefficient has a CI excluding zero in the degrading direction.
> **H3 is supported only if** a mixed-precision configuration lands within **0.02**
> of full-F16 recall at **≤60%** of peak F16 VRAM.

Those exact numbers are negotiable; having numbers is not. The harness on this
branch takes `--ni-margin` and will refuse to emit a non-inferiority verdict unless
the interval is inside it *and* enough paired differences are non-zero.

---

## 9. Smaller but real

- **Reconcile the SPD-RAG stance.** You call it *"a shitcan paper"* and then build
  the architecture from it. Either justify the document-axis decomposition on its
  own terms (it is defensible: isolation is structural, branches are independent
  retrieval universes) or pick MA-RAG as the base. A reviewer will ask why you built
  on something you distrust, and "it brought the problem to my attention" is not an
  answer that survives review.
- **RAGChecker is not future work — it is RQ1.** Group C frames RAGChecker/Ragas as
  "diagnostic tooling the future-work section will reference," but RQ1 (stage
  attribution: retrieval vs. extraction vs. synthesis error budget) is essentially
  what RAGChecker does. Use it, cite it as the method, and you get a validated
  metric suite for free instead of hand-rolling one.
- **Pick a primary between aggressive and hybrid.** The central research question
  asks about "aggressive *or* hybrid"; the contribution list marks hybrid OPTIONAL;
  the title says "Under Aggressive Quantization." Three different primaries. For a
  workshop paper, make **uniform precision × width** the confirmatory result and
  **mixed precision** the actionable follow-on — that also matches what the harness
  can actually run.
- **Power calculation ordering.** §2 says items-per-cell will be confirmed against a
  power calculation "once we have a pilot effect size," but the pilot as designed
  (6/12 questions) cannot estimate an effect size usefully. Size the pilot to
  produce a usable variance estimate, or take the effect size from ACBench /
  the long-context quantization paper (#9) instead.
- **Ethics section.** "None of note" is fine, but say one sentence about the actual
  one: cheaper local agents lower the cost of automated retrieval over private
  document collections. That is a real, non-fabricated consideration for this venue.

---

## 10. Citation corrections

All 14 references resolve to real papers — **zero hallucinations**, which is not the
usual outcome for an AI-assisted reference list. Four need fixing:

| # | Issue | Correction |
| --- | --- | --- |
| 3 | 2604.00901 cited without a title | *Experience as a Compass: Multi-agent RAG with Evolving Orchestration and Agent Prompts* (HERA), Sha Li, 2026-04-01. Your claim about static roles **is** supported by its abstract. |
| 5 | Title wrong | Official title is *Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning* — no `MMOA-RAG:` prefix. MMOA-RAG is the method name. Yiqun Chen, 2025-01-25. |
| 8 | Title truncated | *AWQ: Activation-aware Weight Quantization **for LLM Compression and Acceleration***. Ji Lin, 2023-06-01. |
| 13 | Wrong year | Ragas is arXiv **2023**-09-26 (Shahul Es). "2024" is only correct if you cite the EACL 2024 demo version explicitly. |

Also verified: `NebulAICompany/SPD-RAG` and `yixuantt/MultiHop-RAG` both exist.
Reference #1's authorship ("Akay et al., 2026") is correct.

---

## 11. What the harness can and cannot do for you today

Fixed on this branch and ready:

- uniform precision × width, fixed-evidence and end-to-end conditions
- stage-level attribution (branch / orchestration / synthesis), which is RQ1
- paired bootstrap CIs, a predeclared NI margin, and an interaction contrast that
  now carries an interval
- systems metrics: per-process VRAM and RSS, load time, tokens/s, wall time
- provenance gates that make a fabricated or contaminated run unpublishable

**Not supported, and it is your headline contribution:** *per-role precision
assignment*. The README says so explicitly — "Uniform system precision … Stage-specific
quantization is out of scope." Running `Q4 orchestrator + F16 workers` needs either
two servers resident simultaneously (≈8 GB combined for a 3B — fits on a T4, tight
with two) or a re-load between roles (kills the latency measurement).

That is a design decision, not a bug, so I have not built it. It is the one thing
standing between this repo and the experiment the proposal describes, and it should
be the next piece of work.
