"""One test per bug found in the audit. Named so the intent survives the fix.

Every test here failed against the pre-audit code. The common shape of the
bugs was a metric or a gate that rewarded ABSENCE: an empty denominator scored
1.0, a missing precision satisfied a parity check, a dead agent reported
perfect synthesis, an abstention matched the gold answer as a substring. The
assertions below pin the direction of each of those, because a scoring bug that
flatters degradation is invisible in a study whose entire subject is
degradation.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze as A  # noqa: E402
import coordinator as coord_mod  # noqa: E402
import document_agent as agent_mod  # noqa: E402
import metrics as M  # noqa: E402
import prompts  # noqa: E402
from backends.llamacpp_backend import (FIXTURE_ENV, FixtureBackend,  # noqa: E402
                                       GenerationResult, SamplingConfig)
from fixtures import passing_events_and_predictions, passing_provenance  # noqa: E402
from gates import MIN_INFORMATIVE_PAIRS, evaluate_gates  # noqa: E402
from llm_client import BackendGenerationError, GuardedClient  # noqa: E402
from logging_utils import EventLog, call_id, stable_id  # noqa: E402
from schemas import (CoordinatorOutput, DocumentAgentOutput,  # noqa: E402
                     SupportedFact, extract_json_object, parse_into)

CONDITION = "fixed_verified_evidence"


# -- scoring: abstention and negation ---------------------------------------


def test_regression_abstention_does_not_match_the_gold_answer_as_a_substring():
    """`contains` was a raw substring test on the normalised strings, so gold
    "no" matched inside "not"/"cannot"/"know" and gold "Yes" matched inside
    "Yesterday" -- handing a perfect score to refusals and to answers that said
    the opposite of the gold."""
    assert M.answer_score("I do not know.", "no")["score"] == 0.0
    assert M.answer_score("I cannot answer that.", "no")["score"] == 0.0
    assert M.answer_score("Nothing in the passages says so.", "no")["score"] == 0.0
    assert M.answer_score("No. Yesterday the results were announced.",
                          "Yes")["score"] == 0.0
    # ... while a genuine answer still scores 1.0
    assert M.answer_score("Google", "Google")["score"] == 1.0
    assert M.answer_score("The company is Google.", "Google")["score"] == 1.0
    assert M.answer_score("no", "no")["score"] == 1.0
    assert M.answer_score("Yes", "yes")["score"] == 1.0


def test_regression_containment_is_suppressed_by_negation_and_by_length():
    """Both suppressions are reported, not silently applied: a metric that
    quietly zeroes a hit is unauditable."""
    negated = M.contains_answer("The acquirer is not Google but Microsoft.", "Google")
    assert negated["contains"] == 0.0
    assert negated["negated"] is True
    assert negated["too_long"] is False

    plain = M.contains_answer("The acquirer is Google.", "Google")
    assert plain["contains"] == 1.0 and plain["negated"] is False

    # Listing every candidate entity until one of them is the gold answer is
    # not answering. Above the length guard a containment hit is not credited.
    ramble = "Google " + " ".join(f"company{i}" for i in range(120))
    long_hit = M.contains_answer(ramble, "Google")
    assert long_hit["contains"] == 0.0
    assert long_hit["too_long"] is True
    assert long_hit["pred_tokens"] > long_hit["max_pred_tokens"]

    scored = M.answer_score(ramble, "Google")
    assert scored["contains"] == 0.0
    assert scored["contains_suppressed_too_long"] is True


def test_regression_blank_prediction_against_blank_gold_is_not_a_perfect_score():
    """`float(p == g)` gave a blank prediction 1.0 against a blank gold. A
    question with no gold answer carries no signal and must be EXCLUDED, which
    is what `scorable=False` and `score=None` mean downstream."""
    blank = M.answer_score("", "")
    assert blank["score"] is None
    assert blank["scorable"] is False
    assert M.answer_score("anything at all", "")["scorable"] is False
    assert M.token_f1("", "") == 0.0
    assert M.char_f1("", "") == 0.0
    assert M.token_f1("", "Google") == 0.0
    assert M.token_f1("Google", "") == 0.0


# -- scoring: verbosity and numbers ------------------------------------------


def test_regression_appending_filler_text_cannot_raise_fact_match_score():
    """Coverage was measured over the WHOLE candidate, so appending text could
    only ever raise the score. Quantization changes output length, so a
    length-sensitive metric cannot separate verbosity from quality -- it was
    rewarding whichever precision rambled more."""
    gold = "Cinder Mining listed 940 employees in its 2023 annual report."
    partial = "Cinder Mining listed 940 employees."
    filler = " ".join(f"padding{i} noise{i} extra{i}" for i in range(40))
    padded = f"{partial} {filler} the 2023 annual report was published."

    base = M.fact_match_score(gold, partial)
    assert 0.0 < base < 1.0
    assert M.fact_match_score(gold, padded) == base, \
        "appending text must not raise coverage"
    # the un-windowed measure is what the bug looked like
    assert M.fact_match_score(gold, padded, windowed=False) > base

    # and the property holds at the threshold that decides supported/unsupported
    assert M.fact_supported(gold, padded) == M.fact_supported(gold, partial)


def test_regression_a_wrong_number_zeroes_a_fact_match_rather_than_halving_it():
    """A fact with the wrong number is not the fact. Halving left a wrong-number
    restatement above the 0.6 support threshold whenever the wording matched."""
    gold = "Cinder Mining listed 940 employees in its 2023 annual report."
    wrong = "Cinder Mining listed 512 employees in its 2019 annual report."
    assert M.fact_match_score(gold, wrong) == 0.0
    assert not M.fact_supported(gold, wrong)
    assert M.fact_match_score(gold, gold) == 1.0

    # polarity disagreement is a softer signal, so it halves rather than zeroes
    negated = "Cinder Mining did not list 940 employees in its 2023 annual report."
    assert 0.0 < M.fact_match_score(gold, negated) < 1.0


def test_regression_atomic_fact_precision_is_falsifiable():
    """The old check accepted a match in EITHER direction, so any predicted
    fact whose tokens were a subset of a gold fact counted as supported --
    emitting one-word facts scored precision 1.0."""
    gold = [{"fact_id": "f1",
             "text": "Cinder Mining listed 940 employees in its 2023 annual report."}]
    one_word = M.atomic_fact_precision(["Cinder"], gold)
    assert one_word["precision"] == 0.0
    assert one_word["n_too_short"] == 1

    honest = M.atomic_fact_precision(
        ["Cinder Mining listed 940 employees in its 2023 annual report."], gold)
    assert honest["precision"] == 1.0
    assert honest["n_too_short"] == 0

    fabricated = M.atomic_fact_precision(
        ["Cinder Mining opened a research office in Lisbon last quarter."], gold)
    assert fabricated["precision"] == 0.0


def test_regression_empty_denominators_are_none_not_zero_or_one():
    """A ratio over an empty set is undefined. Reporting 0.0 made "the agent
    emitted nothing" indistinguishable from "the agent was perfectly faithful",
    and dragged aggregates in the flattering direction."""
    assert M.atomic_fact_recall([], "any text")["recall"] is None
    assert M.atomic_fact_precision([], [])["precision"] is None
    assert M.uncited_claim_rate([]) is None
    assert M.unsupported_claim_rate([]) is None
    assert M.orchestration_metrics([], [], [])["any_required_branch_failed"] is None
    assert M.orchestration_metrics([], [], [])["required_document_coverage"] is None
    eff = M.system_efficiency(None, None, None)
    assert eff == {"quality_per_second": None, "quality_per_gb": None,
                   "model_size_gb": None}
    assert M.system_efficiency(0.8, 0.0, 0)["quality_per_second"] is None


# -- orchestration and synthesis ---------------------------------------------


def test_regression_all_agents_dead_is_a_branch_failure_and_undefined_synthesis():
    """Two bugs that pointed the same way. `any_required_branch_failed` was
    `n_ok < len(agent_recalls)`, i.e. `0 < 0 == False` when every agent died,
    so a precision whose branches all collapsed reported a PERFECT branch
    failure probability -- the exact quantity the compounding hypothesis is
    about. And `synthesis_loss` fell back to 0.0 when the agents recovered
    nothing, so the worse the branches did, the better synthesis looked."""
    gold = [{"fact_id": "f1", "document_id": "d1",
             "text": "Borealis will acquire Cinder Mining for 1.8 billion dollars."},
            {"fact_id": "f2", "document_id": "d2",
             "text": "Cinder Mining listed 940 employees in its 2023 annual report."}]
    required = ["d1", "d2"]

    dead = M.orchestration_metrics([], required, [])
    assert dead["any_required_branch_failed"] is True
    assert dead["n_required_agents"] == 2
    assert dead["n_agents_returned"] == 0
    assert dead["n_agents_succeeded"] == 0
    assert dead["required_document_correct_coverage"] == 0.0

    assert M.synthesis_metrics([], "", gold)["synthesis_loss"] is None

    # agents that returned but recovered nothing are the same story
    empty_agents = [{"document_id": d, "supported_facts": [],
                     "insufficient_evidence": True} for d in required]
    empty = M.orchestration_metrics(empty_agents, required, [None, None])
    assert empty["any_required_branch_failed"] is True
    assert M.synthesis_metrics(empty_agents, "", gold)["synthesis_loss"] is None

    # a healthy pipeline still reports 0.0 loss, not None
    live = [{"document_id": "d1", "supported_facts": [
        {"fact": gold[0]["text"], "chunk_ids": ["c1"], "citation_valid": True},
        {"fact": gold[1]["text"], "chunk_ids": ["c2"], "citation_valid": True}]}]
    assert M.synthesis_metrics(
        live, f"{gold[0]['text']} {gold[1]['text']}", gold)["synthesis_loss"] == 0.0


def test_regression_forced_finalize_is_never_counted_as_retrieval_success():
    """A forced finalize is a free retry handed to whichever precision failed.
    Scoring it as a retrieval success attenuates the gap being measured."""
    rescued = {"document_id": "d1", "supported_facts": [],
               "forced_finalize": True, "retrieval_success": True,
               "search_rounds": 2}
    res = M.per_agent_recall(rescued, [])
    assert res["forced_finalize"] is True
    assert res["retrieval_success"] is False

    clean = dict(rescued, forced_finalize=False)
    assert M.per_agent_recall(clean, [])["retrieval_success"] is True

    assert M.orchestration_metrics([rescued], ["d1"], [None])["n_forced_finalize"] == 1


def test_regression_contradiction_count_ignores_question_numbers_and_formatting():
    """Restating a number from the prompt is not a contradiction, and 3 vs 3.0
    or 1,500 vs 1500 are the same number. Both used to be counted, so the
    measure tracked formatting style rather than faithfulness."""
    agents = [{"supported_facts": [{"fact": "The company runs 3.0 sites."},
                                   {"fact": "It employs 1,500 people."}]}]
    assert M.contradiction_count("It runs 3 sites and employs 1500 people.",
                                 agents, [], "") == 0

    # a number the question supplied is not invented by the synthesizer
    assert M.contradiction_count("The 2024 revenue figure was reported.", [], [],
                                 "What happened in 2024?") == 0

    # a genuinely novel number still counts
    detail = M.unsourced_numbers("Revenue was 412 million dollars.", agents, [], "")
    assert detail["unsourced_numbers"] == ["412"]
    assert detail["n_unsourced_numbers"] == 1
    assert M.contradiction_count("Revenue was 412 million dollars.", agents,
                                 ["a conflict"], "") == 2


# -- JSON extraction ---------------------------------------------------------


def test_regression_extract_json_object_returns_the_answer_not_the_schema_echo():
    """Every system prompt ends with a literal JSON example, so a model that
    echoes the schema before answering used to have its ECHO parsed as the
    reply -- yielding `{"instruction": "..."}` scored as a real extraction."""
    echoed = (
        'Understood. The required schema is '
        '{"shared_tasks":[{"task_id":"t1","instruction":"...",'
        '"required_fields":["..."]}],"synthesis_directive":"..."}\n'
        'Here is my answer:\n'
        '{"shared_tasks":[{"task_id":"t1","instruction":"Extract the acquisition '
        'price in dollars","required_fields":["numbers"]}],'
        '"synthesis_directive":"Merge the findings and keep exact numbers."}')
    got = extract_json_object(echoed, CoordinatorOutput)
    assert got is not None
    assert got["shared_tasks"][0]["instruction"] == \
        "Extract the acquisition price in dollars"
    assert got["synthesis_directive"] != "..."

    # the LAST fenced block wins, not the first, for the same reason
    assert extract_json_object('```json\n{"a":1}\n```\ntext\n```json\n{"a":2}\n```') \
        == {"a": 2}

    # an outer object beats one of its own nested fragments
    nested = ('{"question_id":"q1","document_id":"d1","search_queries":[],'
              '"retrieved_chunk_ids":["d1::c0"],'
              '"supported_facts":[{"fact":"x","chunk_ids":["d1::c0"]}],'
              '"insufficient_evidence":false}')
    assert extract_json_object(nested, DocumentAgentOutput)["document_id"] == "d1"

    # a stray brace in prose must not abort the scan
    assert extract_json_object('oops { unbalanced ... then {"a":3}') == {"a": 3}
    assert extract_json_object("no json here") is None
    assert extract_json_object("") is None


def test_regression_parse_into_always_returns_a_result_and_never_raises():
    """`model_cls(**obj)` raises TypeError on non-string keys. A malformed
    generation must never abort the question it belongs to."""
    for text in ('{"1": 2}', "not json", "", "[]", '{"shared_tasks": 5}',
                 '{"shared_tasks": [{"task_id": 1}]}'):
        result = parse_into(CoordinatorOutput, text)
        assert result.ok in (True, False)
        if not result.ok:
            assert result.error


def test_regression_citation_audit_survives_a_schema_round_trip():
    """`citation_valid` is set by the citation audit, never by the model. It was
    not a schema field, so re-validating an agent output dropped it and every
    fact silently became unsupported (rate 1.0)."""
    fact = SupportedFact(fact="x", chunk_ids=["d1::c0"], citation_valid=True,
                         citation_hallucinated=False, citation_absent=False)
    out = DocumentAgentOutput(document_id="d1", supported_facts=[fact])
    round_tripped = DocumentAgentOutput(**out.model_dump()).model_dump()
    assert round_tripped["supported_facts"][0]["citation_valid"] is True


def test_regression_drop_uncited_is_idempotent():
    """The first pass rewrites `chunk_ids` to the surviving ids, so a second
    naive pass saw an empty list and reclassified a HALLUCINATED citation as
    merely uncited -- zeroing `hallucinated_citations` on the entire
    forced-finalize path, which is where hallucinated citations concentrate."""
    out = {"retrieved_chunk_ids": ["d1::c0"],
           "supported_facts": [
               {"fact": "cited", "chunk_ids": ["d1::c0"]},
               {"fact": "hallucinated", "chunk_ids": ["other::c9"]},
               {"fact": "uncited", "chunk_ids": []}]}
    first = agent_mod._drop_uncited(out)
    assert first["hallucinated_citations"] == 1
    assert first["uncited_facts"] == 1
    assert first["citation_audit_done"] is True

    assert agent_mod._drop_uncited(first)["hallucinated_citations"] == 1
    forced = agent_mod._drop_uncited(first, force=True)
    assert forced["hallucinated_citations"] == 1
    assert forced["uncited_facts"] == 1


# -- prompts -----------------------------------------------------------------


def test_regression_fill_is_single_pass_so_model_text_cannot_inject_a_placeholder():
    """Several substituted values are MODEL-GENERATED (coordinator
    instructions, retrieved passages, prior replies). Sequential `str.replace`
    rescans already-substituted text, so an instruction containing the literal
    `{passages}` was expanded with the document body -- corrupting both the
    prompt and its hash, in a study whose central control is prompt hashing."""
    out = prompts.fill("instructions={instructions}\npassages={passages}",
                       instructions="extract everything in {passages}",
                       passages="THE DOCUMENT BODY")
    assert out == ("instructions=extract everything in {passages}\n"
                   "passages=THE DOCUMENT BODY")
    assert out.count("THE DOCUMENT BODY") == 1

    # braces belonging to the embedded JSON schemas survive byte-identically
    assert "{" in prompts.fill(prompts.COORDINATOR_SYSTEM)
    filled = prompts.fill(prompts.DOC_AGENT_FIXED_SYSTEM, question_id="q1",
                          document_id="d1", chunk_id_list='"d1::c0"')
    assert '"question_id":"q1"' in filled
    assert '"supported_facts":[{"fact":"..."' in filled


def test_regression_per_role_token_budgets_are_a_mapping():
    """A single scalar `max_tokens` was hashed into `sampling_hash` while no
    call site ever read it, so the recorded sampling configuration described a
    constraint the run did not have."""
    assert isinstance(prompts.DEFAULT_MAX_TOKENS, dict)
    for role in ("coordinator", "document_agent", "document_agent_turn",
                 "synthesizer"):
        assert isinstance(prompts.DEFAULT_MAX_TOKENS[role], int)
        assert prompts.DEFAULT_MAX_TOKENS[role] > 0


# -- analysis ----------------------------------------------------------------


def test_regression_all_zero_paired_diffs_do_not_yield_non_inferiority():
    """A stage where every paired difference is exactly 0.0 produces a
    zero-width bootstrap interval, which satisfied "the interval is narrow"
    perfectly: zero information yielding the strongest possible verdict. It is
    the signature of a pipeline where every question failed identically at
    every precision."""
    ci = A.paired_bootstrap_ci([0.0] * 8, resamples=500)
    assert ci["n_informative"] == 0
    assert ci["lo"] == ci["hi"] == 0.0

    label = A.verdict(ci, 0.05)["label"]
    assert label != "NON_INFERIOR_WITHIN_MARGIN"
    assert label == "NOT_SIGNIFICANT_BUT_CI_TOO_WIDE"
    assert A.verdict(ci, 0.05)["degenerate"] is True

    # one informative pair short is still not enough
    almost = A.paired_bootstrap_ci([0.0] * 8 + [0.01] * (MIN_INFORMATIVE_PAIRS - 1),
                                   resamples=500)
    assert almost["n_informative"] == MIN_INFORMATIVE_PAIRS - 1
    assert A.verdict(almost, 0.05)["label"] != "NON_INFERIOR_WITHIN_MARGIN"


def test_regression_verdict_on_a_loss_metric_is_not_sign_inverted():
    """`ci` is always over F16 - Q4. For an accuracy metric a positive
    difference means Q4 is worse; for a LOSS metric it means Q4 is better.
    Reading a loss metric with the accuracy convention reports degradation
    exactly when the quantized model improved, and vice versa."""
    assert "synthesis_loss" in A.LOWER_IS_BETTER
    assert "unsupported_claim_rate" in A.LOWER_IS_BETTER
    assert "branch_failure_probability" in A.LOWER_IS_BETTER
    assert "atomic_fact_recall_final" not in A.LOWER_IS_BETTER

    # Q4 has the LOWER loss, i.e. Q4 is better: F16 - Q4 is positive.
    q4_better = {"lo": 0.02, "hi": 0.08, "n": 8, "n_informative": 6}
    assert A.verdict(q4_better, 0.05, higher_is_better=False)["label"] == \
        "NON_INFERIOR_WITHIN_MARGIN"
    assert A.verdict(q4_better, 0.05, higher_is_better=True)["label"] == \
        "DEGRADATION_DETECTED"

    # Q4 has the HIGHER loss, i.e. Q4 is worse: F16 - Q4 is negative.
    q4_worse = {"lo": -0.08, "hi": -0.02, "n": 8, "n_informative": 6}
    assert A.verdict(q4_worse, 0.05, higher_is_better=False)["label"] == \
        "DEGRADATION_DETECTED"

    # `analyse` must pick the direction from LOWER_IS_BETTER, not from the
    # caller remembering to pass it.
    rows = []
    for i in range(6):
        rows.append({"question_id": f"q{i}", "precision": "F16", "width": 2,
                     "synthesis_loss": 0.5})
        rows.append({"question_id": f"q{i}", "precision": "Q4_K_M", "width": 2,
                     "synthesis_loss": 0.9})
        rows.append({"question_id": f"q{i}", "precision": "Q8_0", "width": 2,
                     "synthesis_loss": 0.5})
    res = A.analyse(rows, "synthesis_loss", 300, 0.05)
    assert res["higher_is_better"] is False
    assert res["f16_q4_gap_overall"] == pytest.approx(-0.4)
    assert res["q4_verdict"]["label"] == "DEGRADATION_DETECTED", \
        "Q4 losing 0.4 more facts per question is a degradation, not a win"
    assert res["q8_verdict"]["label"] != "DEGRADATION_DETECTED"

    # the same numbers read as an accuracy metric invert
    accuracy = A.analyse(
        [dict(r, atomic_fact_recall_final=r.pop("synthesis_loss")) for r in rows],
        "atomic_fact_recall_final", 300, 0.05)
    assert accuracy["higher_is_better"] is True
    assert accuracy["q4_verdict"]["label"] != "DEGRADATION_DETECTED"


def test_regression_paired_returns_a_tuple_and_get_metric_returns_none():
    """`paired` returns (diffs, question_ids) so a caller can say WHICH
    questions an interval rests on. `get_metric` returns None for a missing or
    unscorable value: 0.0 silently entered a paired difference as a real
    observation of zero."""
    rows = [{"question_id": "q0", "precision": "F16",
             "atomic_fact_recall_final": 0.8},
            {"question_id": "q0", "precision": "Q4_K_M",
             "atomic_fact_recall_final": 0.5},
            {"question_id": "q1", "precision": "F16",
             "atomic_fact_recall_final": 0.9},
            {"question_id": "q1", "precision": "Q4_K_M",
             "atomic_fact_recall_final": None}]
    result = A.paired(rows, "atomic_fact_recall_final", "F16", "Q4_K_M")
    assert isinstance(result, tuple) and len(result) == 2
    diffs, qids = result
    assert qids == ["q0"], "an unscorable side must drop the PAIR, not count as 0"
    assert diffs == [pytest.approx(0.3)]

    assert A.get_metric({}, "atomic_fact_recall_final") is None
    assert A.get_metric({"atomic_fact_recall_final": None},
                        "atomic_fact_recall_final") is None
    assert A.get_metric({"answer_score": {"score": None}}, "answer_score") is None
    assert A.get_metric({"answer_score": {"score": 0.0}}, "answer_score") == 0.0
    assert A.get_metric({"orchestration": {"any_required_branch_failed": None}},
                        "branch_failure_probability") is None


def test_regression_bootstrap_streams_differ_per_contrast():
    """Every interval used to share one RNG stream, which makes their
    Monte-Carlo errors correlated: deterministic, but not independent
    replicates."""
    a = A._seed_for("atomic_fact_recall_final", "F16", "Q4_K_M")
    b = A._seed_for("atomic_fact_recall_final", "F16", "Q8_0")
    c = A._seed_for("answer_score", "F16", "Q4_K_M")
    assert len({a, b, c}) == 3
    assert A._seed_for("x", "y") == A._seed_for("x", "y")   # still deterministic


def test_regression_mean_of_nothing_is_none():
    """`sum(xs)/len(xs)` over an empty list is a crash; returning 0.0 instead is
    worse, because it aggregates as a real measurement."""
    assert A.mean([]) is None
    assert A.mean([0.5, 0.5]) == 0.5

    import score as S
    assert S.mean([]) is None
    assert S.mean([None, None]) is None
    assert S.mean([1.0, None, 0.0]) == 0.5   # None values are excluded, not zeroed


# -- gates and provenance ----------------------------------------------------


def test_regression_question_failure_event_is_not_fixture_contamination():
    """A question_failure record used to be written with `is_fixture: True`, so
    one transient server timeout permanently poisoned an append-only event log
    and forced SCIENTIFIC_RECOMMENDATION_NOT_AVAILABLE for that stage forever --
    punishing the honest recording of a failure harder than crashing."""
    events, preds = passing_events_and_predictions(1)
    failure = {"call_id": stable_id("failure", "stage_a", "F16", CONDITION, "q0"),
               "stage": "stage_a", "precision": "F16", "condition": CONDITION,
               "role": "question_failure", "question_id": "q0",
               "is_failure_record": True,
               "failure_kind": "backend_generation_error",
               "error": "HTTPError: 502"}
    report = evaluate_gates(events + [failure], preds, passing_provenance(),
                            required_questions=1, condition=CONDITION,
                            verify_model_files=False)
    assert report.passed, report.failures()

    # a real fixture record still blocks
    contaminated = evaluate_gates(events + [dict(failure, is_fixture=True,
                                                 is_failure_record=False)],
                                  preds, passing_provenance(),
                                  required_questions=1, condition=CONDITION,
                                  verify_model_files=False)
    assert any("real_inference_only" in f for f in contaminated.failures())


def test_regression_a_backend_error_is_raised_and_left_out_of_the_resume_index(
        tmp_path):
    """A 502 is infrastructure, not model quality. It used to be parsed as an
    empty generation, scored as a parse failure, and -- worst -- written into
    the resume index, so rerunning the stage inherited the empty answer instead
    of retrying the call."""
    os.environ[FIXTURE_ENV] = "1"

    class ErroringBackend(FixtureBackend):
        def chat(self, messages, max_tokens=None):
            self.calls += 1
            return GenerationResult(text="", error="HTTP 502 Bad Gateway",
                                    backend="llama-server", finish_reason="error")

    backend = ErroringBackend(SamplingConfig(), precision="F16")
    backend.start()
    log = EventLog(tmp_path / "events.jsonl")
    client = GuardedClient(backend, log, "stage_a", "F16", CONDITION)

    with pytest.raises(BackendGenerationError) as exc:
        coord_mod.run_coordinator(client, "q0", "a question")
    cid = call_id("stage_a", "F16", CONDITION, "q0", "coordinator")
    assert exc.value.call_id == cid
    assert exc.value.precision == "F16"

    # the diagnostic event is recorded, but NOT under the failing call_id
    records = list(log.read())
    assert records and records[0]["generation_error"] == "HTTP 502 Bad Gateway"
    assert records[0]["call_id"] == ""
    assert records[0]["failed_call_id"] == cid
    assert not log.completed(cid), "an errored call must be retried, not resumed"

    # no repair attempt against a broken server
    assert backend.calls == 1
    assert records[0]["repair_attempted"] is False


def test_regression_stable_id_without_a_variant_is_byte_identical():
    """`variant` separates run modes that must not share a resume slot. Adding
    it unconditionally would have invalidated every id already on disk."""
    assert stable_id("a", "b", "c") == stable_id("a", "b", "c", variant="")
    assert stable_id("a", "b", "c") != stable_id("a", "b", "c", variant="harness")
    assert call_id("stage_a", "F16", CONDITION, "q0", "coordinator") == \
        call_id("stage_a", "F16", CONDITION, "q0", "coordinator", variant="")
    # attempt is a real component, so a retry is its own record
    assert call_id("stage_a", "F16", CONDITION, "q0", "agent", "d1", 0) != \
        call_id("stage_a", "F16", CONDITION, "q0", "agent", "d1", 1)


def test_regression_load_yaml_refuses_to_fall_back_to_the_mini_parser(monkeypatch):
    """The fallback parser cannot represent the nested `sampling.max_tokens`
    mapping the real config uses, so silently falling back produced a config
    that looked loaded and was wrong."""
    import config as C

    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(RuntimeError, match="PyYAML"):
        C.load_yaml(ROOT / "configs" / "models.yaml")
