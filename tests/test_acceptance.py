"""The ten required acceptance tests, one per brief item.

None of these touch the network or a GGUF. They test the HARNESS. Passing them
proves engineering readiness only -- test 5 and test 10 exist precisely to make
sure that distinction cannot be blurred.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import coordinator as coord_mod
import document_agent as agent_mod
import synthesizer as synth_mod
from backends.llamacpp_backend import (FIXTURE_ENV, FixtureBackend, SamplingConfig,
                                       build_backend)
from evidence import (FixedEvidenceCache, build_store, document_body_sha256,
                      embed_signature, fixed_evidence_key, get_fixed_evidence,
                      make_embedder)
from fixtures import (DOCS, QUESTION, documents_json,
                      passing_events_and_predictions, passing_provenance)
from gates import NOT_AVAILABLE, evaluate_gates, scientific_outcome
from llm_client import GuardedClient, messages_hash
from logging_utils import EventLog, call_id
from retrieval import DocumentIndexStore, DocumentIsolationError

MODELS_CFG = {"embedding": {"model": "test", "revision": "test", "hash_fallback": True}}
RCFG = dict(top_k=2, chunk_tokens=120, overlap_tokens=20)

PRECISIONS = ("F16", "Q8_0", "Q4_K_M")


def make_store(tmp_path, docs=None, cache_name="embed_cache"):
    emb = make_embedder(MODELS_CFG, str(tmp_path / cache_name))
    corpus = docs if docs is not None else documents_json()
    return build_store(corpus, emb, sorted(corpus), RCFG["chunk_tokens"],
                       RCFG["overlap_tokens"]), emb


def fixture_client(tmp_path, precision="F16", stage="stage_a",
                   condition="fixed_verified_evidence", dry_run=False,
                   backend=None, log_name=None):
    os.environ[FIXTURE_ENV] = "1"
    if backend is None:
        backend = build_backend("fixture", None, SamplingConfig())
    backend.precision = precision
    backend.start()
    log = EventLog(tmp_path / (log_name or f"events_{precision}.jsonl"))
    return GuardedClient(backend, log, stage, precision, condition, dry_run=dry_run), log


class PrecisionDependentCoordinator(FixtureBackend):
    """A backend whose coordinator output DIFFERS per precision.

    This is not a pathological mock: a coordinator run separately at each
    precision IS precision-dependent by construction, which is exactly why the
    real harness freezes it once at F16. A test that cannot tell the two apart
    proves nothing.
    """

    def chat(self, messages, max_tokens=None):
        res = super().chat(messages, max_tokens=max_tokens)
        prompt = json.dumps(messages, sort_keys=True)
        if "shared_tasks" in prompt and "synthesis_directive" in prompt:
            res.text = json.dumps({
                "shared_tasks": [{
                    "task_id": "t1",
                    "instruction": f"instruction authored by {self.precision}",
                    "required_fields": ["entities"]}],
                "synthesis_directive": f"directive authored by {self.precision}"})
        return res


# 1 ------------------------------------------------------------------------
def test_1_document_agent_can_only_access_its_assigned_document(tmp_path):
    store, _ = make_store(tmp_path)
    report = store.assert_isolation()
    assert report["isolated"], report["violations"]

    # an agent handed doc_bbb cannot reach doc_ccc's chunks by any route
    bbb = store.get("doc_bbb")
    ccc_chunk = store.get("doc_ccc").chunk_ids[0]
    with pytest.raises(DocumentIsolationError):
        bbb.get_chunk(ccc_chunk)
    hits = store.search("doc_bbb", "how many employees does Cinder Mining have", top_k=5)
    assert hits and all(h.document_id == "doc_bbb" for h in hits)
    assert all(not h.chunk_id.startswith("doc_ccc") for h in hits)
    with pytest.raises(DocumentIsolationError):
        store.get("doc_not_assigned")


# 2 ------------------------------------------------------------------------
def test_2_fixed_evidence_chunk_ids_identical_across_precisions(tmp_path):
    """Chunk selection is frozen once and reused, so it cannot vary by precision.

    The previous version looped over three precision labels while doing exactly
    the same work each time, and iterations 2 and 3 were pure cache hits -- it
    proved a dict returns what was put in it. This rebuilds the index from
    scratch in a fresh DocumentIndexStore with a fresh embedder and a fresh
    cache file, which is the property that actually has to hold when the three
    precision blocks run in separate processes on separate machines.
    """
    sig = None
    per_precision = {}
    for precision in PRECISIONS:
        # a genuinely independent rebuild per precision: new embedder, new
        # store, new cache file. Nothing is carried over except the corpus.
        store, emb = make_store(tmp_path, cache_name=f"embed_cache_{precision}")
        assert isinstance(store, DocumentIndexStore)
        cache = FixedEvidenceCache(tmp_path / f"fixed_evidence_{precision}.json")
        sig = embed_signature(emb)
        ids = []
        for doc in QUESTION["documents"]:
            hits = get_fixed_evidence(store, cache, QUESTION["question_id"],
                                      QUESTION["question"], doc["document_id"],
                                      RCFG["top_k"], RCFG["chunk_tokens"],
                                      RCFG["overlap_tokens"], sig)
            assert hits, "a cold rebuild must actually retrieve"
            ids += [h.chunk_id for h in hits]
        per_precision[precision] = ids
        cache.flush()
    assert len({tuple(v) for v in per_precision.values()}) == 1, per_precision
    assert (tmp_path / "fixed_evidence_F16.json").exists()

    # A shared cache is a HIT, not a recomputation, and must agree with the
    # cold rebuilds rather than quietly diverging from them.
    shared_store, shared_emb = make_store(tmp_path, cache_name="embed_cache_shared")
    shared_cache = FixedEvidenceCache(tmp_path / "fixed_evidence_F16.json")
    reused = []
    for doc in QUESTION["documents"]:
        reused += [h.chunk_id for h in get_fixed_evidence(
            shared_store, shared_cache, QUESTION["question_id"],
            QUESTION["question"], doc["document_id"], RCFG["top_k"],
            RCFG["chunk_tokens"], RCFG["overlap_tokens"],
            embed_signature(shared_emb))]
    assert reused == per_precision["F16"]

    # ... and a CHANGED document body must miss the cache. Chunk ids are stable
    # even when the text behind them is not, so without body_sha256 in the key
    # a rewritten corpus.json would be served stale passages while the chunk-id
    # parity gate still reported green.
    edited = documents_json()
    edited["doc_bbb"]["body"] = DOCS["doc_bbb"].replace("1.8 billion", "2.4 billion")
    edited_store, edited_emb = make_store(tmp_path, docs=edited,
                                          cache_name="embed_cache_edited")
    assert document_body_sha256(edited_store, "doc_bbb") != \
        document_body_sha256(shared_store, "doc_bbb")
    args = (QUESTION["question_id"], "doc_bbb", RCFG["top_k"],
            RCFG["chunk_tokens"], RCFG["overlap_tokens"], sig)
    assert fixed_evidence_key(*args, document_body_sha256(edited_store, "doc_bbb")) != \
        fixed_evidence_key(*args, document_body_sha256(shared_store, "doc_bbb"))
    edited_hits = get_fixed_evidence(
        edited_store, shared_cache, QUESTION["question_id"], QUESTION["question"],
        "doc_bbb", RCFG["top_k"], RCFG["chunk_tokens"], RCFG["overlap_tokens"], sig)
    assert any("2.4 billion" in h.text for h in edited_hits), \
        "a changed body must be re-retrieved, not served from the old key"

    # the untouched document keeps its key, so the cache is not invalidated
    # wholesale on any edit
    assert document_body_sha256(edited_store, "doc_ccc") == \
        document_body_sha256(shared_store, "doc_ccc")


# 3 ------------------------------------------------------------------------
def test_3_frozen_coordinator_makes_agent_prompts_identical_across_precisions(tmp_path):
    """The coordinator is produced ONCE at F16 and reused verbatim, so every
    precision builds byte-identical document-agent prompts.

    The previous version looped over three precision labels and never used the
    loop variable: it called two pure functions with identical arguments three
    times and asserted the results were equal. It could not fail, and it did
    not fail while the harness was running a fresh, precision-dependent
    coordinator whose output fed every downstream prompt.
    """
    store, emb = make_store(tmp_path)
    cache = FixedEvidenceCache(tmp_path / "fe.json")
    sig = embed_signature(emb)
    hits = get_fixed_evidence(store, cache, QUESTION["question_id"],
                              QUESTION["question"], "doc_bbb", RCFG["top_k"],
                              RCFG["chunk_tokens"], RCFG["overlap_tokens"], sig)
    qid, question = QUESTION["question_id"], QUESTION["question"]
    frozen_path = tmp_path / "frozen_coordinator.json"

    def agent_prompt_hash(shared_tasks):
        return messages_hash(agent_mod.build_fixed_messages(
            qid, "doc_bbb", question, coord_mod.format_instructions(shared_tasks),
            hits))

    # -- the control: an unfrozen coordinator IS precision-dependent ---------
    unfrozen = {}
    for precision in PRECISIONS:
        client, _ = fixture_client(tmp_path, precision=precision,
                                   backend=PrecisionDependentCoordinator(),
                                   log_name=f"unfrozen_{precision}.jsonl")
        out = coord_mod.run_coordinator(client, qid, question)
        unfrozen[precision] = agent_prompt_hash(out["shared_tasks"])
    assert len(set(unfrozen.values())) == 3, (
        "the fake backend must actually differ per precision, otherwise this "
        "test cannot detect the bug it exists for")

    # -- freeze once, at F16 -------------------------------------------------
    f16_client, _ = fixture_client(tmp_path, precision="F16",
                                   backend=PrecisionDependentCoordinator(),
                                   log_name="frozen_F16.jsonl")
    record = coord_mod.freeze_coordinator(frozen_path, f16_client, qid, question)
    assert record["frozen"] is True
    assert record["provenance"]["produced_by_precision"] == coord_mod.FREEZE_PRECISION
    assert record["shared_tasks"]

    # -- every precision reads the same bytes --------------------------------
    hashes, task_blobs = set(), set()
    for precision in PRECISIONS:
        loaded = coord_mod.load_frozen_coordinator(frozen_path, qid)
        assert loaded["produced_by_precision"] == "F16"
        task_blobs.add(json.dumps(loaded["shared_tasks"], sort_keys=True))
        hashes.add(agent_prompt_hash(loaded["shared_tasks"]))
    assert len(task_blobs) == 1, "shared_tasks must be byte-identical"
    assert len(hashes) == 1, "document-agent prompts must be identical"
    assert hashes == {unfrozen["F16"]}, "the F16 value is the one that wins"

    # -- freezing is idempotent: a later run cannot overwrite the frozen value
    later_client, _ = fixture_client(tmp_path, precision="F16",
                                     backend=PrecisionDependentCoordinator(),
                                     log_name="frozen_again.jsonl")
    again = coord_mod.freeze_coordinator(frozen_path, later_client, qid, question)
    assert again["shared_tasks"] == record["shared_tasks"]

    # -- and no other precision may produce the frozen value -----------------
    for precision in ("Q8_0", "Q4_K_M"):
        rogue, _ = fixture_client(tmp_path, precision=precision,
                                  backend=PrecisionDependentCoordinator(),
                                  log_name=f"rogue_{precision}.jsonl")
        with pytest.raises(ValueError):
            coord_mod.freeze_coordinator(frozen_path, rogue, "q_unfrozen", question)

    # -- a missing frozen record is an error, never a silent fresh run -------
    with pytest.raises(KeyError):
        coord_mod.load_frozen_coordinator(frozen_path, "q_never_frozen")
    assert coord_mod.load_frozen_coordinator(
        frozen_path, "q_never_frozen", required=False) is None

    # The coordinator prompt no longer carries the supporting-document count:
    # that is len(question["documents"]), i.e. gold evidence width and the
    # study's independent variable, which no deployed system would have.
    with pytest.raises(TypeError):
        coord_mod.build_messages(question, 2)


# 4 ------------------------------------------------------------------------
def test_4_model_hashes_and_common_provenance_are_required(tmp_path):
    def named(prov):
        rep = evaluate_gates([], [], prov, required_questions=0,
                             verify_model_files=False)
        return {g.name: g for g in rep.gates}

    good = passing_provenance()
    assert named(good)["model_hashes_recorded"].passed
    assert named(good)["common_source_checkpoint"].passed

    bad = {"models": {"F16": {"sha256": "FIXTURE"}, "Q8_0": {}, "Q4_K_M": {}}}
    assert not named(bad)["model_hashes_recorded"].passed
    assert not named(bad)["common_source_checkpoint"].passed

    same = {"models": {p: {"sha256": "a" * 64, "derived_from_f16_sha256": "a" * 64}
                       for p in PRECISIONS},
            "common_source": {"f16_gguf_sha256": "a" * 64}}
    assert not named(same)["model_hashes_recorded"].passed, \
        "three precisions with one sha256 is one model relabelled three times"

    # -- the gate now checks the derivation, not a note about it ------------
    # A Q4 whose recorded parent is NOT the F16 in use means the three blocks
    # are not the same checkpoint at three precisions, which is the entire
    # controlled comparison. `bool(provenance["common_source"])` -- the old
    # gate -- passes every one of the following.
    truthy_note_only = {
        "models": {p: {"sha256": s} for p, s in
                   zip(PRECISIONS, ("a" * 64, "b" * 64, "c" * 64))},
        "common_source": {"f16_gguf_sha256": "a" * 64, "note": "converted together"}}
    assert not named(truthy_note_only)["common_source_checkpoint"].passed, \
        "no derived_from_f16_sha256 recorded must fail"

    wrong_parent = passing_provenance()
    wrong_parent["models"]["Q4_K_M"]["derived_from_f16_sha256"] = "d" * 64
    gate = named(wrong_parent)["common_source_checkpoint"]
    assert not gate.passed, "a Q4 derived from a different F16 must fail"
    assert "Q4_K_M" in gate.detail

    half_recorded = passing_provenance()
    del half_recorded["models"]["Q8_0"]["derived_from_f16_sha256"]
    assert not named(half_recorded)["common_source_checkpoint"].passed

    no_f16 = passing_provenance()
    no_f16["models"]["F16"]["sha256"] = ""
    del no_f16["common_source"]["f16_gguf_sha256"]
    assert not named(no_f16)["common_source_checkpoint"].passed, \
        "with no F16 sha256 anywhere there is nothing to compare the parents to"
    assert not named(no_f16)["model_hashes_recorded"].passed

    # the two recorded F16 hashes must agree with each other: a common_source
    # note describing one checkpoint while a different F16 is actually loaded
    # is the failure the whole gate exists to catch
    inconsistent_f16 = passing_provenance()
    inconsistent_f16["models"]["F16"]["sha256"] = "e" * 64
    inconsistent_f16["models"]["Q8_0"]["derived_from_f16_sha256"] = "e" * 64
    inconsistent_f16["models"]["Q4_K_M"]["derived_from_f16_sha256"] = "e" * 64
    assert not named(inconsistent_f16)["common_source_checkpoint"].passed


# 5 ------------------------------------------------------------------------
def test_5_fixture_data_cannot_enter_scientific_analysis(tmp_path):
    # the backend refuses to run at all unless explicitly unlocked
    os.environ.pop(FIXTURE_ENV, None)
    with pytest.raises(PermissionError):
        FixtureBackend().start()

    os.environ[FIXTURE_ENV] = "1"
    client, log = fixture_client(tmp_path)
    coord_mod.run_coordinator(client, QUESTION["question_id"], QUESTION["question"])
    events = list(log.read())
    assert events and all(e["is_fixture"] for e in events)

    # Everything else about this run is impeccable: real hashes, real
    # derivation chain, real embedder, matching prompts. One fixture watermark
    # must still be enough to block a recommendation on its own.
    clean_events, clean_preds = passing_events_and_predictions(1)
    rep = evaluate_gates(clean_events + events, clean_preds,
                         passing_provenance(), required_questions=1,
                         condition="fixed_verified_evidence",
                         verify_model_files=False)
    assert not rep.passed
    assert any("real_inference_only" in f for f in rep.failures())
    assert rep.failures() == [f for f in rep.failures()
                              if "real_inference_only" in f], \
        "the fixture watermark must be the ONLY reason this report fails"
    decision = scientific_outcome({"f16_atomic_fact_recall": 0.9}, rep)
    assert decision["scientific_outcome"] == NOT_AVAILABLE

    # a fixture-watermarked PREDICTION is blocked too, not just an event
    fixture_preds = [{"question_id": "q0", "precision": p,
                      "condition": "fixed_verified_evidence", "is_fixture": True,
                      "evidence_chunk_ids": []} for p in PRECISIONS]
    rep2 = evaluate_gates(clean_events, clean_preds + fixture_preds,
                          passing_provenance(), required_questions=1,
                          condition="fixed_verified_evidence",
                          verify_model_files=False)
    assert not rep2.passed
    assert any("real_inference_only" in f for f in rep2.failures())


# 6 ------------------------------------------------------------------------
def test_6_interrupted_runs_resume_without_duplicating_predictions(tmp_path):
    os.environ[FIXTURE_ENV] = "1"
    path = tmp_path / "events.jsonl"

    def run_once():
        backend = build_backend("fixture", None, SamplingConfig())
        backend.start()
        log = EventLog(path)
        client = GuardedClient(backend, log, "stage_a", "F16",
                               "fixed_verified_evidence")
        coord_mod.run_coordinator(client, QUESTION["question_id"],
                                  QUESTION["question"])
        return log, backend

    log_a, backend_a = run_once()
    n_after_first = len(list(log_a.read()))
    assert backend_a.calls > 0, "the first pass must actually call the model"

    # simulate a hard kill: append a torn line, then rerun the same work
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"call_id": "torn", "role": "coord')
    log_b, backend_b = run_once()
    records = list(log_b.read())
    assert len(records) == n_after_first, "resume must not duplicate a prediction"
    assert backend_b.calls == 0, "resumed call must not hit the model again"
    ids = [r["call_id"] for r in records]
    assert len(ids) == len(set(ids))
    # the deterministic id is stable across processes
    assert ids[0] == call_id("stage_a", "F16", "fixed_verified_evidence",
                             QUESTION["question_id"], "coordinator")


# 7 ------------------------------------------------------------------------
def test_7_malformed_json_is_logged_and_repaired_at_most_once(tmp_path):
    class FlakyBackend(FixtureBackend):
        def __init__(self):
            super().__init__()
            self.texts = ['this is not json at all',
                          '{"final_answer":"1.8 billion dollars","used_document_ids":[],'
                          '"used_fact_ids":[],"unresolved_conflicts":[]}']
            self.n = 0

        def chat(self, messages, max_tokens=None):
            res = super().chat(messages, max_tokens=max_tokens)
            res.text = self.texts[min(self.n, len(self.texts) - 1)]
            self.n += 1
            return res

    os.environ[FIXTURE_ENV] = "1"
    backend = FlakyBackend()
    backend.start()
    log = EventLog(tmp_path / "e.jsonl")
    client = GuardedClient(backend, log, "stage_a", "F16", "fixed_verified_evidence")
    out = synth_mod.run_synthesizer(client, QUESTION["question_id"],
                                    QUESTION["question"], "merge", [])
    rec = list(log.read())[0]
    assert rec["repair_attempted"] is True
    assert rec["repair_used"] is True
    assert rec["raw_text"] == 'this is not json at all'
    assert rec["repair_text"].startswith("{")
    assert backend.n == 2, "exactly one repair attempt, never two"
    assert out["final_answer"] == "1.8 billion dollars"

    # a second failure is NOT retried again
    class AlwaysBad(FlakyBackend):
        def __init__(self):
            super().__init__()
            self.texts = ["nope", "still nope"]

    backend2 = AlwaysBad()
    backend2.start()
    log2 = EventLog(tmp_path / "e2.jsonl")
    client2 = GuardedClient(backend2, log2, "stage_a", "F16", "fixed_verified_evidence")
    synth_mod.run_synthesizer(client2, QUESTION["question_id"], QUESTION["question"],
                              "merge", [])
    assert backend2.n == 2
    rec2 = list(log2.read())[0]
    assert rec2["parse_ok"] is False and rec2["repair_attempted"] is True


# 8 ------------------------------------------------------------------------
def test_8_gpu_offload_is_detected_and_recorded():
    from hardware import cuda_report, detect_gpu_offload
    log = ("ggml_cuda_init: found 1 CUDA devices\n"
           "llm_load_tensors: offloaded 37/37 layers to GPU\n")
    info = detect_gpu_offload(log)
    assert info["offload_detected"] is True
    assert info["layers_offloaded"] == 37 and info["layers_total"] == 37
    assert "37/37" in info["evidence"]

    cpu_only = detect_gpu_offload("llm_load_tensors: offloaded 0/37 layers to GPU\n")
    assert cpu_only["offload_detected"] is False

    rep = cuda_report()
    assert "nvidia_smi" in rep and "gpus" in rep     # never raises off-GPU


# 9 ------------------------------------------------------------------------
def _gates(events, preds, prov=None, required_questions=1):
    return evaluate_gates(events, preds, prov or passing_provenance(),
                          required_questions=required_questions,
                          condition="fixed_verified_evidence",
                          verify_model_files=False)


def test_9_all_three_precisions_must_produce_a_real_generation():
    events, preds = passing_events_and_predictions(1)
    ok = _gates(events, preds)
    assert ok.passed, ok.failures()

    # drop Q4's tokens -> the nonzero-generation gate must fail
    zeroed = [dict(e, generated_tokens=0) if e["precision"] == "Q4_K_M" else e
              for e in events]
    bad = _gates(zeroed, preds)
    assert not bad.passed
    assert any("nonzero_generations_per_precision" in f for f in bad.failures())

    # drop Q4 entirely -> the all-precisions gate must fail
    missing = _gates([e for e in events if e["precision"] != "Q4_K_M"],
                     [p for p in preds if p["precision"] != "Q4_K_M"])
    assert any("all_three_precisions_present" in f for f in missing.failures())

    # -- a parity gate may not pass because data is MISSING ------------------
    # `len(set(values)) > 1` over whatever happened to be present meant a
    # question that ran at only one precision had exactly one value and
    # "passed": parity was guaranteed precisely when the evidence for it was
    # absent. The all-precisions gate does not cover this, because a slot can
    # be short a precision while every precision is present overall.
    two_q_events, two_q_preds = passing_events_and_predictions(2)
    partial_preds = [p for p in two_q_preds
                     if not (p["question_id"] == "q1" and p["precision"] == "Q4_K_M")]
    partial = _gates(two_q_events, partial_preds)
    assert any("fixed_evidence_chunk_parity" in f for f in partial.failures()), \
        "a slot missing a precision must FAIL parity, not pass vacuously"

    partial_events = [e for e in two_q_events
                      if not (e["question_id"] == "q1"
                              and e["precision"] == "Q4_K_M"
                              and e["role"] == "document_agent")]
    assert any("prompt_hash_parity" in f
               for f in _gates(partial_events, two_q_preds).failures())

    # an EMPTY value in a slot is missing data wearing three matching hats
    blanked = [dict(e, prompt_hash="") if e["role"] == "document_agent" else e
               for e in events]
    assert any("prompt_hash_parity" in f for f in _gates(blanked, preds).failures())

    blank_preds = [dict(p, evidence_chunk_ids=[]) for p in preds]
    assert any("fixed_evidence_chunk_parity" in f
               for f in _gates(events, blank_preds).failures())

    # no records at all is a failure too, not "nothing to disagree about"
    assert any("fixed_evidence_chunk_parity" in f for f in _gates([], []).failures())

    # -- and genuine disagreement still fails --------------------------------
    skewed = [dict(p, evidence_chunk_ids=["doc_bbb::c0009"])
              if p["precision"] == "Q4_K_M" else p for p in preds]
    assert any("fixed_evidence_chunk_parity" in f
               for f in _gates(events, skewed).failures())

    drifting = [dict(e, prompt_hash="other") if (e["role"] == "coordinator"
                                                 and e["precision"] == "Q8_0") else e
                for e in events]
    assert any("coordinator_prompt_parity" in f
               for f in _gates(drifting, preds).failures())

    # -- a backend error is infrastructure failure, never model quality ------
    errored = events + [dict(events[0], generation_error="HTTP 502 from llama-server")]
    assert any("no_generation_errors" in f for f in _gates(errored, preds).failures())

    # -- the hash-fallback embedder cannot support a scientific claim --------
    fake_embed = passing_provenance()
    fake_embed["embedding"]["embed_is_real_model"] = False
    assert any("embedding_is_real_model" in f
               for f in _gates(events, preds, fake_embed).failures())

    # -- the three blocks must have differed ONLY in precision ---------------
    mixed_sampling = [dict(e, sampling_hash="different")
                      if e["precision"] == "Q4_K_M" else e for e in events]
    assert any("sampling_identical_across_precisions" in f
               for f in _gates(mixed_sampling, preds).failures())

    partial_offload = passing_provenance()
    partial_offload["blocks"]["Q4_K_M"]["gpu_offload"]["layers_offloaded"] = 12
    assert any("gpu_offload_identical_across_precisions" in f
               for f in _gates(events, preds, partial_offload).failures())


# 10 -----------------------------------------------------------------------
def test_10_stage_a_recommendation_unavailable_until_all_18_predictions():
    for n in range(1, 6):                       # 3..15 predictions
        ev, pr = passing_events_and_predictions(n)
        rep = _gates(ev, pr, required_questions=6)
        assert not rep.passed, f"{len(pr)} predictions must not unlock a recommendation"
        assert any("stage_question_count_complete" in f for f in rep.failures())
        assert scientific_outcome({"f16_atomic_fact_recall": 0.8}, rep)[
            "scientific_outcome"] == NOT_AVAILABLE

    ev, pr = passing_events_and_predictions(6)   # the full 18
    assert len(pr) == 18
    rep = _gates(ev, pr, required_questions=6)
    assert rep.passed, rep.failures()
    out = scientific_outcome({"f16_atomic_fact_recall": 0.8,
                              "q8_atomic_fact_recall": 0.78,
                              "q4_atomic_fact_recall": 0.60,
                              "f16_q4_gap_overall": 0.20,
                              "f16_q8_gap_overall": 0.02,
                              "f16_q4_gap_by_width": {"2": 0.05, "4": 0.30},
                              "f16_q4_gap_by_width_n": {"2": 3, "4": 3},
                              "f16_q4_gap_ci": {"lo": 0.05, "hi": 0.35,
                                                "n_informative": 12}}, rep)
    assert out["scientific_outcome"] != NOT_AVAILABLE

    # 18 predictions that are 17 real ones plus a repeat do NOT count: the gate
    # counts DISTINCT question ids per precision.
    short_ev, short_pr = passing_events_and_predictions(5)
    padded = short_pr + [dict(short_pr[0], evidence_chunk_ids=["doc_bbb::c0000"])
                         for _ in range(3)]
    assert len(padded) == 18
    assert not _gates(short_ev, padded, required_questions=6).passed

    # 18 predictions spread over the right questions but at only two
    # precisions cannot unlock it either, and the parity gates must say so
    # rather than reporting agreement over the precisions that showed up.
    nine_ev, nine_pr = passing_events_and_predictions(9)
    two_prec = [p for p in nine_pr if p["precision"] != "Q4_K_M"]
    assert len(two_prec) == 18
    rep_two = _gates(nine_ev, two_prec, required_questions=6)
    assert not rep_two.passed
    assert any("fixed_evidence_chunk_parity" in f for f in rep_two.failures())
