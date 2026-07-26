"""The ten required acceptance tests, one per brief item.

None of these touch the network or a GGUF. They test the HARNESS. Passing them
proves engineering readiness only -- test 5 and test 10 exist precisely to make
sure that distinction cannot be blurred.
"""
from __future__ import annotations

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
from evidence import (FixedEvidenceCache, build_store, embed_signature,
                      get_fixed_evidence, make_embedder)
from fixtures import DOCS, QUESTION, documents_json
from gates import NOT_AVAILABLE, evaluate_gates, scientific_outcome
from llm_client import GuardedClient, messages_hash
from logging_utils import EventLog, call_id
from retrieval import DocumentIsolationError

MODELS_CFG = {"embedding": {"model": "test", "revision": "test", "hash_fallback": True}}
RCFG = dict(top_k=2, chunk_tokens=120, overlap_tokens=20)


def make_store(tmp_path):
    emb = make_embedder(MODELS_CFG, str(tmp_path / "embed_cache"))
    return build_store(documents_json(), emb, sorted(DOCS), RCFG["chunk_tokens"],
                       RCFG["overlap_tokens"]), emb


def fixture_client(tmp_path, precision="F16", stage="stage_a",
                   condition="fixed_verified_evidence", dry_run=False):
    os.environ[FIXTURE_ENV] = "1"
    backend = build_backend("fixture", None, SamplingConfig(), )
    backend.precision = precision
    backend.start()
    log = EventLog(tmp_path / f"events_{precision}.jsonl")
    return GuardedClient(backend, log, stage, precision, condition, dry_run=dry_run), log


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
    store, emb = make_store(tmp_path)
    cache = FixedEvidenceCache(tmp_path / "fixed_evidence.json")
    sig = embed_signature(emb)
    per_precision = {}
    for precision in ("F16", "Q8_0", "Q4_K_M"):
        ids = []
        for doc in QUESTION["documents"]:
            hits = get_fixed_evidence(store, cache, QUESTION["question_id"],
                                      QUESTION["question"], doc["document_id"],
                                      RCFG["top_k"], RCFG["chunk_tokens"],
                                      RCFG["overlap_tokens"], sig)
            ids += [h.chunk_id for h in hits]
        per_precision[precision] = ids
    assert len({tuple(v) for v in per_precision.values()}) == 1, per_precision
    cache.flush()
    assert (tmp_path / "fixed_evidence.json").exists()


# 3 ------------------------------------------------------------------------
def test_3_prompt_hashes_match_across_precisions(tmp_path):
    store, emb = make_store(tmp_path)
    cache = FixedEvidenceCache(tmp_path / "fe.json")
    sig = embed_signature(emb)
    hits = get_fixed_evidence(store, cache, QUESTION["question_id"],
                              QUESTION["question"], "doc_bbb", RCFG["top_k"],
                              RCFG["chunk_tokens"], RCFG["overlap_tokens"], sig)
    tasks = coord_mod.fallback_tasks(QUESTION["question"])["shared_tasks"]
    hashes = set()
    for precision in ("F16", "Q8_0", "Q4_K_M"):
        msgs = agent_mod.build_fixed_messages(
            QUESTION["question_id"], "doc_bbb", QUESTION["question"],
            coord_mod.format_instructions(tasks), hits)
        hashes.add(messages_hash(msgs))
        coord_msgs = coord_mod.build_messages(QUESTION["question"], 2)
        hashes.add("coord:" + messages_hash(coord_msgs))
    assert len(hashes) == 2, "prompts must be byte-identical across precisions"


# 4 ------------------------------------------------------------------------
def test_4_model_hashes_and_common_provenance_are_required(tmp_path):
    good = {"models": {"F16": {"sha256": "a" * 64}, "Q8_0": {"sha256": "b" * 64},
                       "Q4_K_M": {"sha256": "c" * 64}},
            "common_source": {"f16_gguf_sha256": "a" * 64}}
    rep = evaluate_gates([], [], good, required_questions=0)
    named = {g.name: g for g in rep.gates}
    assert named["model_hashes_recorded"].passed
    assert named["common_source_checkpoint"].passed

    bad = {"models": {"F16": {"sha256": "FIXTURE"}, "Q8_0": {}, "Q4_K_M": {}}}
    rep2 = evaluate_gates([], [], bad, required_questions=0)
    named2 = {g.name: g for g in rep2.gates}
    assert not named2["model_hashes_recorded"].passed
    assert not named2["common_source_checkpoint"].passed

    same = {"models": {p: {"sha256": "a" * 64} for p in ("F16", "Q8_0", "Q4_K_M")},
            "common_source": {"f16_gguf_sha256": "a" * 64}}
    rep3 = evaluate_gates([], [], same, required_questions=0)
    assert not {g.name: g for g in rep3.gates}["model_hashes_recorded"].passed


# 5 ------------------------------------------------------------------------
def test_5_fixture_data_cannot_enter_scientific_analysis(tmp_path):
    # the backend refuses to run at all unless explicitly unlocked
    os.environ.pop(FIXTURE_ENV, None)
    with pytest.raises(PermissionError):
        FixtureBackend().start()

    os.environ[FIXTURE_ENV] = "1"
    client, log = fixture_client(tmp_path)
    coord_mod.run_coordinator(client, QUESTION["question_id"],
                              QUESTION["question"], 2)
    events = list(log.read())
    assert events and all(e["is_fixture"] for e in events)

    preds = [{"question_id": QUESTION["question_id"], "precision": p,
              "condition": "fixed_verified_evidence", "is_fixture": True,
              "evidence_chunk_ids": []} for p in ("F16", "Q8_0", "Q4_K_M")]
    prov = {"models": {p: {"sha256": "a" * 64} for p in ("F16", "Q8_0", "Q4_K_M")},
            "common_source": {"f16_gguf_sha256": "a" * 64}}
    rep = evaluate_gates(events, preds, prov, required_questions=1,
                         condition="fixed_verified_evidence")
    assert not rep.passed
    assert any("real_inference_only" in f for f in rep.failures())
    decision = scientific_outcome({"f16_atomic_fact_recall": 0.9}, rep)
    assert decision["scientific_outcome"] == NOT_AVAILABLE


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
                                  QUESTION["question"], 2)
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
def test_9_all_three_precisions_must_produce_a_real_generation():
    prov = {"models": {p: {"sha256": c * 64} for p, c in
                       (("F16", "a"), ("Q8_0", "b"), ("Q4_K_M", "c"))},
            "common_source": {"f16_gguf_sha256": "a" * 64}}
    events, preds = [], []
    for p in ("F16", "Q8_0", "Q4_K_M"):
        events.append({"precision": p, "backend": "llama-server", "is_fixture": False,
                       "generated_tokens": 20, "condition": "fixed_verified_evidence",
                       "role": "coordinator", "question_id": "q1", "prompt_hash": "h",
                       "document_id": "-"})
        preds.append({"question_id": "q1", "precision": p,
                      "condition": "fixed_verified_evidence", "is_fixture": False,
                      "evidence_chunk_ids": ["doc_bbb::c0000"]})
    ok = evaluate_gates(events, preds, prov, required_questions=1,
                        condition="fixed_verified_evidence")
    assert ok.passed, ok.failures()

    # drop Q4's tokens -> the nonzero-generation gate must fail
    events[-1]["generated_tokens"] = 0
    bad = evaluate_gates(events, preds, prov, required_questions=1,
                         condition="fixed_verified_evidence")
    assert not bad.passed
    assert any("nonzero_generations_per_precision" in f for f in bad.failures())

    # drop Q4 entirely -> the all-precisions gate must fail
    missing = evaluate_gates(events[:2], preds[:2], prov, required_questions=1,
                             condition="fixed_verified_evidence")
    assert any("all_three_precisions_present" in f for f in missing.failures())


# 10 -----------------------------------------------------------------------
def test_10_stage_a_recommendation_unavailable_until_all_18_predictions():
    prov = {"models": {p: {"sha256": c * 64} for p, c in
                       (("F16", "a"), ("Q8_0", "b"), ("Q4_K_M", "c"))},
            "common_source": {"f16_gguf_sha256": "a" * 64}}

    def build(n_questions):
        events, preds = [], []
        for i in range(n_questions):
            for p in ("F16", "Q8_0", "Q4_K_M"):
                qid = f"q{i}"
                events.append({"precision": p, "backend": "llama-server",
                               "is_fixture": False, "generated_tokens": 20,
                               "condition": "fixed_verified_evidence",
                               "role": "coordinator", "question_id": qid,
                               "prompt_hash": "h", "document_id": "-"})
                preds.append({"question_id": qid, "precision": p,
                              "condition": "fixed_verified_evidence",
                              "is_fixture": False,
                              "evidence_chunk_ids": ["doc_bbb::c0000"]})
        return events, preds

    for n in range(1, 6):                       # 3..15 predictions
        ev, pr = build(n)
        rep = evaluate_gates(ev, pr, prov, required_questions=6,
                             condition="fixed_verified_evidence")
        assert not rep.passed, f"{len(pr)} predictions must not unlock a recommendation"
        assert scientific_outcome({"f16_atomic_fact_recall": 0.8}, rep)[
            "scientific_outcome"] == NOT_AVAILABLE

    ev, pr = build(6)                            # the full 18
    assert len(pr) == 18
    rep = evaluate_gates(ev, pr, prov, required_questions=6,
                         condition="fixed_verified_evidence")
    assert rep.passed, rep.failures()
    out = scientific_outcome({"f16_atomic_fact_recall": 0.8,
                              "q8_atomic_fact_recall": 0.78,
                              "q4_atomic_fact_recall": 0.60,
                              "f16_q4_gap_overall": 0.20,
                              "f16_q8_gap_overall": 0.02,
                              "f16_q4_gap_by_width": {"2": 0.05, "4": 0.30},
                              "f16_q4_gap_ci": {"lo": 0.05, "hi": 0.35}}, rep)
    assert out["scientific_outcome"] != NOT_AVAILABLE
