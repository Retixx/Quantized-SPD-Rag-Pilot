"""Whole-pipeline smoke test on the fixture backend.

Proves the three layers wire together, the end-to-end loop respects its two
search rounds and its document boundary, and Stage-C selection is deterministic.
Engineering only -- every record here is watermarked is_fixture.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import coordinator as coord_mod
import document_agent as agent_mod
import synthesizer as synth_mod
from backends.llamacpp_backend import FIXTURE_ENV, FixtureBackend, SamplingConfig
from evidence import (FixedEvidenceCache, build_store, embed_signature,
                      get_fixed_evidence, make_embedder)
from fixtures import DOCS, QUESTION, documents_json
from llm_client import GuardedClient
from logging_utils import EventLog

CFG = {"embedding": {"model": "t", "revision": "t", "hash_fallback": True}}


def _client(tmp_path, condition="fixed_verified_evidence"):
    os.environ[FIXTURE_ENV] = "1"
    b = FixtureBackend(SamplingConfig(), precision="F16")
    b.start()
    return GuardedClient(b, EventLog(tmp_path / "e.jsonl"), "stage_a", "F16",
                         condition), b


def test_three_layers_produce_a_prediction(tmp_path):
    emb = make_embedder(CFG, str(tmp_path / "ec"))
    store = build_store(documents_json(), emb, sorted(DOCS), 120, 20)
    cache = FixedEvidenceCache(tmp_path / "fe.json")
    sig = embed_signature(emb)
    client, backend = _client(tmp_path)

    coord = coord_mod.run_coordinator(client, QUESTION["question_id"],
                                      QUESTION["question"], 2)
    assert coord["shared_tasks"]
    agents = []
    for doc in QUESTION["documents"]:
        hits = get_fixed_evidence(store, cache, QUESTION["question_id"],
                                  QUESTION["question"], doc["document_id"],
                                  2, 120, 20, sig)
        assert hits and all(h.document_id == doc["document_id"] for h in hits)
        out = agent_mod.run_fixed_evidence_agent(
            client, QUESTION["question_id"], doc["document_id"],
            QUESTION["question"], coord["shared_tasks"], hits)
        assert out["document_id"] == doc["document_id"]
        assert set(out["retrieved_chunk_ids"]) == {h.chunk_id for h in hits}
        agents.append(out)
    synth = synth_mod.run_synthesizer(client, QUESTION["question_id"],
                                      QUESTION["question"],
                                      coord["synthesis_directive"], agents)
    assert synth["is_fixture"] is True
    assert synth["facts_available"] == sum(len(a["supported_facts"]) for a in agents)


def test_end_to_end_loop_respects_round_budget_and_isolation(tmp_path):
    emb = make_embedder(CFG, str(tmp_path / "ec"))
    store = build_store(documents_json(), emb, sorted(DOCS), 120, 20)
    client, backend = _client(tmp_path, condition="end_to_end")
    out = agent_mod.run_end_to_end_agent(
        client, store, QUESTION["question_id"], "doc_bbb", QUESTION["question"],
        coord_mod.fallback_tasks(QUESTION["question"])["shared_tasks"],
        top_k=2, max_search_rounds=2)
    assert out["search_rounds"] <= 2
    assert all(c.startswith("doc_bbb") for c in out["retrieved_chunk_ids"])


def test_stage_c_selection_is_deterministic(tmp_path):
    from run_stage import select_stage_c_questions
    manifest = {"stages": {"stage_b": {"questions": [
        {"question_id": f"q{i:02d}", "question": "x", "width": w,
         "documents": [], "answer": ""}
        for i, w in enumerate([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])]}}}
    preds = []
    for i, q in enumerate(manifest["stages"]["stage_b"]["questions"]):
        for prec, s in (("F16", 1.0), ("Q4_K_M", 1.0 - (i % 5) * 0.2)):
            preds.append({"question_id": q["question_id"], "precision": prec,
                          "condition": "fixed_verified_evidence",
                          "answer_score": {"score": s},
                          "atomic_fact_recall_final": s})
    a = [q["question_id"] for q in select_stage_c_questions(preds, manifest)]
    b = [q["question_id"] for q in select_stage_c_questions(preds, manifest)]
    assert a == b and len(a) == 6 and len(set(a)) == 6
    widths = {q["question_id"]: q["width"]
              for q in manifest["stages"]["stage_b"]["questions"]}
    assert sum(1 for q in a[:2] if widths[q] == 2) == 2
    assert sum(1 for q in a[2:4] if widths[q] == 4) == 2
