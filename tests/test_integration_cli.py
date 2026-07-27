"""End-to-end CLI integration: run_stage.py -> score.py -> analyze.py.

Every other test in this suite imports functions and inspects return values.
That is exactly how four blocking bugs shipped: each unit was correct in
isolation and the pipeline that wires them together was not. Specifically, no
test ever ran the three scripts in sequence and read the files they left
behind, so nothing noticed that

  * a harness run wrote into the same directory a real run resumes from, so a
    dry run followed by a real run reported "18/18 predictions" without
    calling the model once;
  * Stage-C selection read keys out of `predictions.json` that only ever exist
    in `scored.json`;
  * `run_coordinator` was invoked per precision, so no set of prompt hashes
    could ever satisfy the parity gate;
  * a fixture-watermarked run could still reach the outcome map.

This test therefore asserts on ARTIFACTS ON DISK, never on hand-written dicts.
It runs on the fixture backend (`SPDQ_ALLOW_FIXTURE=1`) over a 2-question
manifest carved out of the real data, in a temporary results directory, and it
requires no GGUF, no GPU and no network.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gates import NOT_AVAILABLE  # noqa: E402

PY = sys.executable
N_QUESTIONS = 2
PRECISIONS = ("F16", "Q8_0", "Q4_K_M")


def _run(script: str, *args, cwd=ROOT):
    """Invoke one of the pilot scripts in a subprocess, fixture backend unlocked."""
    env = dict(os.environ)
    env["SPDQ_ALLOW_FIXTURE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT / "scripts"), env.get("PYTHONPATH", "")])
    proc = subprocess.run(
        [PY, str(ROOT / "scripts" / script), *[str(a) for a in args]],
        cwd=str(cwd), env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, (
        f"{script} exited {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
    return proc


@pytest.fixture(scope="module")
def mini_inputs(tmp_path_factory):
    """A 2-question manifest carved out of the real MultiHop-RAG pilot data.

    Real data, real chunking, real rubric -- only the question count and the
    backend are reduced. A synthetic corpus here would let a schema drift in
    prepare_dataset.py pass unnoticed.
    """
    yaml = pytest.importorskip("yaml")
    src = ROOT / "data"
    for name in ("pilot_manifest.json", "pilot_documents.json", "gold_facts.json"):
        if not (src / name).exists():
            pytest.skip(f"data/{name} missing; run scripts/prepare_dataset.py")

    out = tmp_path_factory.mktemp("mini_inputs")
    manifest = json.loads((src / "pilot_manifest.json").read_text())
    questions = manifest["stages"]["stage_a"]["questions"][:N_QUESTIONS]
    assert len(questions) == N_QUESTIONS

    (out / "manifest.json").write_text(json.dumps({
        "stages": {"stage_a": {"questions": questions},
                   "stage_b": {"questions": []},
                   "stage_c": {"questions": []}}}))

    documents = json.loads((src / "pilot_documents.json").read_text())
    needed = {d["document_id"] for q in questions for d in q["documents"]}
    (out / "documents.json").write_text(
        json.dumps({d: documents[d] for d in sorted(needed)}))

    gold = json.loads((src / "gold_facts.json").read_text())
    (out / "gold.json").write_text(json.dumps({
        "fact_match_threshold": gold["fact_match_threshold"],
        "questions": {q["question_id"]: gold["questions"][q["question_id"]]
                      for q in questions}}))

    models_cfg = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())
    models_cfg["backend"]["kind"] = "fixture"
    # the real embedder would need a network download; the run is watermarked
    # non-scientific either way and `embedding_is_real_model` records it
    models_cfg["embedding"]["hash_fallback"] = True
    (out / "models.yaml").write_text(yaml.safe_dump(models_cfg))

    stage_cfg = yaml.safe_load((ROOT / "configs" / "stage_a.yaml").read_text())
    (out / "stage_a.yaml").write_text(yaml.safe_dump(stage_cfg))

    return {"dir": out,
            "question_ids": [q["question_id"] for q in questions],
            "document_ids": sorted(needed)}


@pytest.fixture(scope="module")
def harness_run(mini_inputs, tmp_path_factory):
    """run_stage -> score -> analyze, once, against a temporary results dir."""
    d = mini_inputs["dir"]
    results = tmp_path_factory.mktemp("results_root") / "results"

    run = _run("run_stage.py", "--stage", "stage_a", "--backend", "fixture",
               "--results-dir", results,
               "--manifest", d / "manifest.json",
               "--documents", d / "documents.json",
               "--models-config", d / "models.yaml",
               "--stage-config", d / "stage_a.yaml")

    # Harness output is quarantined, so the downstream scripts have to be
    # pointed at the quarantine directory explicitly.
    harness = results / "_harness"
    score = _run("score.py", "--stage", "stage_a", "--results-dir", harness,
                 "--gold", d / "gold.json")
    analyze = _run("analyze.py", "--stage", "stage_a", "--results-dir", harness,
                   "--resamples", "200")
    return {"results": results, "harness": harness, "stage_dir": harness / "stage_a",
            "run": run, "score": score, "analyze": analyze}


def _load(path: Path):
    assert path.exists(), f"expected artifact missing: {path}"
    return json.loads(path.read_text())


# --------------------------------------------------------------------------


def test_pipeline_writes_predictions_scored_and_analysis_to_disk(harness_run,
                                                                 mini_inputs):
    sd = harness_run["stage_dir"]
    predictions = _load(sd / "predictions.json")
    scored = _load(sd / "scored.json")
    analysis = _load(sd / "analysis.json")

    assert len(predictions) == N_QUESTIONS * len(PRECISIONS)
    assert {p["precision"] for p in predictions} == set(PRECISIONS)
    assert {p["question_id"] for p in predictions} == set(mini_inputs["question_ids"])
    assert all(p["condition"] == "fixed_verified_evidence" for p in predictions)

    # scoring consumed every prediction; nothing was dropped for want of a rubric
    assert len(scored) == len(predictions)
    summary = _load(sd / "score_summary.json")
    assert summary["missing_gold_for"] == []
    assert summary["n_scored"] == len(predictions)

    assert analysis["stage"] == "stage_a"
    assert _load(sd / "run_report.json")["completed_predictions"] == len(predictions)
    assert (sd / "events.jsonl").exists()
    assert (sd / "isolation_report.json").exists()
    assert _load(sd / "isolation_report.json")["isolated"] is True


def test_fixture_run_cannot_produce_a_scientific_recommendation(harness_run):
    """The load-bearing assertion of the whole suite, made against the file the
    reader would actually open."""
    analysis = _load(harness_run["stage_dir"] / "analysis.json")
    assert analysis["scientific_outcome"] == NOT_AVAILABLE
    assert analysis["gates"]["all_gates_passed"] is False
    assert any("real_inference_only" in f for f in analysis["gates"]["failures"])
    assert analysis["n_real_predictions"] == 0
    assert analysis["n_excluded_fixture_or_dry_run"] == N_QUESTIONS * len(PRECISIONS)

    # ... and the stage gate that unlocks Stage B stays shut
    gate = _load(harness_run["stage_dir"] / "gate.json")
    assert gate["proceed"] is False
    assert gate["scientific_outcome"] == NOT_AVAILABLE

    # the harness itself, however, is reported as working -- the two statuses
    # are separate axes and conflating them made a clean run read as broken
    assert analysis["engineering_status"] == "HARNESS_READY"


def test_harness_output_is_quarantined_under_harness(harness_run):
    """A harness run must not write anything a real run would resume from."""
    results, harness = harness_run["results"], harness_run["harness"]
    assert harness.is_dir()
    assert (harness / "stage_a" / "predictions.json").exists()

    # nothing at the real-run location
    assert not (results / "stage_a").exists(), \
        "a fixture run must never write into the real stage directory"
    for artifact in ("predictions.json", "events.jsonl", "scored.json",
                     "analysis.json", "gate.json"):
        assert not (results / "stage_a" / artifact).exists()

    # the frozen coordinator is quarantined with the rest of the run
    assert (harness / "frozen_coordinator.json").exists()
    assert not (results / "frozen_coordinator.json").exists()

    # and the run says so out loud
    assert "HARNESS-ONLY" in harness_run["run"].stdout


def test_coordinator_is_frozen_once_and_reused_by_every_precision(harness_run,
                                                                  mini_inputs):
    """Read off disk: one frozen record per question, produced at F16, and every
    precision's document-agent prompt hash identical for the same slot."""
    frozen = _load(harness_run["harness"] / "frozen_coordinator.json")
    assert set(frozen) == set(mini_inputs["question_ids"])
    for record in frozen.values():
        assert record["frozen"] is True
        assert record["provenance"]["produced_by_precision"] == "F16"

    events = [json.loads(line) for line in
              (harness_run["stage_dir"] / "events.jsonl").read_text().splitlines()
              if line.strip()]
    assert events

    # exactly one coordinator generation per question, not one per precision
    coordinator_calls = [e for e in events if e.get("role") == "coordinator"]
    assert len(coordinator_calls) == N_QUESTIONS
    assert {e["precision"] for e in coordinator_calls} == {"F16"}

    # document-agent prompts agree across precisions for every (question, doc)
    slots = {}
    for e in events:
        if e.get("role") != "document_agent":
            continue
        slots.setdefault((e["question_id"], e["document_id"]), {})[
            e["precision"]] = e["prompt_hash"]
    assert slots
    for slot, by_precision in slots.items():
        assert set(by_precision) == set(PRECISIONS), f"{slot} missing a precision"
        assert len(set(by_precision.values())) == 1, f"prompt hashes differ for {slot}"


def test_fixed_evidence_is_identical_across_precisions_on_disk(harness_run):
    predictions = _load(harness_run["stage_dir"] / "predictions.json")
    slots = {}
    for p in predictions:
        slots.setdefault(p["question_id"], {})[p["precision"]] = \
            tuple(sorted(p["evidence_chunk_ids"]))
    for qid, by_precision in slots.items():
        assert set(by_precision) == set(PRECISIONS)
        assert len(set(by_precision.values())) == 1, f"chunk ids differ for {qid}"
        assert all(by_precision[PRECISIONS[0]]), f"{qid} retrieved no evidence"


def test_document_isolation_holds_through_the_real_pipeline(harness_run):
    """Every chunk an agent cited belongs to the document it was assigned."""
    predictions = _load(harness_run["stage_dir"] / "predictions.json")
    for pred in predictions:
        for agent in pred["agent_outputs"]:
            did = agent["document_id"]
            for chunk_id in agent["retrieved_chunk_ids"]:
                assert chunk_id.startswith(did), (
                    f"agent for {did} retrieved {chunk_id}")
            for fact in agent["supported_facts"]:
                for chunk_id in fact["chunk_ids"]:
                    assert chunk_id.startswith(did)


def test_a_dry_run_does_not_seed_the_resume_index_of_a_real_run(mini_inputs,
                                                                tmp_path):
    """The bug: `--dry-run` shared `events.jsonl` and `predictions.json` with
    real runs, and `call_id` has no run-mode component. A dry run followed by a
    real run in the same results directory therefore resume-skipped every
    question, printed "18/18 predictions" and never called the model.

    A real run needs a GGUF, so what is asserted here is the property that
    makes the bug impossible: after a dry run, the directory a real run reads
    its resume index from does not exist, so the index it loads is empty.
    """
    d = mini_inputs["dir"]
    results = tmp_path / "results"

    dry = _run("run_stage.py", "--stage", "stage_a", "--dry-run",
               "--results-dir", results,
               "--manifest", d / "manifest.json",
               "--documents", d / "documents.json",
               "--models-config", d / "models.yaml",
               "--stage-config", d / "stage_a.yaml")
    assert "HARNESS-ONLY" in dry.stdout

    harness_stage = results / "_harness" / "stage_a"
    real_stage = results / "stage_a"
    assert (harness_stage / "predictions.json").exists(), \
        "the dry run must still record what it did, just not where a real run looks"
    assert not real_stage.exists(), \
        "a dry run must not create the real stage directory at all"

    # the resume index a real run would build from that directory is empty
    real_predictions = json.loads((real_stage / "predictions.json").read_text()) \
        if (real_stage / "predictions.json").exists() else []
    resume_keys = {(p["question_id"], p["precision"], p["condition"])
                   for p in real_predictions}
    assert resume_keys == set(), "a real run must have nothing to resume-skip"

    # dry-run rows are watermarked as well as quarantined -- belt and braces,
    # because a user can always copy files into the wrong place
    dry_predictions = json.loads((harness_stage / "predictions.json").read_text())
    assert dry_predictions
    assert all(p["dry_run"] for p in dry_predictions)


def test_stage_c_selection_refuses_to_run_without_scored_stage_b(mini_inputs,
                                                                 tmp_path):
    """Stage C reads `stage_b/scored.json`. It used to read `predictions.json`
    and pull keys that file never contains, so `disagreement()` silently
    returned a constant and the max-disagreement rule collapsed into a
    question_id sort -- producing exactly what having no Stage B data at all
    would produce, with no error and no warning.
    """
    from run_stage import select_stage_c_questions

    manifest = {"stages": {"stage_b": {"questions": [
        {"question_id": f"q{i:02d}", "question": "x", "width": w,
         "documents": [], "answer": ""}
        for i, w in enumerate([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])]}}}

    with pytest.raises(SystemExit):
        select_stage_c_questions([], manifest)

    # predictions.json-shaped rows carry no scores at all: that must be a hard
    # error, not a quiet fallback to alphabetical order
    prediction_shaped = [
        {"question_id": q["question_id"], "precision": p,
         "condition": "fixed_verified_evidence", "final_answer": "...",
         "evidence_chunk_ids": []}
        for q in manifest["stages"]["stage_b"]["questions"] for p in ("F16", "Q4_K_M")]
    with pytest.raises(SystemExit):
        select_stage_c_questions(prediction_shaped, manifest)

    # fixture rows are not scorable data either
    fixture_rows = [dict(r, is_fixture=True, answer_score={"score": 1.0})
                    for r in prediction_shaped]
    with pytest.raises(SystemExit):
        select_stage_c_questions(fixture_rows, manifest)
