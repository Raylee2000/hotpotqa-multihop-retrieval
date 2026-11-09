"""Spec-v2 decisions (D-01 … D-14) as executable checks."""
import dataclasses
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rag_hotpotqa.data.access import SplitIntegrityError, SplitPolicyError, load_split_ids
from rag_hotpotqa.data.documents import (
    SF_OUT_OF_RANGE, build_question, supporting_sentences_input,
)
from rag_hotpotqa.data.splits import ids_hash
from rag_hotpotqa.evaluation.metrics import path_metrics
from rag_hotpotqa.experiments.multihop import build_paths, oracle_block, run_question
from rag_hotpotqa.experiments.runner import (
    ConfigMismatchError, FailureRateError, check_or_write_run_config, run_retriever,
)
from rag_hotpotqa.retrieval.base import Scorer, compose
from rag_hotpotqa.retrieval.bm25 import BM25Scorer, BM25Stats
from rag_hotpotqa.retrieval.dense import DenseScorer
from rag_hotpotqa.retrieval.factory import ModelVersionError, check_hf_cache_inside_project
from rag_hotpotqa.retrieval.similarity import EncodingTrace, Similarity
from rag_hotpotqa.utils.config import load_settings

from conftest import FakeModel, make_record

PROJECT = Path(__file__).resolve().parents[1]
CFG = {"ks": [1, 2, 3, 5], "top_m": 5, "widths": {"P1": 1, "P3": 3, "P5": 5, "PAll": None},
       "e2_width": "P3", "r3_width": 3}
ALL_EXPS = ["O", "E1", "E1P", "R1", "R2", "R3"]


def ten_doc_record(qid="q10", sf=None, answer="Paris", qtype="bridge"):
    words = ["river", "castle", "music", "film", "novel", "station", "bridge", "garden", "museum", "harbor"]
    ctx = []
    for i, w in enumerate(words):
        title = {2: "Alpha", 6: "Beta"}.get(i, f"Doc{i}")
        ctx.append([title, [f"{title} has a {w} and a {words[(i + 3) % 10]}.", f" The {w} is old.", f" It is near Paris {i}."]])
    sf = sf if sf is not None else [["Alpha", 1], ["Alpha", 0], ["Beta", 2]]
    return make_record(qid=qid, qtype=qtype, answer=answer, context=ctx, sf=sf,
                       question="Which river castle music film is near the old station?")


def dense_sim(max_len=64, model=None):
    d = DenseScorer("fake", revision="r1", model=model or FakeModel(), max_seq_length=max_len, name="dense",
                    cache_namespace="test")
    return Similarity("dense", [d]), d


class RecordingScorer(Scorer):
    name = "rec"

    def __init__(self):
        self.queries = []

    def score(self, query, candidates, local_corpus=None):
        self.queries.append(query)
        return np.linspace(1.0, 0.1, len(candidates))


# ---------------------------------------------------------------- D-03 / D-04
def test_e1_query_is_question_plus_body_without_title():
    q = build_question(ten_doc_record())
    rec = RecordingScorer()
    oracle_block(q, q.unlabeled(), Similarity("rec", [rec]), "E1", None)
    a, b = q.gold_pair()
    assert rec.queries[0] == compose(q.question, q.docs[a].body)   # AtoB
    assert rec.queries[1] == compose(q.question, q.docs[b].body)   # BtoA
    assert not rec.queries[0].split("\n\n")[1].startswith("Alpha\n")


def test_e1p_uses_only_A_supporting_sentences_sorted_dedup():
    q = build_question(ten_doc_record(sf=[["Alpha", 1], ["Alpha", 0], ["Alpha", 1], ["Beta", 2]]))
    text, reason, ids = supporting_sentences_input(q, 2)
    assert reason is None and ids == [0, 1]
    assert text == (q.docs[2].sentences[0] + q.docs[2].sentences[1]).strip()
    rec = RecordingScorer()
    oracle_block(q, q.unlabeled(), Similarity("rec", [rec]), "E1P", None)
    assert rec.queries[0] == compose(q.question, text)                     # AtoB: A's sentences only
    assert rec.queries[1] == compose(q.question, q.docs[6].sentences[2].strip())  # BtoA: B's own sentence
    assert "Paris 2" not in rec.queries[0]  # sentence 2 of A is not a supporting fact


def test_e1p_invalid_is_flagged_never_replaced_by_full_text():
    q = build_question(ten_doc_record(sf=[["Alpha", 9], ["Beta", 0]]))
    rec = RecordingScorer()
    m, detail, validity = oracle_block(q, q.unlabeled(), Similarity("rec", [rec]), "E1P", None)
    assert validity == {"AtoB": SF_OUT_OF_RANGE, "BtoA": None}
    assert not any(k.startswith("E1P.AtoB") for k in m) and "E1P.BtoA.recall@1" in m
    assert len(rec.queries) == 1 and q.docs[2].body not in rec.queries[0]
    assert detail["AtoB"]["invalid_reason"] == SF_OUT_OF_RANGE


def test_both_directions_reported_separately():
    q = build_question(ten_doc_record())
    sim, _ = dense_sim()
    row = run_question(q, sim, ALL_EXPS, CFG)
    for exp in ("E1", "E1P"):
        for d in ("AtoB", "BtoA"):
            for k in (1, 2, 3, 5):
                assert f"{exp}.{d}.recall@{k}" in row["metrics"]
    for d in ("AtoB", "BtoA"):
        for f in ("F0", "F1", "F2"):
            assert f"R2.{d}.{f}.recall@5" in row["metrics"]
    assert not any(("forward" in k or "either" in k or "dirmean" in k) for k in row["metrics"])


# ---------------------------------------------------------------- D-05 BM25
def test_bm25_local_statistics_come_from_full_context_for_every_call():
    q = build_question(ten_doc_record())
    qi = q.unlabeled()
    local = BM25Scorer("local")
    sim = Similarity("bm25_local", [local])
    a = q.docs[2]
    cands = [d for d in q.docs if d.index != 2]
    full = BM25Scorer("trainglobal", BM25Stats.from_texts([d.text for d in q.docs]))
    only9 = BM25Scorer("trainglobal", BM25Stats.from_texts([d.text for d in cands]))
    got = sim.s_qa(qi, a, cands, "E1")
    query = compose(q.question, a.body)
    assert np.allclose(got, full.score(query, [d.text for d in cands]))
    assert not np.allclose(got, only9.score(query, [d.text for d in cands]))
    got_ab = sim.s_q_ab(qi, a, cands, "R2")
    exp_ab = full.score(q.question, [compose(a.text, d.text) for d in cands])
    assert np.allclose(got_ab, exp_ab)  # stats from original articles, not from composed A+B texts
    with pytest.raises(ValueError):
        local.score("x", ["y"], None)


def test_bm25_variants_share_tokenizer_formula_params():
    st = BM25Stats.from_texts(["a b", "c"])
    fl, fg = BM25Scorer("local").fingerprint(), BM25Scorer("trainglobal", st).fingerprint()
    for k in ("k1", "b", "tokenizer", "formula"):
        assert fl[k] == fg[k]


# ---------------------------------------------------------------- D-01 / D-02 / D-08 paths
def test_r1_widths_top5_per_A_joint_ranking():
    q = build_question(ten_doc_record())
    sim, _ = dense_sim()
    row = run_question(q, sim, ALL_EXPS, CFG)
    paths = row["detail"]["R1"]["paths"]
    assert {w: len(p) for w, p in paths.items()} == {"P1": 5, "P3": 15, "P5": 25, "PAll": 50}
    for p in paths.values():
        assert all(a != b for a, b, _ in p)
        scores = [s for _, _, s in p]
        assert scores == sorted(scores, reverse=True)
        per_a = {}
        for a, _, _ in p:
            per_a[a] = per_a.get(a, 0) + 1
        assert max(per_a.values()) <= 5
    assert set(map(tuple, (x[:2] for x in paths["P3"]))) <= set(map(tuple, (x[:2] for x in paths["PAll"])))
    # E2 is R1 at P3 by definition; F0 score = s(Q,A) + G0
    s_q = row["detail"]["O"]["scores"]
    a, b, s = paths["P3"][0]
    g = row["detail"]["R1"]["per_A_second_hop"][str(a)]
    assert s == pytest.approx(s_q[a] + g["scores"][g["candidates"].index(b)], abs=1e-5)
    assert row["metrics"]["E2.width"] == "P3"


def test_path_recall_unchanged_for_k_le_m():
    rng = np.random.default_rng(7)
    for _ in range(300):
        s_q = rng.normal(size=10)
        per_a = {a: ([b for b in range(10) if b != a], rng.normal(size=9)) for a in range(10)}
        gold = list(rng.choice(10, 2, replace=False))
        order = list(np.argsort(-s_q))
        for n in (1, 3, 5, 10):
            m5 = build_paths(s_q, per_a, order[:n], 5)
            m9 = build_paths(s_q, per_a, order[:n], 9)
            r5 = path_metrics([(a, b) for a, b, _ in m5], gold)
            r9 = path_metrics([(a, b) for a, b, _ in m9], gold)
            for k in (1, 2, 3, 5):
                assert r5[f"path_recall@{k}"] == r9[f"path_recall@{k}"]


def test_wider_first_hop_can_change_top1():
    s_q = np.array([1.0, 0.9, 0.8, 0.1])
    per_a = {0: ([1, 2, 3], np.array([0.1, 0.1, 0.1])), 1: ([0, 2, 3], np.array([0.1, 0.1, 0.1])),
             2: ([0, 1, 3], np.array([0.1, 0.1, 0.1])), 3: ([0, 1, 2], np.array([5.0, 0.1, 0.1]))}
    assert build_paths(s_q, per_a, [0], 5)[0][:2] != build_paths(s_q, per_a, [0, 1, 2, 3], 5)[0][:2]


# ---------------------------------------------------------------- D-13 label blindness
def test_autonomous_rankings_do_not_depend_on_gold_labels():
    q = build_question(ten_doc_record())
    q_other = build_question(ten_doc_record(sf=[["Doc0", 0], ["Doc9", 1]], answer="something else"))
    assert q.gold_indices != q_other.gold_indices
    sim, _ = dense_sim()
    r1 = run_question(q, sim, ALL_EXPS, CFG)
    r2 = run_question(q_other, sim, ALL_EXPS, CFG)
    assert r1["detail"]["O"]["order"] == r2["detail"]["O"]["order"]
    assert r1["detail"]["R1"]["paths"] == r2["detail"]["R1"]["paths"]
    for s in ("G0", "G1", "G2"):
        assert r1["detail"]["R3"][s]["paths"] == r2["detail"]["R3"][s]["paths"]
    qi = q.unlabeled()
    assert not any(hasattr(qi, f) for f in ("answer", "supporting_facts", "gold_indices"))


# ---------------------------------------------------------------- D-11 truncation ledger
def test_truncation_events_complete_and_survive_cache_hits():
    q = build_question(ten_doc_record())
    model = FakeModel()
    sim, dense = dense_sim(max_len=12, model=model)
    first = run_question(q, sim, ALL_EXPS, CFG)
    calls = model.calls
    second = run_question(q, sim, ALL_EXPS, CFG)
    assert model.calls == calls  # everything served from the embedding cache
    assert first["encoding_events"] == second["encoding_events"] and first["encoding_events"]
    ev = first["encoding_events"]
    types = {(e["exp"], e["type"]) for e in ev}
    for need in [("O", "Q"), ("O", "D"), ("E1", "Q+A"), ("E1P", "Q+SF"), ("R1", "Q+A"), ("R2", "A+B"), ("R3", "A+B")]:
        assert need in types
    for e in ev:
        assert {"exp", "type", "articles", "raw_tokens", "input_tokens", "truncated", "policy", "cache_key"} <= set(e)
        assert e["input_tokens"] <= 10 and (e["truncated"] == (e["raw_tokens"] > e["input_tokens"]) or e["parts"])
    qa = [e for e in ev if e["type"] == "Q+A"]
    assert all(e["policy"] == "query_first" for e in qa) and any(e["truncated"] for e in qa)
    assert all(e["policy"] == "balanced" for e in ev if e["type"] == "A+B")
    assert len({(e["exp"], e["type"], tuple(e["articles"])) for e in ev}) == len(ev)  # de-duplicated


def test_truncated_inputs_record_lost_gold_supporting_facts():
    # Gold Alpha (index 2), SF sentences 0 and 1. Budget = 12 - 2 = 10 whitespace tokens.
    q = build_question(ten_doc_record(sf=[["Alpha", 0], ["Alpha", 1], ["Beta", 2]]))
    sim, _ = dense_sim(max_len=12)
    row = run_question(q, sim, ALL_EXPS, CFG)
    a_id = q.docs[2].doc_id
    ev = {(e["exp"], e["type"], tuple(e["articles"])): e for e in row["encoding_events"]}
    # D: "Alpha | Alpha has a castle and a station. | The castle is old. | ..." -> 10 tokens end at "castle"
    d = ev[("O", "D", (a_id,))]
    assert d["truncated"] and d["gold_sf_truncation"][a_id]["kept"] == [0]
    assert d["gold_sf_truncation"][a_id]["lost_partial"] == [1] and d["gold_sf_lost"]
    # Q+A: the 10-token question uses the whole budget -> A's gold sentences are fully lost
    qa = ev[("E1", "Q+A", (a_id,))]
    assert qa["gold_sf_truncation"][a_id]["lost_full"] == [0, 1]
    # A+B (balanced, 5 tokens per side): first SF sentence partially kept, second lost
    ab = ev[("R2", "A+B", (a_id, q.docs[6].doc_id))]
    assert ab["gold_sf_truncation"][a_id]["lost_partial"] == [0]
    assert ab["gold_sf_truncation"][a_id]["lost_full"] == [1]
    # non-gold articles are never annotated; untruncated events carry no annotation
    assert all("gold_sf_truncation" not in e for e in row["encoding_events"]
               if e["articles"] and not set(e["articles"]) & {a_id, q.docs[6].doc_id})
    assert all("gold_sf_truncation" not in e for e in row["encoding_events"] if not e["truncated"])


def test_sf_annotation_does_not_change_rankings():
    q = build_question(ten_doc_record())
    sim, _ = dense_sim(max_len=12)
    row = run_question(q, sim, ALL_EXPS, CFG)
    sim2, _ = dense_sim(max_len=12)
    from rag_hotpotqa.experiments import multihop

    orig = multihop.annotate_sf_truncation
    multihop.annotate_sf_truncation = lambda *a, **k: None
    try:
        row2 = run_question(q, sim2, ALL_EXPS, CFG)
    finally:
        multihop.annotate_sf_truncation = orig
    def no_time(m):
        return {k: v for k, v in m.items() if not k.endswith(".seconds")}

    assert no_time(row["metrics"]) == no_time(row2["metrics"]) and row["detail"] == row2["detail"]


def test_bm25_has_no_encoder_events():
    q = build_question(ten_doc_record())
    sim = Similarity("bm25_local", [BM25Scorer("local")])
    row = run_question(q, sim, ["O"], CFG)
    assert row["encoding_events"] == [] and "O.fh_mrr" in row["metrics"]


# ---------------------------------------------------------------- D-13 access control
def _fake_project(tmp_path: Path):
    (tmp_path / "configs").mkdir()
    shutil.copy(PROJECT / "configs" / "data.yaml", tmp_path / "configs" / "data.yaml")
    ids = ["a1", "a2", "a3"]
    (tmp_path / "data" / "splits").mkdir(parents=True)
    (tmp_path / "data" / "manifests").mkdir(parents=True)
    (tmp_path / "data" / "splits" / "exploration_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    manifest = {"splits": {"exploration": {"ids_sha256": ids_hash(ids)}, "official_dev": {"ids_sha256": "x"}}}
    (tmp_path / "data" / "manifests" / "splits_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return dataclasses.replace(load_settings(), project_root=tmp_path, data_root=tmp_path / "data")


def test_split_integrity_and_dev_lock(tmp_path):
    s = _fake_project(tmp_path)
    assert load_split_ids(s, "exploration", "oer") == ["a1", "a2", "a3"]
    (tmp_path / "data" / "splits" / "exploration_ids.txt").write_text("a1\na2\na4\n", encoding="utf-8")
    with pytest.raises(SplitIntegrityError):
        load_split_ids(s, "exploration", "oer")
    with pytest.raises(SplitPolicyError, match="locked"):
        load_split_ids(s, "official_dev", "final")
    with pytest.raises(SplitPolicyError):
        load_split_ids(s, "official_dev", "oer")


def test_hf_cache_guard(monkeypatch, tmp_path):
    import huggingface_hub.constants as hfc

    # Keep the cache outside a synthetic project even when pytest's own
    # basetemp is intentionally inside the real writable workspace.
    s = dataclasses.replace(load_settings(), project_root=tmp_path / "project")
    monkeypatch.setattr(hfc, "HF_HUB_CACHE", str(tmp_path))
    with pytest.raises(ModelVersionError, match="outside"):
        check_hf_cache_inside_project(s, require_offline=False)
    monkeypatch.setattr(hfc, "HF_HUB_CACHE", str(s.project_root / ".cache" / "huggingface" / "hub"))
    monkeypatch.setattr(hfc, "HF_HUB_OFFLINE", False)
    with pytest.raises(ModelVersionError, match="offline"):
        check_hf_cache_inside_project(s, require_offline=True)


# ---------------------------------------------------------------- D-14 runner
class FlakyScorer(Scorer):
    name = "flaky"

    def __init__(self, exc_type=ValueError):
        self.exc_type = exc_type

    def score(self, query, candidates, local_corpus=None):
        if "BOOM" in query:
            raise self.exc_type("boom")
        return np.linspace(1.0, 0.1, len(candidates))


STOP = {"min_free_disk_gb": 0, "min_free_ram_gb": 0, "max_failure_rate": 0.5, "min_failures": 10}


def _qs():
    ok1 = build_question(ten_doc_record(qid="ok1"))
    bad = build_question(dict(ten_doc_record(qid="bad"), question="BOOM question"))
    ok2 = build_question(ten_doc_record(qid="ok2"))
    return [ok1, bad, ok2]


def test_nonfatal_failure_logged_retried_and_counted(tmp_path):
    sim = Similarity("flaky", [FlakyScorer()])
    res = run_retriever(_qs(), sim, ["O"], CFG, STOP, tmp_path, tmp_path, None, 2)
    assert res["n_completed"] == 2 and res["n_unresolved_failures"] == 1 and res["unresolved_failure_qids"] == ["bad"]
    fails = (tmp_path / "flaky.failures.jsonl").read_text(encoding="utf-8")
    assert '"qid": "bad"' in fails
    res2 = run_retriever(_qs(), sim, ["O"], CFG, STOP, tmp_path, tmp_path, None, 2)  # resume retries only "bad"
    assert res2["attempted_this_session"] == 1 and res2["n_completed"] == 2
    assert json.loads((tmp_path / "flaky.status.json").read_text(encoding="utf-8"))["total"] == 3


def test_fatal_errors_stop_the_run(tmp_path):
    sim = Similarity("flaky", [FlakyScorer(SplitPolicyError)])
    with pytest.raises(SplitPolicyError):
        run_retriever(_qs(), sim, ["O"], CFG, STOP, tmp_path, tmp_path, None, 2)
    breaker = dict(STOP, min_failures=0, max_failure_rate=0.0)
    with pytest.raises(FailureRateError):
        run_retriever(_qs(), Similarity("flaky2", [FlakyScorer()]), ["O"], CFG, breaker, tmp_path, tmp_path, None, 2)


def test_resume_refuses_changed_config(tmp_path):
    check_or_write_run_config(tmp_path, tmp_path, {"a": 1})
    check_or_write_run_config(tmp_path, tmp_path, {"a": 1})
    with pytest.raises(ConfigMismatchError):
        check_or_write_run_config(tmp_path, tmp_path, {"a": 2})


# ---------------------------------------------------------------- D-13 history protection
def test_spec_v2_outputs_are_separate_from_v1():
    from rag_hotpotqa.utils.config import load_config

    e = load_config("experiments")
    assert e["spec_version"] == 2 and e["run"]["output_root"] != "outputs/pilot"
    assert not e["run"]["report_dir"].endswith("pilot_round1.md")
