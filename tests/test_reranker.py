"""Path reranker learners on synthetic candidate sets: the signal feature must rank the gold pair first."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
torch = pytest.importorskip("torch")

SPEC = importlib.util.spec_from_file_location("rr", Path(__file__).resolve().parents[1] / "scripts" / "train_path_reranker.py")
rr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rr)


def synthetic(tmp_path: Path, name: str, n_q: int, seed: int):
    rng = np.random.default_rng(seed)
    X, y, q, pairs, info = [], [], [], [], []
    feats = ["f_best", "noise", "n_long", "logtok_max", "logtok_min"]
    for i in range(n_q):
        n = 8
        g = rng.integers(n)
        for j in range(n):
            signal = 1.0 if j == g else 0.0
            X.append([signal + rng.normal(0, 0.3), rng.normal(), 0, 5.0, 4.0])
            y.append(int(j == g))
            q.append(i)
            pairs.append((j, j + 1, j))
        info.append({"qid": f"q{i}", "qtype": "bridge" if i % 2 else "comparison", "two_gold": True, "gold_in_cand": True})
    np.savez(tmp_path / f"{name}.npz", X=np.asarray(X, np.float32), y=np.asarray(y, np.int8), q=np.asarray(q, np.int32),
             pairs=np.asarray(pairs, np.int16))
    (tmp_path / f"{name}_meta.json").write_text(json.dumps({"features": feats, "questions": info}), encoding="utf-8")
    return rr.Data(tmp_path / f"{name}.npz", tmp_path / f"{name}_meta.json")


def test_listwise_and_lambdamart_learn_the_signal(tmp_path):
    tr, vs = synthetic(tmp_path, "train", 150, 0), synthetic(tmp_path, "vs", 80, 1)
    Xtr, Xvs = rr.standardise(tr.X, vs.X)
    w = rr.fit_listwise(tr, Xtr, 1e-3, epochs=200)
    assert w[0] > abs(w[1]) * 3
    lw = rr.evaluate(vs, Xvs @ w)
    lm = rr.evaluate(vs, rr.fit_lambdamart(tr, tr.X, rounds=30, depth=2)(vs.X))
    noise = rr.evaluate(vs, vs.X[:, 1])
    assert lw["pr1"] > 0.8 and lm["pr1"] > 0.8 and noise["pr1"] < 0.4
    assert lw["candidate_path_recall"] == 1.0 and lw["conditional_ranking_accuracy"] == lw["pr1"]


def test_pairwise_data_weights_hard_pairs(tmp_path):
    tr = synthetic(tmp_path, "train", 5, 2)
    D, yp, wp = rr.pairwise_data(tr, tr.X)
    assert len(D) == 2 * 5 * 7 and set(np.unique(yp)) == {0.0, 1.0} and set(np.unique(wp)) <= {1.0, 2.0}
