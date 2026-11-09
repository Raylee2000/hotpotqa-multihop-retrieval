"""Tests for evaluation/stats.py (direction-gap analysis, classifier metrics) and data/holdout.py."""
from __future__ import annotations

import json

import numpy as np
import pytest

from rag_hotpotqa.data.access import SplitIntegrityError, SplitPolicyError
from rag_hotpotqa.data.holdout import build_balanced_holdout, ids_from_line_prefix, largest_remainder
from rag_hotpotqa.evaluation.stats import (binary_report, boot_indices, boot_mean, conditional_logit_pairs, ece, holm,
                                           mcnemar_exact, prior_weights)


def test_mcnemar_exact_counts_and_p():
    a = [1] * 10 + [0] * 2 + [1] * 5
    b = [0] * 10 + [1] * 2 + [1] * 5
    r = mcnemar_exact(a, b)
    assert (r["only_a"], r["only_b"]) == (10, 2)
    assert r["p_exact"] == pytest.approx(0.0386, abs=1e-4)  # 2 * P(X <= 2), X ~ Bin(12, 0.5)
    assert mcnemar_exact([1, 0], [1, 0])["p_exact"] == 1.0


def test_holm_step_down():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


def test_boot_mean_matches_direct():
    x = np.arange(20, dtype=float)
    idx = boot_indices(20, 300)
    assert np.allclose(boot_mean(x, idx, chunk=7), x[idx].mean(1))


def test_conditional_logit_recovers_coefficient():
    rng = np.random.default_rng(0)
    n, beta = 20000, np.array([1.0, -0.5])
    x1, x2 = rng.normal(size=(n, 2)), rng.normal(size=(n, 2))
    alpha = rng.normal(size=n) * 2  # question fixed effects, cancel in the conditional likelihood
    y1 = rng.random(n) < 1 / (1 + np.exp(-(alpha + x1 @ beta)))
    y2 = rng.random(n) < 1 / (1 + np.exp(-(alpha + x2 @ beta)))
    m = conditional_logit_pairs(x1, x2, y1, y2)
    assert m["identifiable"] and m["converged"]
    assert np.allclose(m["beta"], beta, atol=0.1)


def test_conditional_logit_flags_collinear_swap_variables():
    # start / target properties swap between the two instances -> diffs are exact negatives -> rank deficient
    rng = np.random.default_rng(1)
    pa, pb = rng.integers(0, 2, 500), rng.integers(0, 2, 500)
    x1 = np.c_[pa, pb]  # instance 1: start = a, target = b
    x2 = np.c_[pb, pa]  # instance 2: start = b, target = a
    y1, y2 = rng.integers(0, 2, 500), rng.integers(0, 2, 500)
    m = conditional_logit_pairs(x1, x2, y1, y2)
    assert m["identifiable"] is False and m["rank"] == 1


def test_binary_report_basic_and_weights():
    y = np.array([0, 0, 0, 1, 1, 1])
    p1 = np.array([0.1, 0.2, 0.7, 0.8, 0.9, 0.4])
    r = binary_report(y, p1, ("bridge", "comparison"))
    assert r["confusion"]["counts"] == [[2, 1], [1, 2]]
    assert r["accuracy"] == pytest.approx(4 / 6)
    assert r["balanced_accuracy"] == pytest.approx(2 / 3)
    assert r["per_class"]["comparison"]["precision"] == pytest.approx(2 / 3)
    # prior reweighting: accuracy = sum_c prior_c * recall_c
    w = prior_weights(y, (0.8, 0.2))
    rw = binary_report(y, p1, ("bridge", "comparison"), w=w)
    assert rw["accuracy"] == pytest.approx(0.8 * 2 / 3 + 0.2 * 2 / 3)
    tpr_c, fpr = 2 / 3, 1 / 3
    assert rw["per_class"]["comparison"]["precision"] == pytest.approx(0.2 * tpr_c / (0.2 * tpr_c + 0.8 * fpr))
    assert w.mean() == pytest.approx(1.0)


def test_threshold_is_strict_like_sklearn_predict():
    r = binary_report(np.array([0, 1]), np.array([0.5, 0.9]), ("bridge", "comparison"))
    assert r["confusion"]["counts"] == [[1, 0], [0, 1]]


def test_ece_perfect_and_off():
    y = np.array([0, 1] * 50)
    assert ece(y, np.full(100, 0.5)) == pytest.approx(0.0)
    assert ece(np.ones(10), np.full(10, 0.7)) == pytest.approx(0.3)


def _meta():
    meta = {}
    for i in range(40):
        meta[f"c{i:02d}"] = ("comparison", ("easy", "medium", "hard")[i % 3])
    for i in range(300):
        meta[f"b{i:03d}"] = ("bridge", ("easy", "medium", "hard")[i % 3])
    return meta


def test_balanced_holdout_disjoint_balanced_deterministic():
    meta = _meta()
    assigned = {f"c{i:02d}" for i in range(10)} | {f"b{i:03d}" for i in range(100)}
    h1 = build_balanced_holdout(meta, assigned, 42)
    h2 = build_balanced_holdout(dict(reversed(list(meta.items()))), assigned, 42)
    assert h1 == h2  # input-order independent
    ids = set(h1["minority_ids"]) | set(h1["majority_ids"])
    assert not ids & assigned
    assert len(h1["minority_ids"]) == 30 == len(h1["majority_ids"])  # all free comparison, 1:1
    for lv in ("easy", "medium", "hard"):
        assert sum(meta[i][1] == lv for i in h1["minority_ids"]) == sum(meta[i][1] == lv for i in h1["majority_ids"])
    assert build_balanced_holdout(meta, assigned, 7)["majority_ids"] != h1["majority_ids"]


def test_balanced_holdout_never_reuses_assigned_when_pool_short():
    meta = _meta()
    assigned = {f"b{i:03d}" for i in range(290)}
    with pytest.raises(ValueError):
        build_balanced_holdout(meta, assigned, 42)


def test_largest_remainder_sums():
    q = largest_remainder(10, {"a": 1, "b": 1, "c": 1})
    assert sum(q.values()) == 10 and max(q.values()) - min(q.values()) <= 1


def test_ids_from_line_prefix_reads_only_id(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text("\n".join(json.dumps({"_id": f"id{i}", "question": "SECRET"}, sort_keys=True) for i in range(3)) + "\n", encoding="utf-8")
    assert ids_from_line_prefix(p) == ["id0", "id1", "id2"]
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"question": "q", "_id": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(SplitIntegrityError):
        ids_from_line_prefix(bad)


def test_holdout_purpose_enforced(tmp_path):
    from rag_hotpotqa.data.holdout import load_holdout_ids

    class S:  # minimal settings stand-in
        project_root = tmp_path
    with pytest.raises(SplitPolicyError):
        load_holdout_ids(S(), "qtype_balanced_holdout_v1", "train")
