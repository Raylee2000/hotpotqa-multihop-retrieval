"""Metric primitives and the MDR evaluator on hand-built cases."""
import numpy as np
import pytest

from rag_hotpotqa.evaluation.mdr_metrics import mdr_question_metrics
from rag_hotpotqa.evaluation.metrics import (
    aggregate, first_hop_metrics, hit_at_k, path_metrics, rank_paths, reciprocal_rank, second_hop_metrics,
)


def test_hit_and_rr():
    assert hit_at_k(1, 1) == 1 and hit_at_k(2, 1) == 0 and hit_at_k(None, 3) == 0
    assert hit_at_k(9, None) == 1
    assert reciprocal_rank(4) == 0.25 and reciprocal_rank(None) == 0.0


def test_first_hop_metrics():
    m = first_hop_metrics([3, 1, 0, 2], gold=[0, 1], ks=(1, 2, 3, 5))
    assert m["gold_ranks"] == [3, 2] and m["best_gold_rank"] == 2
    assert m["any_gold@1"] == 0 and m["any_gold@2"] == 1
    assert m["all_gold@2"] == 0 and m["all_gold@3"] == 1
    assert "all_gold@1" not in m  # undefined for two gold articles
    assert m["mrr"] == 0.5


def test_topk_larger_than_pool():
    m = first_hop_metrics([1, 0], gold=[0, 1], ks=(1, 2, 3, 5))
    assert m["all_gold@5"] == 1
    s = second_hop_metrics([5], gold_b=5, ks=(1, 3, 5))
    assert s["recall@5"] == 1 and s["pool_size"] == 1 and s["rank"] == 1


def test_path_metrics_unordered_match_directed_entries():
    ranked = [(2, 0), (1, 0), (0, 1)]
    m = path_metrics(ranked, gold=[0, 1], ks=(1, 2, 3, 5))
    assert m["gold_path_rank"] == 2  # (1,0) is a correct (unordered) match
    assert m["path_recall@1"] == 0 and m["path_recall@2"] == 1 and m["path_recall@5"] == 1
    assert m["n_paths"] == 3 and m["gold_path_coverage"] == 1
    # A->C is wrong
    assert path_metrics([(0, 2)], gold=[0, 1], ks=(1,))["path_recall@1"] == 0


def test_rank_paths_rejects_same_doc_and_breaks_ties():
    with pytest.raises(ValueError):
        rank_paths({(1, 1): 0.5})
    assert rank_paths({(2, 0): 1.0, (0, 2): 1.0, (1, 0): 2.0}) == [(1, 0), (0, 2), (2, 0)]
    with pytest.raises(ValueError):
        rank_paths({(0, 1): float("nan")})


def test_aggregate_groups_and_missing():
    rows = [{"qtype": "bridge", "metrics": {"m": 1}}, {"qtype": "bridge", "metrics": {"m": 0}},
            {"qtype": "comparison", "metrics": {}}]
    a = aggregate(rows, ["m"], metrics_field="metrics")
    assert a["overall"]["m"] == 0.5 and a["overall"]["m__n"] == 2
    assert a["bridge"]["m"] == 0.5 and a["comparison"]["m"] is None


def test_mdr_metrics_hand_cases():
    sp = ["X", "Y"]
    assert mdr_question_metrics([("X", "Y")], sp) == {"PR": 1, "P-EM": 1, "1-Recall": 1, "Path Recall": 1}
    assert mdr_question_metrics([("Y", "X")], sp)["Path Recall"] == 1
    assert mdr_question_metrics([("X", "Z"), ("W", "Y")], sp) == {"PR": 1, "P-EM": 1, "1-Recall": 1, "Path Recall": 0}
    assert mdr_question_metrics([("Z", "X")], sp) == {"PR": 1, "P-EM": 0, "1-Recall": 0, "Path Recall": 0}
    with pytest.raises(ValueError):
        mdr_question_metrics([("X", "Y")], ["X", "X"])


def test_our_path_recall_equals_mdr_path_recall():
    titles = ["X", "Y", "Z", "W"]
    rng = np.random.default_rng(0)
    for _ in range(200):
        paths = [tuple(int(x) for x in rng.choice(4, 2, replace=False)) for _ in range(5)]
        for k in (1, 2, 3, 5):
            ours = path_metrics(paths, gold=[0, 1], ks=(k,))[f"path_recall@{k}"]
            mdr = mdr_question_metrics([(titles[a], titles[b]) for a, b in paths[:k]], ["X", "Y"])["Path Recall"]
            assert ours == mdr
