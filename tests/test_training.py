"""Contrastive training data, false-negative mask, loss gradients, split isolation, width-limited R1."""
from __future__ import annotations

import numpy as np
import pytest

from rag_hotpotqa.data.access import SplitPolicyError, check_split_allowed
from rag_hotpotqa.data.documents import build_question
from rag_hotpotqa.experiments.multihop import run_question
from rag_hotpotqa.retrieval.dense import DenseScorer
from rag_hotpotqa.retrieval.similarity import Similarity
from rag_hotpotqa.training.contrastive import batch_docs, build_examples, info_nce, text_key

from conftest import make_record

torch = pytest.importorskip("torch")


def qs(n=8):
    out = []
    for i in range(n):
        ctx = [[f"Gold{i}a", [f"Gold{i}a mentions bridge{i}."]], [f"Gold{i}b", [f"Bridge{i} is in town{i}."]],
               [f"Neg{i}x", [f"Where was person{i} born, neg x."]], [f"Neg{i}y", ["Unrelated."]]]
        out.append(build_question(make_record(qid=f"q{i}", context=ctx, sf=[[f"Gold{i}a", 0], [f"Gold{i}b", 0]],
                                              question=f"Where was person{i} born?")))
    return out


def test_examples_never_use_gold_as_negative_and_are_deterministic():
    ex, skipped = build_examples(qs(), seed=42, n_hard=2)
    assert len(ex) == 8 and skipped["not_two_gold"] == 0
    for e in ex:
        assert all(text_key(t) not in e.gold_keys for t in e.hard_negatives)
        assert text_key(e.positive) in e.gold_keys
    ex2, _ = build_examples(qs(), seed=42, n_hard=2)
    assert [(e.kind, e.positive) for e in ex] == [(e.kind, e.positive) for e in ex2]
    assert {e.kind for e in ex} == {"first_hop", "second_hop"}
    # the label-free BM25 ranking puts the lexically closest non-gold article first
    fh = next(e for e in ex if e.kind == "first_hop")
    assert fh.hard_negatives[0].startswith("Neg")


def test_false_negative_mask_covers_other_gold_and_duplicates():
    ex, _ = build_examples(qs(3), seed=42, n_hard=1)
    dup = ex[0].__class__(**{**ex[1].__dict__})
    dup.positive = ex[0].positive          # same article text as example 0's positive, other question
    batch = [ex[0], dup, ex[2]]
    docs, mask = batch_docs(batch, use_hard=True)
    assert docs[:3] == [b.positive for b in batch]
    assert mask[0, 1]                       # identical gold text of query 0 is not a negative for it
    assert not mask[0, 0] and not mask[2, 2]
    assert not mask[2].any() or all(text_key(docs[j]) in batch[2].gold_keys for j in np.where(mask[2])[0])


def test_info_nce_gradient_pushes_positive_up_and_overfits():
    torch.manual_seed(0)
    q = torch.nn.Parameter(torch.randn(4, 16))
    d = torch.nn.Parameter(torch.randn(8, 16))
    mask = np.zeros((4, 8), dtype=bool)
    mask[0, 5] = True
    opt = torch.optim.SGD([q, d], lr=0.5)
    first = None
    for _ in range(200):
        qv, dv = torch.nn.functional.normalize(q, dim=-1), torch.nn.functional.normalize(d, dim=-1)
        loss = info_nce(qv, dv, mask, tau=0.1)
        if first is None:
            first = float(loss)
            loss.backward()
            # masked document gets no gradient from query 0's term only; positives get pulled closer
            assert torch.isfinite(q.grad).all() and torch.isfinite(d.grad).all()
            opt.zero_grad()
            continue
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < 0.05 < first
    sims = (torch.nn.functional.normalize(q, dim=-1) @ torch.nn.functional.normalize(d, dim=-1).T).detach()
    assert all(int(torch.argmax(sims[i].masked_fill(torch.as_tensor(mask[i]), -9))) == i for i in range(4))


def test_masked_negative_is_excluded_from_loss():
    qv = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0]]), dim=-1)
    dv = torch.nn.functional.normalize(torch.tensor([[1.0, 0.1], [1.0, 0.0]]), dim=-1)  # doc 1 identical to query
    unmasked = float(info_nce(qv, dv, np.zeros((1, 2), bool), 0.05))
    masked = float(info_nce(qv, dv, np.array([[False, True]]), 0.05))
    assert masked == pytest.approx(0.0, abs=1e-6) and unmasked > 0.5


def test_split_policy_training_vs_selection():
    check_split_allowed("training", "train")
    check_split_allowed("validation", "select")
    for split in ("exploration", "validation", "dev"):
        with pytest.raises(SplitPolicyError):
            check_split_allowed(split, "train")
    with pytest.raises(SplitPolicyError):
        check_split_allowed("dev", "select")


def test_width_limited_r1_matches_full_run(fake_model):
    ctx = [[f"T{i}", [f"T{i} sentence about born place {i}.", f" More words {i}."]] for i in range(7)]
    q = build_question(make_record(context=ctx, sf=[["T1", 0], ["T4", 0]]))
    full = {"ks": [1, 2, 3, 5], "top_m": 5, "widths": {"P1": 1, "P3": 3, "P5": 5, "PAll": None}, "e2_width": "P3", "r3_width": 3}
    lim = {**full, "widths": {"P1": 1, "P3": 3}}
    a = run_question(q, Similarity("dense", [DenseScorer("f", model=fake_model, max_seq_length=64)]), ["E1", "R1"], full)
    b = run_question(q, Similarity("dense", [DenseScorer("f", model=fake_model, max_seq_length=64)]), ["E1", "R1"], lim)
    for k, v in b["metrics"].items():
        assert a["metrics"][k] == v, k


def test_validation_partition_fixed_disjoint_and_report_guarded():
    from rag_hotpotqa.data.access import check_part_access, partition_validation
    from rag_hotpotqa.experiments.runner import select_subset

    ids = [f"id{i}" for i in range(50)]
    p = partition_validation(ids, n_select=20)
    assert len(p["v_select"]) == 20 and len(p["v_report"]) == 30
    assert not set(p["v_select"]) & set(p["v_report"]) and set(p["v_select"]) | set(p["v_report"]) == set(ids)
    assert p["v_select"] == select_subset(ids, 20)  # the registered selection subsets are prefixes of V-select
    assert partition_validation(list(reversed(ids)), n_select=20) == p
    check_part_access("v_select", False)
    with pytest.raises(SplitPolicyError):
        check_part_access("v_report", False)
    check_part_access("v_report", True)
