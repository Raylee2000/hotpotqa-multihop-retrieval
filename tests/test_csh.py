"""Conditional second-hop examples: predicted evidence only (gold SF never an input), masks, margin loss."""
from __future__ import annotations

import dataclasses

import pytest

from rag_hotpotqa.data.documents import build_question
from rag_hotpotqa.training.contrastive import text_key
from rag_hotpotqa.training.csh import build_csh_examples, conditional_margin, csh_batch, evidence_text

from conftest import make_record

torch = pytest.importorskip("torch")


def qs(n=4):
    out = []
    for i in range(n):
        ctx = [[f"A{i}", [f"A{i} was founded by B{i}.", " It sells shoes.", " It is old."]],
               [f"B{i}", [f"B{i} was born in Paris."]], [f"N{i}", ["Noise one.", " Noise two."]], [f"M{i}", ["Other."]]]
        out.append(build_question(make_record(qid=f"q{i}", context=ctx, sf=[[f"A{i}", 0], [f"B{i}", 0]],
                                              question=f"Where was the founder of A{i} born?")))
    return out


def preds_for(questions):
    p = {}
    for q in questions:
        p[(q.qid, 0)] = ([0, 1, 2], [0.2, 0.9, 0.1])  # the model prefers a NON-SF sentence of A
        p[(q.qid, 1)] = ([0], [0.8])
        p[(q.qid, 2)] = ([0, 1], [0.6, 0.3])
    return p


def test_evidence_is_predicted_top_k_in_document_order():
    q = qs(1)[0]
    p = preds_for([q])
    assert evidence_text(q, 0, p, k=1) == "It sells shoes."
    assert evidence_text(q, 0, p, k=2) == "A0 was founded by B0. It sells shoes."
    assert evidence_text(q, 3, p, k=2) is None


def test_examples_ignore_gold_sf_and_use_wrong_first_hop():
    questions = qs()
    p = preds_for(questions)
    kw = dict(preds=p, k=1, seed=42, epoch=0, mode="both", wrong_first={q.qid: 2 for q in questions},
              hard={(q.qid, a): [3, 2] for q in questions for a in (0, 1)}, n_hard=1)
    ex, skipped = build_csh_examples(questions, **kw)
    assert len(ex) == 4 and skipped["no_evidence"] == 0
    # changing the gold supporting facts must not change any query (they are not an input)
    altered = [dataclasses.replace(q, supporting_facts=[(q.docs[0].title, 2), (q.docs[1].title, 0)]) for q in questions]
    ex2, _ = build_csh_examples(altered, **kw)
    assert [e.query for e in ex] == [e.query for e in ex2]
    for e in ex:
        assert e.positive.startswith(("A", "B")) and e.wrong_query.endswith("Noise one.")
        assert e.hard_negatives and not e.hard_negatives[0].startswith(("A", "B"))
        assert text_key(e.positive) in e.gold_keys


def test_both_directions_occur_and_silver_mode_drops_undecided():
    questions = qs(12)
    p = {(q.qid, d): ([0], [0.5]) for q in questions for d in (0, 1)}
    ex, _ = build_csh_examples(questions, p, 1, 42, 0, "both", {}, {}, 0)
    assert {e.a for e in ex} == {0, 1}
    ex_s, sk = build_csh_examples(questions, p, 1, 42, 0, "silver", {}, {}, 0, silver={"q0": (0, 1)})
    assert [e.qid for e in ex_s] == ["q0"] and sk["no_direction"] == 11


def test_mask_removes_other_gold_including_a():
    questions = qs(2)
    p = preds_for(questions)
    ex, _ = build_csh_examples(questions, p, 1, 42, 0, "both", {}, {(q.qid, a): [0, 1, 2] for q in questions for a in (0, 1)}, 1)
    # make example 1's hard negative the same text as example 0's A article
    ex[1].hard_negatives = [questions[0].docs[ex[0].a].text]
    docs, mask = csh_batch(ex, use_hard=True)
    assert len(docs) == 4 and docs[3] == questions[0].docs[ex[0].a].text  # positives, then hard negatives in order
    assert mask[0, 3] and not mask[0, 0] and not mask[1, 3] and not mask[0, 2]


def test_conditional_margin_gradient_direction():
    pos = torch.tensor([0.5, 0.5], requires_grad=True)
    wrong = torch.tensor([0.45, 0.2], requires_grad=True)
    loss = conditional_margin(pos, wrong, 0.1)
    loss.backward()
    assert pytest.approx(float(loss), abs=1e-6) == 0.025
    assert pos.grad[0] < 0 and wrong.grad[0] > 0 and pos.grad[1] == 0  # only the violating example pushes
