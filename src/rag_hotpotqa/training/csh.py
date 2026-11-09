"""Conditional second-hop (CSH) training: query = Q ⊕ predicted evidence of A, positive = the other gold B.

Predicted evidence Ŝ_k(A) = the k sentences of A with the highest SF-selector scores, in document order. On Training
the scores are cross-fitted (the selector never saw the question); gold supporting facts are NEVER an input here
(they only trained the selector). Every example also carries a wrong first hop A' (BM25-Local top non-gold article
for Q, label-free ranking) whose evidence query q'' feeds the conditional margin loss

    L = InfoNCE(q', {B} ∪ in-batch ∪ hard negatives; false-negative mask)  +  λ · max(0, m − s(q', B) + s(q'', B))

i.e. the evidence of the right A must lead to B more strongly than the evidence of a wrong A'.
Direction: "both" picks (g1→g2) or (g2→g1) per question and epoch from sha256(seed, qid, epoch) — no silver label;
"silver" uses the derived hop order (S-HOP-LINK / S-HOP-QTITLE) and drops undecided questions.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..data.documents import Question
from .contrastive import text_key
from .sf import article_sentences

Preds = Mapping[tuple[str, int], tuple[Sequence[int], Sequence[float]]]  # (qid, doc) -> (sentence ids, scores)


def evidence_text(q: Question, doc: int, preds: Preds, k: int) -> str | None:
    """Top-k predicted sentences of article `doc`, in document order; None if there are no predictions."""
    p = preds.get((q.qid, doc))
    if p is None or not len(p[0]):
        return None
    sent_ids, scores = list(p[0]), np.asarray(p[1], float)
    keep = sorted(sent_ids[i] for i in np.argsort(-scores, kind="stable")[:k])
    text = dict(article_sentences(q, doc))
    return " ".join(text[i] for i in keep if i in text) or None


def compose_query(question: str, evidence: str, joiner: str = "\n\n") -> str:
    return question + joiner + evidence


@dataclass
class CSHExample:
    qid: str
    a: int
    b: int
    query: str                 # Q ⊕ Ŝ_k(A)
    positive: str              # text of B (title + body)
    wrong_query: str | None    # Q ⊕ Ŝ_k(A') for the conditional margin loss
    hard_negatives: list[str]
    gold_keys: frozenset[str] = field(default_factory=frozenset)


def direction(qid: str, seed: int, epoch: int) -> int:
    return int(hashlib.sha256(f"{seed}:{qid}:{epoch}".encode()).hexdigest()[:8], 16) % 2


def build_csh_examples(questions: Sequence[Question], preds: Preds, k: int, seed: int, epoch: int, mode: str,
                       wrong_first: Mapping[str, int], hard: Mapping[tuple[str, int], Sequence[int]], n_hard: int,
                       silver: Mapping[str, tuple[int, int]] | None = None, joiner: str = "\n\n") -> tuple[list[CSHExample], dict]:
    """hard[(qid, a)] = label-free ranking of candidate negatives for the query built from a (non-gold indices first
    to last); wrong_first[qid] = A' index. Only bridge questions with two gold articles are used."""
    out, skipped = [], {"not_bridge_two_gold": 0, "no_direction": 0, "no_evidence": 0}
    for q in questions:
        if q.qtype != "bridge" or not q.is_two_gold:
            skipped["not_bridge_two_gold"] += 1
            continue
        g1, g2 = q.gold_indices
        if mode == "silver":
            if silver is None or q.qid not in silver:
                skipped["no_direction"] += 1
                continue
            a, b = silver[q.qid]
        else:
            a, b = (g1, g2) if direction(q.qid, seed, epoch) == 0 else (g2, g1)
        ev = evidence_text(q, a, preds, k)
        if ev is None:
            skipped["no_evidence"] += 1
            continue
        wq = None
        if q.qid in wrong_first:
            wev = evidence_text(q, wrong_first[q.qid], preds, k)
            wq = compose_query(q.question, wev, joiner) if wev else None
        gold = set(q.gold_indices)
        negs = [i for i in hard.get((q.qid, a), []) if i not in gold][:n_hard]
        keys = frozenset(text_key(q.docs[i].text) for i in q.gold_indices)
        out.append(CSHExample(q.qid, a, b, compose_query(q.question, ev, joiner), q.docs[b].text, wq,
                              [q.docs[i].text for i in negs], keys))
    return out, skipped


def csh_batch(batch: Sequence[CSHExample], use_hard: bool) -> tuple[list[str], np.ndarray]:
    """Documents (positives first, then hard negatives) and mask[i, j] = True when document j is a gold article of
    example i's question other than its own positive (never a negative; this also removes A itself)."""
    docs = [e.positive for e in batch]
    if use_hard:
        docs += [t for e in batch for t in e.hard_negatives]
    keys = [text_key(t) for t in docs]
    mask = np.zeros((len(batch), len(docs)), dtype=bool)
    for i, e in enumerate(batch):
        for j, kk in enumerate(keys):
            if j != i and kk in e.gold_keys:
                mask[i, j] = True
    return docs, mask


def conditional_margin(pos_sim, wrong_sim, margin: float):
    """mean max(0, m − s(q', B) + s(q'', B)) over examples that have a wrong-first-hop query (torch tensors)."""
    import torch

    return torch.clamp(margin - pos_sim + wrong_sim, min=0).mean()
