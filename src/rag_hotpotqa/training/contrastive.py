"""Retrieval contrastive fine-tuning data and loss (Training split only).

Example (one per Training question per epoch, chosen deterministically from sha256(seed, qid)):
    first_hop   query = Q                       positive = one gold article (title + body)
    second_hop  query = Q + joiner + A body     positive = the other gold article B
Hard negatives: non-gold context articles of the same question, ranked by BM25-Local against the
query (static, label-free ranking; the other gold article is never a negative).

Loss (InfoNCE / multiple-negatives ranking with temperature tau), for query i with positive p_i:

    L_i = -log  exp(s(q_i, p_i) / tau) / sum_{d in D_i} exp(s(q_i, d) / tau)

D_i = {p_i} ∪ {in-batch positives and hard negatives of other examples} ∪ {own hard negatives}
      minus every document whose text equals a gold article of question i (false-negative mask).
Hard negatives enter only as extra softmax terms; they get no extra weight. Since
dL/ds(q,d) = softmax_d / tau ≥ 0 for negatives, the gradient pushes a negative down in proportion to
how close it already is, and the mask removes known false negatives (the other gold article, the same
article text under another question). Non-gold articles that are actually relevant remain possible
label noise (logged as a risk, not corrected).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..data.documents import Question
from ..retrieval.base import compose, rank_order
from ..retrieval.bm25 import BM25Scorer


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class Example:
    qid: str
    qtype: str
    kind: str
    query: str
    positive: str
    hard_negatives: list[str]
    gold_keys: frozenset[str] = field(default_factory=frozenset)


def build_examples(questions: Sequence[Question], seed: int, n_hard: int, joiner: str = "\n\n") -> tuple[list[Example], dict]:
    bm25 = BM25Scorer("local")
    out, skipped = [], {"not_two_gold": 0, "empty_body": 0, "no_negative": 0}
    for q in questions:
        if not q.is_two_gold:
            skipped["not_two_gold"] += 1
            continue
        h = int(hashlib.sha256(f"{seed}:{q.qid}".encode()).hexdigest()[:8], 16)
        kind = "first_hop" if h % 2 == 0 else "second_hop"
        g = q.gold_indices[(h >> 1) % 2]
        other = next(i for i in q.gold_indices if i != g)
        if kind == "first_hop":
            query, pos = q.question, q.docs[g].text
        else:
            if not q.docs[g].body:
                skipped["empty_body"] += 1
                continue
            query, pos = compose(q.question, q.docs[g].body, joiner), q.docs[other].text
        cands = [d for d in q.docs if d.index not in q.gold_indices]
        if not cands:
            skipped["no_negative"] += 1
            continue
        sc = bm25.score(query, [d.text for d in cands], local_corpus=[d.text for d in q.docs])
        hard = [cands[i].text for i in rank_order(sc)[:n_hard]]
        gold_keys = frozenset(text_key(q.docs[i].text) for i in q.gold_indices)
        out.append(Example(q.qid, q.qtype, kind, query, pos, hard, gold_keys))
    return out, skipped


def batch_docs(batch: Sequence[Example], use_hard: bool) -> tuple[list[str], np.ndarray]:
    """Documents of a batch (positives first, then hard negatives) and the false-negative mask
    mask[i, j] = True when document j must not count as a negative for query i."""
    docs = [e.positive for e in batch]
    if use_hard:
        docs += [t for e in batch for t in e.hard_negatives]
    keys = [text_key(t) for t in docs]
    mask = np.zeros((len(batch), len(docs)), dtype=bool)
    for i, e in enumerate(batch):
        for j, k in enumerate(keys):
            if j != i and k in e.gold_keys:
                mask[i, j] = True
    return docs, mask


def info_nce(qv, dv, mask, tau: float):
    """Mean InfoNCE over the batch; positives are documents 0..B-1 (torch tensors, L2-normalised)."""
    import torch
    import torch.nn.functional as F

    logits = qv @ dv.T / tau
    logits = logits.masked_fill(torch.as_tensor(mask, device=logits.device), float("-inf"))
    return F.cross_entropy(logits, torch.arange(qv.shape[0], device=logits.device))
