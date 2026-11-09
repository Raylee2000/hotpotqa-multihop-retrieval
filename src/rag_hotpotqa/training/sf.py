"""Supporting-fact (sentence) selection: pair construction, cross-encoder scoring, metrics.

Labels are the OFFICIAL HotpotQA supporting facts (sentence level). The model is a cross-encoder over
"[CLS] question [SEP] sentence [SEP]" (mean pooling + linear head). This is sentence selection, not token-level
keyword extraction.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from ..data.documents import Question
from ..retrieval.base import rank_order
from ..retrieval.bm25 import BM25Scorer

KS = (1, 2, 3, 5)


def article_sentences(q: Question, idx: int) -> list[tuple[int, str]]:
    """(sentence id, stripped text) of the non-empty sentences of context article idx."""
    return [(i, s.strip()) for i, s in enumerate(q.docs[idx].sentences) if s.strip()]


def sf_ids(q: Question, idx: int) -> set[int]:
    return {s for t, s in q.supporting_facts if t == q.docs[idx].title}


def bm25_nongold(q: Question, bm25: BM25Scorer | None = None) -> list[int]:
    """Non-gold context articles ranked by BM25-Local against the question (label-free ranking)."""
    bm25 = bm25 or BM25Scorer("local")
    cands = [d for d in q.docs if d.index not in q.gold_indices]
    if not cands:
        return []
    sc = bm25.score(q.question, [d.text for d in cands], local_corpus=[d.text for d in q.docs])
    return [cands[i].index for i in rank_order(sc)]


def train_pairs(q: Question, bm25: BM25Scorer | None = None) -> list[tuple[str, str, int]]:
    """(question, sentence, label): every sentence of both gold articles, every sentence of the BM25-top non-gold
    article (hard negatives), and 2 sentences of other non-gold articles (sha256 choice)."""
    out = []
    for g in q.gold_indices:
        lab = sf_ids(q, g)
        out += [(q.question, t, int(i in lab)) for i, t in article_sentences(q, g)]
    ng = bm25_nongold(q, bm25)
    if ng:
        out += [(q.question, t, 0) for _, t in article_sentences(q, ng[0])]
        rest = [(d, i, t) for d in ng[1:] for i, t in article_sentences(q, d)]
        rest.sort(key=lambda r: hashlib.sha256(f"{q.qid}:{r[0]}:{r[1]}".encode()).hexdigest())
        out += [(q.question, t, 0) for _, _, t in rest[:2]]
    return out


def score_pairs(enc, head, tok, questions: Sequence[str], sents: Sequence[str], batch: int = 128) -> np.ndarray:
    """Sigmoid scores of a cross-encoder (enc: HF AutoModel, head: Linear(hidden, 1)); no grad."""
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(sents), batch):
            b = tok(list(questions[i:i + batch]), list(sents[i:i + batch]), truncation="longest_first", max_length=128,
                    padding=True, return_tensors="pt")
            h = enc(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            out.append(torch.sigmoid(head((h * m).sum(1) / m.sum(1)).squeeze(-1)).numpy())
    return np.concatenate(out) if out else np.zeros(0)


def ranking_metrics(items: Sequence[tuple[np.ndarray, np.ndarray]]) -> dict:
    """items: (labels, scores) per unit (question or article) with at least one positive."""
    out = {f"P@{k}": [] for k in KS} | {f"R@{k}": [] for k in KS} | {f"all_in_top{k}": [] for k in KS}
    for y, s in items:
        if y.sum() == 0:
            continue
        top = y[np.argsort(-s, kind="stable")]
        for k in KS:
            out[f"P@{k}"].append(top[:k].sum() / k)
            out[f"R@{k}"].append(top[:k].sum() / y.sum())
            out[f"all_in_top{k}"].append(int(top[:k].sum() == y.sum()))
    return {k: round(float(np.mean(v)), 4) for k, v in out.items() if v} | {"n_units": len(out["P@1"])}


def threshold_metrics(y, pred) -> dict:
    y, pred = np.asarray(y), np.asarray(pred)
    tp, fp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    pr, rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {"precision": round(pr, 4), "recall": round(rc, 4), "f1": round(2 * pr * rc / max(1e-9, pr + rc), 4),
            "n_pairs": int(len(y)), "positives": int(y.sum())}


def select_evidence(sentences: Sequence[tuple[int, str]], scores: np.ndarray, k: int) -> list[int]:
    """Sentence ids of the top-k scored sentences, returned in document order (label-free)."""
    top = np.argsort(-np.asarray(scores), kind="stable")[:k]
    return sorted(sentences[i][0] for i in top)
