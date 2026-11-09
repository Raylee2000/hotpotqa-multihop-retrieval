"""Re-implementation of the metrics printed by MDR's scripts/eval/eval_mhop_retrieval.py.

Source read on 2026-10-31 (facebookresearch/multihop_dense_retrieval, main). Per
question, with ``sp`` = the two gold titles and the top-``topk`` predicted chains:

    retrieved_titles = union of hop-1 and hop-2 titles over all topk chains
    hop1_titles      = union of hop-1 titles over all topk chains
    PR          = 1 if ANY gold title in retrieved_titles
    P-EM        = 1 if ALL gold titles in retrieved_titles
    1-Recall    = 1 if ANY gold title in hop1_titles
    Path Recall = 1 if ANY chain p has set(p) == set(sp)

Matching is by title string; chains are order-insensitive; MDR does not drop
chains with hop1 == hop2. These are reproduced exactly here so our predictions
can be scored with MDR's definitions (and MDR predictions with ours).
"""
from __future__ import annotations

from typing import Sequence

TitlePath = tuple[str, str]


def mdr_question_metrics(chains: Sequence[TitlePath], sp: Sequence[str]) -> dict[str, int]:
    if len(set(sp)) != 2:
        raise ValueError("MDR evaluator asserts exactly two distinct gold titles")
    retrieved = {t for c in chains for t in c}
    hop1 = {c[0] for c in chains}
    covered = [t in retrieved for t in sp]
    return {
        "PR": int(any(covered)),
        "P-EM": int(all(covered)),
        "1-Recall": int(any(t in hop1 for t in sp)),
        "Path Recall": int(any(set(c) == set(sp) for c in chains)),
    }


def mdr_aggregate(per_question: Sequence[dict[str, int]]) -> dict[str, float]:
    n = len(per_question)
    keys = ("PR", "P-EM", "1-Recall", "Path Recall")
    return {k: (sum(q[k] for q in per_question) / n if n else 0.0) for k in keys} | {"n": n}


def paths_to_titles(paths: Sequence[tuple[int, int]], titles: Sequence[str]) -> list[TitlePath]:
    return [(titles[a], titles[b]) for a, b in paths]
