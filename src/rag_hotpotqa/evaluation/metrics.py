"""Metric primitives (spec v2). The definitions below are normative (reported PR@1: README.md §08).

Three metric families, never mixed in one table:
- first-hop article retrieval  (ranking of all context articles by s(Q, D))
- Oracle second-hop retrieval   (ranking of the B candidates given a gold A)
- two-hop path retrieval        (ranking of directed paths (A, B))

Conventions
- Ranks are 1-based. ``None`` rank = not in the ranked list (Recall 0, RR 0).
- A path is an ordered pair (A, B), A != B. Path correctness is the
  unordered match {A, B} == gold set; (A, B) and (B, A) remain distinct
  entries of the ranking and are never merged before ranking.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence

Path = tuple[int, int]
KS = (1, 2, 3, 5)


def hit_at_k(rank: int | None, k: int | None) -> int:
    """1 if rank <= k. ``k=None`` means any finite rank."""
    if rank is None:
        return 0
    return int(k is None or rank <= k)


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def rank_of(item: int, order: Sequence[int]) -> int | None:
    try:
        return list(order).index(item) + 1
    except ValueError:
        return None


def first_hop_metrics(order: Sequence[int], gold: Sequence[int], ks: Iterable[int] = KS) -> dict[str, Any]:
    """order: all context indices ranked by s(Q, D)."""
    ranks = [rank_of(g, order) for g in gold]
    finite = [r for r in ranks if r is not None]
    best = min(finite) if finite else None
    out: dict[str, Any] = {"gold_ranks": ranks, "best_gold_rank": best, "mrr": reciprocal_rank(best)}
    for k in ks:
        out[f"any_gold@{k}"] = hit_at_k(best, k)
        if k >= len(gold):
            out[f"all_gold@{k}"] = int(all(hit_at_k(r, k) for r in ranks))
    return out


def second_hop_metrics(order: Sequence[int], gold_b: int, ks: Iterable[int] = KS) -> dict[str, Any]:
    """order: candidate B indices (A excluded) ranked by a second-hop score."""
    r = rank_of(gold_b, order)
    out: dict[str, Any] = {"rank": r, "rr": reciprocal_rank(r), "pool_size": len(order)}
    for k in ks:
        out[f"recall@{k}"] = hit_at_k(r, k)
    return out


def rank_paths(path_scores: dict[Path, float]) -> list[Path]:
    """Sort directed paths by score desc; ties by (A index, B index) asc. A == B never allowed."""
    for (a, b), s in path_scores.items():
        if a == b:
            raise ValueError(f"path with A == B: {(a, b)}")
        if s != s or s in (float("inf"), float("-inf")):
            raise ValueError(f"non-finite path score for {(a, b)}")
    return sorted(path_scores, key=lambda p: (-path_scores[p], p[0], p[1]))


def path_metrics(ranked: Sequence[Path], gold: Sequence[int], ks: Iterable[int] = KS) -> dict[str, Any]:
    gold_set = set(gold)
    r = next((i + 1 for i, p in enumerate(ranked) if set(p) == gold_set), None)
    out: dict[str, Any] = {"n_paths": len(ranked), "gold_path_rank": r, "gold_path_rr": reciprocal_rank(r),
                           "gold_path_coverage": int(r is not None)}
    for k in ks:
        out[f"path_recall@{k}"] = hit_at_k(r, k)
    return out


def aggregate(rows: Sequence[dict[str, Any]], metric_keys: Sequence[str], group_key: str = "qtype",
              metrics_field: str | None = None) -> dict[str, Any]:
    """Mean of each metric over rows that define it, overall and per group.
    ``key__n`` reports the denominator whenever it differs from the group size."""

    def get(r):
        return r[metrics_field] if metrics_field else r

    def _mean(rs: Sequence[dict[str, Any]]) -> dict[str, Any]:
        res: dict[str, Any] = {"n": len(rs)}
        for m in metric_keys:
            vals = [get(r)[m] for r in rs if get(r).get(m) is not None]
            res[m] = round(sum(vals) / len(vals), 4) if vals else None
            if len(vals) != len(rs):
                res[f"{m}__n"] = len(vals)
        return res

    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        groups[str(r.get(group_key))].append(r)
    return {"overall": _mean(rows), **{g: _mean(v) for g, v in sorted(groups.items())}}
