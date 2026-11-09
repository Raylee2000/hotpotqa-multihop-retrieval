"""Deterministic, stratified, ID-based Train splits.

Algorithm (seed-only dependent, independent of file row order):
1. Sort all official IDs within each stratum (type x level).
2. Permute each stratum with numpy PCG64(seed).
3. Allocate each split's per-stratum quota by largest remainder so the split
   size is exact and stratum proportions follow the full Train.
4. Take Exploration, then Training, then Validation from the front of each
   permuted stratum, so splits are disjoint by construction.
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np

SPLIT_ORDER = ("exploration", "training", "validation")


def stratum_of(rec: dict[str, Any], keys: Iterable[str]) -> str:
    return "|".join(str(rec[k]) for k in keys)


def ids_hash(ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def _quotas(total: int, stratum_sizes: dict[str, int], pop: int) -> dict[str, int]:
    raw = {s: total * n / pop for s, n in stratum_sizes.items()}
    base = {s: int(np.floor(v)) for s, v in raw.items()}
    remaining = total - sum(base.values())
    # Largest remainder; ties broken by stratum name for determinism.
    order = sorted(raw, key=lambda s: (-(raw[s] - base[s]), s))
    for s in order[:remaining]:
        base[s] += 1
    return base


def make_splits(records: list[dict[str, Any]], sizes: dict[str, int], seed: int,
                strata_keys: tuple[str, ...] = ("type", "level")) -> dict[str, list[str]]:
    ids = [r["_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question IDs in source data")
    if sum(sizes.values()) > len(ids):
        raise ValueError("Requested split sizes exceed available questions")
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for r in records:
        by_stratum[stratum_of(r, strata_keys)].append(r["_id"])
    rng = np.random.Generator(np.random.PCG64(seed))
    permuted = {}
    for s in sorted(by_stratum):
        members = sorted(by_stratum[s])
        permuted[s] = [members[i] for i in rng.permutation(len(members))]
    stratum_sizes = {s: len(v) for s, v in permuted.items()}
    cursor = {s: 0 for s in permuted}
    out: dict[str, list[str]] = {}
    for name in SPLIT_ORDER:
        if name not in sizes:
            continue
        q = _quotas(sizes[name], stratum_sizes, len(ids))
        chosen: list[str] = []
        for s in sorted(permuted):
            take = permuted[s][cursor[s]: cursor[s] + q[s]]
            if len(take) < q[s]:
                raise ValueError(f"Stratum {s} exhausted while building {name}")
            cursor[s] += q[s]
            chosen.extend(take)
        out[name] = sorted(chosen)
    return out


def validate_splits(splits: dict[str, list[str]], expected_sizes: dict[str, int],
                    forbidden: dict[str, set[str]] | None = None) -> dict[str, Any]:
    """Raise on any integrity violation; return a summary of checks."""
    checks: dict[str, Any] = {}
    for name, ids in splits.items():
        if len(ids) != len(set(ids)):
            raise ValueError(f"{name}: duplicate IDs")
        if len(ids) != expected_sizes[name]:
            raise ValueError(f"{name}: size {len(ids)} != {expected_sizes[name]}")
        checks[f"{name}_size"] = len(ids)
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = set(splits[a]) & set(splits[b])
            if inter:
                raise ValueError(f"{a} and {b} overlap on {len(inter)} IDs")
            checks[f"overlap_{a}_{b}"] = 0
    for fname, fids in (forbidden or {}).items():
        for name, ids in splits.items():
            inter = set(ids) & fids
            if inter:
                raise ValueError(f"{name} overlaps {fname} on {len(inter)} IDs")
            checks[f"overlap_{name}_{fname}"] = 0
    return checks


def distribution(records_by_id: dict[str, dict[str, Any]], ids: Iterable[str], key: str) -> dict[str, float]:
    c = Counter(records_by_id[i][key] for i in ids)
    n = sum(c.values())
    return {k: round(v / n, 4) for k, v in sorted(c.items())}
