"""Fixed evaluation-only holdouts drawn from the official Train questions not assigned to any split.

The 50,000 assigned IDs (Exploration / Training / Validation) and the frozen split manifest are never changed.
A holdout has its own manifest under data/manifests/ holding the full ID list and its SHA256; it may be read only
with purpose ``holdout_eval`` (evaluation only: no training, no hyper-parameter or threshold selection).

Official Dev is never parsed here: its IDs are taken from the first characters of each canonical line (records are
written with sorted keys, so ``_id`` is always the first field); no other field is decoded.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from ..utils.config import Settings
from .access import SplitIntegrityError, SplitPolicyError
from .splits import ids_hash

HOLDOUT_PURPOSE = "holdout_eval"
_ID_PREFIX = re.compile(r'^\{"_id": "([^"]+)"')


def ids_from_line_prefix(path: Path | str) -> list[str]:
    """IDs of a canonical JSONL file read from the first 64 characters of each line only."""
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            m = _ID_PREFIX.match(line[:64])
            if not m:
                raise SplitIntegrityError(f"{path}: line does not start with the _id field")
            out.append(m.group(1))
    return out


def largest_remainder(total: int, weights: dict[str, int]) -> dict[str, int]:
    s = sum(weights.values())
    raw = {k: total * v / s for k, v in weights.items()}
    q = {k: int(np.floor(v)) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - q[k]), k))[: total - sum(q.values())]:
        q[k] += 1
    return q


def build_balanced_holdout(meta: dict[str, tuple[str, str]], assigned: set[str], seed: int = 42,
                           minority: str = "comparison", majority: str = "bridge") -> dict:
    """meta: id -> (type, level) for every official Train question. All unassigned ``minority`` questions are taken
    (N); N ``majority`` questions are drawn without replacement, with per-level quotas equal to the minority's
    per-level counts (so the two classes are 1:1 within every level). Within a level, the sorted candidate IDs are
    permuted with numpy PCG64(seed) and the first quota are taken. Deterministic and input-order independent."""
    free = sorted(i for i in meta if i not in assigned)
    mino = [i for i in free if meta[i][0] == minority]
    pool: dict[str, list[str]] = {}
    for i in free:
        if meta[i][0] == majority:
            pool.setdefault(meta[i][1], []).append(i)
    quota: dict[str, int] = {}
    for i in mino:
        quota[meta[i][1]] = quota.get(meta[i][1], 0) + 1
    short = {lv: (q, len(pool.get(lv, []))) for lv, q in quota.items() if len(pool.get(lv, [])) < q}
    if short:
        raise ValueError(f"majority pool too small for level quotas: {short}")
    maj = []
    for lv in sorted(quota):
        cand = sorted(pool[lv])
        perm = np.random.Generator(np.random.PCG64(seed)).permutation(len(cand))
        maj.extend(cand[j] for j in perm[: quota[lv]])
    return {"n_unassigned": len(free), "minority_ids": sorted(mino), "majority_ids": sorted(maj),
            "level_quotas": dict(sorted(quota.items())),
            "majority_pool_by_level": {lv: len(v) for lv, v in sorted(pool.items())}}


def load_holdout_ids(settings: Settings, name: str, purpose: str) -> list[str]:
    if purpose != HOLDOUT_PURPOSE:
        raise SplitPolicyError(f"holdout {name!r} may only be read with purpose {HOLDOUT_PURPOSE!r}, got {purpose!r}")
    man = json.loads((settings.project_root / "data" / "manifests" / f"{name}.json").read_text(encoding="utf-8"))
    ids = man["ids"]
    if ids_hash(ids) != man["ids_sha256"]:
        raise SplitIntegrityError(f"{name}: ID hash does not match its manifest")
    return ids
