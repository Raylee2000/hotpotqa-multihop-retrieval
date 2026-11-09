"""Reciprocal Rank Fusion over the same candidate list."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import ranks_from_scores


def rrf_scores(component_scores: Sequence[Sequence[float]], k: int = 60,
               tie_keys: Sequence | None = None) -> np.ndarray:
    """S(d) = sum_m 1 / (k + r_m(d)), ranks 1-based, ties within a component
    broken by ``tie_keys`` (default: candidate position = stable doc index)."""
    n = len(component_scores[0])
    if any(len(s) != n for s in component_scores):
        raise ValueError("all components must score the same candidates")
    out = np.zeros(n, dtype=np.float64)
    for s in component_scores:
        ranks = ranks_from_scores(s, tie_keys)
        out += np.array([1.0 / (k + r) for r in ranks])
    return out
