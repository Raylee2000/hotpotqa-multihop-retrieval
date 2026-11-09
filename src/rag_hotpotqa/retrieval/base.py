"""Scorer interface and deterministic ranking."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class NonFiniteScoreError(ValueError):
    """A similarity score was NaN or +/-Inf."""


def check_finite(scores: np.ndarray, where: str = "") -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise NonFiniteScoreError(f"Non-finite similarity scores {where}: {scores[~np.isfinite(scores)][:5]}")
    return scores


def rank_order(scores: Sequence[float], tie_keys: Sequence | None = None) -> list[int]:
    """Positions sorted by score descending; ties broken by ``tie_keys`` ascending
    (default: original position, i.e. stable document index)."""
    s = check_finite(np.asarray(scores))
    keys = list(range(len(s))) if tie_keys is None else list(tie_keys)
    return sorted(range(len(s)), key=lambda i: (-s[i], keys[i]))


def ranks_from_scores(scores: Sequence[float], tie_keys: Sequence | None = None) -> list[int]:
    """1-based rank of each position under :func:`rank_order`."""
    order = rank_order(scores, tie_keys)
    ranks = [0] * len(order)
    for r, i in enumerate(order, start=1):
        ranks[i] = r
    return ranks


class Scorer(ABC):
    """s(X, Y): scores one query text against a list of candidate texts."""

    name: str = "scorer"

    def prefetch(self, texts: Sequence[str], role: str = "doc") -> None:  # noqa: B027
        """Optional: batch-precompute anything needed for these texts."""

    @abstractmethod
    def score(self, query: str, candidates: Sequence[str], local_corpus: Sequence[str] | None = None) -> np.ndarray:
        """``local_corpus`` = the question's full context article texts; used only
        by scorers whose statistics are per question (BM25-Local)."""

    def token_info(self, text: str, composed_from: tuple[str, str, str] | None = None) -> dict | None:
        """Encoder input accounting for ``text`` (None for non-neural scorers).
        ``composed_from`` = (kind, left, right) when ``text`` was produced by
        compose_q_plus_a (kind "q_plus_a") or compose_a_plus_b (kind "a_plus_b")."""
        return None

    def compose_q_plus_a(self, q: str, a: str) -> str:
        return compose(q, a)

    def compose_a_plus_b(self, a: str, b: str) -> str:
        return compose(a, b)

    def fingerprint(self) -> dict:
        return {"name": self.name}


DEFAULT_JOINER = "\n\n"


def compose(left: str, right: str, joiner: str = DEFAULT_JOINER) -> str:
    left, right = left.strip(), right.strip()
    if not left:
        return right
    if not right:
        return left
    return f"{left}{joiner}{right}"
