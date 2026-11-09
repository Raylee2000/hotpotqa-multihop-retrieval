"""Deterministic K-fold assignment of Training questions for cross-fitted predictions.

A model that produces "predicted" inputs for Training question q (SF sentences, question type, ...) is trained
only on the other folds, so predictions on Training are as noisy as on unseen questions.
"""
from __future__ import annotations

import hashlib


def fold_of(qid: str, k: int = 2) -> int:
    return int(hashlib.sha256(f"fold:{qid}".encode()).hexdigest()[:8], 16) % k
