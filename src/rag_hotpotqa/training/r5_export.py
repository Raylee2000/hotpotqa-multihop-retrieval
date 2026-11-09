"""Write freshly trained R5 LambdaMART trees straight to the pickle-free ``r5_trees.npz`` format.

The research run kept the fitted sklearn trees in a joblib file and the release maintainer converted them once
(tools/maintainer/convert_r5_joblib.py). A training run from this repository never writes a pickle: the trees
returned by ``fit_lambdamart`` (scripts/train_path_reranker.py, unchanged) are exported here with the same array
layout, and the export is accepted only if the numpy implementation (rag_hotpotqa.ranking.r5_trees.R5Trees, the
class the public ranking stage loads) equals sklearn exactly — tree by tree and in total — on every check matrix,
and again after reloading the written file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..ranking.r5_trees import FORMAT, R5Trees


def trees_to_arrays(trees: Sequence) -> tuple[dict, np.ndarray]:
    from sklearn.tree import DecisionTreeRegressor

    parts = {k: [] for k in ("left", "right", "feature", "threshold", "value")}
    offsets = [0]
    for t in trees:
        if type(t) is not DecisionTreeRegressor or t.n_outputs_ != 1 or t.tree_.value.shape[1:] != (1, 1):
            raise ValueError("expected single-output sklearn DecisionTreeRegressor trees")
        tr = t.tree_
        parts["left"].append(tr.children_left.astype(np.int64))
        parts["right"].append(tr.children_right.astype(np.int64))
        parts["feature"].append(tr.feature.astype(np.int64))
        parts["threshold"].append(tr.threshold.astype(np.float64))
        parts["value"].append(tr.value[:, 0, 0].astype(np.float64))
        offsets.append(offsets[-1] + tr.node_count)
    return {k: np.concatenate(v) for k, v in parts.items()}, np.asarray(offsets, dtype=np.int64)


def _check(conv: R5Trees, trees: Sequence, lr: float, Z: np.ndarray, label: str) -> np.ndarray:
    X32 = np.ascontiguousarray(Z, dtype=np.float32)
    for i, t in enumerate(trees):
        if not np.array_equal(conv.predict_tree(i, X32), t.predict(Z)):
            raise ValueError(f"tree {i} differs from sklearn on {label}")
    ref = lr * sum(t.predict(Z) for t in trees)
    if not np.array_equal(conv.predict(Z), ref):
        raise ValueError(f"total score differs from sklearn on {label}")
    return ref


def export_r5(trees: Sequence, lr: float, features: list[str], out: Path, checks: Iterable[tuple[str, np.ndarray]],
              meta_extra: dict | None = None, n_random: int = 5000) -> dict:
    """Write ``out`` (fail if it exists) after the exactness checks; returns a verification report."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists — refusing to overwrite")
    arrays, offsets = trees_to_arrays(trees)
    meta = {"format": FORMAT, "lr": float(lr), "features": list(features), "n_trees": len(trees), **(meta_extra or {})}
    conv = R5Trees(meta["lr"], features=meta["features"], offsets=offsets, **arrays)
    report = {"random_inputs": n_random}
    _check(conv, trees, lr, np.random.default_rng(0).normal(size=(n_random, len(features))), "random inputs")
    refs = {}
    for label, Z in checks:
        refs[label] = _check(conv, trees, lr, Z, label)
        report[f"rows_{label}"] = int(len(Z))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, meta_json=np.asarray(json.dumps(meta, sort_keys=True)), offsets=offsets, **arrays)
    back = R5Trees.load(out)
    for label, Z in checks:
        if not np.array_equal(back.predict(Z), refs[label]):
            raise ValueError(f"reloaded {out.name} differs on {label}")
    report["status"] = "PASS"
    report["check"] = "numpy R5Trees == sklearn, tree by tree and in total (bitwise), before and after reload"
    return report
