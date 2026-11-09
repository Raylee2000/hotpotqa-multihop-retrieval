"""Pickle-free inference for the frozen R5 LambdaMART path reranker.

The original artifact (``model.joblib``) is a dict ``{"lr": float, "trees": [sklearn DecisionTreeRegressor, ...]}``
and was scored as ``lr * sum(t.predict(Z) for t in trees)`` (scripts/overnight_v3/rank.py in the research repo).
For the public release each tree was exported once, in the maintainer's trusted environment, to plain numpy
arrays (``r5_trees.npz``, loaded with ``allow_pickle=False``). This module reproduces sklearn's prediction exactly:

* sklearn casts the input to float32 and sends a sample left when ``x[feature] <= threshold`` (threshold float64);
* a regression leaf returns ``value[node, 0, 0]``;
* tree predictions are accumulated in the same order as Python's ``sum`` (starting from 0) and then multiplied
  by the learning rate.

The converted file was verified against the original model before release (tools/maintainer/convert_r5_joblib.py;
the original joblib SHA256 is recorded in THIRD_PARTY_LICENSES.md).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FORMAT = "rag_hotpotqa.r5_trees.v1"


class R5Trees:
    def __init__(self, lr: float, left, right, feature, threshold, value, offsets, features: list[str]):
        self.lr = float(lr)
        self.left, self.right, self.feature = left, right, feature
        self.threshold, self.value, self.offsets = threshold, value, offsets
        self.features = list(features)

    @property
    def n_trees(self) -> int:
        return len(self.offsets) - 1

    @classmethod
    def load(cls, path: Path) -> "R5Trees":
        path = Path(path)
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta_json"]))
            if meta.get("format") != FORMAT:
                raise ValueError(f"{path}: unexpected format {meta.get('format')!r}")
            arrays = {k: z[k].copy() for k in ("left", "right", "feature", "threshold", "value", "offsets")}
        if arrays["left"].dtype.kind != "i" or arrays["threshold"].dtype != np.float64 or arrays["value"].dtype != np.float64:
            raise ValueError(f"{path}: unexpected array dtypes")
        return cls(meta["lr"], features=meta["features"], **arrays)

    def predict_tree(self, t: int, X32: np.ndarray) -> np.ndarray:
        a, b = int(self.offsets[t]), int(self.offsets[t + 1])
        left, right = self.left[a:b], self.right[a:b]
        feat, thr, val = self.feature[a:b], self.threshold[a:b], self.value[a:b]
        node = np.zeros(len(X32), dtype=np.int64)
        rows = np.arange(len(X32))
        while True:
            internal = left[node] != -1
            if not internal.any():
                return val[node]
            go_left = X32[rows, np.where(internal, feat[node], 0)] <= thr[node]
            node = np.where(internal, np.where(go_left, left[node], right[node]), node)

    def predict(self, Z: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z)
        if Z.ndim != 2 or Z.shape[1] != len(self.features):
            raise ValueError(f"expected {len(self.features)} feature columns, got shape {Z.shape}")
        if not np.isfinite(Z).all():
            raise ValueError("non-finite feature values")
        X32 = np.ascontiguousarray(Z, dtype=np.float32)
        acc = 0
        for t in range(self.n_trees):
            acc = acc + self.predict_tree(t, X32)
        return self.lr * acc
