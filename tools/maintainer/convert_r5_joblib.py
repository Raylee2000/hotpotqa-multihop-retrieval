"""MAINTAINER-ONLY, one-time conversion of the frozen R5 joblib artifact to a pickle-free .npz.

This script is NOT part of the public workflow and is never called by run.py. It loads a joblib (pickle) file,
which executes code, so it must only be run by the maintainer on the artifact they produced themselves:

    python tools/maintainer/convert_r5_joblib.py --i-trust-this-file <artifact_dir> <out.npz>

<artifact_dir> is the original COMPLETE artifact directory (model.joblib, feature_schema.json,
inmemory_scores_v_select.npy). Verification (all must hold, otherwise nothing is written):
  1. every tree is a single-output sklearn DecisionTreeRegressor;
  2. on random float inputs and on the original V-select feature matrix (if given), the numpy implementation
     equals sklearn exactly, tree by tree and in total;
  3. the total equals the stored in-memory V-select scores bit for bit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from rag_hotpotqa.ranking.r5_trees import FORMAT, R5Trees  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-trust-this-file", action="store_true", required=True)
    ap.add_argument("artifact", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--v-select-npz", type=Path, help="original candidates v_select.npz (feature matrix X)")
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists")
    import joblib
    from sklearn.tree import DecisionTreeRegressor

    model = joblib.load(args.artifact / "model.joblib")
    schema = json.loads((args.artifact / "feature_schema.json").read_text(encoding="utf-8"))
    assert set(model) == {"lr", "trees"}, sorted(model)
    parts = {k: [] for k in ("left", "right", "feature", "threshold", "value")}
    offsets = [0]
    for t in model["trees"]:
        assert type(t) is DecisionTreeRegressor and t.n_outputs_ == 1
        tr = t.tree_
        assert tr.value.shape[1:] == (1, 1)
        parts["left"].append(tr.children_left.astype(np.int64))
        parts["right"].append(tr.children_right.astype(np.int64))
        parts["feature"].append(tr.feature.astype(np.int64))
        parts["threshold"].append(tr.threshold.astype(np.float64))
        parts["value"].append(tr.value[:, 0, 0].astype(np.float64))
        offsets.append(offsets[-1] + tr.node_count)
    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    meta = {"format": FORMAT, "lr": float(model["lr"]), "features": schema["features"], "n_trees": len(model["trees"]),
            "source_model_sha256": hashlib.sha256((args.artifact / "model.joblib").read_bytes()).hexdigest()}
    conv = R5Trees(meta["lr"], features=meta["features"], offsets=np.asarray(offsets, dtype=np.int64), **arrays)

    def check(Z, label):
        X32 = np.ascontiguousarray(Z, dtype=np.float32)
        for i, t in enumerate(model["trees"]):
            if not np.array_equal(conv.predict_tree(i, X32), t.predict(Z)):
                raise SystemExit(f"tree {i} differs on {label}")
        ref = model["lr"] * sum(t.predict(Z) for t in model["trees"])
        if not np.array_equal(conv.predict(Z), ref):
            raise SystemExit(f"total differs on {label}")
        return ref

    rng = np.random.default_rng(0)
    check(rng.normal(size=(5000, len(meta["features"]))), "random inputs")
    report = {"random_inputs": 5000}
    if args.v_select_npz:
        with np.load(args.v_select_npz, allow_pickle=False) as z:
            X = z["X"]
        meta_path = args.v_select_npz.with_name(args.v_select_npz.stem + "_meta.json")
        cand_order = json.loads(meta_path.read_text(encoding="utf-8"))["features"]
        if cand_order != schema["candidate_feature_order"]:
            raise SystemExit("candidate feature order differs from the artifact schema")
        Z = X[:, [cand_order.index(f) for f in schema["features"]]]
        ref = check(Z, "V-select candidates")
        stored = np.load(args.artifact / "inmemory_scores_v_select.npy", allow_pickle=False)
        if not np.array_equal(ref, stored) or not np.array_equal(conv.predict(Z), stored):
            raise SystemExit("differs from the stored in-memory V-select scores")
        report["v_select_rows"] = int(len(Z))
    np.savez(args.out, meta_json=np.asarray(json.dumps(meta, sort_keys=True)), offsets=np.asarray(offsets, dtype=np.int64), **arrays)
    back = R5Trees.load(args.out)
    if args.v_select_npz and not np.array_equal(back.predict(Z), stored):
        raise SystemExit("reloaded file differs")
    print(json.dumps({"PASS": True, **report, **meta, "out_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
