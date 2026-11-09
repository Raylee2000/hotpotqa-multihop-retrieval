"""Ranking stage of the public evaluation: score fixed candidates with R5_full and R0; labels enter metrics only.

Same definitions as the formal scorer (scripts/overnight_v3/rank.py, Matrix B ranking + the Matrix A R0 row):
  R5_full = lr * sum(tree predictions) on the artifact's 27 features (pickle-free r5_trees.npz, see
            src/rag_hotpotqa/ranking/r5_trees.py), R0 = the raw path score f_best on the same candidates.
Candidates are read with the original Data class of scripts/train_path_reranker.py (only the class is used).
Usage: python scripts/rank_public.py --base <work>/<part> --part <part>
Output (fail-if-exists): <base>/ranking/per_question.jsonl, <base>/ranking/methods.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_hotpotqa.ranking.path_rank import rank_one  # noqa: E402
from rag_hotpotqa.ranking.r5_trees import R5Trees  # noqa: E402
from rag_hotpotqa.release.integrity import model_dir, model_rel, verify_models  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--part", required=True)
    args = ap.parse_args()
    out = args.base / "ranking"
    if out.exists():
        raise SystemExit(f"{out} exists - refusing to overwrite")
    verify_models(["r5_full"])
    R5_DIR = model_dir("r5_full")
    spec = importlib.util.spec_from_file_location("original_rr", ROOT / "scripts/train_path_reranker.py")
    rr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rr)
    c = args.base / "candidates/general_csh"
    d = rr.Data(c / f"{args.part}.npz", c / f"{args.part}_meta.json")
    schema = json.loads((R5_DIR / "feature_schema.json").read_text(encoding="utf-8"))
    if schema["candidate_feature_order"] != d.feats:
        raise SystemExit("candidate feature order differs from the R5 feature schema")
    model = R5Trees.load(R5_DIR / "r5_trees.npz")
    if model.features != schema["features"]:
        raise SystemExit("R5 tree features differ from the feature schema")
    scores = {"R5_full": model.predict(d.cols(schema["features"])), "R0": d.cols(["f_best"])[:, 0]}
    rows = []
    for i, info in enumerate(d.meta["questions"]):
        if not info["two_gold"]:
            raise RuntimeError("Non-two-gold question requires explicit denominator policy")
        idx = d.groups[i]
        rows.append({"qid": info["qid"], "qtype": info["qtype"], "n_cand": len(idx),
                     "methods": {name: rank_one(d, idx, s) for name, s in scores.items()}})
    out.mkdir(parents=True)
    with open(out / "per_question.jsonl", "x", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + "\n")
    (out / "methods.json").write_text(json.dumps({"methods": list(scores), "r5": f"{model_rel('r5_full')}/r5_trees.npz"}, indent=1) + "\n",
                                      encoding="utf-8", newline="\n")
    print(f"PASS ranking {len(rows)}/{len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
