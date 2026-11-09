"""Train the R5_full path reranker (LambdaMART, feature set `full`) on fixed candidate tables.

Original logic, not re-implemented: Data, evaluate and fit_lambdamart are imported from scripts/train_path_reranker.py
(copied unchanged from the research repository). The research run fitted R5 inside train_path_reranker.py with the
grid {(rounds, depth)} = (100, 3), (200, 4), (300, 3) and chose by V-select PR@1, ties by MRR (`pick`, the rule of
that script); with the formal candidates the choice was rounds=200, depth=4 (outputs/path_reranker_v1/general_csh).
This script runs the same R5 grid and rule for the `full` feature set only (R1–R4 and the no_title control are
research comparisons, not part of the Full method). `--quick` uses the original smoke grid (20, 3).

The chosen trees are written directly in the pickle-free release format (models/r5_full layout:
r5_trees.npz + feature_schema.json) after a bitwise numpy-vs-sklearn check on random inputs and on the V-select
candidates (rag_hotpotqa.training.r5_export). No joblib / pickle file is written.

Usage: python scripts/train_r5.py --cand-run general_csh --run-name r5_full [--quick]
Output (fail-if-exists): outputs/path_reranker_v1/<run-name>/{r5_trees.npz, feature_schema.json, result.json}
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_hotpotqa.training.r5_export import export_r5  # noqa: E402
from rag_hotpotqa.utils.config import load_settings, safe_path  # noqa: E402
from rag_hotpotqa.utils.io import atomic_write_json, sha256_file  # noqa: E402
from rag_hotpotqa.utils.runtime import git_commit, peak_rss_mb  # noqa: E402

ORIGINAL = ROOT / "scripts" / "train_path_reranker.py"
GRID = [(100, 3), (200, 4), (300, 3)]       # train_path_reranker.py main(): grids["lm"]
QUICK_GRID = [(20, 3)]                      # same script, --quick


def load_original():
    spec = importlib.util.spec_from_file_location("train_path_reranker_original", ORIGINAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    settings = load_settings()
    root = settings.project_root
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand-run", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--quick", action="store_true", help="original smoke grid (20 rounds, depth 3)")
    args = ap.parse_args()
    cdir = root / "outputs" / "path_candidates_v1" / args.cand_run
    out = safe_path(root / "outputs" / "path_reranker_v1" / args.run_name, root)
    if (out / "result.json").exists():
        raise SystemExit(f"{out} is complete — refusing to overwrite (use a new run name)")
    if (out / "r5_trees.npz").exists():
        raise SystemExit(f"{out} has a partial result — remove it or use a new run name")
    rr = load_original()
    tr = rr.Data(cdir / "train.npz", cdir / "train_meta.json")
    vs = rr.Data(cdir / "v_select.npz", cdir / "v_select_meta.json")
    if tr.feats != vs.feats:
        raise SystemExit("STOP: train / v_select feature lists differ")
    fs = tr.feats                               # feature set "full" (train_path_reranker.py: feats_full = tr.feats)
    raw_tr, raw_vs = tr.cols(fs), vs.cols(fs)   # R5 uses raw (unstandardised) features, as the original
    grid = QUICK_GRID if args.quick else GRID
    cands, fitted = [], {}
    t0 = time.perf_counter()
    for r, dp in grid:
        t = time.perf_counter()
        pred = rr.fit_lambdamart(tr, raw_tr, r, dp)
        ev = rr.evaluate(vs, pred(raw_vs))
        key = f"rounds={r},depth={dp}"
        cands.append((key, ev))
        fitted[key] = pred
        print(json.dumps({"hyper": key, "v_select_pr1": ev["pr1"], "rr": ev["rr"], "fit_s": round(time.perf_counter() - t, 1)}), flush=True)
    best_key, best = max(cands, key=lambda c: (c[1]["pr1"], c[1]["rr"]))   # pick(): V-select PR@1, ties MRR
    fit_s = time.perf_counter() - t0
    predictor = fitted[best_key]
    free = dict(zip(predictor.__code__.co_freevars, (c.cell_contents for c in predictor.__closure__)))
    trees, lr = free["trees"], free["lr"]
    out.mkdir(parents=True, exist_ok=True)
    report = export_r5(trees, lr, fs, out / "r5_trees.npz", [("v_select_candidates", raw_vs)],
                       meta_extra={"source": f"trained by scripts/train_r5.py on outputs/path_candidates_v1/{args.cand_run}",
                                   "hyper": best_key})
    schema = {"features": fs, "candidate_feature_order": vs.feats, "standardisation": None,
              "note": "R5 uses raw (unstandardised) features, as the original code"}
    (out / "feature_schema.json").write_text(json.dumps(schema, indent=1), encoding="utf-8")
    for _, ev in cands:
        ev.pop("_pr1_vector", None)
    res = {"method": "R5_lambdamart__full", "selection": "V-select PR@1, ties MRR (train_path_reranker.py pick)",
           "grid": {k: ev["pr1"] for k, ev in cands}, "hyper": best_key, "lr": lr, "n_trees": len(trees),
           "v_select": {k: best[k] for k in ("pr1", "pr3", "pr5", "rr", "candidate_path_recall", "n_two_gold")},
           "export_check": report, "cand_run": args.cand_run,
           "candidate_files_sha256": {n: sha256_file(cdir / n) for n in ("train.npz", "train_meta.json", "v_select.npz", "v_select_meta.json")},
           "train_ids_sha256": tr.meta["ids_sha256"], "n_train_questions": len(tr.meta["questions"]), "n_train_rows": int(len(tr.y)),
           "v_select_ids_sha256": vs.meta["ids_sha256"], "upstream": {k: tr.meta[k] for k in ("first_model", "first_weights_sha256",
                                                                                              "second_model", "second_weights_sha256",
                                                                                              "sf_run", "qtype_run")},
           "training_code_sha256": sha256_file(ORIGINAL), "fit_seconds_total": round(fit_s, 1),
           "peak_rss_mb": round(peak_rss_mb(), 1), "git_commit": git_commit(root), "utc": datetime.now(timezone.utc).isoformat()}
    atomic_write_json(out / "result.json", res, root)
    print(json.dumps({"chosen": best_key, "v_select_pr1": best["pr1"], "export": report["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
