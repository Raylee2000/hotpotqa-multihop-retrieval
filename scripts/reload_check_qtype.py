"""Independent-process reload check for a QType frozen-LR artifact written by export_qtype_artifact.py.

Port of the research scripts/artifact_registry/reload_check_qtype.py (tolerances unchanged):
  (a) reload vs in-memory probabilities of the fitting process: max |diff| <= 1e-6
  (b) reload vs the qtype run's prediction files (pred_v_select.jsonl, pred_train_crossfit.jsonl; 6 decimals):
      max |diff| <= 1e-5 and 0 label flips at 0.5 — every qid must be present (no default 0.5)
COMPLETE.json is written only if (a) and (b) pass (rag_hotpotqa.utils.artifacts.mark_complete).
Usage: python scripts/reload_check_qtype.py <artifact_dir> --qtype-run RUN [--n-train N] [--n-val N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_hotpotqa.data.access import load_split_records  # noqa: E402
from rag_hotpotqa.training.folds import fold_of  # noqa: E402
from rag_hotpotqa.utils.artifacts import ArtifactError, compare_predictions, mark_complete, verify_manifest  # noqa: E402
from rag_hotpotqa.utils.config import load_settings  # noqa: E402

TOL_INMEM, TOL_ORIG = 1e-6, 1e-5


def p_of(npz, e):
    return 1 / (1 + np.exp(e @ npz["coef"][0] + npz["intercept"][0]))   # predict_proba[:, 0] for classes [0, 1]


def stats(p, ref):
    d = np.abs(p - ref)
    return {"n": int(len(d)), "mean_abs_diff": float(d.mean()), "max_abs_diff": float(d.max()), "n_above_tol": int((d > TOL_ORIG).sum()),
            "label_flips": int(((p >= 0.5) != (ref >= 0.5)).sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact", type=Path)
    ap.add_argument("--qtype-run", required=True)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-val", type=int, default=None)
    args = ap.parse_args()
    art = args.artifact.resolve()
    out = {"status": "FAIL", "tolerances": {"reload_vs_inmemory": TOL_INMEM, "reload_vs_original": TOL_ORIG}}
    try:
        man = verify_manifest(art)
        schema = json.loads((art / man["files"]["feature_schema"]["path"]).read_text(encoding="utf-8"))
        es = schema["encoder"]
        st = load_settings()  # project-local HF cache before transformers is imported
        sys.path.insert(0, str(ROOT / "scripts"))
        from export_qtype_artifact import qtype_ids

        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(es["torch_threads"])
        tr_ids, va_ids = qtype_ids(st, args.n_train, args.n_val)
        tr = load_split_records(st, "training", "train", tr_ids)
        va = load_split_records(st, "validation", "select", va_ids)
        m = SentenceTransformer(es["model"], revision=es["revision"], device="cpu")
        m.max_seq_length = es["max_seq_length"]
        e_tr = m.encode([r["question"] for r in tr], batch_size=es["batch_size"], normalize_embeddings=es["normalize_embeddings"])
        e_va = m.encode([r["question"] for r in va], batch_size=es["batch_size"], normalize_embeddings=es["normalize_embeddings"])
        full = np.load(art / man["files"]["lr_full"]["path"], allow_pickle=False)
        fold = {f: np.load(art / man["files"][f"lr_fold{f}"]["path"], allow_pickle=False) for f in (0, 1)}
        p_va = p_of(full, e_va)
        folds = np.array([fold_of(r["_id"]) for r in tr])
        p_cf = np.zeros(len(tr))
        for f in (0, 1):
            p_cf[folds == f] = p_of(fold[f], e_tr[folds == f])
        out["reload_vs_inmemory"] = {
            "v_select_max_abs_diff": compare_predictions(np.load(art / man["files"]["inmemory_v_select"]["path"]), p_va, TOL_INMEM),
            "train_crossfit_max_abs_diff": compare_predictions(np.load(art / man["files"]["inmemory_train_crossfit"]["path"]), p_cf, TOL_INMEM)}
        run = ROOT / "outputs" / "qtype_v1" / args.qtype_run
        ov = {json.loads(x)["qid"]: json.loads(x)["p_bridge"] for x in open(run / "pred_v_select.jsonl", encoding="utf-8")}
        oc = {json.loads(x)["qid"]: json.loads(x)["p_bridge"] for x in open(run / "pred_train_crossfit.jsonl", encoding="utf-8")}
        miss = [r["_id"] for r in va if r["_id"] not in ov] + [r["_id"] for r in tr if r["_id"] not in oc]
        if miss:
            raise ArtifactError(f"{len(miss)} qids missing in the qtype run's prediction files (no default 0.5 is used)")
        sv = stats(p_va, np.array([ov[r["_id"]] for r in va]))
        sc = stats(p_cf, np.array([oc[r["_id"]] for r in tr]))
        out[f"reload_vs_qtype_run_{args.qtype_run}"] = {"v_select": sv, "train_crossfit": sc}
        ok = all(s["max_abs_diff"] <= TOL_ORIG and s["label_flips"] == 0 for s in (sv, sc))
        out["status"] = "PASS" if ok else "FAIL"
        if not ok:
            out["error"] = "reload does not reproduce the qtype run's probabilities within the fixed tolerance"
    except ArtifactError as e:
        out["error"] = str(e)
    (art / "reload_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    if out["status"] == "PASS":
        mark_complete(art, out)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
