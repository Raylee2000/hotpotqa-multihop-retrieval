"""Save the QType frozen-LR classifier of a train_qtype_classifier.py run as a COMPLETE model artifact.

Port of the research script scripts/artifact_registry/refit_qtype_frozen_lr.py, which produced the formal
models/qtype_frozen_lr (the original qtype_v2 run never saved its LR weights, so the research refitted them under the
identical conditions). Same conditions here: torch threads 8, set_seed(42), pinned MiniLM with max_seq_length from
configs/retrieval.yaml, encode(batch_size=128, normalize_embeddings=True), Training rows in sha256 order
(select_subset), V-select rows as in train_qtype_classifier.py, LogisticRegression(C, max_iter=2000,
random_state=42), P(bridge) = predict_proba[:, 0], fold-f model fitted on fold != f. No search: C is read from the
qtype run's result.json (chosen there by the pre-registered grid {0.1, 1, 10} on V-select macro-F1), and the
Training / V-select id hashes must equal the qtype run's.
Differences from the research script: the qtype run, output directory and subset sizes are arguments (the research
script hard-coded qtype_v2 and asserted C == 10); the independent reload check compares with that qtype run's
prediction files (scripts/reload_check_qtype.py); the reranker-feature impact report on the old candidate tables is
not produced (those tables do not exist yet at this point of the pipeline).
Usage: python scripts/export_qtype_artifact.py --qtype-run qtype_v2 --out outputs/qtype_artifacts_v1/qtype_v2_frozen_lr
Output (fail-if-exists): the --out directory, with the models/qtype_frozen_lr file set.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_hotpotqa.data.access import load_split_ids, load_split_records, load_validation_part  # noqa: E402
from rag_hotpotqa.data.splits import ids_hash  # noqa: E402
from rag_hotpotqa.experiments.runner import select_subset  # noqa: E402
from rag_hotpotqa.retrieval.factory import check_hf_cache_inside_project, verify_dense_weights  # noqa: E402
from rag_hotpotqa.training.folds import fold_of  # noqa: E402
from rag_hotpotqa.utils.artifacts import write_manifest  # noqa: E402
from rag_hotpotqa.utils.config import load_config, load_settings, safe_path  # noqa: E402
from rag_hotpotqa.utils.io import sha256_file  # noqa: E402
from rag_hotpotqa.utils.runtime import git_commit, peak_rss_mb, set_seed  # noqa: E402

LABELS = ["bridge", "comparison"]
THREADS = 8


def save_lr(path: Path, clf, C):
    np.savez(path, coef=clf.coef_, intercept=clf.intercept_, classes=clf.classes_, C=np.array(C), n_iter=clf.n_iter_)


def qtype_ids(st, n_train, n_val):
    """The id selection of train_qtype_classifier.py (identical expressions)."""
    tr_ids = select_subset(load_split_ids(st, "training", "train"), n_train)
    va_ids = (load_validation_part(st, "v_select") if n_val is None
              else select_subset(load_split_ids(st, "validation", "select"), n_val))
    return tr_ids, va_ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qtype-run", required=True, help="outputs/qtype_v1/<run> written by train_qtype_classifier.py")
    ap.add_argument("--out", required=True, help="artifact directory (repository-relative)")
    ap.add_argument("--n-train", type=int, default=None, help="same value as the qtype run (smoke)")
    ap.add_argument("--n-val", type=int, default=None, help="same value as the qtype run (smoke)")
    args = ap.parse_args()
    st = load_settings()  # first: sets the project-local HF cache before transformers is imported
    root = st.project_root
    out = safe_path(root / args.out, root)
    if out.exists():
        raise SystemExit(f"{out} exists — refusing to overwrite")
    run = root / "outputs" / "qtype_v1" / args.qtype_run
    src = json.loads((run / "result.json").read_text(encoding="utf-8"))
    C = src["frozen_linear"]["C"]
    import torch
    from sentence_transformers import SentenceTransformer
    from sklearn import __version__ as skv
    from sklearn.linear_model import LogisticRegression

    torch.set_num_threads(THREADS)
    set_seed(42)
    d = load_config("retrieval", st)["dense"]
    check_hf_cache_inside_project(st, True)
    w = verify_dense_weights(d["model"], d["revision"], d["weights_sha256"])
    tr_ids, va_ids = qtype_ids(st, args.n_train, args.n_val)
    if ids_hash(tr_ids) != src["train_ids_sha256"] or ids_hash(va_ids) != src["val_ids_sha256"]:
        raise SystemExit(f"STOP: Training / V-select ids differ from the {args.qtype_run} run (pass the same --n-train / --n-val)")
    tr = load_split_records(st, "training", "train", tr_ids)
    va = load_split_records(st, "validation", "select", va_ids)
    y_tr = [LABELS.index(r["type"]) for r in tr]
    model = SentenceTransformer(d["model"], revision=d["revision"], device="cpu")
    model.max_seq_length = d["max_seq_length"]
    t = time.perf_counter()
    e_tr = model.encode([r["question"] for r in tr], batch_size=128, normalize_embeddings=True)
    e_va = model.encode([r["question"] for r in va], batch_size=128, normalize_embeddings=True)
    enc_s = time.perf_counter() - t
    t = time.perf_counter()
    clf = LogisticRegression(C=C, max_iter=2000, random_state=42).fit(e_tr, y_tr)
    p_va = clf.predict_proba(e_va)[:, 0]
    folds = np.array([fold_of(r["_id"]) for r in tr])
    p_tr_cf = np.zeros(len(tr))
    fold_clfs = {}
    for f in (0, 1):
        fold_clfs[f] = LogisticRegression(C=C, max_iter=2000, random_state=42).fit(e_tr[folds != f], np.asarray(y_tr)[folds != f])
        p_tr_cf[folds == f] = fold_clfs[f].predict_proba(e_tr[folds == f])[:, 0]
    fit_s = time.perf_counter() - t
    out.mkdir(parents=True)
    save_lr(out / "lr_full.npz", clf, C)
    for f in (0, 1):
        save_lr(out / f"lr_fold{f}.npz", fold_clfs[f], C)
    np.save(out / "inmemory_p_bridge_v_select.npy", p_va)
    np.save(out / "inmemory_p_bridge_train_crossfit.npy", p_tr_cf)
    (out / "feature_schema.json").write_text(json.dumps({
        "input": "question text only", "encoder": {"model": d["model"], "revision": d["revision"], "weights_sha256": w["weights_sha256"],
                                                   "max_seq_length": d["max_seq_length"], "normalize_embeddings": True, "batch_size": 128,
                                                   "torch_threads": THREADS},
        "features": "384-d sentence embedding (pooling of the pinned SentenceTransformer)", "scaler": None,
        "classes": LABELS, "output": "P(bridge) = predict_proba[:, 0]",
        "row_order": {"v_select": "as train_qtype_classifier.py", "train_crossfit": "Training ids in sha256 order (select_subset)"},
        "folds": "training.folds.fold_of(qid); model fold f fitted on fold != f, predicts fold f"}, indent=1), encoding="utf-8")
    write_manifest(out, {"lr_full": out / "lr_full.npz", "lr_fold0": out / "lr_fold0.npz", "lr_fold1": out / "lr_fold1.npz",
                         "feature_schema": out / "feature_schema.json", "inmemory_v_select": out / "inmemory_p_bridge_v_select.npy",
                         "inmemory_train_crossfit": out / "inmemory_p_bridge_train_crossfit.npy"},
                   status=f"fitted under the conditions of the qtype run {args.qtype_run}; equivalence with that run's "
                          "prediction files is established by reload_check.json",
                   model_class="sklearn LogisticRegression on frozen MiniLM question embeddings",
                   hyperparameters={"C": C, "max_iter": 2000, "random_state": 42,
                                    "selection": f"C chosen in outputs/qtype_v1/{args.qtype_run} (grid 0.1/1/10, V-select macro-F1)"},
                   training_code={"replicates": "scripts/train_qtype_classifier.py (frozen_linear)",
                                  "file_sha256": sha256_file(root / "scripts/train_qtype_classifier.py")},
                   data={"train_ids_sha256": ids_hash(tr_ids), "v_select_ids_sha256": ids_hash(va_ids), "n_train": len(tr),
                         "n_v_select": len(va), "labels": "official HotpotQA type", "n_train_arg": args.n_train, "n_val_arg": args.n_val},
                   original={"run": f"outputs/qtype_v1/{args.qtype_run}", "provenance": src["provenance"],
                             "pred_v_select_sha256": sha256_file(run / "pred_v_select.jsonl"),
                             "pred_train_crossfit_sha256": sha256_file(run / "pred_train_crossfit.jsonl")},
                   environment={"python": sys.version.split()[0], "sklearn": skv, "numpy": np.__version__, "torch": torch.__version__},
                   provenance={"git_commit": git_commit(root), "command": " ".join(sys.argv), "utc": datetime.now(timezone.utc).isoformat()},
                   cost={"encode_seconds": round(enc_s, 1), "fit_seconds": round(fit_s, 1), "peak_rss_mb": round(peak_rss_mb(), 1)})
    print(json.dumps({"C": C, "encode_s": round(enc_s, 1), "fit_s": round(fit_s, 1)}))
    cmd = [sys.executable, "-X", "utf8", str(root / "scripts/reload_check_qtype.py"), str(out), "--qtype-run", args.qtype_run]
    for flag, v in (("--n-train", args.n_train), ("--n-val", args.n_val)):
        if v is not None:
            cmd += [flag, str(v)]
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
