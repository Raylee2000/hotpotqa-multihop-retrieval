"""[Public release copy] QType frozen MiniLM + logistic regression.

Copied from the research repository scripts/train_qtype_classifier.py; training logic, hyper-parameters, data selection,
seeds and checkpoint-selection rules are unchanged. Changes:
  - `git status` provenance via rag_hotpotqa.training.provenance.git_dirty (no failure without git)

Original docstring:
Question-type classification (bridge / comparison) from the question text only.

Labels: HotpotQA `type` field (the project has no separate "B/C" definition; B/C = bridge/comparison).
Two clearly separated models:
  frozen_linear  frozen pinned MiniLM sentence embedding + logistic regression (NOT encoder fine-tuning);
                 C chosen on Validation from {0.1, 1, 10} (pre-registered grid)
  finetune       MiniLM encoder + linear head, all weights trained (cross-entropy, 1 epoch, lr 2e-5, bs 32)
Training split: fitting / gradient updates. Validation: V-select (first 2,000, fixed partition) for C selection and
evaluation; V-report only with --final-report. Dev: never. Also exported: P(bridge) on V-select, and 2-fold
cross-fitted P(bridge) on Training (frozen+LR; each fold predicted by a model fitted on the other fold) for
downstream routing / reranker features. Calibration: Brier score and 10-bin ECE.
Usage: python scripts/train_qtype_classifier.py --run-name qtype_v1 [--n-train N] [--n-val N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_hotpotqa.data.access import load_split_ids, load_split_records, load_validation_part  # noqa: E402
from rag_hotpotqa.training.folds import fold_of  # noqa: E402
from rag_hotpotqa.data.splits import ids_hash  # noqa: E402
from rag_hotpotqa.experiments.runner import select_subset  # noqa: E402
from rag_hotpotqa.retrieval.factory import check_hf_cache_inside_project, verify_dense_weights  # noqa: E402
from rag_hotpotqa.utils.config import load_config, load_settings, safe_path  # noqa: E402
from rag_hotpotqa.training.provenance import git_dirty  # noqa: E402
from rag_hotpotqa.utils.io import atomic_write_json, sha256_file  # noqa: E402
from rag_hotpotqa.utils.runtime import get_logger, git_commit, peak_rss_mb, set_seed  # noqa: E402

LABELS = ("bridge", "comparison")


def metrics(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    cm = [[int(((y == i) & (p == j)).sum()) for j in range(2)] for i in range(2)]
    rec = [cm[i][i] / max(1, sum(cm[i])) for i in range(2)]
    prec = [cm[i][i] / max(1, cm[0][i] + cm[1][i]) for i in range(2)]
    f1 = [2 * a * b / (a + b) if a + b else 0.0 for a, b in zip(prec, rec)]
    return {"n": int(len(y)), "accuracy": round(float((y == p).mean()), 4), "macro_f1": round(float(np.mean(f1)), 4),
            "per_class_recall": dict(zip(LABELS, map(lambda v: round(v, 4), rec))),
            "per_class_precision": dict(zip(LABELS, map(lambda v: round(v, 4), prec))),
            "confusion_matrix_rows_true_cols_pred": {"labels": LABELS, "matrix": cm}}


def calibration(y, p_bridge) -> dict:
    """Brier score and 10-bin expected calibration error of P(bridge) (label 0 = bridge)."""
    y_b = (np.asarray(y) == 0).astype(float)
    p = np.asarray(p_bridge, float)
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(abs(y_b[bins == b].mean() - p[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
    return {"brier": round(float(((p - y_b) ** 2).mean()), 5), "ece_10bin": round(float(ece), 5)}


def main() -> int:
    settings = load_settings()
    root = settings.project_root
    rcfg = load_config("retrieval", settings)
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-val", type=int, default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--skip-finetune", action="store_true")
    ap.add_argument("--final-report", action="store_true", help="also evaluate V-report (only after all choices are frozen)")
    ap.add_argument("--eval-exploration", action="store_true",
                    help="also predict Exploration 10K with the frozen+LR model (inference only, after C is fixed; no choice uses it)")
    args = ap.parse_args()
    run_dir = safe_path(root / "outputs" / "qtype_v1" / args.run_name, root)
    run_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger(log_file=run_dir / "train.log")
    dirty = git_dirty(root)
    prov = {"started_utc": datetime.now(timezone.utc).isoformat(), "git_commit_at_start": git_commit(root),
            "dirty_files_at_start": dirty, "argv": sys.argv}

    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    torch.set_num_threads(args.threads)
    set_seed(42)
    d = rcfg["dense"]
    check_hf_cache_inside_project(settings, True)
    weights = verify_dense_weights(d["model"], d["revision"], d["weights_sha256"])
    tr_ids = select_subset(load_split_ids(settings, "training", "train"), args.n_train)
    va_ids = (load_validation_part(settings, "v_select") if args.n_val is None
              else select_subset(load_split_ids(settings, "validation", "select"), args.n_val))
    tr = load_split_records(settings, "training", "train", tr_ids)
    va = load_split_records(settings, "validation", "select", va_ids)
    xq_tr, y_tr = [r["question"] for r in tr], [LABELS.index(r["type"]) for r in tr]
    xq_va, y_va = [r["question"] for r in va], [LABELS.index(r["type"]) for r in va]
    res = {"provenance": prov, "train_ids_sha256": ids_hash(tr_ids), "val_ids_sha256": ids_hash(va_ids),
           "label_source": "HotpotQA `type` field", "class_balance_train": dict(zip(LABELS, np.bincount(y_tr).tolist())),
           "class_balance_val": dict(zip(LABELS, np.bincount(y_va).tolist())), "base_weights": weights}

    # ---- frozen encoder + logistic regression
    model = SentenceTransformer(d["model"], revision=d["revision"], device="cpu")
    model.max_seq_length = d["max_seq_length"]
    t = time.perf_counter()
    e_tr = model.encode(xq_tr, batch_size=128, normalize_embeddings=True)
    e_va = model.encode(xq_va, batch_size=128, normalize_embeddings=True)
    grid = {}
    for c in (0.1, 1.0, 10.0):
        clf = LogisticRegression(C=c, max_iter=2000, random_state=42).fit(e_tr, y_tr)
        grid[c] = (metrics(y_va, clf.predict(e_va)), clf)
    best_c = max(grid, key=lambda c: (grid[c][0]["macro_f1"], -c))
    res["frozen_linear"] = {"C_grid_val_macro_f1": {str(c): grid[c][0]["macro_f1"] for c in grid}, "C": best_c,
                            "train": metrics(y_tr, grid[best_c][1].predict(e_tr)), "validation": grid[best_c][0],
                            "seconds": round(time.perf_counter() - t, 1),
                            "note": "frozen encoder; only the logistic-regression head is fitted"}
    log.info("frozen_linear C=%s val %s", best_c, grid[best_c][0])
    np.save(run_dir / "frozen_val_pred.npy", grid[best_c][1].predict(e_va))
    p_va = grid[best_c][1].predict_proba(e_va)[:, 0]
    res["frozen_linear"]["validation_calibration"] = calibration(y_va, p_va)
    # 2-fold cross-fitted P(bridge) on Training with the selected C
    folds = np.array([fold_of(r["_id"]) for r in tr])
    p_tr_cf = np.zeros(len(tr))
    for f in (0, 1):
        clf = LogisticRegression(C=best_c, max_iter=2000, random_state=42).fit(e_tr[folds != f], np.asarray(y_tr)[folds != f])
        p_tr_cf[folds == f] = clf.predict_proba(e_tr[folds == f])[:, 0]
    res["frozen_linear"]["train_crossfit"] = {**metrics(y_tr, (p_tr_cf < 0.5).astype(int)), **calibration(y_tr, p_tr_cf)}
    with open(run_dir / "pred_train_crossfit.jsonl", "w", encoding="utf-8") as fh:
        for r, p in zip(tr, p_tr_cf):
            fh.write(json.dumps({"qid": r["_id"], "p_bridge": round(float(p), 6), "fold": fold_of(r["_id"])}) + "\n")
    with open(run_dir / "pred_v_select.jsonl", "w", encoding="utf-8") as fh:
        for r, p in zip(va, p_va):
            fh.write(json.dumps({"qid": r["_id"], "p_bridge": round(float(p), 6)}) + "\n")
    if args.final_report:
        rp = load_split_records(settings, "validation", "select", load_validation_part(settings, "v_report", True, "qtype final report"))
        e_rp = model.encode([r["question"] for r in rp], batch_size=128, normalize_embeddings=True)
        y_rp = [LABELS.index(r["type"]) for r in rp]
        res["frozen_linear"]["v_report"] = {**metrics(y_rp, grid[best_c][1].predict(e_rp)),
                                            **calibration(y_rp, grid[best_c][1].predict_proba(e_rp)[:, 0])}
    if args.eval_exploration:
        ex = load_split_records(settings, "exploration", "oer", load_split_ids(settings, "exploration", "oer"))
        e_ex = model.encode([r["question"] for r in ex], batch_size=128, normalize_embeddings=True)
        y_ex = np.array([LABELS.index(r["type"]) for r in ex])
        p_ex = grid[best_c][1].predict_proba(e_ex)[:, 0]
        yhat = (p_ex < 0.5).astype(int)
        rng = np.random.default_rng(42)
        boot = rng.integers(0, len(ex), (1000, len(ex)))
        ci = lambda f: [round(float(np.percentile([f(b) for b in boot], q)), 4) for q in (2.5, 97.5)]
        rec = lambda c: (lambda b: float((yhat[b][y_ex[b] == c] == c).mean()))
        res["frozen_linear"]["exploration_heldout"] = {
            **metrics(y_ex, yhat), **calibration(y_ex, p_ex),
            "ci95_accuracy": ci(lambda b: float((yhat[b] == y_ex[b]).mean())),
            "ci95_recall_bridge": ci(rec(0)), "ci95_recall_comparison": ci(rec(1)),
            "by_level": {lv: metrics(y_ex[m], yhat[m]) for lv in ("easy", "medium", "hard")
                         for m in [np.array([r["level"] == lv for r in ex])] if m.any()},
            "note": "inference only; C, threshold 0.5 and features were fixed on Training / V-select before this"}
        with open(run_dir / "pred_exploration.jsonl", "w", encoding="utf-8") as fh:
            for r, p in zip(ex, p_ex):
                fh.write(json.dumps({"qid": r["_id"], "p_bridge": round(float(p), 6), "type": r["type"]}) + "\n")
        log.info("exploration (held-out) %s", res["frozen_linear"]["exploration_heldout"])

    # ---- full fine-tuning: encoder + linear head
    if not args.skip_finetune:
        head = torch.nn.Linear(model.get_sentence_embedding_dimension(), 2)
        params = list(model.parameters()) + list(head.parameters())
        opt = torch.optim.AdamW(params, lr=2e-5, weight_decay=0.01)
        bs, steps = 32, len(xq_tr) // 32
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / max(1, steps // 10), max(0.0, (steps - s) / max(1, steps - steps // 10))))
        order = np.random.Generator(np.random.PCG64(42)).permutation(len(xq_tr))
        model.train()
        t = time.perf_counter()
        losses = []
        for s in range(steps):
            idx = order[s * bs:(s + 1) * bs]
            emb = model(model.tokenize([xq_tr[i] for i in idx]))["sentence_embedding"]
            loss = F.cross_entropy(head(emb), torch.tensor([y_tr[i] for i in idx]))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            losses.append(float(loss))
            if (s + 1) % 100 == 0:
                log.info("finetune step %d/%d loss(last100) %.4f", s + 1, steps, float(np.mean(losses[-100:])))
        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(xq_va), 256):
                emb = model(model.tokenize(xq_va[i:i + 256]))["sentence_embedding"]
                preds.extend(head(emb).argmax(-1).tolist())
        res["finetune"] = {"steps": steps, "batch_size": bs, "lr": 2e-5, "epochs": 1, "validation": metrics(y_va, preds),
                           "loss_first100": round(float(np.mean(losses[:100])), 4) if losses else None,
                           "loss_last100": round(float(np.mean(losses[-100:])), 4) if losses else None,
                           "seconds": round(time.perf_counter() - t, 1)}
        out = run_dir / "finetuned_encoder"
        model.save(str(out))
        torch.save(head.state_dict(), run_dir / "head.pt")
        res["finetune"]["encoder_sha256"] = sha256_file(out / "model.safetensors")
        np.save(run_dir / "finetune_val_pred.npy", np.array(preds))
        log.info("finetune val %s", res["finetune"]["validation"])
    res["peak_rss_mb"] = round(peak_rss_mb(), 1)
    atomic_write_json(run_dir / "result.json", res, root)
    print(json.dumps({k: res[k] for k in ("frozen_linear", "finetune") if k in res}, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
