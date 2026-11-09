"""[Public release copy] Supporting Fact Selector, 2-fold cross-fit.

Copied from the research repository scripts/train_sf_selector_cf.py; training logic, hyper-parameters, data selection,
seeds and checkpoint-selection rules are unchanged. Changes:
  - resume checkpoints (written by this script) are read with torch.load(weights_only=True) instead of weights_only=False
  - `git status` provenance via rag_hotpotqa.training.provenance.git_dirty (no failure without git)
  - final head saved as head.safetensors (same tensors; the public inference loads safetensors) instead of head.pt

Original docstring:
Cross-fitted supporting-sentence selector (official SF labels), resumable, with live curves.

Fold f model: trained on Training questions with fold_of(qid) != f (2 folds), full fine-tuning of the pinned
MiniLM as a cross-encoder pair classifier (BCE, pos_weight = #neg/#pos), 1 epoch. After training it exports
  pred_train_fold{f}.jsonl   scores for Training questions IN fold f (unseen by this model): every sentence of
                             every context article (inputs for conditional second-hop training — predicted
                             evidence of A and of a wrong first hop A' — and for path-reranker features)
  pred_v_select.jsonl        scores for every sentence of every context article of V-select (Validation)
Validation predictions = mean of the two fold models (scripts/eval_sf_selector_cf.py). Gold SF are labels only.
Usage: python scripts/train_sf_selector_cf.py --run-name sf_cf_v1 --fold 0   (then --fold 1)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_hotpotqa.data.access import load_split_ids, load_split_records, load_validation_part  # noqa: E402
from rag_hotpotqa.data.documents import build_question  # noqa: E402
from rag_hotpotqa.data.splits import ids_hash  # noqa: E402
from rag_hotpotqa.experiments.runner import check_or_write_run_config, select_subset  # noqa: E402
from rag_hotpotqa.retrieval.bm25 import BM25Scorer  # noqa: E402
from rag_hotpotqa.retrieval.factory import check_hf_cache_inside_project, verify_dense_weights  # noqa: E402
from rag_hotpotqa.training import monitor  # noqa: E402
from rag_hotpotqa.training.folds import fold_of  # noqa: E402
from rag_hotpotqa.training.sf import article_sentences, score_pairs, train_pairs  # noqa: E402
from rag_hotpotqa.utils.config import load_config, load_settings, safe_path  # noqa: E402
from rag_hotpotqa.training.provenance import git_dirty  # noqa: E402
from rag_hotpotqa.utils.io import atomic_write_json, sha256_file  # noqa: E402
from rag_hotpotqa.utils.runtime import get_logger, git_commit, peak_rss_mb, set_seed  # noqa: E402

LR, BS, MAXLEN, SEED = 2e-5, 32, 128, 42


def export(path: Path, rows) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def main() -> int:
    settings = load_settings()
    root = settings.project_root
    rcfg = load_config("retrieval", settings)
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--fold", type=int, choices=[0, 1], required=True)
    ap.add_argument("--n-train", type=int, default=None, help="Training subset before the fold filter (smoke)")
    ap.add_argument("--n-val", type=int, default=None, help="V-select prefix (smoke); default all 2,000")
    ap.add_argument("--val-loss-n", type=int, default=200)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--plot-every", type=int, default=50)
    ap.add_argument("--abort-after-step", type=int, default=None, help="testing only")
    args = ap.parse_args()
    run_dir = safe_path(root / "outputs" / "sf_selector_v2" / args.run_name / f"fold{args.fold}", root)
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    log = get_logger(log_file=run_dir / "train.log")
    started = datetime.now(timezone.utc).isoformat()
    dirty = git_dirty(root)
    with open(run_dir / "provenance.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"session_started_utc": started, "git_commit_at_start": git_commit(root),
                             "dirty_files_at_start": dirty, "argv": sys.argv}) + "\n")

    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(args.threads)
    set_seed(SEED)
    d = rcfg["dense"]
    check_hf_cache_inside_project(settings, True)
    weights = verify_dense_weights(d["model"], d["revision"], d["weights_sha256"])
    tmpl = rcfg["document"]["template"]
    all_tr = select_subset(load_split_ids(settings, "training", "train"), args.n_train)
    tr_qs = [build_question(r, tmpl) for r in load_split_records(settings, "training", "train", all_tr)]
    fit_qs = [q for q in tr_qs if fold_of(q.qid) != args.fold]
    out_qs = [q for q in tr_qs if fold_of(q.qid) == args.fold]
    vs_ids = load_validation_part(settings, "v_select")
    vs_ids = vs_ids if args.n_val is None else vs_ids[:args.n_val]
    vs_qs = [build_question(r, tmpl) for r in load_split_records(settings, "validation", "select", vs_ids)]
    bm25 = BM25Scorer("local")
    t = time.perf_counter()
    pairs = [p for q in fit_qs for p in train_pairs(q, bm25)]
    val_pairs = [p for q in vs_qs[:args.val_loss_n] for p in train_pairs(q, bm25)]
    n_pos = sum(p[2] for p in pairs)
    pos_weight = (len(pairs) - n_pos) / max(1, n_pos)
    steps = len(pairs) // BS
    run_config = {"fold": args.fold, "n_train": args.n_train, "fit_ids_sha256": ids_hash([q.qid for q in fit_qs]),
                  "export_ids_sha256": ids_hash([q.qid for q in out_qs]), "v_select_ids_sha256": ids_hash(vs_ids),
                  "n_pairs": len(pairs), "positives": n_pos, "pos_weight": round(pos_weight, 5), "steps": steps,
                  "lr": LR, "batch_size": BS, "max_length": MAXLEN, "label_source": "official HotpotQA supporting_facts",
                  "base_weights_sha256": weights["weights_sha256"], "val_loss_questions": args.val_loss_n}
    check_or_write_run_config(run_dir, root, run_config)
    log.info("fold %d: fit %d questions, %d pairs (%d pos, pos_weight %.3f), %d steps; export %d Training questions, "
             "%d V-select (%.0f s)", args.fold, len(fit_qs), len(pairs), n_pos, pos_weight, steps, len(out_qs), len(vs_qs),
             time.perf_counter() - t)

    st = SentenceTransformer(d["model"], revision=d["revision"], device="cpu")
    enc, tok = st[0].auto_model, st.tokenizer
    head = torch.nn.Linear(enc.config.hidden_size, 1)
    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
    warm = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    step, last = 0, run_dir / "ckpt" / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=True)
        enc.load_state_dict(ck["enc"])
        head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step = ck["step"]
        torch.set_rng_state(ck["torch_rng"])
        log.info("resumed from step %d", step)
        with open(run_dir / "train_log.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"event": "resumed", "from_step": step, "utc": started}) + "\n")

    def save_ckpt():
        tmp = last.with_suffix(".tmp")
        torch.save({"enc": enc.state_dict(), "head": head.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "step": step, "torch_rng": torch.get_rng_state()}, tmp)
        os.replace(tmp, last)

    def logits(qs_, ss_):
        b = tok(list(qs_), list(ss_), truncation="longest_first", max_length=MAXLEN, padding=True, return_tensors="pt")
        h = enc(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        return head((h * m).sum(1) / m.sum(1)).squeeze(-1)

    pw = torch.tensor([pos_weight])
    val_path = run_dir / "val_loss.jsonl"
    done_val = {json.loads(x)["step"] for x in open(val_path, encoding="utf-8") if x.strip()} if val_path.exists() else set()

    def maybe_val(force=False):
        if not val_pairs or step in done_val or not (force or step % args.val_every == 0):
            return
        t0 = time.perf_counter()
        tot = 0.0
        with torch.random.fork_rng(), torch.no_grad():
            enc.eval()
            for i in range(0, len(val_pairs), 128):
                vb = val_pairs[i:i + 128]
                tot += float(F.binary_cross_entropy_with_logits(logits([x[0] for x in vb], [x[1] for x in vb]),
                                                                torch.tensor([float(x[2]) for x in vb]), pos_weight=pw,
                                                                reduction="sum"))
        enc.train()
        rec = {"step": step, "val_loss": round(tot / len(val_pairs), 5), "n_examples": len(val_pairs),
               "seconds": round(time.perf_counter() - t0, 1)}
        with open(val_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        done_val.add(step)
        log.info("validation loss step %d: %.4f", step, rec["val_loss"])

    def plot():
        try:
            monitor.render(run_dir, title=f"SF selector {args.run_name} fold {args.fold}")
        except Exception as exc:  # never stop training for a plot
            log.warning("live plot failed: %s", exc)

    order = np.random.Generator(np.random.PCG64(SEED)).permutation(len(pairs))
    enc.train()
    maybe_val(force=step == 0)
    plot()
    t_run, step0 = time.perf_counter(), step
    while step < steps:
        b = [pairs[i] for i in order[step * BS:(step + 1) * BS]]
        t0 = time.perf_counter()
        loss = F.binary_cross_entropy_with_logits(logits([x[0] for x in b], [x[1] for x in b]),
                                                  torch.tensor([float(x[2]) for x in b]), pos_weight=pw)
        opt.zero_grad()
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        if not math.isfinite(gn) or not math.isfinite(float(loss)):
            raise RuntimeError(f"non-finite loss/gradient at step {step}")
        opt.step()
        sched.step()
        step += 1
        with open(run_dir / "train_log.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"step": step, "epoch": 0, "loss": round(float(loss), 5), "grad_norm": round(gn, 4),
                                 "lr": sched.get_last_lr()[0], "seconds": round(time.perf_counter() - t0, 3)}) + "\n")
        if step % args.save_every == 0 or step % args.val_every == 0 or step == steps:
            save_ckpt()
            log.info("step %d/%d loss %.4f %.2f s/step RSS peak %.0f MB", step, steps, float(loss),
                     (time.perf_counter() - t_run) / max(1, step - step0), peak_rss_mb())
        maybe_val(force=step == steps)
        if step % args.plot_every == 0 or step == steps:
            plot()
        if args.abort_after_step is not None and step >= args.abort_after_step:
            log.info("abort-after-step %d: hard exit (resume test)", step)
            os._exit(3)

    final = run_dir / "final"
    if not (final / "head.safetensors").exists():
        from safetensors.torch import save_file

        enc.save_pretrained(final)
        tok.save_pretrained(final)
        save_file({k: v.contiguous() for k, v in head.state_dict().items()}, str(final / "head.safetensors"))
    enc.eval()
    t = time.perf_counter()
    if not (run_dir / f"pred_train_fold{args.fold}.jsonl").exists():
        def train_rows():
            for q in out_qs:
                for idx in range(len(q.docs)):
                    ss = article_sentences(q, idx)
                    sc = score_pairs(enc, head, tok, [q.question] * len(ss), [s for _, s in ss])
                    yield {"qid": q.qid, "doc": idx, "sent_ids": [i for i, _ in ss], "scores": [round(float(x), 5) for x in sc]}
        export(run_dir / f"pred_train_fold{args.fold}.jsonl", train_rows())
    if not (run_dir / "pred_v_select.jsonl").exists():
        def val_rows():
            for q in vs_qs:
                for doc in q.docs:
                    ss = article_sentences(q, doc.index)
                    sc = score_pairs(enc, head, tok, [q.question] * len(ss), [s for _, s in ss])
                    yield {"qid": q.qid, "doc": doc.index, "sent_ids": [i for i, _ in ss], "scores": [round(float(x), 5) for x in sc]}
        export(run_dir / "pred_v_select.jsonl", val_rows())
    atomic_write_json(run_dir / "result.json", {"run_config": run_config, "steps": step,
                                                "encoder_sha256": sha256_file(final / "model.safetensors"),
                                                "export_seconds": round(time.perf_counter() - t, 1),
                                                "peak_rss_mb": round(peak_rss_mb(), 1),
                                                "finished_utc": datetime.now(timezone.utc).isoformat()}, root)
    log.info("fold %d finished (%d steps); exports written", args.fold, step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
