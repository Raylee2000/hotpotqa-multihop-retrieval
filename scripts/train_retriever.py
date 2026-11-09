"""[Public release copy] General Encoder (variant hardneg).

Copied from the research repository scripts/train_retriever.py; training logic, hyper-parameters, data selection,
seeds and checkpoint-selection rules are unchanged. Changes:
  - resume checkpoints (written by this script) are read with torch.load(weights_only=True) instead of weights_only=False
  - `git status` provenance via rag_hotpotqa.training.provenance.git_dirty (no failure without git)

Original docstring:
Contrastive fine-tuning of the pinned MiniLM on the Training split (resumable, checkpointed).

Smoke:  python scripts/train_retriever.py --run-name smoke_hardneg --variant hardneg --n-train 200 --max-steps 12 --eval-n 50 --eval-every 6
Formal: python scripts/train_retriever.py --run-name hardneg_v1 --variant hardneg
Resume: re-run the identical command (continues from ckpt/last.pt; a changed config is refused).
LoRA:   add --adapter lora [--lora-targets qv --lora-r 8 --lora-alpha 16 --lora-dropout 0.1 --lr 2e-4]; base frozen, checkpoints hold the adapter only, snapshots are merged plain SentenceTransformers (same pooling).
Live monitoring: train_metrics.csv / val_metrics.csv / selection_metrics.csv and live/index.html (SVG curves,
auto-refresh) are rebuilt every --plot-every steps; Validation loss on --val-n fixed V-select examples every --val-every.
Resume test: --abort-after-step K hard-exits (os._exit) after step K, like a kill; not part of the run config.
Gradient updates use Training questions only; Validation is read with purpose "select" for checkpoint
selection; Dev is never read.
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

from rag_hotpotqa.data.access import load_split_ids, load_split_records  # noqa: E402
from rag_hotpotqa.data.documents import build_question  # noqa: E402
from rag_hotpotqa.data.splits import ids_hash  # noqa: E402
from rag_hotpotqa.experiments.multihop import run_question  # noqa: E402
from rag_hotpotqa.experiments.runner import check_or_write_run_config, select_subset  # noqa: E402
from rag_hotpotqa.retrieval.dense import DenseScorer  # noqa: E402
from rag_hotpotqa.retrieval.factory import check_hf_cache_inside_project, verify_dense_weights  # noqa: E402
from rag_hotpotqa.retrieval.similarity import Similarity  # noqa: E402
from rag_hotpotqa.training.contrastive import batch_docs, build_examples, info_nce  # noqa: E402
from rag_hotpotqa.training import monitor  # noqa: E402
from rag_hotpotqa.training.lora import TARGETS, adapter_state, apply_lora, load_adapter_state, merged_copy  # noqa: E402
from rag_hotpotqa.utils.config import load_config, load_settings, load_yaml, safe_path  # noqa: E402
from rag_hotpotqa.training.provenance import git_dirty  # noqa: E402
from rag_hotpotqa.utils.io import atomic_write_json, sha256_file  # noqa: E402
from rag_hotpotqa.utils.runtime import get_logger, git_commit, peak_rss_mb, set_seed  # noqa: E402


def selection_eval(model, questions, ecfg) -> dict:
    """O + E1 on the fixed Validation selection subset with the current weights (no disk cache)."""
    scorer = DenseScorer("finetune", model=model, max_seq_length=model.max_seq_length, normalize=True,
                         batch_size=64, cache_dir=None, joiner="\n\n", name="dense")
    sim = Similarity("dense", [scorer], query_article="body")
    cfg = {"ks": ecfg["ks"], **ecfg["paths"]}
    texts = [x for q in questions for x in sim.texts_for(q.unlabeled(), qa_contents=[q.docs[g].body for g in q.gold_indices])[0]]
    scorer.prefetch(texts)
    rows = [run_question(q, sim, ["E1"], cfg) for q in questions]
    mean = lambda k: float(np.mean([r["metrics"][k] for r in rows]))  # noqa: E731
    m = {"O.fh_mrr": mean("O.fh_mrr"), "O.fh_all_gold@2": mean("O.fh_all_gold@2"), "O.fh_any_gold@1": mean("O.fh_any_gold@1"),
         "E1.AtoB.recall@1": mean("E1.AtoB.recall@1"), "E1.BtoA.recall@1": mean("E1.BtoA.recall@1")}
    m["score"] = (m["O.fh_mrr"] + m["O.fh_all_gold@2"] + (m["E1.AtoB.recall@1"] + m["E1.BtoA.recall@1"]) / 2) / 3
    return {k: round(v, 5) for k, v in m.items()}


def main() -> int:
    settings = load_settings()
    root = settings.project_root
    tcfg = load_yaml(root / "configs" / "training_v1.yaml")
    rcfg = load_config("retrieval", settings)
    ecfg = load_config("experiments", settings)
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--variant", choices=sorted(tcfg["variants"]), required=True)
    ap.add_argument("--n-train", type=int, default=None, help="Training subset (smoke tests)")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--eval-n", type=int, default=tcfg["selection"]["validation_subset"])
    ap.add_argument("--eval-every", type=int, default=tcfg["selection"]["eval_every_steps"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=tcfg["checkpoint"]["save_every_steps"])
    ap.add_argument("--adapter", choices=["none", "lora"], default="none")
    ap.add_argument("--lora-targets", choices=sorted(TARGETS), default="qv")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--lora-dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=None, help="override the config learning rate")
    ap.add_argument("--val-n", type=int, default=256, help="fixed Validation examples for the Validation loss (0 = off)")
    ap.add_argument("--val-every", type=int, default=125)
    ap.add_argument("--plot-every", type=int, default=25)
    ap.add_argument("--abort-after-step", type=int, default=None, help="testing only: hard exit after this step")
    args = ap.parse_args()
    run_dir = safe_path(root / tcfg["output_root"] / args.run_name, root)
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    log = get_logger(log_file=run_dir / "train.log")
    started = datetime.now(timezone.utc).isoformat()
    with open(run_dir / "provenance.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"session_started_utc": started, "git_commit_at_start": git_commit(root),
                             "dirty_files_at_start": git_dirty(root), "argv": sys.argv}) + "\n")

    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(args.threads)
    set_seed(tcfg["seed"])
    d = rcfg["dense"]
    hf = check_hf_cache_inside_project(settings, True)
    weights = verify_dense_weights(d["model"], d["revision"], d["weights_sha256"])

    train_ids = select_subset(load_split_ids(settings, "training", "train"), args.n_train)
    train_qs = [build_question(r, rcfg["document"]["template"]) for r in load_split_records(settings, "training", "train", train_ids)]
    sel_ids = select_subset(load_split_ids(settings, "validation", "select"), args.eval_n)
    sel_qs = [build_question(r, rcfg["document"]["template"]) for r in load_split_records(settings, "validation", "select", sel_ids)]
    val_ids = select_subset(load_split_ids(settings, "validation", "select"), args.val_n) if args.val_n else []
    val_qs = [build_question(r, rcfg["document"]["template"]) for r in load_split_records(settings, "validation", "select", val_ids)] if val_ids else []
    variant = tcfg["variants"][args.variant]
    examples, skipped = build_examples(train_qs, tcfg["seed"], tcfg["data"]["n_hard"], rcfg["compose"]["joiner"])
    bs = tcfg["batch_size"]
    steps_per_epoch = len(examples) // bs
    total_steps = steps_per_epoch * tcfg["epochs"] if args.max_steps is None else min(args.max_steps, steps_per_epoch * tcfg["epochs"])
    run_config = {"training_config_sha256": sha256_file(root / "configs" / "training_v1.yaml"), "variant": args.variant,
                  "use_hard": variant["use_hard"], "n_train": args.n_train, "max_steps": args.max_steps,
                  "train_ids_sha256": ids_hash(train_ids), "selection_ids_sha256": ids_hash(sel_ids),
                  "eval_every": args.eval_every, "save_every": args.save_every, "base_weights_sha256": weights["weights_sha256"],
                  "revision": d["revision"], "n_examples": len(examples), "total_steps": total_steps}
    # Validation loss: same example construction as Training (1 BM25 hard negative) for every variant, so curves compare
    val_examples = build_examples(val_qs, tcfg["seed"], tcfg["data"]["n_hard"], rcfg["compose"]["joiner"])[0] if val_qs else []
    if args.val_n:
        run_config["val_loss"] = {"n_questions": args.val_n, "n_examples": len(val_examples), "every": args.val_every,
                                  "val_ids_sha256": ids_hash(val_ids), "use_hard": True}
    lr = tcfg["optimizer"]["lr"] if args.lr is None else args.lr
    if args.lr is not None:  # only present when used, so full fine-tuning configs are unchanged
        run_config["lr"] = lr
    if args.adapter == "lora":
        run_config["adapter"] = {"type": "lora", "targets": args.lora_targets, "r": args.lora_r,
                                 "alpha": args.lora_alpha, "dropout": args.lora_dropout}
    check_or_write_run_config(run_dir, root, run_config)
    kinds = {k: sum(e.kind == k for e in examples) for k in ("first_hop", "second_hop")}
    log.info("train %s variant=%s examples=%d %s skipped=%s steps=%d", args.run_name, args.variant, len(examples),
             kinds, skipped, total_steps)

    model = SentenceTransformer(d["model"], revision=d["revision"], device="cpu")
    model.max_seq_length = tcfg["max_seq_length"]
    lora = args.adapter == "lora"
    if lora:
        wrapped = apply_lora(model, args.lora_targets, args.lora_r, args.lora_alpha, args.lora_dropout)
        log.info("LoRA on %d modules; trainable %d of %d parameters", len(wrapped),
                 sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters()))
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=tcfg["optimizer"]["weight_decay"])
    warm = max(1, int(tcfg["optimizer"]["warmup_ratio"] * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (total_steps - s) / max(1, total_steps - warm))))
    step = 0
    last = run_dir / "ckpt" / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=True)
        if lora:
            load_adapter_state(model, ck["model"])
        else:
            model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step = ck["step"]
        torch.set_rng_state(ck["torch_rng"])
        log.info("resumed from step %d", step)
        with open(run_dir / "train_log.jsonl", "a", encoding="utf-8") as fh:  # later records of a step supersede earlier ones
            fh.write(json.dumps({"event": "resumed", "from_step": step, "utc": started}) + "\n")

    def save_ckpt():
        tmp = last.with_suffix(".tmp")
        torch.save({"model": adapter_state(model) if lora else model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                    "torch_rng": torch.get_rng_state()}, tmp)
        os.replace(tmp, last)

    sel_path = run_dir / "selection.jsonl"
    done_evals = {json.loads(x)["step"] for x in open(sel_path, encoding="utf-8")} if sel_path.exists() else set()

    def maybe_eval(force=False):
        if step in done_evals or not (force or step % args.eval_every == 0):
            return
        t = time.perf_counter()
        snap = run_dir / f"snapshot_step{step:05d}"
        with torch.random.fork_rng():  # evaluation / snapshots must not change the training RNG (resume skips done evals)
            m = selection_eval(model, sel_qs, ecfg)
            if lora:  # merged plain model for evaluation; the adapter alone is kept next to it
                merged_copy(model).save(str(snap))
                torch.save(adapter_state(model), snap / "adapter.pt")
            else:
                model.save(str(snap))
        model.train()
        rec = {"step": step, "metrics": m, "seconds": round(time.perf_counter() - t, 1),
               "weights_sha256": sha256_file(snap / "model.safetensors"), "snapshot": snap.relative_to(root).as_posix()}
        with open(sel_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        done_evals.add(step)
        log.info("selection eval step %d: %s (%.0f s)", step, m, rec["seconds"])

    val_path = run_dir / "val_loss.jsonl"
    done_val = {json.loads(x)["step"] for x in open(val_path, encoding="utf-8") if x.strip()} if val_path.exists() else set()

    def maybe_val_loss(force=False):
        if not val_examples or step in done_val or not (force or step % args.val_every == 0):
            return
        t = time.perf_counter()
        tot, n = 0.0, 0
        with torch.random.fork_rng(), torch.no_grad():  # no gradient, no effect on the training RNG
            model.eval()
            for i in range(0, len(val_examples), bs):
                vb = val_examples[i:i + bs]
                vdocs, vmask = batch_docs(vb, True)
                vq = F.normalize(model(model.tokenize([e.query for e in vb]))["sentence_embedding"], dim=-1)
                vd = F.normalize(model(model.tokenize(vdocs))["sentence_embedding"], dim=-1)
                tot += float(info_nce(vq, vd, vmask, tcfg["tau"])) * len(vb)
                n += len(vb)
        model.train()
        rec = {"step": step, "val_loss": round(tot / n, 5), "n_examples": n, "seconds": round(time.perf_counter() - t, 1)}
        with open(val_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        done_val.add(step)
        log.info("validation loss step %d: %.4f (%d examples, %.0f s)", step, rec["val_loss"], n, rec["seconds"])

    def plot():
        try:
            monitor.render(run_dir, title=f"{args.run_name} ({args.variant}{', LoRA' if lora else ', full FT'})")
        except Exception as exc:  # plotting must never stop training
            log.warning("live plot failed: %s", exc)

    maybe_eval(force=step == 0)
    maybe_val_loss(force=step == 0)
    plot()
    model.train()
    step_start = step
    log_path = run_dir / "train_log.jsonl"
    t_run = time.perf_counter()
    while step < total_steps:
        epoch, pos = divmod(step, steps_per_epoch)
        order = np.random.Generator(np.random.PCG64(tcfg["seed"] + epoch)).permutation(len(examples))
        batch = [examples[i] for i in order[pos * bs:(pos + 1) * bs]]
        docs, mask = batch_docs(batch, variant["use_hard"])
        t = time.perf_counter()
        qv = F.normalize(model(model.tokenize([e.query for e in batch]))["sentence_embedding"], dim=-1)
        dv = F.normalize(model(model.tokenize(docs))["sentence_embedding"], dim=-1)
        loss = info_nce(qv, dv, mask, tcfg["tau"])
        opt.zero_grad()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["optimizer"]["grad_clip"]))
        if not math.isfinite(gnorm) or not math.isfinite(float(loss)):
            raise RuntimeError(f"non-finite loss/gradient at step {step}: loss={float(loss)} grad_norm={gnorm}")
        if step == 0:
            log.info("gradient sanity: grad_norm=%.4f params_without_grad=%d of %d", gnorm,
                     sum(g is None for g in grads), len(grads))
        with torch.no_grad():
            pos_sim = float((qv * dv[: len(batch)]).sum(-1).mean())
        opt.step()
        sched.step()
        step += 1
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"step": step, "epoch": epoch, "loss": round(float(loss), 5), "grad_norm": round(gnorm, 4),
                                 "lr": sched.get_last_lr()[0], "pos_cos": round(pos_sim, 4), "n_docs": len(docs),
                                 "masked": int(mask.sum()), "seconds": round(time.perf_counter() - t, 3)}) + "\n")
        if step % args.save_every == 0 or step % args.eval_every == 0 or step == total_steps:
            save_ckpt()
            el = time.perf_counter() - t_run
            log.info("step %d/%d loss %.4f %.2f s/step RSS peak %.0f MB", step, total_steps, float(loss),
                     el / max(1, step - step_start), peak_rss_mb())
        maybe_eval(force=step == total_steps)
        maybe_val_loss(force=step == total_steps)
        if step % args.plot_every == 0 or step % args.val_every == 0 or step == total_steps:
            plot()
        if args.abort_after_step is not None and step >= args.abort_after_step:
            log.info("abort-after-step %d: hard exit (resume test)", step)
            os._exit(3)
    sels = [json.loads(x) for x in open(sel_path, encoding="utf-8")]
    best = max(sels, key=lambda r: (r["metrics"]["score"], -r["step"]))
    atomic_write_json(run_dir / "result.json", {"run": args.run_name, "variant": args.variant, "steps": step,
                                                "best": best, "selection_history": sels, "skipped": skipped,
                                                "example_kinds": kinds, "hf_cache": hf, "base_weights": weights,
                                                "peak_rss_mb": round(peak_rss_mb(), 1),
                                                "finished_utc": datetime.now(timezone.utc).isoformat()}, root)
    log.info("finished; best step %d score %.4f", best["step"], best["metrics"]["score"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
