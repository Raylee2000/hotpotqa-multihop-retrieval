"""[Public release copy] Conditional Second-Hop Encoder (variant main).

Copied from the research repository scripts/train_second_hop.py; training logic, hyper-parameters, data selection,
seeds and checkpoint-selection rules are unchanged. Changes:
  - resume checkpoints (written by this script) are read with torch.load(weights_only=True) instead of weights_only=False
  - `git status` provenance via rag_hotpotqa.training.provenance.git_dirty (no failure without git)

Original docstring:
Conditional second-hop (CSH) retriever training: Q ⊕ predicted evidence(A) -> B (design §13.3).

Stages (each resumable):
  mine   once per --sf-run: for every Training bridge question (cross-fitted SF predictions) and V-select bridge
         question (mean of the fold models), both directions: label-free rankings of the non-gold articles against
         q' = Q ⊕ Ŝ_k(A): frozen MiniLM (max_seq_length 128 for mining only) and BM25-Local.
         -> outputs/csh_v1/mining_<sf-run>_k<k>/{train,v_select}.jsonl
  train  InfoNCE (τ 0.05, false-negative mask) + λ·conditional margin (λ 0.5, m 0.1), 1 epoch, bs 16, AdamW,
         Full FT (lr 2e-5) or LoRA (--adapter lora, lr 2e-4). Hard negatives per example: dense top-1 and BM25 top-1
         (next BM25 article if identical).
  select every --eval-every steps on V-select (first --eval-n questions, bridge, both directions): conditional
         R@1 / MRR with predicted-evidence queries (QV2); snapshots; Validation loss on fixed V-select examples.
Variants: main (hard + cond), nocond, inbatch (no hard, no cond), silverdir (main, silver hop order only).
Gold SF are never an input; Validation gold is used only to compute metrics / the Validation loss.
Usage: python scripts/train_second_hop.py --run-name csh_main_v1 --variant main --sf-run sf_cf_v1
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
from rag_hotpotqa.retrieval.base import rank_order  # noqa: E402
from rag_hotpotqa.retrieval.bm25 import BM25Scorer  # noqa: E402
from rag_hotpotqa.retrieval.factory import check_hf_cache_inside_project, verify_dense_weights  # noqa: E402
from rag_hotpotqa.training import monitor  # noqa: E402
from rag_hotpotqa.training.contrastive import info_nce  # noqa: E402
from rag_hotpotqa.training.csh import build_csh_examples, compose_query, conditional_margin, csh_batch, evidence_text  # noqa: E402
from rag_hotpotqa.training.folds import fold_of  # noqa: E402
from rag_hotpotqa.training.lora import TARGETS, adapter_state, apply_lora, load_adapter_state, merged_copy  # noqa: E402
from rag_hotpotqa.training.sf import bm25_nongold  # noqa: E402
from rag_hotpotqa.utils.config import load_config, load_settings, safe_path  # noqa: E402
from rag_hotpotqa.training.provenance import git_dirty  # noqa: E402
from rag_hotpotqa.utils.io import atomic_write_json, sha256_file  # noqa: E402
from rag_hotpotqa.utils.runtime import get_logger, git_commit, peak_rss_mb, set_seed  # noqa: E402

VARIANTS = {"main": {"use_hard": True, "n_hard": 2, "cond": True, "mode": "both"},
            "nocond": {"use_hard": True, "n_hard": 2, "cond": False, "mode": "both"},
            "inbatch": {"use_hard": False, "n_hard": 0, "cond": False, "mode": "both"},
            "silverdir": {"use_hard": True, "n_hard": 2, "cond": True, "mode": "silver"}}
TAU, LAMBDA, MARGIN, BS, SEED, MINE_LEN = 0.05, 0.5, 0.1, 16, 42, 128


def load_preds(path: Path, into: dict | None = None, average: bool = False) -> dict:
    out = {} if into is None else into
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        key = (r["qid"], r["doc"])
        if average and key in out:
            out[key] = (r["sent_ids"], [(a + b) / 2 for a, b in zip(out[key][1], r["scores"])])
        else:
            out[key] = (r["sent_ids"], r["scores"])
    return out


def mine(path: Path, qs, preds, k: int, model, log) -> dict:
    """{(qid, a): {"dense": [...], "bm25": [...]}} for both gold directions; appends to path (resumable)."""
    done = {}
    if path.exists():
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            done[(r["qid"], r["a"])] = {"dense": r["dense"], "bm25": r["bm25"]}
    todo = [q for q in qs if q.qtype == "bridge" and q.is_two_gold and (q.qid, q.gold_indices[0]) not in done]
    bm25 = BM25Scorer("local")
    t = time.perf_counter()
    with open(path, "a", encoding="utf-8") as fh:
        for n, q in enumerate(todo):
            cands = [d.index for d in q.docs if d.index not in q.gold_indices]
            evs = {a: evidence_text(q, a, preds, k) for a in q.gold_indices}
            if not cands or all(v is None for v in evs.values()):
                continue
            dvec = model.encode([q.docs[i].text for i in cands], batch_size=64, normalize_embeddings=True)
            for a, ev in evs.items():
                if ev is None:
                    continue
                qp = compose_query(q.question, ev)
                qv = model.encode([qp], normalize_embeddings=True)[0]
                dense = [cands[i] for i in rank_order(dvec @ qv)]
                sc = bm25.score(qp, [q.docs[i].text for i in cands], local_corpus=[d.text for d in q.docs])
                rec = {"qid": q.qid, "a": a, "dense": dense, "bm25": [cands[i] for i in rank_order(sc)]}
                fh.write(json.dumps(rec) + "\n")
                done[(q.qid, a)] = {"dense": rec["dense"], "bm25": rec["bm25"]}
            if (n + 1) % 1000 == 0:
                fh.flush()
                log.info("mining %s: %d/%d questions (%.1f q/s)", path.name, n + 1, len(todo), (n + 1) / (time.perf_counter() - t))
    return done


def combine_hard(mined: dict, n_hard: int) -> dict:
    """dense top-1, then BM25 top-1 (skipping an article already chosen), up to n_hard."""
    out = {}
    for key, m in mined.items():
        pick = []
        for src in (m["dense"], m["bm25"]):
            nxt = next((i for i in src if i not in pick), None)
            if nxt is not None:
                pick.append(nxt)
        out[key] = pick[:n_hard]
    return out


def main() -> int:
    settings = load_settings()
    root = settings.project_root
    rcfg = load_config("retrieval", settings)
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--sf-run", required=True, help="cross-fitted SF selector run (outputs/sf_selector_v2/<run>)")
    ap.add_argument("--k", type=int, default=2, help="predicted evidence sentences (pre-registered 2)")
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--eval-n", type=int, default=1000, help="first V-select questions for selection")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--val-n", type=int, default=256)
    ap.add_argument("--val-every", type=int, default=125)
    ap.add_argument("--plot-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--adapter", choices=["none", "lora"], default="none")
    ap.add_argument("--lora-targets", choices=sorted(TARGETS), default="qv")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--lora-dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--mine-only", action="store_true")
    ap.add_argument("--abort-after-step", type=int, default=None, help="testing only")
    args = ap.parse_args()
    var = VARIANTS[args.variant]
    lora = args.adapter == "lora"
    lr = args.lr if args.lr is not None else (2e-4 if lora else 2e-5)
    run_dir = safe_path(root / "outputs" / "csh_v1" / args.run_name, root)
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
    tr_ids = select_subset(load_split_ids(settings, "training", "train"), args.n_train)
    tr_qs = [build_question(r, tmpl) for r in load_split_records(settings, "training", "train", tr_ids)]
    vs_ids = load_validation_part(settings, "v_select")
    vs_qs = [build_question(r, tmpl) for r in load_split_records(settings, "validation", "select", vs_ids)]
    sf_dir = root / "outputs" / "sf_selector_v2" / args.sf_run
    tr_preds = {}
    for f in (0, 1):
        load_preds(sf_dir / f"fold{f}" / f"pred_train_fold{f}.jsonl", tr_preds)
    tr_preds = {k: v for k, v in tr_preds.items() if fold_of(k[0]) in (0, 1)}
    vs_preds = load_preds(sf_dir / "fold0" / "pred_v_select.jsonl")
    load_preds(sf_dir / "fold1" / "pred_v_select.jsonl", vs_preds, average=True)
    silver = {}
    sil_path = root / "outputs" / "silver_v1" / "silver_labels.jsonl"
    if var["mode"] == "silver":
        for line in open(sil_path, encoding="utf-8"):
            r = json.loads(line)
            if r["hop_rule"] in ("S-HOP-LINK", "S-HOP-QTITLE"):
                silver[r["qid"]] = (r["first"], r["second"])

    mdir = root / "outputs" / "csh_v1" / f"mining_{args.sf_run}_k{args.k}"
    mdir.mkdir(parents=True, exist_ok=True)
    miner = SentenceTransformer(d["model"], revision=d["revision"], device="cpu")
    miner.max_seq_length = MINE_LEN
    t = time.perf_counter()
    mined_tr = mine(mdir / ("train.jsonl" if args.n_train is None else f"train_n{args.n_train}.jsonl"), tr_qs, tr_preds, args.k, miner, log)
    mined_vs = mine(mdir / "v_select.jsonl", vs_qs, vs_preds, args.k, miner, log)
    log.info("mining ready: %d Training / %d V-select (question, A) queries (%.0f s)", len(mined_tr), len(mined_vs), time.perf_counter() - t)
    del miner
    if args.mine_only:
        return 0

    wrong_tr = {q.qid: bm25_nongold(q)[0] for q in tr_qs if q.qtype == "bridge" and q.is_two_gold and len(q.docs) > 2}
    examples, skipped = build_csh_examples(tr_qs, tr_preds, args.k, SEED, 0, var["mode"], wrong_tr,
                                           combine_hard(mined_tr, var["n_hard"]), var["n_hard"], silver)
    val_qs = [q for q in vs_qs[:args.val_n]]
    wrong_vs = {q.qid: bm25_nongold(q)[0] for q in val_qs if q.qtype == "bridge" and q.is_two_gold and len(q.docs) > 2}
    val_examples, _ = build_csh_examples(val_qs, vs_preds, args.k, SEED, 0, "both", wrong_vs, combine_hard(mined_vs, 2), 2)
    sel_qs = [q for q in vs_qs[:args.eval_n] if q.qtype == "bridge" and q.is_two_gold]
    steps = len(examples) // BS
    run_config = {"variant": args.variant, **var, "sf_run": args.sf_run, "k": args.k, "n_train": args.n_train,
                  "train_ids_sha256": ids_hash(tr_ids), "n_examples": len(examples), "skipped": skipped, "steps": steps,
                  "tau": TAU, "lambda": LAMBDA, "margin": MARGIN, "batch_size": BS, "lr": lr, "epochs": 1,
                  "selection": {"v_select_first": args.eval_n, "n_bridge": len(sel_qs), "every": args.eval_every},
                  "val_loss": {"v_select_first": args.val_n, "n_examples": len(val_examples), "every": args.val_every},
                  "base_weights_sha256": weights["weights_sha256"], "mining_max_seq_length": MINE_LEN}
    if lora:
        run_config["adapter"] = {"type": "lora", "targets": args.lora_targets, "r": args.lora_r, "alpha": args.lora_alpha,
                                 "dropout": args.lora_dropout}
    check_or_write_run_config(run_dir, root, run_config)
    log.info("CSH %s: %d examples (skipped %s), %d steps, %d selection questions, %d val examples", args.variant,
             len(examples), skipped, steps, len(sel_qs), len(val_examples))

    model = SentenceTransformer(d["model"], revision=d["revision"], device="cpu")
    model.max_seq_length = d["max_seq_length"]
    if lora:
        apply_lora(model, args.lora_targets, args.lora_r, args.lora_alpha, args.lora_dropout)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    warm = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    step, last = 0, run_dir / "ckpt" / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=True)
        load_adapter_state(model, ck["model"]) if lora else model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step = ck["step"]
        torch.set_rng_state(ck["torch_rng"])
        log.info("resumed from step %d", step)
        with open(run_dir / "train_log.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"event": "resumed", "from_step": step, "utc": started}) + "\n")

    def save_ckpt():
        tmp = last.with_suffix(".tmp")
        torch.save({"model": adapter_state(model) if lora else model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step, "torch_rng": torch.get_rng_state()}, tmp)
        os.replace(tmp, last)

    def enc(texts):
        return F.normalize(model(model.tokenize(list(texts)))["sentence_embedding"], dim=-1)

    def batch_loss(batch, use_hard, cond):
        docs, mask = csh_batch(batch, use_hard)
        qv, dv = enc([e.query for e in batch]), enc(docs)
        loss = info_nce(qv, dv, mask, TAU)
        parts = {"nce": float(loss)}
        idx = [i for i, e in enumerate(batch) if e.wrong_query]
        if cond and idx:
            wv = enc([batch[i].wrong_query for i in idx])
            pos = (qv[idx] * dv[idx]).sum(-1)
            cm = conditional_margin(pos, (wv * dv[idx]).sum(-1), MARGIN)
            loss = loss + LAMBDA * cm
            parts["cond"] = float(cm)
        return loss, parts, qv, dv

    sel_path, val_path = run_dir / "selection.jsonl", run_dir / "val_loss.jsonl"
    done_sel = {json.loads(x)["step"] for x in open(sel_path, encoding="utf-8") if x.strip()} if sel_path.exists() else set()
    done_val = {json.loads(x)["step"] for x in open(val_path, encoding="utf-8") if x.strip()} if val_path.exists() else set()

    def selection_metrics() -> dict:
        r1, r5, rr = [], [], []
        with torch.no_grad():
            model.eval()
            for q in sel_qs:
                dv = enc([doc.text for doc in q.docs])
                for a in q.gold_indices:
                    ev = evidence_text(q, a, vs_preds, args.k)
                    if ev is None:
                        continue
                    b = next(i for i in q.gold_indices if i != a)
                    s = (dv @ enc([compose_query(q.question, ev)])[0]).numpy()
                    cands = [i for i in range(len(q.docs)) if i != a]
                    order = [cands[i] for i in rank_order(s[cands])]
                    rank = order.index(b) + 1
                    r1.append(rank == 1)
                    r5.append(rank <= 5)
                    rr.append(1 / rank)
        model.train()
        m = {"QV2.cond_recall@1": float(np.mean(r1)), "QV2.cond_recall@5": float(np.mean(r5)), "QV2.cond_mrr": float(np.mean(rr)),
             "n_queries": len(r1)}
        m["score"] = m["QV2.cond_recall@1"]
        return {k: round(v, 5) if isinstance(v, float) else v for k, v in m.items()}

    def maybe_select(force=False):
        if step in done_sel or not (force or step % args.eval_every == 0):
            return
        t0 = time.perf_counter()
        snap = run_dir / f"snapshot_step{step:05d}"
        with torch.random.fork_rng():
            m = selection_metrics()
            if lora:
                merged_copy(model).save(str(snap))
                torch.save(adapter_state(model), snap / "adapter.pt")
            else:
                model.save(str(snap))
        rec = {"step": step, "metrics": m, "seconds": round(time.perf_counter() - t0, 1),
               "weights_sha256": sha256_file(snap / "model.safetensors"), "snapshot": snap.relative_to(root).as_posix()}
        with open(sel_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        done_sel.add(step)
        log.info("selection step %d: %s (%.0f s)", step, m, rec["seconds"])

    def maybe_val(force=False):
        if not val_examples or step in done_val or not (force or step % args.val_every == 0):
            return
        t0 = time.perf_counter()
        tot = n = 0
        with torch.random.fork_rng(), torch.no_grad():
            model.eval()
            for i in range(0, len(val_examples), BS):
                vb = val_examples[i:i + BS]
                loss, _, _, _ = batch_loss(vb, True, True)  # same objective for every variant: comparable curves
                tot += float(loss) * len(vb)
                n += len(vb)
        model.train()
        rec = {"step": step, "val_loss": round(tot / n, 5), "n_examples": n, "seconds": round(time.perf_counter() - t0, 1)}
        with open(val_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        done_val.add(step)
        log.info("validation loss step %d: %.4f", step, rec["val_loss"])

    def plot():
        try:
            monitor.render(run_dir, title=f"CSH {args.run_name} ({args.variant}{', LoRA' if lora else ', full FT'})")
        except Exception as exc:
            log.warning("live plot failed: %s", exc)

    maybe_select(force=step == 0)
    maybe_val(force=step == 0)
    plot()
    model.train()
    order = np.random.Generator(np.random.PCG64(SEED)).permutation(len(examples))
    t_run, step0 = time.perf_counter(), step
    while step < steps:
        batch = [examples[i] for i in order[step * BS:(step + 1) * BS]]
        t0 = time.perf_counter()
        loss, parts, qv, dv = batch_loss(batch, var["use_hard"], var["cond"])
        opt.zero_grad()
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        if not math.isfinite(gn) or not math.isfinite(float(loss)):
            raise RuntimeError(f"non-finite loss/gradient at step {step}")
        with torch.no_grad():
            pos_cos = float((qv * dv[: len(batch)]).sum(-1).mean())
        opt.step()
        sched.step()
        step += 1
        with open(run_dir / "train_log.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"step": step, "epoch": 0, "loss": round(float(loss), 5), **{k: round(v, 5) for k, v in parts.items()},
                                 "grad_norm": round(gn, 4), "lr": sched.get_last_lr()[0], "pos_cos": round(pos_cos, 4),
                                 "seconds": round(time.perf_counter() - t0, 3)}) + "\n")
        if step % args.save_every == 0 or step % args.eval_every == 0 or step == steps:
            save_ckpt()
            log.info("step %d/%d loss %.4f %s %.2f s/step RSS peak %.0f MB", step, steps, float(loss), parts,
                     (time.perf_counter() - t_run) / max(1, step - step0), peak_rss_mb())
        maybe_select(force=step == steps)
        maybe_val(force=step == steps)
        if step % args.plot_every == 0 or step == steps:
            plot()
        if args.abort_after_step is not None and step >= args.abort_after_step:
            log.info("abort-after-step %d: hard exit (resume test)", step)
            os._exit(3)
    sels = [json.loads(x) for x in open(sel_path, encoding="utf-8")]
    best = max(sels, key=lambda r: (r["metrics"]["score"], -r["step"]))
    atomic_write_json(run_dir / "result.json", {"run": args.run_name, "variant": args.variant, "steps": step, "best": best,
                                                "selection_history": sels, "run_config": run_config,
                                                "peak_rss_mb": round(peak_rss_mb(), 1),
                                                "finished_utc": datetime.now(timezone.utc).isoformat()}, root)
    log.info("finished; best step %d score %.4f", best["step"], best["metrics"]["score"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
