"""[Public release copy] R5 training / selection candidates (Training 10,000 + V-select).

Copied from the research repository scripts/gen_path_candidates.py; training logic, hyper-parameters, data selection,
seeds and checkpoint-selection rules are unchanged. Changes:
  - none

Original docstring:
Fixed candidate path sets + order-invariant features for the path reranker (design §5, §13.5).

Candidates per question (label-free; titles are NOT used to generate candidates, so a no-title reranker is fair):
  bridge-style  Top-5 first-hop A by s1(Q, d) x Top-5 B by s2(QV2(A), d) over d != A  (directed generation)
  parallel      all pairs among the Top-5 first-hop articles
merged into unordered pairs {x, y}. Gold articles are never added; a question whose gold pair is missing simply has
no positive (Candidate Path Recall measures this).
Features per pair (both directions kept as features; the gold label is order-invariant):
  dense: s1(Q,x), s1(Q,y) (max/min), s2(QV2(A),B) per direction, F = s1(Q,A)+s2(QV2(A),B) best/other direction,
         direction gap, first-hop ranks (min/max)
  lexical: BM25-Local of Q vs each article (normalised per question), BM25 of QV2(A) vs B (best direction)
  length: log tokens (max/min), number of articles over 254 tokens
  SF selector: max predicted sentence score per article (max/min)
  question type: P(bridge) (cross-fitted on Training) and products with the dense features
  generation flags: produced by the bridge-style / parallel generator
  title (separate group "title_*", dropped in the no-title control): title(y) in body(x) / title(x) in body(y),
         number of the two titles found in Q
QV2(A) = Q ⊕ top-k predicted SF sentences of A (cross-fitted on Training, mean of folds on V-select); if A has no
predicted sentences, Q ⊕ body(A) is used and the flag `qv2_fallback` is set.
Output: outputs/path_candidates_v1/<run>/<split>.npz (+ meta json).
Usage: python scripts/gen_path_candidates.py --run-name frozen --split v_select --sf-run sf_cf_v1 --qtype-run qtype_v2
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_hotpotqa.data.access import load_split_ids, load_split_records, load_validation_part  # noqa: E402
from rag_hotpotqa.data.documents import build_question  # noqa: E402
from rag_hotpotqa.data.splits import ids_hash  # noqa: E402
from rag_hotpotqa.experiments.runner import select_subset  # noqa: E402
from rag_hotpotqa.retrieval.base import rank_order  # noqa: E402
from rag_hotpotqa.retrieval.bm25 import BM25Scorer  # noqa: E402
from rag_hotpotqa.retrieval.factory import check_hf_cache_inside_project, verify_dense_weights  # noqa: E402
from rag_hotpotqa.training.csh import compose_query, evidence_text  # noqa: E402
from rag_hotpotqa.training.silver import find_title  # noqa: E402
from rag_hotpotqa.utils.config import load_config, load_settings, safe_path  # noqa: E402
from rag_hotpotqa.utils.io import atomic_write_json, sha256_file  # noqa: E402
from rag_hotpotqa.utils.runtime import get_logger, git_commit, peak_rss_mb  # noqa: E402

K_FIRST, K_SECOND, K_PAR = 5, 5, 5
FEATURES = ["f_best", "f_other", "dir_gap", "s1_max", "s1_min", "s2_best", "s2_other", "rank_min", "rank_max",
            "bm25_max", "bm25_min", "bm25_cond_best", "logtok_max", "logtok_min", "n_long", "sf_max", "sf_min",
            "p_bridge", "pb_x_f_best", "pb_x_s1_min", "pc_x_s1_min", "gen_bridge", "gen_parallel", "qv2_fallback",
            "title_link_any", "title_link_both", "title_in_q"]


def load_preds(path: Path, into: dict, average: bool) -> None:
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        key = (r["qid"], r["doc"])
        if average and key in into:
            into[key] = (r["sent_ids"], [(a + b) / 2 for a, b in zip(into[key][1], r["scores"])])
        else:
            into[key] = (r["sent_ids"], r["scores"])


def load_model(spec: str, root: Path, d: dict):
    from sentence_transformers import SentenceTransformer

    if spec == "frozen":
        wsha = verify_dense_weights(d["model"], d["revision"], d["weights_sha256"])["weights_sha256"]
        m = SentenceTransformer(d["model"], revision=d["revision"], device="cpu")
    else:
        p = safe_path(root / spec, root)
        m, wsha = SentenceTransformer(str(p), device="cpu"), sha256_file(p / "model.safetensors")
    m.max_seq_length = d["max_seq_length"]
    return m, wsha


def main() -> int:
    settings = load_settings()
    root = settings.project_root
    rcfg = load_config("retrieval", settings)
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--split", choices=["train", "v_select"], required=True)
    ap.add_argument("--n", type=int, default=None, help="Training: first n by sha256 (default 10000); V-select: prefix")
    ap.add_argument("--first-model", default="frozen")
    ap.add_argument("--second-model", default=None, help="default: same as --first-model")
    ap.add_argument("--sf-run", required=True)
    ap.add_argument("--qtype-run", required=True)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    out_dir = safe_path(root / "outputs" / "path_candidates_v1" / args.run_name, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger(log_file=out_dir / f"{args.split}.log")

    import torch

    torch.set_num_threads(args.threads)
    d = rcfg["dense"]
    check_hf_cache_inside_project(settings, True)
    m1, w1 = load_model(args.first_model, root, d)
    m2, w2 = (m1, w1) if args.second_model in (None, args.first_model) else load_model(args.second_model, root, d)
    tmpl = rcfg["document"]["template"]
    sf_dir = root / "outputs" / "sf_selector_v2" / args.sf_run
    qt_dir = root / "outputs" / "qtype_v1" / args.qtype_run
    preds: dict = {}
    if args.split == "train":
        ids = select_subset(load_split_ids(settings, "training", "train"), args.n or 10000)
        recs = load_split_records(settings, "training", "train", ids)
        for f in (0, 1):
            load_preds(sf_dir / f"fold{f}" / f"pred_train_fold{f}.jsonl", preds, average=False)
        pb = {json.loads(x)["qid"]: json.loads(x)["p_bridge"] for x in open(qt_dir / "pred_train_crossfit.jsonl", encoding="utf-8")}
    else:
        ids = load_validation_part(settings, "v_select")
        ids = ids if args.n is None else ids[:args.n]
        recs = load_split_records(settings, "validation", "select", ids)
        for f in (0, 1):
            load_preds(sf_dir / f"fold{f}" / "pred_v_select.jsonl", preds, average=f == 1)
        pb = {json.loads(x)["qid"]: json.loads(x)["p_bridge"] for x in open(qt_dir / "pred_v_select.jsonl", encoding="utf-8")}
    qs = [build_question(r, tmpl) for r in recs]
    tok = m1.tokenizer
    bm25 = BM25Scorer("local")
    rows, labels, qidx, pairs, qinfo = [], [], [], [], []
    t = time.perf_counter()
    for n, q in enumerate(qs):
        docs = [doc.text for doc in q.docs]
        nd = len(docs)
        if nd < 2:
            continue
        e1 = m1.encode(docs + [q.question], batch_size=64, normalize_embeddings=True)
        dv1, s1 = e1[:nd], e1[:nd] @ e1[nd]
        qv2_txt, fallback = [], []
        for a in range(nd):
            ev = evidence_text(q, a, preds, args.k)
            fallback.append(ev is None)
            qv2_txt.append(compose_query(q.question, ev if ev else q.docs[a].body))
        e2 = m2.encode(qv2_txt + (docs if m2 is not m1 else []), batch_size=64, normalize_embeddings=True)
        qv2 = e2[:nd]
        dv2 = dv1 if m2 is m1 else e2[nd:]
        s2 = qv2 @ dv2.T  # s2[a, b] = s(QV2(a), b)
        rank = np.empty(nd, int)
        rank[rank_order(s1)] = np.arange(1, nd + 1)
        bm = np.asarray(bm25.score(q.question, docs, local_corpus=docs), float)
        bm = bm / bm.max() if bm.max() > 0 else bm
        ntok = [len(tok.tokenize(x)) for x in docs]
        sfmax = [max(preds[(q.qid, i)][1]) if (q.qid, i) in preds and len(preds[(q.qid, i)][1]) else 0.0 for i in range(nd)]
        cand: dict[frozenset, set] = {}
        for a in rank_order(s1)[:K_FIRST]:
            others = [b for b in range(nd) if b != a]
            for j in rank_order(s2[a, others])[:K_SECOND]:
                cand.setdefault(frozenset((a, others[j])), set()).add("bridge")
        for x, y in combinations(rank_order(s1)[:K_PAR], 2):
            cand.setdefault(frozenset((x, y)), set()).add("parallel")
        p = pb.get(q.qid, 0.5)
        gold = frozenset(q.gold_indices) if q.is_two_gold else None
        bm_cond_cache = {}
        for key, gen in cand.items():
            x, y = sorted(key)
            fxy, fyx = s1[x] + s2[x, y], s1[y] + s2[y, x]
            a, b = (x, y) if fxy >= fyx else (y, x)
            if (a, b) not in bm_cond_cache:
                sc = bm25.score(qv2_txt[a], [docs[b]], local_corpus=docs)
                bm_cond_cache[(a, b)] = float(sc[0])  # raw BM25, log1p-compressed below
            link_xy = find_title(q.docs[y].title, q.docs[x].body) is not None
            link_yx = find_title(q.docs[x].title, q.docs[y].body) is not None
            tq = int(find_title(q.docs[x].title, q.question) is not None) + int(find_title(q.docs[y].title, q.question) is not None)
            f = {"f_best": max(fxy, fyx), "f_other": min(fxy, fyx), "dir_gap": abs(fxy - fyx),
                 "s1_max": max(s1[x], s1[y]), "s1_min": min(s1[x], s1[y]), "s2_best": s2[a, b], "s2_other": s2[b, a],
                 "rank_min": min(rank[x], rank[y]), "rank_max": max(rank[x], rank[y]), "bm25_max": max(bm[x], bm[y]),
                 "bm25_min": min(bm[x], bm[y]), "bm25_cond_best": math.log1p(max(0.0, bm_cond_cache[(a, b)])),
                 "logtok_max": math.log(max(ntok[x], ntok[y]) + 1), "logtok_min": math.log(min(ntok[x], ntok[y]) + 1),
                 "n_long": int(ntok[x] > 254) + int(ntok[y] > 254), "sf_max": max(sfmax[x], sfmax[y]), "sf_min": min(sfmax[x], sfmax[y]),
                 "p_bridge": p, "pb_x_f_best": p * max(fxy, fyx), "pb_x_s1_min": p * min(s1[x], s1[y]),
                 "pc_x_s1_min": (1 - p) * min(s1[x], s1[y]), "gen_bridge": int("bridge" in gen), "gen_parallel": int("parallel" in gen),
                 "qv2_fallback": int(fallback[a]), "title_link_any": int(link_xy or link_yx), "title_link_both": int(link_xy and link_yx),
                 "title_in_q": tq}
            rows.append([float(f[k]) for k in FEATURES])
            labels.append(int(gold is not None and key == gold))
            qidx.append(len(qinfo))
            pairs.append((x, y, a))
        qinfo.append({"qid": q.qid, "qtype": q.qtype, "two_gold": q.is_two_gold, "n_cand": len(cand),
                      "gold_in_cand": bool(gold is not None and gold in cand)})
        if (n + 1) % 500 == 0:
            log.info("%s: %d/%d questions (%.2f q/s)", args.split, n + 1, len(qs), (n + 1) / (time.perf_counter() - t))
    np.savez_compressed(out_dir / f"{args.split}.npz", X=np.asarray(rows, np.float32), y=np.asarray(labels, np.int8),
                        q=np.asarray(qidx, np.int32), pairs=np.asarray(pairs, np.int16))
    meta = {"split": args.split, "n_questions": len(qinfo), "ids_sha256": ids_hash(ids), "features": FEATURES,
            "first_model": args.first_model, "first_weights_sha256": w1, "second_model": args.second_model or args.first_model,
            "second_weights_sha256": w2, "sf_run": args.sf_run, "qtype_run": args.qtype_run, "k": args.k,
            "candidates": {"k_first": K_FIRST, "k_second": K_SECOND, "k_parallel": K_PAR, "titles_used_for_generation": False},
            "questions": qinfo, "git_commit": git_commit(root), "utc": datetime.now(timezone.utc).isoformat(),
            "seconds": round(time.perf_counter() - t, 1), "peak_rss_mb": round(peak_rss_mb(), 1)}
    atomic_write_json(out_dir / f"{args.split}_meta.json", meta, root)
    cr = np.mean([qi["gold_in_cand"] for qi in qinfo if qi["two_gold"]])
    log.info("%s done: %d questions, %d candidate pairs, candidate path recall %.4f", args.split, len(qinfo), len(rows), cr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
