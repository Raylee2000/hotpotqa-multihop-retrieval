"""[final_eval_v1 GENERIC ENTRY: --part {v_select, v_report, bc_holdout} via rag_hotpotqa.data.eval_parts.load_eval_part;
v_report / bc_holdout need --authorisation (FROZEN, approved). Outputs: <out-root>/<part>/. Diff against the source shows the only changes.]

V-select 2K first-hop baselines with the unchanged Phase 1 implementation (experiment O, spec v2).

Retrievers: bm25_local, bm25_trainglobal, dense (frozen), hybrid_local, hybrid_trainglobal — built exactly as
retrieval.factory.build_retrievers (same BM25 parameters, TrainGlobal IDF cache, RRF k) — plus general_best
(hardneg_v1 step 1869) as a Dense scorer, built as scripts/eval_retriever_validation.py does. Candidate set = the
question's context articles for every retriever (TrainGlobal differs only in the IDF statistics).
The requested partition is loaded through the authorisation gate. A historical
prefix-run batching protocol can be replayed with --prefill-questions and a fresh
--cache-dir; the reference V-select protocol used a 20-question smoke run.
Output (fail-if-exists): <out-root>/<part>/<out-name>/.
The prefill count is a batching parameter, not a selection or training step.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rag_hotpotqa.data.eval_parts import PARTS, load_eval_part, select_eval_subset  # noqa: E402
from rag_hotpotqa.data.documents import build_question  # noqa: E402
from rag_hotpotqa.data.splits import ids_hash  # noqa: E402
from rag_hotpotqa.evaluation.summarize import load_slim, summarize_retriever  # noqa: E402
from rag_hotpotqa.experiments.runner import _prefetch, run_retriever  # noqa: E402
from rag_hotpotqa.retrieval.bm25 import BM25Scorer  # noqa: E402
from rag_hotpotqa.retrieval.dense import DenseScorer  # noqa: E402
from rag_hotpotqa.retrieval.factory import check_hf_cache_inside_project, load_or_build_trainglobal_stats, verify_dense_weights  # noqa: E402
from rag_hotpotqa.retrieval.similarity import Similarity  # noqa: E402
from rag_hotpotqa.utils.config import load_config, load_settings, safe_path  # noqa: E402
from rag_hotpotqa.utils.io import atomic_write_json, sha256_file  # noqa: E402
from rag_hotpotqa.utils.runtime import get_logger, git_commit, peak_rss_mb, set_seed  # noqa: E402

from rag_hotpotqa.release.integrity import model_rel  # noqa: E402

# public release layout: models/general_encoder (was outputs/training_v1/hardneg_v1/snapshot_step01869), or the
# retrained General Encoder when the run uses a trained model manifest (RAG_HOTPOTQA_MODELS_MANIFEST)
GENERAL_BEST = model_rel("general_encoder")
ORDER = ["bm25_local", "bm25_trainglobal", "dense", "hybrid_local", "hybrid_trainglobal", "general_best"]


def prefill_embeddings(questions, sims, n: int, chunk_size: int) -> None:
    """Replay a declared prefix run's batching, using unlabeled encoder inputs.

    A historical smoke run populated the reference cache before its full run.
    CPU float32 embeddings can depend on padded batch shape. Replay that
    protocol explicitly, rather than loading reference predictions or vectors.
    """
    if not 0 <= n <= len(questions):
        raise ValueError("prefill count must be between zero and the evaluation size")
    for name in ("dense", "general_best"):
        for start in range(0, n, chunk_size):
            _prefetch(questions[start:min(start + chunk_size, n)], sims[name], ["O"], {})
            for c in sims[name].components:
                c.cache.flush()


def main() -> int:
    settings = load_settings()
    root = settings.project_root
    rcfg, dcfg, ecfg = (load_config(n, settings) for n in ("retrieval", "data", "experiments"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="smoke: V-select prefix")
    ap.add_argument("--out-name", default="baselines")
    ap.add_argument("--part", choices=PARTS, default="v_select")
    ap.add_argument("--ids-file", default=None)
    ap.add_argument("--authorisation", default=None)
    ap.add_argument("--out-root", default="outputs/final_eval_v1")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--cache-dir", default=".cache/embeddings_final_eval_v1",
                    help="embedding cache under the project; use a new directory for protocol replay")
    ap.add_argument("--prefill-questions", type=int, default=0,
                    help="explicit prefix-run batch protocol (historical V-select reference: 20)")
    args = ap.parse_args()
    run_dir = safe_path(root / args.out_root / args.part / args.out_name, root)
    if run_dir.exists():
        raise SystemExit(f"{run_dir} exists — refusing to overwrite")
    cache = safe_path(root / args.cache_dir, root)
    if args.prefill_questions < 0:
        raise SystemExit("prefill count must be nonnegative")
    if args.prefill_questions and cache.exists() and any(cache.iterdir()):
        raise SystemExit("protocol replay requires a fresh embedding cache; refusing existing cache")
    run_dir.mkdir(parents=True)
    log = get_logger(log_file=run_dir / "run.log")
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(args.threads)
    set_seed(settings.random_seed)
    ids, recs = load_eval_part(settings, args.part, args.authorisation)
    ids, recs = select_eval_subset(ids, recs, args.ids_file)
    if args.n is not None:
        ids, recs = ids[:args.n], recs[:args.n]
    if args.prefill_questions > len(ids):
        raise SystemExit("prefill count exceeds evaluation size")
    qs = [build_question(r, rcfg["document"]["template"]) for r in recs]
    joiner, qa, b, d = rcfg["compose"]["joiner"], rcfg["document"]["query_article"], rcfg["bm25"], rcfg["dense"]
    tg_stats, tg_meta = load_or_build_trainglobal_stats(settings, rcfg, dcfg)
    bm = {"bm25_local": BM25Scorer("local", None, b["k1"], b["b"], joiner, stats_version=b["local"]["stats_version"], name="bm25_local"),
          "bm25_trainglobal": BM25Scorer("trainglobal", tg_stats, b["k1"], b["b"], joiner, stats_version=b["trainglobal"]["stats_version"],
                                         name="bm25_trainglobal")}
    check_hf_cache_inside_project(settings, True)
    base = verify_dense_weights(d["model"], d["revision"], d["weights_sha256"])
    common = dict(max_seq_length=d["max_seq_length"], normalize=d["normalize"], batch_size=d["batch_size"], cache_dir=cache,
                  joiner=joiner, q_plus_a_policy=rcfg["compose"]["q_plus_a"], a_plus_b_policy=rcfg["compose"]["a_plus_b"])
    frozen = DenseScorer(d["model"], revision=d["revision"], device="cpu", name="dense", cache_namespace=d["cache_namespace"], **common)
    gpath = safe_path(root / GENERAL_BEST, root)
    gsha = sha256_file(gpath / "model.safetensors")
    gen = DenseScorer(f"finetuned:{GENERAL_BEST}", revision=gsha[:16], model=SentenceTransformer(str(gpath), device="cpu"),
                      name="dense", cache_namespace="vselect_eval_v1_general_best", **common)
    sims = {"bm25_local": Similarity("bm25_local", [bm["bm25_local"]], query_article=qa),
            "bm25_trainglobal": Similarity("bm25_trainglobal", [bm["bm25_trainglobal"]], query_article=qa),
            "dense": Similarity("dense", [frozen], query_article=qa),
            "hybrid_local": Similarity("hybrid_local", [bm["bm25_local"], frozen], fusion="rrf", rrf_k=rcfg["hybrid"]["k"], query_article=qa),
            "hybrid_trainglobal": Similarity("hybrid_trainglobal", [bm["bm25_trainglobal"], frozen], fusion="rrf", rrf_k=rcfg["hybrid"]["k"],
                                             query_article=qa),
            "general_best": Similarity("general_best", [gen], query_article=qa)}
    exp_cfg = {"ks": ecfg["ks"], **ecfg["paths"]}
    prefill_embeddings(qs, sims, args.prefill_questions, ecfg["run"]["chunk_size"])
    runs, summaries, timing = {}, {}, {}
    for name in ORDER:
        t = time.perf_counter()
        enc_id = None if name.startswith("bm25") else (f"{d['model']}@{d['revision']}" if name != "general_best" else f"general_best@{gsha[:16]}")
        runs[name] = run_retriever(qs, sims[name], ["O"], exp_cfg, ecfg["stop"], run_dir, root, enc_id, ecfg["run"]["chunk_size"])
        timing[name] = round(time.perf_counter() - t, 1)
        summaries[name] = summarize_retriever(load_slim(run_dir / f"{name}.jsonl"), ["O"], exp_cfg)
        log.info("%s done in %.1f s", name, timing[name])
    atomic_write_json(run_dir / "summary.json", summaries, root)
    atomic_write_json(run_dir / "manifest.json", {
        "split": args.part, "n_questions": len(qs), "question_ids_sha256": ids_hash(ids), "experiments": ["O"],
        "retrievers": ORDER, "candidate_scope": "the question's context articles (distractor setting)",
        "bm25": {"k1": b["k1"], "b": b["b"], "trainglobal_stats": tg_meta}, "hybrid_rrf_k": rcfg["hybrid"]["k"],
        "dense": {"frozen_weights_sha256": base["weights_sha256"], "general_best": GENERAL_BEST, "general_best_sha256": gsha,
                  "max_seq_length": d["max_seq_length"], "cache_dir": cache.relative_to(root).as_posix()},
        "fingerprints": {n: s.fingerprint() for n, s in sims.items()}, "runtime_seconds": timing,
        "encoding_protocol": {"prefill_questions": args.prefill_questions, "chunk_size": ecfg["run"]["chunk_size"],
                              "batch_size": d["batch_size"], "threads": args.threads, "normalize": d["normalize"],
                              "dtype": "float32", "prefix_ids_sha256": ids_hash(ids[:args.prefill_questions])},
        "runs": {n: {k: v for k, v in r.items() if k in ("n_completed", "n_unresolved_failures", "peak_rss_mb")} for n, r in runs.items()},
        "artifacts_sha256": {f"{n}.jsonl": sha256_file(run_dir / f"{n}.jsonl") for n in ORDER},
        "git_commit": git_commit(root), "utc": datetime.now(timezone.utc).isoformat(), "peak_rss_mb": round(peak_rss_mb(), 1)}, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
