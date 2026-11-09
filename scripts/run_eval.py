"""Public evaluation runner: the formal Phase 6–8 protocol for the six baselines and the final pipeline.

Protocol kept from the research runner (scripts/overnight_v3/runner.py, execute_attempt / validate_attempt):
  * questions in the canonical order of the part, cut into fixed 128-question shards;
  * every shard runs in a new directory with its own empty embedding cache; the baseline batching replays a
    20-question prefill (the formal protocol), threads 8, HF/Transformers offline;
  * stages: baselines -> live SF / QType inputs -> candidates (General Encoder + CSH, 27 features) -> ranking
    (R5_full, R0); every stage output is validated before the next shard;
  * metrics: PR@1 of the top-1 unordered pair (baselines: both gold in the top-2 articles), question-paired
    bootstrap 2,000 / seed 42, exact McNemar, Holm (src/rag_hotpotqa/evaluation/campaign_stats.py, unchanged).
Differences from the research runner (release gate also described in README.md §08): the 285-file research freeze is replaced
by configs/release_manifest.json + configs/models.json + the data lock (a failed check still stops the run); no git
add/commit; no Windows Task Scheduler; R5 uses the pickle-free tree file; Matrix A pipeline ablations are not run.

Usage:
  python scripts/run_eval.py --part official_dev_distractor [--part v_report ...] [--shards N] [--stages all|baselines]
                             [--models-manifest outputs/trained_models/<profile>/models.json]
Output (fail-if-exists): outputs/runs/<run-name>/
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_hotpotqa.release.integrity import (MODELS_MANIFEST_ENV, RELEASE_MANIFEST, IntegrityError, model_rel,  # noqa: E402
                                            models_manifest_path, read_json, sha256_path, verify_base_model, verify_data,
                                            verify_models, verify_release_manifest)

PARTS = ["official_dev_distractor", "v_report", "bc_holdout"]
BASELINES = ["bm25_local", "bm25_trainglobal", "dense", "hybrid_local", "hybrid_trainglobal", "general_best"]
SHARD = 128


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def unique_rows(path: Path, expected: list[str]) -> list[dict]:
    rs = rows(path)
    ids = [r["qid"] for r in rs]
    if ids != list(expected) or len(ids) != len(set(ids)):
        raise RuntimeError(f"ID/order/completeness mismatch {path}")
    return rs


def preflight() -> dict:
    from rag_hotpotqa.utils.config import load_settings

    man = verify_release_manifest()
    verify_models()
    st = load_settings()
    base = verify_base_model(st)
    lock = verify_data(full=True)
    return {"release_manifest_sha256": sha256_path(ROOT / RELEASE_MANIFEST), "code_files_verified": len(man["files"]),
            "models_manifest": models_manifest_path(), "models_manifest_sha256": sha256_path(ROOT / models_manifest_path()),
            "models_verified": list(read_json(models_manifest_path())["models"]), "base_model": base, "data_lock": "PASS",
            "lock": lock}


def run_stage(cmd: list[str], log: Path) -> None:
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
               PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    t = time.perf_counter()
    with open(log, "xb") as f:
        rc = subprocess.run([sys.executable, "-B", "-X", "utf8", *map(str, cmd)], cwd=ROOT, env=env,
                            stdout=f, stderr=subprocess.STDOUT).returncode
    (log.with_suffix(".runtime.json")).write_text(json.dumps({"seconds": time.perf_counter() - t, "exit_code": rc}) + "\n", encoding="utf-8")
    if rc:
        raise RuntimeError(f"stage failed (exit {rc}); see {log.relative_to(ROOT)}")


def validate_shard(b: Path, part: str, ids: list[str], stages: str) -> None:
    from rag_hotpotqa.data.splits import ids_hash

    doc_counts = None
    for n in BASELINES:
        r = unique_rows(b / f"baselines/{n}.jsonl", ids)
        current = {x["qid"]: x["n_docs"] for x in r}
        if doc_counts is None:
            doc_counts = current
        assert current == doc_counts
        assert all(2 <= x["n_docs"] <= 10 and sorted(x["detail"]["O"]["order"]) == list(range(x["n_docs"])) for x in r)
        assert json.loads((b / f"baselines/{n}.status.json").read_text(encoding="utf-8"))["failed_this_session"] == 0
    if stages == "baselines":
        return
    live = b / "live_inputs"
    unique_rows(live / "qtype.jsonl", ids)
    for fold in (0, 1):
        sf = rows(live / f"sf_fold{fold}.jsonl")
        keys = [(r["qid"], r["doc"]) for r in sf]
        assert keys == [(q, d) for q in ids for d in range(doc_counts[q])] and len(keys) == len(set(keys))
        assert all(len(r["sent_ids"]) == len(r["scores"]) and np.isfinite(r["scores"]).all() for r in sf)
    c = b / "candidates/general_csh"
    meta = json.loads((c / f"{part}_meta.json").read_text(encoding="utf-8"))
    assert [q["qid"] for q in meta["questions"]] == ids and meta["ids_sha256"] == ids_hash(ids)
    with np.load(c / f"{part}.npz", allow_pickle=False) as z:
        assert z["X"].shape[1] == 27 and np.isfinite(z["X"]).all()
        assert set(np.unique(z["y"])) <= {0, 1}
        for i in range(len(ids)):
            idx = np.flatnonzero(z["q"] == i)
            pairs = [tuple(p[:2]) for p in z["pairs"][idx]]
            assert len(pairs) == len(set(pairs)) and z["y"][idx].sum() <= 1
            assert all(0 <= a < bb < doc_counts[ids[i]] for a, bb in pairs)
    ranked = unique_rows(b / "ranking/per_question.jsonl", ids)
    assert all(set(r["methods"]) == {"R5_full", "R0"} for r in ranked)


def shard_rows(b: Path, ids: list[str], stages: str) -> list[dict]:
    """Same conversion as the research runner's analysis_rows (baseline PR@1 = O.fh_all_gold@2)."""
    if stages == "baselines":
        by_id = {q: {"qid": q, "qtype": None, "methods": {}} for q in ids}
    else:
        by_id = {r["qid"]: r for r in rows(b / "ranking/per_question.jsonl")}
    for n in BASELINES:
        for r in rows(b / f"baselines/{n}.jsonl"):
            m, order = r["metrics"], r["detail"]["O"]["order"]
            hit = int(m["O.fh_all_gold@2"])
            by_id[r["qid"]]["qtype"] = by_id[r["qid"]]["qtype"] or r["qtype"]
            by_id[r["qid"]]["methods"][f"baseline/{n}"] = {"pr1": hit, "cand": hit, "rr": float(hit), "top1_pair": sorted(order[:2]),
                                                           "candidate_definition": "one top-2 unordered article pair"}
    return [by_id[q] for q in ids]


def run_part(run_dir: Path, part: str, auth: Path, n_shards: int | None, stages: str, lock: dict) -> dict:
    from rag_hotpotqa.data.eval_parts import load_eval_part
    from rag_hotpotqa.data.splits import ids_hash
    from rag_hotpotqa.evaluation.campaign_stats import markdown, summarize
    from rag_hotpotqa.utils.config import load_settings

    ids, recs = load_eval_part(load_settings(), part, str(auth))
    del recs
    if len(ids) != lock["parts"][part]["n"] or ids_hash(ids) != lock["parts"][part]["ids_sha256"]:
        raise IntegrityError(f"{part}: evaluation IDs differ from the lock")
    starts = list(range(0, len(ids), SHARD))[: n_shards or None]
    out_rows, t0 = [], time.perf_counter()
    for k, start in enumerate(starts):
        sid = ids[start:start + SHARD]
        shard = run_dir / part / f"shard_{start // SHARD:05d}"
        shard.mkdir(parents=True)
        ids_file = shard / "ids.txt"
        ids_file.write_text("\n".join(sid) + "\n", encoding="utf-8", newline="\n")
        work = shard / "work"
        common = ["--part", part, "--out-root", work.relative_to(ROOT), "--ids-file", ids_file.relative_to(ROOT),
                  "--authorisation", auth.relative_to(ROOT)]
        cmds = [("baselines", ["scripts/final_eval/run_baselines.py", *common, "--cache-dir", (shard / "cache").relative_to(ROOT),
                               "--prefill-questions", str(min(20, len(sid)))])]
        if stages == "all":
            cmds += [("live_inputs", ["scripts/final_eval/live_inputs.py", *common]),
                     ("candidates", ["scripts/final_eval/gen_candidates.py", "--split", part, "--out-root", work.relative_to(ROOT),
                                     "--ids-file", ids_file.relative_to(ROOT), "--authorisation", auth.relative_to(ROOT),
                                     "--run-name", "general_csh", "--first-model", model_rel("general_encoder"),
                                     "--second-model", model_rel("csh"), "--sf-run", "sf_cf_v1", "--qtype-run", "qtype_v2"]),
                     ("ranking", ["scripts/rank_public.py", "--base", (work / part).relative_to(ROOT), "--part", part])]
        for stage, cmd in cmds:
            print(f"{now()} {part} shard {k + 1}/{len(starts)} {stage}", flush=True)
            run_stage(cmd, shard / f"{stage}.console.log")
        validate_shard(work / part, part, sid, stages)
        out_rows += shard_rows(work / part, sid, stages)
    pdir = run_dir / part
    with open(pdir / "per_question.jsonl", "x", encoding="utf-8", newline="\n") as f:
        for r in out_rows:
            f.write(json.dumps(r, allow_nan=False) + "\n")
    summary = summarize(out_rows, full="R5_full" if stages == "all" else "baseline/general_best")
    summary["provenance"] = {"part": part, "questions": len(out_rows), "of_total": len(ids), "shards": len(starts),
                             "complete_part": len(out_rows) == len(ids), "stages": stages, "seconds": time.perf_counter() - t0,
                             "qtype_counts": dict(Counter(r["qtype"] for r in out_rows))}
    (pdir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n")
    (pdir / "report.md").write_text(markdown(f"{part} — public release run", summary, json.dumps(summary["provenance"])),
                                    encoding="utf-8", newline="\n")
    return summary["provenance"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", action="append", choices=PARTS, required=True)
    ap.add_argument("--shards", type=int, default=None, help="only the first N shards (smoke test: 1)")
    ap.add_argument("--stages", choices=["all", "baselines"], default="all")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--models-manifest", default=None,
                    help="repository-relative model manifest of retrained weights (written by python run.py train); "
                         "default: configs/models.json (released weights in models/)")
    args = ap.parse_args()
    if args.models_manifest:
        os.environ[MODELS_MANIFEST_ENV] = args.models_manifest  # inherited by every stage subprocess
    run_dir = ROOT / "outputs/runs" / (args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    try:
        pre = preflight()
    except (IntegrityError, Exception) as e:  # noqa: BLE001 - any preflight failure stops before evaluation
        print(f"STOP preflight: {e}", file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True, exist_ok=False)
    lock = pre.pop("lock")
    auth = run_dir / "authorisation.json"
    auth.write_text(json.dumps({"status": "FROZEN", "approved_parts": args.part,
                                "approved_by": "public release run after release-manifest, model and data verification",
                                "approved_utc": now(), "freeze_manifest": RELEASE_MANIFEST,
                                "freeze_manifest_sha256": pre["release_manifest_sha256"]}, indent=1) + "\n", encoding="utf-8")
    (run_dir / "preflight.json").write_text(json.dumps(pre, indent=1) + "\n", encoding="utf-8")
    results = {}
    try:
        for part in args.part:
            results[part] = run_part(run_dir, part, auth, args.shards, args.stages, lock)
    except (IntegrityError, RuntimeError, AssertionError) as e:
        print(f"STOP: {type(e).__name__}: {e}", file=sys.stderr)
        return 3
    (run_dir / "RUN_COMPLETE.json").write_text(json.dumps({"utc": now(), "parts": results}, indent=1) + "\n", encoding="utf-8")
    print(f"PASS {run_dir.relative_to(ROOT).as_posix()}  (compare: python run.py compare --run {run_dir.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
