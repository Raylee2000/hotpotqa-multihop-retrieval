"""Unattended, resumable runner for spec-v2 O/E/R (D-14).

Failure policy
- Non-fatal (per question): any ordinary exception while computing one
  question. It is appended to failures.jsonl and the run continues. The
  question is NOT written to the results file, so it is retried on resume and
  always counted in the reported denominators (attempted vs completed).
- Fatal (stop the run): split-policy / split-integrity violations, model or
  cache version mismatch, configuration change on resume, path-safety
  violations, insufficient RAM / disk, MemoryError, and a failure rate above
  ``stop.max_failure_rate`` (circuit breaker).
Progress is checkpointed per question (fsync'ed JSONL) and per chunk
(embedding cache flush + status.json heartbeat).
"""
from __future__ import annotations

import hashlib
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ..data.access import SplitIntegrityError, SplitPolicyError
from ..data.documents import Question
from ..retrieval.base import rank_order
from ..retrieval.dense import CacheMismatchError
from ..retrieval.factory import ModelVersionError
from ..retrieval.similarity import Similarity
from ..utils.config import PathSafetyError
from ..utils.io import ResumableJsonl, atomic_write_json, read_jsonl, stable_json
from ..utils.runtime import check_resources, current_rss_mb, get_logger, peak_rss_mb
from .multihop import SPEC_VERSION, max_width, run_question


class ConfigMismatchError(RuntimeError):
    """Resume attempted with a configuration different from the original run (fatal)."""


class ResourceError(RuntimeError):
    """Not enough RAM or disk to continue safely (fatal)."""


class FailureRateError(RuntimeError):
    """Too many per-question failures (fatal circuit breaker)."""


FATAL = (SplitPolicyError, SplitIntegrityError, ModelVersionError, CacheMismatchError, ConfigMismatchError,
         PathSafetyError, ResourceError, FailureRateError, MemoryError, KeyboardInterrupt)


def select_subset(ids: Sequence[str], n: int | None) -> list[str]:
    """Deterministic subset: the n IDs with smallest sha256(id); n=None keeps all (in that order)."""
    ordered = sorted(ids, key=lambda i: hashlib.sha256(i.encode()).hexdigest())
    return ordered if n is None else ordered[:n]


def check_or_write_run_config(run_dir: Path, root: Path, config: dict[str, Any]) -> None:
    path = run_dir / "run_config.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if stable_json(old["config"]) != stable_json(config):
            raise ConfigMismatchError(f"{path} differs from the current configuration; use a new run name")
        return
    atomic_write_json(path, {"created_utc": datetime.now(timezone.utc).isoformat(), "config": config}, root)


def _prefetch(chunk: Sequence[Question], sim: Similarity, experiments: list[str], cfg: dict) -> None:
    """Batch-encode texts the chunk will need (performance only; scores unchanged)."""
    needs_qa = any(e in experiments for e in ("E1", "R1", "R2", "R3"))
    n_max = max_width(cfg) if "R1" in experiments else None
    limited = needs_qa and n_max is not None and "R3" not in experiments  # width-limited runs (validation eval)
    texts: list[list[str]] = [[] for _ in sim.components]
    for q in chunk:
        qi = q.unlabeled()
        docs = qi.docs
        if limited:
            top = rank_order(sim.s_q(qi, qi.docs, "prefetch"))[:n_max] if "R1" in experiments else []
            docs = [d for d in qi.docs if d.index in set(top) | set(q.gold_indices)]
        qa = [d.body if sim.query_article == "body" else d.text for d in docs] if needs_qa else []
        ab = [q.docs[g] for g in q.gold_indices] if "R2" in experiments else []
        for i, ts in enumerate(sim.texts_for(qi, qa_contents=qa, ab_from=ab)):
            texts[i].extend(ts)
    sim.prefetch_texts(texts)
    if "R3" in experiments:
        texts = [[] for _ in sim.components]
        for q in chunk:
            qi = q.unlabeled()
            top = rank_order(sim.s_q(qi, qi.docs, "prefetch"))[: cfg["r3_width"]]
            for i, ts in enumerate(sim.texts_for(qi, ab_from=[qi.docs[a] for a in top])):
                texts[i].extend(ts)
        sim.prefetch_texts(texts)


def run_retriever(questions: Sequence[Question], sim: Similarity, experiments: list[str], cfg: dict,
                  stop: dict, run_dir: Path, root: Path, encoder_id: str | None, chunk_size: int) -> dict[str, Any]:
    log = get_logger()
    store = ResumableJsonl(run_dir / f"{sim.name}.jsonl", root, key_field="qid", keep_records=False)
    failures_path = run_dir / f"{sim.name}.failures.jsonl"
    todo = [q for q in questions if q.qid not in store]
    log.info("[%s] experiments=%s done=%d todo=%d", sim.name, experiments, len(questions) - len(todo), len(todo))
    attempted = failed = 0
    t0 = time.perf_counter()
    status_path = run_dir / f"{sim.name}.status.json"
    for start in range(0, len(todo), chunk_size):
        try:
            check_resources(run_dir, stop["min_free_disk_gb"], stop["min_free_ram_gb"])
        except RuntimeError as exc:
            raise ResourceError(str(exc)) from exc
        chunk = todo[start:start + chunk_size]
        _prefetch(chunk, sim, experiments, cfg)
        for q in chunk:
            attempted += 1
            try:
                row = run_question(q, sim, experiments, cfg, encoder_id)
                store.append(row)
            except FATAL:
                raise
            except Exception as exc:  # noqa: BLE001 - non-fatal per-question failure
                failed += 1
                with open(failures_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"qid": q.qid, "retriever": sim.name, "utc": datetime.now(timezone.utc).isoformat(),
                                         "error": f"{type(exc).__name__}: {exc}",
                                         "trace": traceback.format_exc(limit=4)}, ensure_ascii=False) + "\n")
                log.error("[%s] question %s failed (non-fatal): %s", sim.name, q.qid, exc)
                if failed > max(stop["min_failures"], stop["max_failure_rate"] * attempted):
                    raise FailureRateError(f"{failed} failures in {attempted} attempts") from exc
        for c in sim.components:
            cache = getattr(c, "cache", None)
            if cache is not None:
                cache.flush()
        elapsed = time.perf_counter() - t0
        rate = attempted / elapsed if elapsed else 0.0
        remaining = len(todo) - attempted
        status = {"retriever": sim.name, "completed_total": len(store.done), "total": len(questions),
                  "attempted_this_session": attempted, "failed_this_session": failed,
                  "q_per_s": round(rate, 3), "eta_s": round(remaining / rate, 1) if rate else None,
                  "rss_mb": round(current_rss_mb(), 1), "peak_rss_mb": round(peak_rss_mb(), 1),
                  "utc": datetime.now(timezone.utc).isoformat()}
        atomic_write_json(status_path, status, root)
        log.info("[%s] %d/%d done, failed %d, %.2f q/s, ETA %s s, RSS %.0f MB", sim.name, status["completed_total"],
                 len(questions), failed, rate, status["eta_s"], status["rss_mb"])
    unresolved = sorted({f["qid"] for f in read_jsonl(failures_path)} - set(store.done)) if failures_path.exists() else []
    return {"retriever": sim.name, "experiments": experiments, "n_questions": len(questions),
            "n_completed": len(store.done), "n_unresolved_failures": len(unresolved), "unresolved_failure_qids": unresolved,
            "attempted_this_session": attempted, "failed_this_session": failed,
            "wall_seconds_this_session": round(time.perf_counter() - t0, 2),
            "peak_rss_mb": round(peak_rss_mb(), 1), "spec_version": SPEC_VERSION,
            "rows_path": (run_dir / f"{sim.name}.jsonl").relative_to(root).as_posix()}
