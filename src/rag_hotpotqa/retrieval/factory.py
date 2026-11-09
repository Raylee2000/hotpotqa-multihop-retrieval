"""Build the spec-v2 retrievers from configs/retrieval.yaml, with model / cache guards."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..data.documents import render_text
from ..data.loader import load_canonical_jsonl
from ..utils.config import Settings, load_config
from ..utils.io import sha256_file
from .bm25 import FORMULA_ID, TOKENIZER_ID, BM25Scorer, BM25Stats
from .dense import DenseScorer
from .similarity import Similarity


class ModelVersionError(RuntimeError):
    """The loaded model / cache location differs from the pinned configuration (fatal)."""


def load_or_build_trainglobal_stats(settings: Settings, rcfg: dict, dcfg: dict) -> tuple[BM25Stats, dict[str, Any]]:
    """IDF and avgdl over all Train context articles (label-free text), de-duplicated
    by sha256 of the rendered text. Never reads answers, supporting facts or Dev."""
    tg = rcfg["bm25"]["trainglobal"]
    train_path = settings.project_root / dcfg["canonical"]["train_path"]
    template = rcfg["document"]["template"]
    train_sha = sha256_file(train_path)
    key_src = [train_sha, template, TOKENIZER_ID, tg["stats_version"], tg["dedup"]]
    key = hashlib.sha256(json.dumps(key_src).encode()).hexdigest()[:16]
    cache = settings.cache_root / "bm25" / f"trainglobal_{tg['stats_version']}_{key}.json.gz"
    if cache.exists():
        stats = BM25Stats.load(cache)
    else:
        seen: dict[str, str] = {}
        for rec in load_canonical_jsonl(train_path):
            for title, sents in rec["context"]:
                text = render_text(title, sents, template)
                seen.setdefault(hashlib.sha256(text.encode("utf-8")).hexdigest(), text)
        stats = BM25Stats.from_texts(seen.values())
        cache.parent.mkdir(parents=True, exist_ok=True)
        stats.save(cache)
    meta = {"stats_version": tg["stats_version"], "dedup": tg["dedup"], "train_canonical_sha256": train_sha,
            "cache_file": cache.relative_to(settings.project_root).as_posix(), "cache_key": key,
            "n_docs": stats.n_docs, "avgdl": round(stats.avgdl, 4), "vocab": len(stats.df),
            "tokenizer": TOKENIZER_ID, "formula": FORMULA_ID}
    return stats, meta


def check_hf_cache_inside_project(settings: Settings, require_offline: bool) -> dict[str, Any]:
    import huggingface_hub.constants as hfc

    cache = Path(hfc.HF_HUB_CACHE).resolve()
    try:
        cache.relative_to(settings.project_root.resolve())
    except ValueError as exc:
        raise ModelVersionError(f"HF cache {cache} is outside the project (other projects' models must not be used)") from exc
    if require_offline and not hfc.HF_HUB_OFFLINE:
        raise ModelVersionError("formal runs require HF offline mode (ALLOW_REMOTE_MODEL_DOWNLOAD=false)")
    return {"hf_hub_cache": cache.relative_to(settings.project_root.resolve()).as_posix(), "offline": bool(hfc.HF_HUB_OFFLINE)}


def verify_dense_weights(model_name: str, revision: str, expected_sha: str) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(model_name, revision=revision, local_files_only=True))
    weights = path / "model.safetensors"
    got = sha256_file(weights)
    if got != expected_sha:
        raise ModelVersionError(f"{model_name}@{revision} weights sha256 {got} != pinned {expected_sha}")
    return {"name": model_name, "revision": revision, "weights_sha256": got, "weights_bytes": weights.stat().st_size}


def build_retrievers(settings: Settings, device: str, names: list[str], require_offline: bool = True
                     ) -> tuple[dict[str, Similarity], dict[str, Any]]:
    rcfg = load_config("retrieval", settings)
    dcfg = load_config("data", settings)
    specs = rcfg["retrievers"]
    unknown = set(names) - set(specs)
    if unknown:
        raise ValueError(f"unknown retrievers {unknown}")
    joiner = rcfg["compose"]["joiner"]
    qa = rcfg["document"]["query_article"]
    b = rcfg["bm25"]
    info: dict[str, Any] = {}

    needed_bm25 = {n if specs[n]["type"] == "bm25" else specs[n].get("bm25") for n in names} - {None}
    bm25: dict[str, BM25Scorer] = {}
    for bn in sorted(needed_bm25):
        scope = specs[bn]["scope"]
        if scope == "trainglobal":
            stats, info["bm25_trainglobal_stats"] = load_or_build_trainglobal_stats(settings, rcfg, dcfg)
        else:
            stats = None
            info["bm25_local_stats"] = {"stats_version": b["local"]["stats_version"],
                                        "corpus": "all context articles of the question"}
        bm25[bn] = BM25Scorer(scope, stats, b["k1"], b["b"], joiner, stats_version=b[scope]["stats_version"], name=bn)

    dense = None
    if any(specs[n]["type"] in {"dense", "hybrid"} for n in names):
        info["hf_cache"] = check_hf_cache_inside_project(settings, require_offline)
        d = rcfg["dense"]
        info["dense_model"] = verify_dense_weights(d["model"], d["revision"], d["weights_sha256"])
        dense = DenseScorer(d["model"], revision=d["revision"], device=device, max_seq_length=d["max_seq_length"],
                            normalize=d["normalize"], batch_size=d["batch_size"],
                            cache_dir=settings.cache_root / "embeddings", joiner=joiner,
                            q_plus_a_policy=rcfg["compose"]["q_plus_a"], a_plus_b_policy=rcfg["compose"]["a_plus_b"],
                            name="dense", cache_namespace=d["cache_namespace"])
        info["dense_model"]["tokenizer"] = dense.tokenizer_id()
        info["dense_model"]["embedding_cache"] = f".cache/embeddings/{dense.cache.fp_hash}"

    sims: dict[str, Similarity] = {}
    for n in names:
        t = specs[n]["type"]
        if t == "bm25":
            sims[n] = Similarity(n, [bm25[n]], query_article=qa)
        elif t == "dense":
            sims[n] = Similarity(n, [dense], query_article=qa)
        else:
            sims[n] = Similarity(n, [bm25[specs[n]["bm25"]], dense], fusion="rrf", rrf_k=rcfg["hybrid"]["k"],
                                 query_article=qa)
    return sims, info
