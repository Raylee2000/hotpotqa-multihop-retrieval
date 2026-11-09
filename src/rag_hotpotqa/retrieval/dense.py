"""Dense bi-encoder scorer (sentence-transformers) with a fingerprinted embedding cache."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .base import NonFiniteScoreError, Scorer, check_finite, compose


class CacheMismatchError(RuntimeError):
    """Cache shard was produced by a different model/config fingerprint."""


class EmbeddingCache:
    """In-memory map text-hash -> vector, persisted as atomic .npz shards under a
    directory named by the model fingerprint. Every shard also stores the
    fingerprint and is rejected on mismatch."""

    def __init__(self, directory: Path | None, fingerprint: dict[str, Any]):
        self.fp = fingerprint
        self.fp_hash = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]
        self.dir = directory / self.fp_hash if directory else None
        self.mem: dict[str, np.ndarray] = {}
        self.pending: dict[str, np.ndarray] = {}
        self.dim: int | None = None
        if self.dir and self.dir.exists():
            for shard in sorted(self.dir.glob("shard_*.npz")):
                self._load_shard(shard)

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load_shard(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as z:
            fp = json.loads(str(z["fingerprint"]))
            if fp != self.fp:
                raise CacheMismatchError(f"{path} fingerprint {fp} != {self.fp}")
            keys, vecs = z["keys"], z["vectors"]
            self._check_dim(vecs.shape[1])
            for k, v in zip(keys, vecs):
                self.mem[str(k)] = v

    def _check_dim(self, d: int) -> None:
        if self.dim is None:
            self.dim = d
        elif d != self.dim:
            raise CacheMismatchError(f"embedding dim {d} != cached dim {self.dim}")

    def get(self, text: str) -> np.ndarray | None:
        return self.mem.get(self.key(text))

    def put(self, text: str, vec: np.ndarray) -> None:
        self._check_dim(vec.shape[-1])
        k = self.key(text)
        self.mem[k] = vec
        self.pending[k] = vec

    def flush(self) -> None:
        if not self.dir or not self.pending:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(self.pending.keys()))
        vecs = np.stack(list(self.pending.values())).astype(np.float32)
        idx = len(list(self.dir.glob("shard_*.npz")))
        target = self.dir / f"shard_{idx:05d}.npz"
        tmp = self.dir / f"shard_{idx:05d}.tmp.npz"
        np.savez(tmp, keys=keys, vectors=vecs, fingerprint=np.array(json.dumps(self.fp, sort_keys=True)))
        os.replace(tmp, target)
        self.pending.clear()


class DenseScorer(Scorer):
    """s(X, Y) = cos(E(X), E(Y)) with a single shared encoder (no prefixes)."""

    def __init__(self, model_name: str, revision: str | None = None, device: str = "cpu",
                 max_seq_length: int = 256, normalize: bool = True, batch_size: int = 64,
                 cache_dir: Path | None = None, joiner: str = "\n\n",
                 q_plus_a_policy: str = "query_first", a_plus_b_policy: str = "balanced",
                 model: Any = None, name: str = "dense", cache_namespace: str = "default"):
        self.name = name
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name, revision=revision, device=device)
        self.model = model
        self.model.max_seq_length = max_seq_length
        self.model_name, self.revision = model_name, revision
        self.max_seq_length, self.normalize, self.batch_size = max_seq_length, normalize, batch_size
        self.joiner, self.qa_policy, self.ab_policy = joiner, q_plus_a_policy, a_plus_b_policy
        self.cache_namespace = cache_namespace
        self.tokenizer = getattr(model, "tokenizer", None)
        self.cache = EmbeddingCache(cache_dir, self.fingerprint())
        self.truncation_stats = {"texts": 0, "truncated": 0}
        self._ntok: dict[str, int] = {}
        self._composed: dict[str, dict[str, Any]] = {}

    # ---- encoding -----------------------------------------------------
    def tokenizer_id(self) -> str:
        if self.tokenizer is None:
            return "none"
        try:
            import transformers

            tv = transformers.__version__
        except ImportError:  # pragma: no cover
            tv = "?"
        return f"{type(self.tokenizer).__name__}@{self.model_name}@{self.revision}|transformers={tv}"

    def fingerprint(self) -> dict:
        return {"name": self.name, "model": self.model_name, "revision": self.revision,
                "max_seq_length": self.max_seq_length, "normalize": self.normalize,
                "namespace": self.cache_namespace}

    def token_info(self, text: str, composed_from: tuple[str, str, str] | None = None) -> dict:
        """Deterministic encoder-input accounting for ``text`` (independent of
        cache hits). Token counts exclude special tokens; budget = max_seq_length
        minus special tokens. For composed texts, ``parts`` gives each side's
        raw and kept token counts and ``policy`` the composition rule."""
        budget = self._budget()
        n_in = self._n_tokens(text)
        comp = None
        if composed_from is not None:
            kind, left, right = composed_from
            policy = {"q_plus_a": self.qa_policy, "a_plus_b": self.ab_policy}[kind]
            key = (policy, left, right)
            if key not in self._composed:
                self._compose(left, right, policy)
            comp = self._composed[key]
        if comp is not None:
            raw = sum(p[0] for p in comp["parts"]) + comp["joiner_tokens"]
            truncated = any(p[0] != p[1] for p in comp["parts"]) or n_in > budget
            policy = comp["policy"]
            parts = comp["parts"]
            chars = comp["chars"]
        else:
            raw, truncated, policy, parts = n_in, n_in > budget, "encoder_max_seq_length", None
            chars = [[len(text), self._kept_chars(text, budget) if truncated else len(text)]]
        # chars: per part [original characters, characters actually kept] (label-free;
        # used after retrieval to check whether gold supporting facts were cut off).
        return {"raw_tokens": raw, "input_tokens": min(n_in, budget), "truncated": bool(truncated),
                "policy": policy, "parts": parts, "chars": chars, "cache_key": EmbeddingCache.key(text)[:16]}

    def _kept_chars(self, text: str, max_tokens: int) -> int:
        """Characters of ``text`` the encoder actually sees when it truncates to ``max_tokens``."""
        if self.tokenizer is None:
            return len(text)
        enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if len(enc["input_ids"]) <= max_tokens:
            return len(text)
        return enc["offset_mapping"][max_tokens - 1][1]

    def _encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=self.normalize,
                                 convert_to_numpy=True, show_progress_bar=False)
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[0] != len(texts):
            raise ValueError(f"unexpected embedding shape {vecs.shape}")
        if not np.all(np.isfinite(vecs)):
            raise NonFiniteScoreError("non-finite embedding values")
        return vecs

    def prefetch(self, texts: Sequence[str], role: str = "doc") -> None:
        missing = list(dict.fromkeys(t for t in texts if self.cache.get(t) is None))
        if missing:
            if self.tokenizer is not None:
                for t in missing:
                    self.truncation_stats["texts"] += 1
                    if self._n_tokens(t) > self._budget():
                        self.truncation_stats["truncated"] += 1
            for t, v in zip(missing, self._encode(missing)):
                self.cache.put(t, v)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.prefetch(texts)
        return np.stack([self.cache.get(t) for t in texts])

    def score(self, query: str, candidates: Sequence[str], local_corpus: Sequence[str] | None = None) -> np.ndarray:
        q = self.embed([query])[0]
        c = self.embed(list(candidates))
        if c.shape[1] != q.shape[0]:
            raise ValueError(f"embedding dim mismatch {c.shape[1]} vs {q.shape[0]}")
        return check_finite(c @ q, self.name)

    # ---- token-budgeted composition ------------------------------------
    def _budget(self) -> int:
        # [CLS] + [SEP] for BERT-style tokenizers.
        n_special = self.tokenizer.num_special_tokens_to_add(pair=False) if self.tokenizer else 2
        return self.max_seq_length - n_special

    def _n_tokens(self, text: str) -> int:
        n = self._ntok.get(text)
        if n is None:
            n = len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
            if len(self._ntok) > 500_000:
                self._ntok.clear()
            self._ntok[text] = n
        return n

    def _truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if len(enc["input_ids"]) <= max_tokens:
            return text
        return text[: enc["offset_mapping"][max_tokens - 1][1]]

    def _compose(self, left: str, right: str, policy: str) -> str:
        if self.tokenizer is None:
            return compose(left, right, self.joiner)
        budget = self._budget() - self._n_tokens(self.joiner) if self.joiner.strip() else self._budget()
        if policy == "query_first":
            left_t = self._truncate(left, budget)
            right_t = self._truncate(right, budget - self._n_tokens(left_t))
        elif policy == "balanced":
            nl, nr = self._n_tokens(left), self._n_tokens(right)
            half = budget // 2
            # Give unused budget from a short side to the other side.
            l_budget = nl if nl <= half else max(half, budget - nr)
            r_budget = budget - min(nl, l_budget)
            left_t, right_t = self._truncate(left, l_budget), self._truncate(right, r_budget)
        else:
            raise ValueError(f"unknown composition policy {policy}")
        out = compose(left_t, right_t, self.joiner)
        if len(self._composed) > 500_000:
            self._composed.clear()
        self._composed[(policy, left, right)] = {
            "policy": policy,
            "parts": [[self._n_tokens(left), self._n_tokens(left_t)], [self._n_tokens(right), self._n_tokens(right_t)]],
            "chars": [[len(left), len(left_t)], [len(right), len(right_t)]],
            "joiner_tokens": self._n_tokens(self.joiner) if self.joiner.strip() else 0,
        }
        return out

    def compose_q_plus_a(self, q: str, a: str) -> str:
        return self._compose(q, a, self.qa_policy)

    def compose_a_plus_b(self, a: str, b: str) -> str:
        return self._compose(a, b, self.ab_policy)
