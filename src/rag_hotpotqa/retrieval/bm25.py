"""Okapi BM25 with fixed tokenizer, formula and parameters; two statistics scopes.

- local:       IDF and avgdl from the question's own context articles (every
               context article of that question, exactly once, as rendered
               "title\\nbody"), independent of which candidates a call scores.
- trainglobal: IDF and avgdl from the de-duplicated set of all context articles
               of the official Train file (label-free text).
Retrieval scope is the question's candidates in both cases.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, OrderedDict
from typing import Iterable, Sequence

import numpy as np

from .base import Scorer, check_finite, compose

_TOKEN = re.compile(r"\w+", re.UNICODE)
TOKENIZER_ID = "nfkc_lower_regex_w+_v1"
FORMULA_ID = "okapi_bm25_lucene_idf_v1"
LOCAL_STATS_MEMO = 256


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(unicodedata.normalize("NFKC", text).lower())


class BM25Stats:
    """Document frequencies and average length over a fixed corpus."""

    def __init__(self, n_docs: int, df: dict[str, int], avgdl: float):
        if n_docs <= 0 or avgdl <= 0:
            raise ValueError("BM25 stats need a non-empty corpus")
        self.n_docs, self.df, self.avgdl = n_docs, df, avgdl

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> "BM25Stats":
        df: Counter[str] = Counter()
        n, total = 0, 0
        for t in texts:
            toks = tokenize(t)
            df.update(set(toks))
            n += 1
            total += len(toks)
        return cls(n, dict(df), total / max(n, 1))

    def idf(self, term: str) -> float:
        d = self.df.get(term, 0)
        return math.log(1.0 + (self.n_docs - d + 0.5) / (d + 0.5))

    def save(self, path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump({"n_docs": self.n_docs, "avgdl": self.avgdl, "df": self.df}, fh)
        tmp.replace(path)

    @classmethod
    def load(cls, path) -> "BM25Stats":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            d = json.load(fh)
        return cls(d["n_docs"], d["df"], d["avgdl"])


class BM25Scorer(Scorer):
    def __init__(self, scope: str, stats: BM25Stats | None = None, k1: float = 1.2, b: float = 0.75,
                 joiner: str = "\n\n", stats_version: str | None = None, name: str | None = None):
        if scope not in {"local", "trainglobal"}:
            raise ValueError(f"unknown BM25 scope {scope}")
        if scope == "trainglobal" and stats is None:
            raise ValueError("trainglobal BM25 needs precomputed stats")
        self.scope, self.stats, self.k1, self.b, self.joiner = scope, stats, k1, b, joiner
        self.stats_version = stats_version
        self.name = name or f"bm25_{scope}"
        self._local: OrderedDict[str, BM25Stats] = OrderedDict()

    def _stats_for(self, local_corpus: Sequence[str] | None) -> BM25Stats:
        if self.scope == "trainglobal":
            return self.stats
        if not local_corpus:
            raise ValueError("BM25-Local requires the question's full context (local_corpus)")
        key = hashlib.sha256("\x1e".join(local_corpus).encode("utf-8")).hexdigest()
        st = self._local.get(key)
        if st is None:
            st = BM25Stats.from_texts(local_corpus)
            self._local[key] = st
            if len(self._local) > LOCAL_STATS_MEMO:
                self._local.popitem(last=False)
        return st

    def _score_tokens(self, q_toks: list[str], d_toks: list[str], stats: BM25Stats) -> float:
        if not q_toks or not d_toks:
            return 0.0
        tf = Counter(d_toks)
        dl = len(d_toks)
        norm = self.k1 * (1.0 - self.b + self.b * dl / stats.avgdl)
        s = 0.0
        for t in q_toks:  # repeated query tokens contribute repeatedly
            f = tf.get(t, 0)
            if f:
                s += stats.idf(t) * f * (self.k1 + 1.0) / (f + norm)
        return s

    def score(self, query: str, candidates: Sequence[str], local_corpus: Sequence[str] | None = None) -> np.ndarray:
        stats = self._stats_for(local_corpus)
        q = tokenize(query)
        out = np.array([self._score_tokens(q, tokenize(c), stats) for c in candidates], dtype=np.float64)
        return check_finite(out, self.name)

    def compose_q_plus_a(self, q: str, a: str) -> str:
        return compose(q, a, self.joiner)

    def compose_a_plus_b(self, a: str, b: str) -> str:
        return compose(a, b, self.joiner)

    def fingerprint(self) -> dict:
        return {"name": self.name, "scope": self.scope, "k1": self.k1, "b": self.b, "tokenizer": TOKENIZER_ID,
                "formula": FORMULA_ID, "stats_version": self.stats_version,
                "n_docs": self.stats.n_docs if self.stats else "per_question"}
