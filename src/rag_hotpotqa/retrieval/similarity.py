"""The research similarity s(X, Y) in its uses, for one retriever (spec v2).

    s_q(Q, D)              = s(Q, D)
    s_qa(Q, A, B)          = s(Q + A_query, B)      A_query = A's body (no title), or E1' sentences
    s_q_ab(Q, A, B)        = s(Q, A + B)            candidate side: rendered articles (title + body)

A retriever is one component scorer (BM25 or dense) or the RRF fusion of
several; under RRF every component ranks exactly the same candidate list.
Candidates are passed in stable document-index order, so position doubles as
the tie-breaker. Every call takes the question's :class:`QueryInput` (no
labels) so BM25-Local statistics always come from the full context.

Encoding events (D-11) are recorded per (experiment, input type, article ids)
from deterministic token accounting, so embedding-cache hits never lose them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..data.documents import Document, QueryInput
from .base import Scorer, check_finite
from .hybrid import rrf_scores


@dataclass
class EncodingTrace:
    """Per-question ledger of encoder inputs (dense components only)."""

    events: list[dict[str, Any]] = field(default_factory=list)
    _seen: set = field(default_factory=set)

    def add(self, component: str, exp: str, itype: str, article_ids: Sequence[str], info: dict | None) -> None:
        if info is None:
            return
        key = (component, exp, itype, tuple(article_ids))
        if key in self._seen:
            return
        self._seen.add(key)
        self.events.append({"component": component, "exp": exp, "type": itype, "articles": list(article_ids),
                            "raw_tokens": info["raw_tokens"], "input_tokens": info["input_tokens"],
                            "truncated": info["truncated"], "policy": info["policy"], "parts": info["parts"],
                            "chars": info.get("chars"), "cache_key": info["cache_key"]})
        # optional fields of long-input encoders (truncation ablation); absent for the T0 scorer
        for k in ("n_units", "unit_overflow", "t0_identical", "n_chunks", "selection"):
            if k in info:
                self.events[-1][k] = info[k]


class Similarity:
    def __init__(self, name: str, components: Sequence[Scorer], fusion: str = "single", rrf_k: int = 60,
                 query_article: str = "body"):
        if fusion == "single" and len(components) != 1:
            raise ValueError("single fusion needs exactly one component")
        if fusion not in {"single", "rrf"}:
            raise ValueError(f"unknown fusion {fusion}")
        if query_article not in {"body", "text"}:
            raise ValueError(f"unknown query_article {query_article}")
        self.name, self.components, self.fusion, self.rrf_k = name, list(components), fusion, rrf_k
        self.query_article = query_article

    def _combine(self, per_component: list[np.ndarray]) -> np.ndarray:
        if self.fusion == "single":
            return check_finite(per_component[0], self.name)
        return rrf_scores(per_component, self.rrf_k)

    @staticmethod
    def _local(qi: QueryInput) -> list[str]:
        return [d.text for d in qi.docs]

    def _log(self, trace: EncodingTrace | None, c: Scorer, exp: str, itype: str, ids: Sequence[str], text: str,
             composed_from: tuple[str, str, str] | None = None) -> None:
        if trace is not None:
            trace.add(c.name, exp, itype, ids, c.token_info(text, composed_from))

    def _log_docs(self, trace, c, exp, docs: Sequence[Document]) -> None:
        for d in docs:
            self._log(trace, c, exp, "D", [d.doc_id], d.text)

    def s_q(self, qi: QueryInput, docs: Sequence[Document], exp: str, trace: EncodingTrace | None = None) -> np.ndarray:
        out = []
        for c in self.components:
            self._log(trace, c, exp, "Q", [], qi.question)
            self._log_docs(trace, c, exp, docs)
            out.append(c.score(qi.question, [d.text for d in docs], self._local(qi)))
        return self._combine(out)

    def s_qa(self, qi: QueryInput, a: Document, docs: Sequence[Document], exp: str,
             trace: EncodingTrace | None = None, a_content: str | None = None, itype: str = "Q+A") -> np.ndarray:
        """s(Q + A, B). ``a_content`` defaults to A's body; E1' passes A's gold sentences."""
        if a_content is None:
            content = a.body if self.query_article == "body" else a.text
        else:
            content = a_content
        out = []
        for c in self.components:
            query = c.compose_q_plus_a(qi.question, content)
            self._log(trace, c, exp, itype, [a.doc_id], query, ("q_plus_a", qi.question, content))
            self._log_docs(trace, c, exp, docs)
            out.append(c.score(query, [d.text for d in docs], self._local(qi)))
        return self._combine(out)

    def s_q_ab(self, qi: QueryInput, a: Document, docs: Sequence[Document], exp: str,
               trace: EncodingTrace | None = None) -> np.ndarray:
        out = []
        for c in self.components:
            self._log(trace, c, exp, "Q", [], qi.question)
            cands = []
            for d in docs:
                t = c.compose_a_plus_b(a.text, d.text)
                self._log(trace, c, exp, "A+B", [a.doc_id, d.doc_id], t, ("a_plus_b", a.text, d.text))
                cands.append(t)
            out.append(c.score(qi.question, cands, self._local(qi)))
        return self._combine(out)

    # ---- batch prefetch (performance only; never changes scores) ----------
    def texts_for(self, qi: QueryInput, qa_contents: Sequence[str] = (), ab_from: Sequence[Document] = ()) -> list[list[str]]:
        out = []
        for c in self.components:
            texts = [qi.question] + [d.text for d in qi.docs]
            texts += [c.compose_q_plus_a(qi.question, t) for t in qa_contents]
            texts += [c.compose_a_plus_b(a.text, d.text) for a in ab_from for d in qi.docs if d.index != a.index]
            out.append(texts)
        return out

    def prefetch_texts(self, per_component: Sequence[Sequence[str]]) -> int:
        for c, texts in zip(self.components, per_component):
            c.prefetch(list(texts))
        return sum(len(t) for t in per_component)

    def fingerprint(self) -> dict:
        return {"name": self.name, "fusion": self.fusion, "rrf_k": self.rrf_k, "query_article": self.query_article,
                "components": [c.fingerprint() for c in self.components]}
