"""Document representation, stable IDs, gold-document mapping, Oracle inputs.

Retrieval unit = one context article (title + all sentences), not a sentence.

Spec v2 (D-03): the two gold articles are named A and B by their position in
the official context list (A = lower index). This naming is independent of
the answer and of any model output; both directions A->B and B->A are always
evaluated and reported separately.
"""
from __future__ import annotations

import hashlib
import re
import string
from dataclasses import dataclass, field
from typing import Any

DEFAULT_TEMPLATE = "{title}\n{body}"

DIRECTIONS = ("AtoB", "BtoA")

# E1' invalid-input reasons (D-04).
SF_MISSING = "sf_missing_for_article"
SF_OUT_OF_RANGE = "sf_sentence_index_out_of_range"
SF_EMPTY = "sf_sentence_empty"
ARTICLE_BODY_EMPTY = "article_body_empty"


@dataclass(frozen=True)
class Document:
    doc_id: str          # "<qid>#<context index:02d>": unique & stable within the dataset
    index: int           # position in the official context list (tie-breaker)
    title: str
    sentences: tuple[str, ...]
    text: str            # candidate-side rendering: title + body
    body: str            # article body only ("".join(sentences).strip()); query-side rendering of A
    content_key: str     # sha256 of (title, sentences): identifies the same article across questions


@dataclass(frozen=True)
class QueryInput:
    """Everything an autonomous retriever may see: no answer, no supporting facts, no gold."""

    qid: str
    question: str
    docs: tuple[Document, ...]


@dataclass
class Question:
    qid: str
    question: str
    answer: str
    qtype: str
    level: str
    docs: list[Document]
    supporting_facts: list[tuple[str, int]]
    gold_indices: list[int]              # context indices of gold docs (resolved, deduped, ascending)
    missing_gold_titles: list[str]       # supporting titles absent from context
    ambiguous_gold_titles: list[str]     # supporting titles appearing >1 time in context
    flags: dict[str, Any] = field(default_factory=dict)

    @property
    def n_gold(self) -> int:
        return len(self.gold_indices)

    @property
    def is_two_gold(self) -> bool:
        return self.n_gold == 2 and not self.missing_gold_titles and not self.ambiguous_gold_titles

    def unlabeled(self) -> QueryInput:
        return QueryInput(self.qid, self.question, tuple(self.docs))

    def gold_pair(self) -> tuple[int, int]:
        """(A, B) = gold context indices in ascending order (D-03)."""
        if not self.is_two_gold:
            raise ValueError(f"{self.qid}: gold pair undefined (n_gold={self.n_gold})")
        a, b = sorted(self.gold_indices)
        return a, b

    def oracle_directions(self) -> list[tuple[str, int, int]]:
        a, b = self.gold_pair()
        return [("AtoB", a, b), ("BtoA", b, a)]


def render_text(title: str, sentences: list[str] | tuple[str, ...], template: str = DEFAULT_TEMPLATE) -> str:
    body = "".join(sentences).strip()
    return template.format(title=title, body=body).strip()


def content_key(title: str, sentences: list[str] | tuple[str, ...]) -> str:
    h = hashlib.sha256()
    h.update(title.encode("utf-8"))
    for s in sentences:
        h.update(b"\x1f")
        h.update(s.encode("utf-8"))
    return h.hexdigest()


_PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    """SQuAD/HotpotQA-style normalisation (lowercase, strip punctuation/articles/space)."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def historical_answer_heuristic_direction(q: Question) -> str:
    """Spec-v1 descriptive statistic only (data_analysis.json). Never used by
    any experiment: spec v2 forbids deriving hop direction from the answer."""
    if not q.is_two_gold:
        return "unknown"
    if q.qtype == "comparison":
        return "symmetric"
    na = normalize_answer(q.answer)
    if not na or na in {"yes", "no"}:
        return "unknown"
    has = [f" {na} " in f" {normalize_answer(' '.join(q.docs[i].sentences))} " for i in q.gold_indices]
    return "answer_heuristic" if has[0] != has[1] else "unknown"


def build_question(rec: dict[str, Any], template: str = DEFAULT_TEMPLATE) -> Question:
    qid = rec["_id"]
    docs = [
        Document(
            doc_id=f"{qid}#{i:02d}",
            index=i,
            title=title,
            sentences=tuple(sents),
            text=render_text(title, sents, template),
            body="".join(sents).strip(),
            content_key=content_key(title, sents),
        )
        for i, (title, sents) in enumerate(rec["context"])
    ]
    title_to_idx: dict[str, list[int]] = {}
    for d in docs:
        title_to_idx.setdefault(d.title, []).append(d.index)

    gold_titles: list[str] = []
    for title, _sid in rec["supporting_facts"]:
        if title not in gold_titles:
            gold_titles.append(title)
    gold, missing, ambiguous = [], [], []
    for t in gold_titles:
        idxs = title_to_idx.get(t, [])
        if not idxs:
            missing.append(t)
        elif len(idxs) > 1:
            ambiguous.append(t)
        else:
            gold.append(idxs[0])

    sent_out_of_range = [
        [t, sid] for t, sid in rec["supporting_facts"]
        if t in title_to_idx and len(title_to_idx[t]) == 1 and not (0 <= sid < len(docs[title_to_idx[t][0]].sentences))
    ]
    return Question(
        qid=qid,
        question=rec["question"],
        answer=rec["answer"],
        qtype=rec["type"],
        level=rec["level"],
        docs=docs,
        supporting_facts=[(t, int(s)) for t, s in rec["supporting_facts"]],
        gold_indices=sorted(gold),
        missing_gold_titles=missing,
        ambiguous_gold_titles=ambiguous,
        flags={
            "empty_question": not rec["question"].strip(),
            "empty_docs": [d.index for d in docs if not d.body],
            "duplicate_titles": sorted(t for t, v in title_to_idx.items() if len(v) > 1),
            "sf_sentence_out_of_range": sent_out_of_range,
        },
    )


def article_body_input(q: Question, a: int) -> tuple[str | None, str | None]:
    """E1 query-side article content: body of A (no title, D-04)."""
    body = q.docs[a].body
    return (body, None) if body else (None, ARTICLE_BODY_EMPTY)


def supporting_sentences_input(q: Question, a: int) -> tuple[str | None, str | None, list[int]]:
    """E1' query-side content: A's own gold supporting sentences (D-04).

    Sentence indices are de-duplicated and sorted ascending; sentences are
    joined exactly like the article body. Any out-of-range, missing or empty
    sentence makes the whole direction invalid; full text is never substituted.
    Returns (text | None, invalid_reason | None, sentence_ids).
    """
    doc = q.docs[a]
    ids = sorted({sid for title, sid in q.supporting_facts if title == doc.title})
    if not ids:
        return None, SF_MISSING, []
    if any(not (0 <= sid < len(doc.sentences)) for sid in ids):
        return None, SF_OUT_OF_RANGE, ids
    sents = [doc.sentences[sid] for sid in ids]
    if any(not s.strip() for s in sents):
        return None, SF_EMPTY, ids
    return "".join(sents).strip(), None, ids
