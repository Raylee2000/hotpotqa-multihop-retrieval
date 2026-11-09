"""Derived (silver) labels for bridge questions — NOT official HotpotQA gold.

Only HotpotQA `type`, `supporting_facts` and `answer` are official gold. Hop order and the bridge entity are
derived here by fixed rules; every label carries its rule id. Use: Training supervision only (never an input
at Validation / inference time).

    S-HOP-LINK    title(X) occurs in an SF sentence of gold Y and not vice versa  ->  A = Y (first), B = X
    S-HOP-QTITLE  (only if S-HOP-LINK undecided) exactly one gold title occurs in the question -> it is A
    S-HOP-ANS     consistency check only: the answer occurs in the SF of exactly one gold article -> it is B
    S-ENT-LINK    bridge entity = the occurrence of title(B) in A's SF sentences (only with S-HOP-LINK)

Title matching: case-insensitive substring of the title with parenthetical qualifiers removed
("Paris (band)" -> "paris"); titles shorter than 3 characters never match.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from ..data.documents import Question, normalize_answer


def core_title(title: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\([^)]*\)", " ", title)).strip()


def find_title(title: str, text: str) -> tuple[int, int] | None:
    """Character span of the title's core form in text (case-insensitive), or None."""
    t = core_title(title)
    if len(t) < 3:
        return None
    m = re.search(re.escape(t), text, flags=re.IGNORECASE)
    return (m.start(), m.end()) if m else None


def sf_sentences(q: Question, idx: int) -> list[tuple[int, str]]:
    d = q.docs[idx]
    sf = {s for t, s in q.supporting_facts if t == d.title}
    return [(i, d.sentences[i]) for i in sorted(sf) if i < len(d.sentences)]


@dataclass
class SilverBridge:
    qid: str
    first: int | None          # context index of A
    second: int | None         # context index of B
    hop_rule: str | None       # "S-HOP-LINK" | "S-HOP-QTITLE" | None
    link_dirs: int             # number of gold->gold title links found (0, 1, 2)
    ans_second: int | None     # S-HOP-ANS: index of the gold article holding the answer (check only)
    entity: dict | None        # S-ENT-LINK: {"text", "sentence_id", "start", "end", "rule"}

    def to_json(self) -> dict:
        return asdict(self)


def _first_link(q: Question, src: int, dst: int) -> tuple[int, int, int] | None:
    """(sentence id, start, end) of the first occurrence of title(dst) in src's SF sentences."""
    for sid, text in sf_sentences(q, src):
        span = find_title(q.docs[dst].title, text)
        if span:
            return sid, span[0], span[1]
    return None


def derive_bridge(q: Question) -> SilverBridge | None:
    """Silver hop order / bridge entity for a two-gold bridge question; None for other questions."""
    if q.qtype != "bridge" or not q.is_two_gold:
        return None
    g1, g2 = q.gold_indices
    l12, l21 = _first_link(q, g1, g2), _first_link(q, g2, g1)  # l12: title(g2) inside g1 -> g1 is A
    first = second = rule = None
    entity = None
    if (l12 is None) != (l21 is None):
        first, second = (g1, g2) if l12 else (g2, g1)
        rule = "S-HOP-LINK"
        sid, s, e = l12 or l21
        text = q.docs[first].sentences[sid]
        entity = {"text": text[s:e], "sentence_id": sid, "start": s, "end": e, "rule": "S-ENT-LINK"}
    else:
        inq = [i for i in (g1, g2) if find_title(q.docs[i].title, q.question)]
        if len(inq) == 1:
            first = inq[0]
            second = g2 if first == g1 else g1
            rule = "S-HOP-QTITLE"
    ans_second = None
    a = normalize_answer(q.answer)
    if a and a not in ("yes", "no"):
        holds = [i for i in (g1, g2) if a in normalize_answer(" ".join(t for _, t in sf_sentences(q, i)))]
        if len(holds) == 1:
            ans_second = holds[0]
    return SilverBridge(q.qid, first, second, rule, int(l12 is not None) + int(l21 is not None), ans_second, entity)
