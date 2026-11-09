"""Per-question computation of the spec-v2 O / E / R experiments for one retriever.

Notation:
    first hop   s(Q, A)                     over all context articles
    G0          s(Q + A, B)                 over articles != A  (A on the query side = body only)
    G1          s(Q, A + B)                 over articles != A  (pair re-encoding)
    G2          s(Q, B)                     over articles != A
    F0(A,B)     s(Q,A) + s(Q+A,B)           R1 / E2 path score, R2 formula
    F1(A,B)     s(Q,A) + s(Q,B)             R2 only
    F2(A,B)     s(Q,A) + s(Q,A+B)           R2 only

Isolation: everything that produces an *autonomous* ranking (first hop, R1/E2
paths, R3 paths) receives only a :class:`QueryInput` (no answer, no supporting
facts, no gold). Gold labels enter only (a) the Oracle blocks E1, E1', R2,
which are allowed to use Exploration gold articles, and (b) metric computation.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from ..data.documents import (
    Question, QueryInput, article_body_input, supporting_sentences_input,
)
from ..evaluation.mdr_metrics import mdr_question_metrics, paths_to_titles
from ..evaluation.metrics import KS, first_hop_metrics, path_metrics, rank_paths, second_hop_metrics
from ..retrieval.base import rank_order
from ..retrieval.similarity import EncodingTrace, Similarity

SPEC_VERSION = 2
STRATEGIES = ("G0", "G1", "G2")


def _r(x: float) -> float:
    return round(float(x), 6)


# ------------------------------------------------------------ label-free core
def first_hop(qi: QueryInput, sim: Similarity, trace: EncodingTrace | None) -> tuple[np.ndarray, list[int]]:
    scores = sim.s_q(qi, qi.docs, "O", trace)
    return scores, rank_order(scores)


def second_hop(qi: QueryInput, sim: Similarity, a: int, strategy: str, exp: str, trace: EncodingTrace | None,
               a_content: str | None = None, itype: str = "Q+A") -> tuple[list[int], np.ndarray]:
    """Scores of every candidate B != A (in context order) under a second-hop strategy."""
    cands = [d for d in qi.docs if d.index != a]
    doc_a = qi.docs[a]
    if strategy == "G0":
        g = sim.s_qa(qi, doc_a, cands, exp, trace, a_content=a_content, itype=itype)
    elif strategy == "G1":
        g = sim.s_q_ab(qi, doc_a, cands, exp, trace)
    elif strategy == "G2":
        g = sim.s_q(qi, cands, exp, trace)
    else:
        raise ValueError(strategy)
    return [d.index for d in cands], g


def ordered(cand_idx: list[int], scores: np.ndarray) -> list[int]:
    return [cand_idx[i] for i in rank_order(scores)]


def build_paths(s_q: np.ndarray, per_a: dict[int, tuple[list[int], np.ndarray]], a_set: list[int],
                top_m: int) -> list[tuple[int, int, float]]:
    """Keep the Top-m B of every A in ``a_set`` (by the second-hop score), score each
    directed path as s(Q,A) + G(A,B), and rank all paths jointly."""
    scores: dict[tuple[int, int], float] = {}
    for a in a_set:
        cand_idx, g = per_a[a]
        for pos in rank_order(g)[:top_m]:
            scores[(a, cand_idx[pos])] = float(s_q[a]) + float(g[pos])
    return [(a, b, scores[(a, b)]) for a, b in rank_paths(scores)]


def autonomous_second_hops(qi: QueryInput, sim: Similarity, a_set: list[int], strategy: str, exp: str,
                           trace: EncodingTrace | None) -> dict[int, tuple[list[int], np.ndarray]]:
    return {a: second_hop(qi, sim, a, strategy, exp, trace) for a in a_set}


# ------------------------------------------------------------ blocks
def o_block(q: Question, s_q: np.ndarray, order: list[int]) -> tuple[dict, dict]:
    m = first_hop_metrics(order, q.gold_indices, KS)
    metrics = {f"O.fh_{k}": v for k, v in m.items() if k != "gold_ranks"}
    detail = {"order": order, "scores": [_r(x) for x in s_q], "gold_ranks": m["gold_ranks"]}
    return metrics, detail


def oracle_block(q: Question, qi: QueryInput, sim: Similarity, exp: str, trace: EncodingTrace | None) -> tuple[dict, dict, dict]:
    """E1 (query = Q + A body) or E1' (query = Q + A's gold supporting sentences)."""
    metrics, detail, validity = {}, {}, {}
    for direction, a, b in q.oracle_directions():
        if exp == "E1":
            content, reason = article_body_input(q, a)
            sids, itype = None, "Q+A"
        else:
            content, reason, sids = supporting_sentences_input(q, a)
            itype = "Q+SF"
        validity[direction] = reason
        d: dict[str, Any] = {"A": a, "B": b, "invalid_reason": reason}
        if sids is not None:
            d["sf_sentence_ids"] = sids
        if reason is None:
            cand_idx, g = second_hop(qi, sim, a, "G0", exp, trace, a_content=content, itype=itype)
            order = ordered(cand_idx, g)
            m = second_hop_metrics(order, b, KS)
            metrics.update({f"{exp}.{direction}.{k}": v for k, v in m.items() if k != "pool_size"})
            d.update({"candidates": cand_idx, "scores": [_r(x) for x in g], "order": order})
        detail[direction] = d
    return metrics, detail, validity


def r2_block(q: Question, qi: QueryInput, sim: Similarity, s_q: np.ndarray, trace) -> tuple[dict, dict, dict]:
    metrics, detail, validity = {}, {}, {}
    for direction, a, b in q.oracle_directions():
        _, reason = article_body_input(q, a)
        validity[direction] = reason
        d: dict[str, Any] = {"A": a, "B": b, "invalid_reason": reason}
        if reason is None:
            const = float(s_q[a])  # s(Q, A) is constant for a fixed gold A
            for f, strat in (("F0", "G0"), ("F1", "G2"), ("F2", "G1")):
                cand_idx, g = second_hop(qi, sim, a, strat, "R2", trace)
                sc = const + g
                order = ordered(cand_idx, sc)
                m = second_hop_metrics(order, b, KS)
                metrics.update({f"R2.{direction}.{f}.{k}": v for k, v in m.items() if k != "pool_size"})
                d[f] = {"candidates": cand_idx, "scores": [_r(x) for x in sc], "order": order}
        detail[direction] = d
    return metrics, detail, validity


def _paths_detail(paths) -> list:
    return [[a, b, _r(s)] for a, b, s in paths]


def max_width(cfg: dict) -> int | None:
    """Largest configured first-hop width; None when PAll is configured."""
    ws = list(cfg["widths"].values())
    return None if any(w is None for w in ws) else max(ws)


def r1_block(q: Question, qi: QueryInput, sim: Similarity, s_q: np.ndarray, order: list[int], cfg: dict,
             trace) -> tuple[dict, dict, dict]:
    """R1 first-hop width P1/P3/P5/PAll with fixed Top-m B per A and F0; returns
    (metrics, detail, paths_by_width) — E2 is R1 at cfg['e2_width'] (D-08)."""
    n_max = max_width(cfg)
    a_super = order if n_max is None else order[:n_max]  # PAll configured => every A (spec-v2 runs)
    per_a = autonomous_second_hops(qi, sim, a_super, "G0", "R1", trace)
    metrics, paths_by_width = {}, {}
    for wname, n in cfg["widths"].items():
        a_set = order if n is None else order[:n]
        paths = build_paths(s_q, per_a, a_set, cfg["top_m"])
        paths_by_width[wname] = paths
        m = path_metrics([(a, b) for a, b, _ in paths], q.gold_indices, KS)
        metrics.update({f"R1.{wname}.{k}": v for k, v in m.items()})
        metrics[f"R1.{wname}.n_first_hop"] = len(a_set)
    detail = {"per_A_second_hop": {str(a): {"candidates": c, "scores": [_r(x) for x in g]} for a, (c, g) in per_a.items()},
              "paths": {w: _paths_detail(p) for w, p in paths_by_width.items()}}
    return metrics, detail, {"per_a": per_a, "paths": paths_by_width}


def e2_metrics(q: Question, order: list[int], r1_state: dict, cfg: dict) -> dict:
    """E2 = R1 at the fixed E2 width; E2-specific descriptive metrics only."""
    paths = r1_state["paths"][cfg["e2_width"]]
    out: dict[str, Any] = {"E2.width": cfg["e2_width"]}
    a1 = order[0]
    out["E2.top1A_is_gold"] = int(a1 in q.gold_indices)
    if a1 in q.gold_indices:
        other = next(g for g in q.gold_indices if g != a1)
        c, g = r1_state["per_a"][a1]
        m = second_hop_metrics(ordered(c, g), other, KS)
        out.update({f"E2.cond_{k}": v for k, v in m.items() if k.startswith("recall@") or k == "rr"})
    titles = [d.title for d in q.docs]
    gold_titles = [titles[i] for i in q.gold_indices]
    for k in KS:
        chains = paths_to_titles([(a, b) for a, b, _ in paths[:k]], titles)
        for name, v in mdr_question_metrics(chains, gold_titles).items():
            out[f"E2.mdr_{name}@{k}"] = v
    return out


def r3_block(q: Question, qi: QueryInput, sim: Similarity, s_q: np.ndarray, order: list[int], cfg: dict,
             r1_state: dict, trace) -> tuple[dict, dict]:
    a_set = order[:cfg["r3_width"]]
    metrics, detail = {}, {}
    gold_in_a = [a for a in a_set if a in q.gold_indices]
    for strat in STRATEGIES:
        t0 = time.perf_counter()
        per_a = autonomous_second_hops(qi, sim, a_set, strat, "R3", trace)
        paths = build_paths(s_q, per_a, a_set, cfg["top_m"])
        metrics[f"R3.{strat}.seconds"] = time.perf_counter() - t0
        m = path_metrics([(a, b) for a, b, _ in paths], q.gold_indices, KS)
        metrics.update({f"R3.{strat}.{k}": v for k, v in m.items()})
        if gold_in_a:  # candidate generation from the highest-ranked retrieved gold A
            a = gold_in_a[0]
            other = next(g for g in q.gold_indices if g != a)
            c, g = per_a[a]
            mm = second_hop_metrics(ordered(c, g), other, KS)
            metrics.update({f"R3.{strat}.cand_{k}": v for k, v in mm.items() if k.startswith("recall@") or k in ("rr", "rank")})
        detail[strat] = {"per_A_second_hop": {str(a): {"candidates": c, "scores": [_r(x) for x in g]}
                                              for a, (c, g) in per_a.items()},
                         "paths": _paths_detail(paths)}
    return metrics, detail


def _gold_sf_sentences(q: Question, idx: int) -> list[tuple[int, str]]:
    doc = q.docs[idx]
    ids = sorted({sid for t, sid in q.supporting_facts if t == doc.title and 0 <= sid < len(doc.sentences)})
    return [(i, doc.sentences[i].strip()) for i in ids if doc.sentences[i].strip()]


def _locate_loss(content: str, kept: int, sents: list[tuple[int, str]]) -> dict[str, list[int]]:
    """Classify each gold sentence by where it lies relative to the kept prefix of ``content``."""
    out = {"kept": [], "lost_full": [], "lost_partial": [], "unlocated": []}
    pos = 0
    for sid, s in sents:
        start = content.find(s, pos)
        if start < 0:
            start = content.find(s)
        if start < 0:
            out["unlocated"].append(sid)
            continue
        end = start + len(s)
        pos = end
        out["lost_full" if start >= kept else "lost_partial" if end > kept else "kept"].append(sid)
    return out


def annotate_sf_truncation(q: Question, events: list[dict[str, Any]], query_article: str = "body") -> None:
    """Evaluation-only annotation (runs after all rankings are fixed): for every
    truncated encoder input that contains a gold article, record which of that
    article's gold supporting sentences were cut off (fully or partially)."""
    gold = {q.docs[i].doc_id: i for i in q.gold_indices}
    by_id = {d.doc_id: d for d in q.docs}
    for e in events:
        if not e.get("truncated") or not e.get("chars"):
            continue
        arts = e["articles"]
        if e["type"] == "D":
            contents, part_arts = [by_id[arts[0]].text], [arts[0]]
        elif e["type"] == "Q+A":
            d = by_id[arts[0]]
            contents, part_arts = [q.question, d.body if query_article == "body" else d.text], [None, arts[0]]
        elif e["type"] == "Q+SF":
            text, _, _ = supporting_sentences_input(q, by_id[arts[0]].index)
            contents, part_arts = [q.question, text or ""], [None, arts[0]]
        elif e["type"] == "A+B":
            contents, part_arts = [by_id[arts[0]].text, by_id[arts[1]].text], arts
        else:
            continue
        info = {}
        for content, art, (raw_c, kept_c) in zip(contents, part_arts, e["chars"]):
            if art is None or art not in gold:
                continue
            loss = _locate_loss(content.strip(), kept_c, _gold_sf_sentences(q, gold[art]))
            info[art] = {"part_truncated": kept_c < raw_c, **loss}
        if info:
            e["gold_sf_truncation"] = info
            e["gold_sf_lost"] = any(v["lost_full"] or v["lost_partial"] for v in info.values())


def run_question(q: Question, sim: Similarity, experiments: list[str], cfg: dict,
                 encoder_id: str | None = None, clock: Callable[[], float] = time.perf_counter) -> dict[str, Any]:
    qi = q.unlabeled()
    trace = EncodingTrace()
    row: dict[str, Any] = {"spec_version": SPEC_VERSION, "qid": q.qid, "qtype": q.qtype, "level": q.level,
                           "n_docs": len(q.docs), "retriever": sim.name, "encoder": encoder_id,
                           "experiments": experiments, "two_gold": q.is_two_gold}
    metrics: dict[str, Any] = {}
    detail: dict[str, Any] = {}
    validity: dict[str, Any] = {}
    timing: dict[str, float] = {}
    if q.is_two_gold:
        a, b = q.gold_pair()
        row["gold"] = {"A": a, "B": b, "A_id": q.docs[a].doc_id, "B_id": q.docs[b].doc_id}
    t = clock()
    s_q, order = first_hop(qi, sim, trace)
    m, detail["O"] = o_block(q, s_q, order)
    metrics.update(m)
    timing["O"] = clock() - t
    if q.is_two_gold:
        for exp in ("E1", "E1P"):
            if exp in experiments:
                t = clock()
                m, detail[exp], validity[exp] = oracle_block(q, qi, sim, exp, trace)
                metrics.update(m)
                timing[exp] = clock() - t
        if "R2" in experiments:
            t = clock()
            m, detail["R2"], validity["R2"] = r2_block(q, qi, sim, s_q, trace)
            metrics.update(m)
            timing["R2"] = clock() - t
        r1_state = None
        if "R1" in experiments:
            t = clock()
            m, detail["R1"], r1_state = r1_block(q, qi, sim, s_q, order, cfg, trace)
            metrics.update(m)
            metrics.update(e2_metrics(q, order, r1_state, cfg))
            timing["R1_E2"] = clock() - t
        if "R3" in experiments:
            t = clock()
            m, detail["R3"] = r3_block(q, qi, sim, s_q, order, cfg, r1_state, trace)
            metrics.update(m)
            timing["R3"] = clock() - t
    annotate_sf_truncation(q, trace.events, sim.query_article)
    extra = getattr(sim, "annotate_events", None)  # e.g. sentence-selection retention (truncation ablation)
    if extra is not None:
        extra(q, trace.events)
    row.update({"metrics": metrics, "validity": validity, "detail": detail, "timing": timing,
                "encoding_events": trace.events})
    return row
