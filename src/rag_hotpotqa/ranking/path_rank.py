"""Per-question path ranking metrics, copied verbatim from the formal scorer (scripts/overnight_v3/rank.py,
``rank_one``, research repo commit 6366c64) so that public results are computed with the identical definition.

PR@1 = 1 iff the top-scored unordered candidate pair is the gold article pair; a question whose candidate set has
no gold pair (or no candidates) scores 0. Ties are broken by candidate order (stable sort).
"""
from __future__ import annotations

import numpy as np


def rank_one(d,idx,s):
    if not len(idx):
        return {"pr1":0,"pr3":0,"pr5":0,"rr":0.,"cand":0,"gold_rank":None,"top1_pair":None,"top5":[],"one_hop_error":False}
    order=np.argsort(-s[idx],kind="stable")
    positive=np.flatnonzero(d.y[idx])
    rank=int(np.flatnonzero(order==positive[0])[0])+1 if len(positive) else None
    top=[int(x) for x in d.pairs[idx[order[0]]][:2]]
    gold=set(d.pairs[idx[positive[0]]][:2]) if len(positive) else set()
    return {"pr1":int(rank==1),"pr3":int(rank is not None and rank<=3),"pr5":int(rank is not None and rank<=5),
            "rr":1/rank if rank else 0.,"cand":int(bool(len(positive))),"gold_rank":rank,"top1_pair":top,
            "top5":[{"pair":[int(x) for x in d.pairs[idx[j]][:2]],"score":float(s[idx[j]])} for j in order[:5]],
            "one_hop_error":bool(rank!=1 and gold and len(gold & set(top))==1)}
