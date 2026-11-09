"""Order-invariant path reranking on FIXED candidate sets (design §5, §13.4–13.5).

Trained on Training candidates (outputs/path_candidates_v1/<run>/train.npz), selected and evaluated on V-select.
Gold pair = unordered {A, B}; directional evidence enters only as features.
Methods
  R0  raw path score: f_best = max over directions of s1(Q,A) + s2(QV2(A),B)          (no training)
  R1  length-calibrated: logistic regression on f_best + length features + f_best x length
  R2  pointwise logistic regression, all features (standardised); C on V-select
  R3  pairwise (RankNet-style) linear model on (gold − wrong) differences; high-score-wrong and one-hop-correct
      wrong pairs weighted x2 (Training labels only); C on V-select
  R4  listwise softmax over each question's candidates (linear, torch, full batch); L2 on V-select
  R5  LambdaMART (own implementation: gradient-boosted sklearn regression trees on NDCG lambdas; LightGBM/XGBoost are
      not installed); rounds/depth on V-select
Each trained method in two feature sets: `full` and `no_title` (drops title_* = HotpotQA-specific title matching).
Metrics (two-gold questions; a question without the gold pair in its candidates counts as a miss):
  Candidate Path Recall, Path Recall@1/3/5, Conditional Ranking Accuracy (PR@1 | gold in candidates), MRR,
  Ranking Gap (best wrong score − gold score, method units; share > 0), gold rank, one-hop-correct share of errors,
  PR@1 by number of long (>254-token) gold articles; paired bootstrap CIs vs R0.
Usage: python scripts/train_path_reranker.py --cand-run frozen --run-name rerank_frozen_v1
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_hotpotqa.utils.config import load_settings, safe_path  # noqa: E402
from rag_hotpotqa.utils.io import atomic_write_json, atomic_write_text  # noqa: E402
from rag_hotpotqa.utils.runtime import git_commit  # noqa: E402

N_BOOT, SEED = 1000, 42
LEN_FEATS = ["logtok_max", "logtok_min", "n_long"]


def ci(x):
    x = np.asarray(x, float)
    m = x[np.random.default_rng(SEED).integers(0, len(x), (N_BOOT, len(x)))].mean(1)
    return {"n": len(x), "mean": round(float(x.mean()), 4), "ci95": [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]}


class Data:
    def __init__(self, npz: Path, meta: Path):
        z = np.load(npz)
        self.X, self.y, self.q, self.pairs = z["X"].astype(np.float64), z["y"].astype(int), z["q"], z["pairs"]
        self.meta = json.loads(meta.read_text(encoding="utf-8"))
        self.feats = self.meta["features"]
        self.groups = [np.flatnonzero(self.q == i) for i in range(len(self.meta["questions"]))]

    def cols(self, names):
        return self.X[:, [self.feats.index(n) for n in names]]


def evaluate(d: Data, scores: np.ndarray) -> dict:
    per = {"pr1": [], "pr3": [], "pr5": [], "rr": [], "cand": [], "cond": [], "gap": [], "gold_rank": [], "one_hop_err": [],
           "qtype": [], "n_long_gold": []}
    il = d.feats.index("n_long")
    for qi, idx in enumerate(d.groups):
        info = d.meta["questions"][qi]
        if not info["two_gold"]:
            continue
        per["qtype"].append(info["qtype"])
        y, s = d.y[idx], scores[idx]
        order = np.argsort(-s, kind="stable")
        has = bool(y.sum())
        per["cand"].append(int(has))
        if not has:
            for k in ("pr1", "pr3", "pr5", "rr"):
                per[k].append(0.0)
            per["n_long_gold"].append(-1)
            continue
        g = int(np.flatnonzero(y)[0])
        r = int(np.flatnonzero(order == g)[0]) + 1
        per["pr1"].append(float(r == 1))
        per["pr3"].append(float(r <= 3))
        per["pr5"].append(float(r <= 5))
        per["rr"].append(1 / r)
        per["cond"].append(float(r == 1))
        per["gold_rank"].append(r)
        wrong = s[y == 0]
        per["gap"].append(float(wrong.max() - s[g]) if len(wrong) else -1.0)
        per["n_long_gold"].append(int(d.X[idx[g], il]))
        if r > 1:  # one-hop-correct error: the top-1 wrong pair shares exactly one article with the gold pair
            gx, gy = set(d.pairs[idx[g]][:2]), set(d.pairs[idx[order[0]]][:2])
            per["one_hop_err"].append(float(len(gx & gy) == 1))
    out = {"n_two_gold": len(per["qtype"]), "candidate_path_recall": round(float(np.mean(per["cand"])), 4)}
    for k in ("pr1", "pr3", "pr5", "rr"):
        out[k] = round(float(np.mean(per[k])), 4)
    out["conditional_ranking_accuracy"] = round(float(np.mean(per["cond"])), 4)
    gap = np.asarray(per["gap"])
    out["ranking_gap"] = {"mean": round(float(gap.mean()), 4), "median": round(float(np.median(gap)), 4), "share_positive": round(float((gap > 0).mean()), 4)}
    out["gold_rank_mean"] = round(float(np.mean(per["gold_rank"])), 3)
    out["one_hop_correct_share_of_errors"] = round(float(np.mean(per["one_hop_err"])), 4) if per["one_hop_err"] else None
    qt = np.asarray(per["qtype"])
    out["by_type"] = {t: {"pr1": round(float(np.mean(np.asarray(per["pr1"])[qt == t])), 4), "n": int((qt == t).sum())} for t in ("bridge", "comparison")}
    nl = np.asarray(per["n_long_gold"])
    out["pr1_by_long_gold_articles"] = {str(k): {"pr1": round(float(np.mean(np.asarray(per["pr1"])[nl == k])), 4), "n": int((nl == k).sum())}
                                        for k in (0, 1, 2) if (nl == k).any()}
    out["_pr1_vector"] = per["pr1"]
    return out


def standardise(tr, *others):
    mu, sd = tr.mean(0), tr.std(0) + 1e-9
    return [(tr - mu) / sd] + [(o - mu) / sd for o in others]


def fit_lr(Xtr, ytr, C, weight=None):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(C=C, max_iter=3000, class_weight="balanced" if weight is None else None).fit(Xtr, ytr, sample_weight=weight)


def pairwise_data(d: Data, X):
    """(gold − wrong) differences; wrong pairs scoring above gold on f_best or sharing one gold article get weight 2."""
    fb = d.X[:, d.feats.index("f_best")]
    diffs, w = [], []
    for idx in d.groups:
        y = d.y[idx]
        if y.sum() != 1:
            continue
        g = idx[np.flatnonzero(y)[0]]
        gold_set = set(d.pairs[g][:2])
        for j in idx[y == 0]:
            hard = fb[j] >= fb[g] or len(gold_set & set(d.pairs[j][:2])) == 1
            diffs.append(X[g] - X[j])
            w.append(2.0 if hard else 1.0)
    D = np.asarray(diffs)
    return np.vstack([D, -D]), np.r_[np.ones(len(D)), np.zeros(len(D))], np.r_[w, w]


def fit_listwise(d: Data, X, l2: float, epochs: int = 300):
    import torch

    torch.manual_seed(SEED)
    groups = [idx for idx in d.groups if d.y[idx].sum() == 1]
    Xt = torch.tensor(X, dtype=torch.float32)
    w = torch.zeros(X.shape[1], requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.05)
    seg = torch.tensor(np.concatenate([np.full(len(idx), i) for i, idx in enumerate(groups)]), dtype=torch.long)
    flat = torch.tensor(np.concatenate(groups), dtype=torch.long)
    gold = torch.tensor([int(np.flatnonzero(d.y[idx])[0]) + sum(len(g) for g in groups[:i]) for i, idx in enumerate(groups)])
    for _ in range(epochs):
        s = Xt[flat] @ w
        with torch.no_grad():  # per-group max for a stable log-sum-exp (constant w.r.t. the gradient)
            mx = torch.zeros(len(groups)).index_reduce_(0, seg, s, "amax", include_self=False)
        lse = torch.log(torch.zeros(len(groups)).index_add_(0, seg, torch.exp(s - mx[seg]))) + mx
        loss = (lse - s[gold]).mean() + l2 * (w ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w.detach().numpy()


def fit_lambdamart(d: Data, X, rounds: int, depth: int, lr: float = 0.1):
    """Gradient-boosted trees on LambdaRank gradients (NDCG, one relevant pair per question)."""
    from sklearn.tree import DecisionTreeRegressor

    groups = [idx for idx in d.groups if d.y[idx].sum() == 1]
    rows = np.concatenate(groups)
    Xr = X[rows]
    pos = {r: i for i, r in enumerate(rows)}
    F = np.zeros(len(rows))
    trees = []
    disc = lambda r: 1 / np.log2(1 + r)  # noqa: E731
    for _ in range(rounds):
        grad, hess = np.zeros(len(rows)), np.zeros(len(rows))
        for idx in groups:
            loc = np.array([pos[r] for r in idx])
            s = F[loc]
            g = int(np.flatnonzero(d.y[idx])[0])
            ranks = np.empty(len(idx), int)
            ranks[np.argsort(-s, kind="stable")] = np.arange(1, len(idx) + 1)
            others = np.arange(len(idx)) != g
            dn = np.abs(disc(ranks[g]) - disc(ranks[others]))
            rho = 1 / (1 + np.exp(s[g] - s[others]))
            grad[loc[g]] -= (rho * dn).sum()
            grad[loc[others]] += rho * dn
            h = rho * (1 - rho) * dn
            hess[loc[g]] += h.sum()
            hess[loc[others]] += h
        hess = np.maximum(hess, 1e-6)
        t = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=50, random_state=SEED).fit(Xr, -grad / hess, sample_weight=hess)
        F += lr * t.predict(Xr)
        trees.append(t)
    return lambda Z: lr * sum(t.predict(Z) for t in trees)


def main() -> int:
    settings = load_settings()
    root = settings.project_root
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand-run", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--quick", action="store_true", help="smaller grids (smoke)")
    args = ap.parse_args()
    cdir = root / "outputs" / "path_candidates_v1" / args.cand_run
    out = safe_path(root / "outputs" / "path_reranker_v1" / args.run_name, root)
    out.mkdir(parents=True, exist_ok=True)
    tr, vs = Data(cdir / "train.npz", cdir / "train_meta.json"), Data(cdir / "v_select.npz", cdir / "v_select_meta.json")
    feats_full = tr.feats
    feats_not = [f for f in feats_full if not f.startswith("title_")]
    res = {"cand_run": args.cand_run, "train_meta": {k: tr.meta[k] for k in tr.meta if k != "questions"},
           "v_select_meta": {k: vs.meta[k] for k in vs.meta if k != "questions"}, "git_commit": git_commit(root),
           "utc": datetime.now(timezone.utc).isoformat(), "selection": "every hyper-parameter chosen by V-select PR@1 (ties: MRR)",
           "methods": {}}
    r0 = vs.cols(["f_best"])[:, 0]
    base = evaluate(vs, r0)
    res["methods"]["R0_raw"] = base
    grids = {"C": [0.1, 1.0] if args.quick else [0.01, 0.1, 1.0, 10.0], "l2": [1e-3] if args.quick else [1e-4, 1e-3, 1e-2],
             "lm": [(20, 3)] if args.quick else [(100, 3), (200, 4), (300, 3)]}

    def pick(cands):
        best = max(cands, key=lambda c: (c[1]["pr1"], c[1]["rr"]))
        best[1]["hyper"] = best[0]
        best[1]["grid"] = {str(c[0]): c[1]["pr1"] for c in cands}
        return best[1]

    len_cols = ["f_best"] + LEN_FEATS
    Xtr, Xvs = standardise(tr.cols(len_cols), vs.cols(len_cols))
    Xtr = np.c_[Xtr, Xtr[:, :1] * Xtr[:, 1:2]]
    Xvs = np.c_[Xvs, Xvs[:, :1] * Xvs[:, 1:2]]
    res["methods"]["R1_length_calibrated"] = pick([(C, evaluate(vs, fit_lr(Xtr, tr.y, C).decision_function(Xvs))) for C in grids["C"]])
    for fs_name, fs in (("full", feats_full), ("no_title", feats_not)):
        Xtr, Xvs = standardise(tr.cols(fs), vs.cols(fs))
        res["methods"][f"R2_pointwise_lr__{fs_name}"] = pick([(C, evaluate(vs, fit_lr(Xtr, tr.y, C).decision_function(Xvs))) for C in grids["C"]])
        Dp, yp, wp = pairwise_data(tr, Xtr)
        from sklearn.linear_model import LogisticRegression

        cands = []
        for C in grids["C"]:
            m = LogisticRegression(C=C, max_iter=3000, fit_intercept=False).fit(Dp, yp, sample_weight=wp)
            cands.append((C, evaluate(vs, Xvs @ m.coef_[0])))
        res["methods"][f"R3_pairwise__{fs_name}"] = pick(cands)
        res["methods"][f"R4_listwise__{fs_name}"] = pick([(l2, evaluate(vs, Xvs @ fit_listwise(tr, Xtr, l2))) for l2 in grids["l2"]])
        raw_tr, raw_vs = tr.cols(fs), vs.cols(fs)
        res["methods"][f"R5_lambdamart__{fs_name}"] = pick([(f"rounds={r},depth={dp}", evaluate(vs, fit_lambdamart(tr, raw_tr, r, dp)(raw_vs)))
                                                            for r, dp in grids["lm"]])
    v0 = np.asarray(base["_pr1_vector"])
    for name, m in res["methods"].items():
        m["pr1_diff_vs_R0"] = ci(np.asarray(m["_pr1_vector"]) - v0) if name != "R0_raw" else None
    for m in res["methods"].values():
        m.pop("_pr1_vector")
    atomic_write_json(out / "result.json", res, root)
    lines = [f"# Path reranker `{args.run_name}` on V-select (candidates `{args.cand_run}`)", "",
             "Selection and evaluation both on V-select (selection-biased; V-report is held out for the final evaluation).", "",
             "| method | hyper | cand recall | PR@1 | ΔPR@1 vs R0 [95% CI] | PR@5 | cond. acc | MRR | gap>0 | bridge PR@1 | comparison PR@1 |",
             "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for name, m in res["methods"].items():
        dlt = m["pr1_diff_vs_R0"]
        dtxt = "–" if dlt is None else "{:+.4f} [{:+.4f}, {:+.4f}]".format(dlt["mean"], *dlt["ci95"])
        lines.append(f"| {name} | {m.get('hyper', '–')} | {m['candidate_path_recall']} | {m['pr1']} | "
                     f"{dtxt} | {m['pr5']} | "
                     f"{m['conditional_ranking_accuracy']} | {m['rr']} | {m['ranking_gap']['share_positive']} | "
                     f"{m['by_type']['bridge']['pr1']} | {m['by_type']['comparison']['pr1']} |")
    atomic_write_text(out / "summary.md", "\n".join(lines) + "\n", root)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
