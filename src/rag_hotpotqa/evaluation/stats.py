"""Small statistics helpers shared by the analysis scripts (numpy / scipy only).

- question-level bootstrap (shared resample indices, chunked), percentile CIs
- McNemar exact test, Holm step-down correction, Wald z from a bootstrap SE
- conditional logistic regression for matched pairs (one success per informative pair), Newton-Raphson
- binary classification metrics with optional sample weights (weights = prior reweighting)
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy import stats as _st

N_BOOT, SEED = 2000, 42


def boot_indices(n: int, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, (n_boot, n), dtype=np.int32)


def boot_stat(idx: np.ndarray, fn: Callable[[np.ndarray], float]) -> np.ndarray:
    """fn applied to each resample (fn receives an index array)."""
    return np.array([fn(b) for b in idx])


def boot_mean(x, idx: np.ndarray, chunk: int = 250) -> np.ndarray:
    x = np.asarray(x, float)
    return np.concatenate([x[idx[i:i + chunk]].mean(1) for i in range(0, len(idx), chunk)])


def pct_ci(boot: np.ndarray, nd: int = 4) -> list[float]:
    return [round(float(np.percentile(boot, 2.5)), nd), round(float(np.percentile(boot, 97.5)), nd)]


def wald_p(est: float, boot: np.ndarray) -> float:
    se = float(np.std(boot, ddof=1))
    return float(2 * _st.norm.sf(abs(est) / se)) if se > 0 else float("nan")


def mcnemar_exact(y_a, y_b) -> dict:
    """Paired binary outcomes; b = a only, c = b only; exact two-sided binomial p."""
    a, b = np.asarray(y_a, int), np.asarray(y_b, int)
    only_a, only_b = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n = only_a + only_b
    p = float(_st.binomtest(min(only_a, only_b), n, 0.5).pvalue) if n else 1.0
    return {"only_a": only_a, "only_b": only_b, "p_exact": p}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p values (monotone, capped at 1)."""
    keys = sorted(pvals, key=lambda k: pvals[k])
    m, out, run = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * pvals[k]))
        out[k] = run
    return out


def conditional_logit_pairs(x1: np.ndarray, x2: np.ndarray, y1, y2, max_iter: int = 100, tol: float = 1e-10) -> dict:
    """Matched-pair conditional logit: P(instance 1 succeeds | exactly one succeeds) = sigmoid(beta . (x1 - x2)).

    Only discordant pairs carry information. Returns beta, model SE, n_discordant, convergence and diagnostics
    (rank, condition number of the differenced design, and whether the fit looks separated)."""
    y1, y2 = np.asarray(y1, int), np.asarray(y2, int)
    keep = y1 != y2
    d = (np.asarray(x1, float) - np.asarray(x2, float))[keep]
    z = y1[keep].astype(float)
    k = d.shape[1]
    diag = {"n_discordant": int(keep.sum()), "rank": int(np.linalg.matrix_rank(d)) if len(d) else 0, "k": k}
    diag["cond_number"] = float(np.linalg.cond(d)) if len(d) >= k and diag["rank"] == k else float("inf")
    if diag["rank"] < k:
        return {**diag, "identifiable": False, "beta": None, "se": None, "converged": False}
    beta = np.zeros(k)
    converged = False
    for _ in range(max_iter):
        p = 1 / (1 + np.exp(-d @ beta))
        g = d.T @ (z - p)
        h = (d * (p * (1 - p))[:, None]).T @ d
        step = np.linalg.solve(h, g)
        beta += step
        if np.max(np.abs(step)) < tol:
            converged = True
            break
    p = 1 / (1 + np.exp(-d @ beta))
    h = (d * (p * (1 - p))[:, None]).T @ d
    se = np.sqrt(np.diag(np.linalg.inv(h)))
    return {**diag, "identifiable": True, "beta": beta.tolist(), "se": se.tolist(), "converged": converged,
            "separation_suspected": bool(np.max(np.abs(beta)) > 15)}


# ---- binary classification (label 0 / 1, score = P(label 1)) ----------------------------------------------

def _w(y, w):
    return np.ones(len(y)) if w is None else np.asarray(w, float)


def ece(y, p, w=None, bins: int = 10) -> float:
    """10-bin expected calibration error of P(label 1), equal-width bins, optional sample weights."""
    y, p, w = np.asarray(y, float), np.asarray(p, float), _w(y, w)
    b = np.minimum((p * bins).astype(int), bins - 1)
    tot = w.sum()
    return float(sum(abs(np.average(y[b == i], weights=w[b == i]) - np.average(p[b == i], weights=w[b == i])) * w[b == i].sum() / tot
                     for i in range(bins) if w[b == i].sum() > 0))


def binary_report(y, p1, labels: Sequence[str], threshold: float = 0.5, w=None) -> dict:
    """Metrics for labels (0, 1) given P(label 1); prediction = 1 iff p1 > threshold (= sklearn ``predict`` at 0.5).

    With weights the result is the weighted (re-weighted) estimate of the same quantities."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y, p1, w = np.asarray(y, int), np.asarray(p1, float), _w(y, w)
    yhat = (p1 > threshold).astype(int)
    cm = np.array([[w[(y == i) & (yhat == j)].sum() for j in (0, 1)] for i in (0, 1)])
    per = {}
    for c in (0, 1):
        tp, fn, fp = cm[c, c], cm[c].sum() - cm[c, c], cm[:, c].sum() - cm[c, c]
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per[labels[c]] = {"precision": float(prec), "recall": float(rec),
                          "f1": float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0, "support": float(cm[c].sum())}
    sup = np.array([per[labels[c]]["support"] for c in (0, 1)])
    f1 = np.array([per[labels[c]]["f1"] for c in (0, 1)])
    return {
        "n": int(len(y)), "accuracy": float(np.trace(cm) / cm.sum()),
        "balanced_accuracy": float(np.mean([per[l]["recall"] for l in labels])),
        "macro_f1": float(f1.mean()), "weighted_f1": float((f1 * sup).sum() / sup.sum()), "per_class": per,
        "confusion": {"labels_rows_true_cols_pred": list(labels), "counts": cm.round(6).tolist(),
                      "row_percent": (cm / cm.sum(1, keepdims=True) * 100).round(2).tolist()},
        "roc_auc": float(roc_auc_score(y, p1, sample_weight=w)),
        f"pr_auc_positive_{labels[1]}": float(average_precision_score(y, p1, sample_weight=w)),
        f"pr_auc_positive_{labels[0]}": float(average_precision_score(1 - y, 1 - p1, sample_weight=w)),
        "brier": float(np.average((p1 - y) ** 2, weights=w)), "ece_10bin": ece(y, p1, w),
    }


def prior_weights(y, target_prior: Sequence[float]) -> np.ndarray:
    """Per-example weights that re-weight the sample's class mix to ``target_prior`` (class-conditional
    behaviour held fixed); the weights average to 1."""
    y = np.asarray(y, int)
    share = np.bincount(y, minlength=2) / len(y)
    return np.array([target_prior[c] / share[c] for c in y])
