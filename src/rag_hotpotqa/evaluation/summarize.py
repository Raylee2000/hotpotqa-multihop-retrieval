"""Aggregate spec-v2 per-question rows into O / E / R tables (all numbers auto-generated)."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..data.documents import DIRECTIONS
from ..utils.io import read_jsonl
from .metrics import KS, aggregate

STRATEGIES = ("G0", "G1", "G2")
FORMULAS = ("F0", "F1", "F2")


def o_keys() -> list[str]:
    return ([f"O.fh_any_gold@{k}" for k in KS] + [f"O.fh_all_gold@{k}" for k in KS if k >= 2]
            + ["O.fh_mrr", "O.fh_best_gold_rank"])


def oracle_keys(exp: str, d: str) -> list[str]:
    return [f"{exp}.{d}.recall@{k}" for k in KS] + [f"{exp}.{d}.rr", f"{exp}.{d}.rank"]


def r1_keys(w: str) -> list[str]:
    return ([f"R1.{w}.path_recall@{k}" for k in KS]
            + [f"R1.{w}.gold_path_rr", f"R1.{w}.gold_path_rank", f"R1.{w}.gold_path_coverage", f"R1.{w}.n_paths",
               f"R1.{w}.n_first_hop"])


def e2_keys() -> list[str]:
    return (["E2.top1A_is_gold"] + [f"E2.cond_recall@{k}" for k in KS] + ["E2.cond_rr"]
            + [f"E2.mdr_{m}@{k}" for k in KS for m in ("PR", "P-EM", "1-Recall", "Path Recall")])


def r2_keys(d: str, f: str) -> list[str]:
    return [f"R2.{d}.{f}.recall@{k}" for k in KS] + [f"R2.{d}.{f}.rr", f"R2.{d}.{f}.rank"]


def r3_keys(s: str) -> list[str]:
    return ([f"R3.{s}.path_recall@{k}" for k in KS] + [f"R3.{s}.gold_path_rr", f"R3.{s}.gold_path_coverage"]
            + [f"R3.{s}.cand_recall@{k}" for k in KS] + [f"R3.{s}.cand_rr", f"R3.{s}.seconds"])


TRUNC_CATS = ("not_truncated", "truncated_sf_kept", "truncated_sf_lost")


def _cat(events: list[dict[str, Any]]) -> str | None:
    """Category of the gold-relevant input(s): worst case over the given events."""
    if not events:
        return None
    if any(e.get("gold_sf_lost") for e in events):
        return "truncated_sf_lost"
    if any(e["truncated"] for e in events):
        return "truncated_sf_kept"
    return "not_truncated"


def _slim(row: dict[str, Any]) -> dict[str, Any]:
    """Keep what aggregation needs; drop large detail."""
    ev = row.get("encoding_events", [])
    gold = row.get("gold") or {}
    ids = {"AtoB": (gold.get("A_id"), gold.get("B_id")), "BtoA": (gold.get("B_id"), gold.get("A_id"))}
    cats: dict[str, str | None] = {}
    for d, (a_id, b_id) in ids.items():
        for exp, itype in (("E1", "Q+A"), ("E1P", "Q+SF")):
            cats[f"{exp}.{d}"] = _cat([e for e in ev if e["exp"] == exp and e["type"] == itype and e["articles"] == [a_id]])
        # R2-F2: the gold-pair candidate A+B (query is Q, never truncated in practice)
        cats[f"R2.{d}.F2"] = _cat([e for e in ev if e["exp"] == "R2" and e["type"] == "A+B" and e["articles"] == [a_id, b_id]])
    # R3-G1: gold-pair A+B inputs that exist (only when a gold article is among the Top-3 A)
    pair = {gold.get("A_id"), gold.get("B_id")}
    cats["R3.G1"] = _cat([e for e in ev if e["exp"] == "R3" and e["type"] == "A+B" and set(e["articles"]) == pair])
    return {"qid": row["qid"], "qtype": row["qtype"], "level": row["level"], "n_docs": row["n_docs"],
            "metrics": row["metrics"], "validity": row.get("validity", {}), "timing": row.get("timing", {}),
            "events": [(e["exp"], e["type"], e["truncated"], "gold_sf_truncation" in e, bool(e.get("gold_sf_lost")))
                       for e in ev],
            "gold_input_category": cats}


def load_slim(path: Path) -> list[dict[str, Any]]:
    return [_slim(r) for r in read_jsonl(path)]


def pair_table(rows: Sequence[dict[str, Any]], d: str) -> dict[str, Any]:
    """E1 vs E1' paired outcomes per k on the common valid sample of direction d."""
    out = {}
    for k in KS:
        c: Counter[str] = Counter()
        for r in rows:
            m = r["metrics"]
            x, y = m.get(f"E1.{d}.recall@{k}"), m.get(f"E1P.{d}.recall@{k}")
            if x is None or y is None:
                continue
            c[{(1, 1): "both_success", (1, 0): "E1_only", (0, 1): "E1P_only", (0, 0): "both_fail"}[(x, y)]] += 1
        out[f"@{k}"] = {"n_common": sum(c.values()), **{key: c.get(key, 0) for key in
                                                         ("both_success", "E1_only", "E1P_only", "both_fail")}}
    return out


def invalid_counts(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Counter] = defaultdict(Counter)
    examples: dict[str, list] = defaultdict(list)
    for r in rows:
        for exp, dirs in r["validity"].items():
            for d, reason in dirs.items():
                key = f"{exp}.{d}"
                out[key]["valid" if reason is None else reason] += 1
                if reason is not None and len(examples[key]) < 20:
                    examples[key].append(r["qid"])
    return {k: dict(v) for k, v in out.items()} | {"invalid_examples": dict(examples)}


def truncation_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, Counter] = defaultdict(Counter)
    by_qtype: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        for exp, itype, tr, gold_inv, sf_lost in r["events"]:
            for c in (by_type[f"{exp}|{itype}"], by_qtype[r["qtype"]]):
                c["events"] += 1
                c["truncated"] += int(tr)
                c["truncated_with_gold_article"] += int(tr and gold_inv)
                c["truncated_gold_sf_lost"] += int(sf_lost)

    def fin(c: dict[str, Counter]) -> dict[str, Any]:
        return {k: {"events": v["events"], "truncated": v["truncated"],
                    "ratio": round(v["truncated"] / v["events"], 4) if v["events"] else None,
                    "truncated_with_gold_article": v["truncated_with_gold_article"],
                    "truncated_gold_sf_lost": v["truncated_gold_sf_lost"]} for k, v in sorted(c.items())}

    # Performance split by the category of the gold-relevant input (D-11 + SF-loss requirement).
    targets = {f"E1.{d}": [f"E1.{d}.recall@1", f"E1.{d}.recall@3"] for d in DIRECTIONS}
    targets |= {f"E1P.{d}": [f"E1P.{d}.recall@1", f"E1P.{d}.recall@3"] for d in DIRECTIONS}
    targets |= {f"R2.{d}.F2": [f"R2.{d}.F2.recall@1", f"R2.{d}.F2.recall@3", f"R2.{d}.F2.rr"] for d in DIRECTIONS}
    targets |= {"R3.G1": ["R3.G1.path_recall@1", "R3.G1.path_recall@3", "R3.G1.cand_recall@1"]}
    split_perf: dict[str, Any] = {}
    for key, metric_keys in targets.items():
        if not any(r["gold_input_category"].get(key) is not None for r in rows):
            continue
        per_cat = {}
        for cat in TRUNC_CATS:
            sub = [r for r in rows if r["gold_input_category"].get(key) == cat]
            entry: dict[str, Any] = {"n_questions": len(sub)}
            for m in metric_keys:
                vals = [r["metrics"][m] for r in sub if r["metrics"].get(m) is not None]
                entry[m] = round(sum(vals) / len(vals), 4) if vals else None
                entry[f"{m}__n"] = len(vals)
            per_cat[cat] = entry
        per_cat["no_gold_pair_input"] = sum(r["gold_input_category"].get(key) is None for r in rows)
        split_perf[key] = per_cat
    out = {"by_experiment_and_input_type": fin(by_type), "by_question_type": fin(by_qtype)}
    if split_perf:
        out["performance_by_gold_input_truncation"] = split_perf
    return out


def summarize_retriever(rows: Sequence[dict[str, Any]], experiments: Iterable[str], cfg: dict) -> dict[str, Any]:
    exps = set(experiments)
    out: dict[str, Any] = {"n_rows": len(rows), "O": aggregate(rows, o_keys(), metrics_field="metrics")}
    if exps & {"E1", "E1P"}:
        for exp in ("E1", "E1P"):
            if exp in exps:
                out[exp] = {d: aggregate(rows, oracle_keys(exp, d), metrics_field="metrics") for d in DIRECTIONS}
        out["validity"] = invalid_counts(rows)
        if {"E1", "E1P"} <= exps:
            out["E1_vs_E1P_pairs"] = {d: pair_table(rows, d) for d in DIRECTIONS}
    if "R1" in exps:
        out["R1"] = {w: aggregate(rows, r1_keys(w), metrics_field="metrics") for w in cfg["widths"]}
        out["E2"] = aggregate(rows, e2_keys(), metrics_field="metrics")
    if "R2" in exps:
        out["R2"] = {d: {f: aggregate(rows, r2_keys(d, f), metrics_field="metrics") for f in FORMULAS} for d in DIRECTIONS}
    if "R3" in exps:
        out["R3"] = {s: aggregate(rows, r3_keys(s), metrics_field="metrics") for s in STRATEGIES}
    out["truncation"] = truncation_summary(rows)
    timing_keys = sorted({k for r in rows for k in r["timing"]})
    out["timing_mean_s"] = {k: round(sum(r["timing"].get(k, 0.0) for r in rows) / max(len(rows), 1), 5)
                            for k in timing_keys}
    return out


def md_table(section: dict[str, Any], keys: Sequence[str], strip: str = "") -> str:
    groups = ["overall"] + [g for g in section if g != "overall"]
    lines = ["| metric | " + " | ".join(f"{g} (n={section[g]['n']})" for g in groups) + " |",
             "|---|" + "---:|" * len(groups)]
    for k in keys:
        cells = []
        for g in groups:
            v = section[g].get(k)
            n = section[g].get(f"{k}__n")
            cell = "–" if v is None else f"{v:.4f}"
            if n is not None and v is not None:
                cell += f" (n={n})"
            cells.append(cell)
        lines.append(f"| `{k.replace(strip, '') if strip else k}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)
