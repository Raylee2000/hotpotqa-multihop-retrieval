"""MAINTAINER-ONLY: extract the formal reference results used by `python run.py compare`.

Reads the sealed formal analysis outputs of the research repository (JSON only) and keeps, per question, the methods
covered by this release: R5_full, R0 (V-report and B/C only; Official Dev ran Matrix B without R0) and the six
baselines. Values are copied unchanged. Also extracts the formal summary for those methods.
Usage: python tools/maintainer/build_reference_results.py --research-root <research repo>
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARTS = ["official_dev_distractor", "v_report", "bc_holdout"]


def keep(name: str) -> bool:
    return name in ("R5_full", "R0") or name.startswith("baseline/")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--research-root", type=Path, required=True)
    args = ap.parse_args()
    out = ROOT / "results" / "reference"
    out.mkdir(parents=True, exist_ok=True)
    index = {}
    for part in PARTS:
        src = args.research_root / "outputs/overnight_v3/results" / f"{part}_analysis"
        target = out / f"{part}.per_question.jsonl.gz"
        if target.exists():
            raise SystemExit(f"{target} exists")
        n = 0
        with open(src / "per_question_metrics.jsonl", encoding="utf-8") as f, open(target, "xb") as raw, \
                gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as g:
            for line in f:
                r = json.loads(line)
                row = {"qid": r["qid"], "qtype": r["qtype"], "n_cand": r.get("n_cand"),
                       "methods": {k: v for k, v in r["methods"].items() if keep(k)}}
                g.write((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
                n += 1
        s = json.loads((src / "summary.json").read_text(encoding="utf-8"))
        strata = {}
        for g_name, st in s["strata"].items():
            strata[g_name] = {"n": st["n"], "methods": {k: v for k, v in st["methods"].items() if keep(k)},
                              "full_vs": {k: v for k, v in st["full_vs"].items() if keep(k)}}
        summ = {"part": part, "n_questions": s["n_questions"], "statistics": s["statistics"], "strata": strata,
                "provenance": {k: s["provenance"][k] for k in ("matrix", "question_ids_sha256", "freeze_manifest_sha256",
                                                                "frozen_git_commit", "qtype_counts")},
                "source": f"outputs/overnight_v3/results/{part}_analysis/summary.json (research repository)",
                "source_sha256": hashlib.sha256((src / "summary.json").read_bytes()).hexdigest()}
        (out / f"{part}.summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                                                  encoding="utf-8", newline="\n")
        index[part] = {"questions": n, "per_question_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                       "source_per_question_sha256": hashlib.sha256((src / "per_question_metrics.jsonl").read_bytes()).hexdigest()}
        print(part, n)
    (out / "INDEX.json").write_text(json.dumps(index, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
