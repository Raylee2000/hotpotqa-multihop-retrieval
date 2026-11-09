"""Compare a public run with the formal research results (results/reference/).

Per question and method, these fields must be identical to the formal output: pr1, cand, gold_rank, top1_pair and,
for R5_full / R0, the top-5 pairs and their float scores. Verdicts:
  EXACT      every compared question and field is identical;
  DIFFERENT  at least one difference (listed; nothing is rounded or tolerated to turn it into EXACT);
  NO_REFERENCE  the method has no formal reference on this part (R0 on Official Dev: Matrix B did not run R0).
For a complete part, the aggregate PR@1 counts are also compared with the formal summary.
Usage: python scripts/compare_results.py --run <run-name>
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "results/reference"
FIELDS = ("pr1", "cand", "gold_rank", "top1_pair", "top5")


def load_ref(part: str) -> dict:
    with gzip.open(REF / f"{part}.per_question.jsonl.gz", "rt", encoding="utf-8") as f:
        return {r["qid"]: r for r in map(json.loads, f)}


def compare_part(run_dir: Path, part: str) -> dict:
    run_rows = [json.loads(x) for x in open(run_dir / part / "per_question.jsonl", encoding="utf-8") if x.strip()]
    ref = load_ref(part)
    methods = sorted({m for r in run_rows for m in r["methods"]})
    out = {"questions": len(run_rows), "methods": {}}
    for m in methods:
        n = diff = 0
        examples = []
        for r in run_rows:
            rr = ref.get(r["qid"])
            if rr is None:
                raise SystemExit(f"{part}: question {r['qid']} is not in the formal reference")
            if m not in rr["methods"]:
                continue
            n += 1
            a, b = r["methods"][m], rr["methods"][m]
            bad = [k for k in FIELDS if (k in a or k in b) and a.get(k) != b.get(k)]
            if r["qtype"] != rr["qtype"]:
                bad.append("qtype")
            if bad:
                diff += 1
                if len(examples) < 10:
                    examples.append({"qid": r["qid"], "fields": bad})
        verdict = "NO_REFERENCE" if n == 0 else ("EXACT" if diff == 0 else "DIFFERENT")
        out["methods"][m] = {"verdict": verdict, "compared": n, "different": diff, "examples": examples}
    summ = json.loads((run_dir / part / "summary.json").read_text(encoding="utf-8"))
    ref_s = json.loads((REF / f"{part}.summary.json").read_text(encoding="utf-8"))
    if summ["provenance"]["complete_part"]:
        agg = {}
        for m, e in ref_s["strata"]["overall"]["methods"].items():
            got = summ["strata"]["overall"]["methods"].get(m)
            if got is not None:
                agg[m] = {"formal_correct": int(e["pr1"]["sum"]), "run_correct": int(got["pr1"]["sum"]),
                          "formal_ci95": e["pr1"]["ci95"], "run_ci95": got["pr1"]["ci95"],
                          "verdict": "EXACT" if e["pr1"] == got["pr1"] else "DIFFERENT"}
        out["aggregate"] = agg
    else:
        out["aggregate"] = "not compared: partial run (smoke / --shards); per-question comparison only"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run_dir = ROOT / "outputs/runs" / args.run
    parts = [p.name for p in run_dir.iterdir() if (p / "per_question.jsonl").exists()]
    report = {p: compare_part(run_dir, p) for p in sorted(parts)}
    (run_dir / "comparison.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    ok = True
    for p, r in report.items():
        print(f"== {p}: {r['questions']} questions")
        for m, e in r["methods"].items():
            print(f"   {m:28s} {e['verdict']:12s} compared {e['compared']}, different {e['different']}")
            ok &= e["verdict"] != "DIFFERENT"
        if isinstance(r["aggregate"], dict):
            for m, e in r["aggregate"].items():
                print(f"   aggregate {m:18s} {e['verdict']:9s} formal {e['formal_correct']} / run {e['run_correct']}")
                ok &= e["verdict"] == "EXACT"
        else:
            print(f"   aggregate: {r['aggregate']}")
    print("RESULT:", "MATCHES FORMAL RESULTS" if ok else "DIFFERS FROM FORMAL RESULTS (see comparison.json)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
