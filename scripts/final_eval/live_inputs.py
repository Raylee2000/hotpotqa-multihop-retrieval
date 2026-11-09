"""[final_eval_v1 GENERIC ENTRY: --part {v_select, v_report, bc_holdout} via rag_hotpotqa.data.eval_parts.load_eval_part;
v_report / bc_holdout need --authorisation (FROZEN, approved). Outputs: <out-root>/<part>/. Diff against the source shows the only changes.]

(Comparison with historical V-select prediction files moved to scripts/final_eval/verify_vselect.py.)

Stage 3 step 1: LIVE upstream predictions for V-select from raw questions + context (no historical prediction read
for inference): SF-selector sentence scores of both cross-fit folds (sf_cf_v1 final/ weights, training/sf.py
score_pairs, rounded to 5 decimals exactly as the original export, scripts/train_sf_selector_cf.py:212-222) and P(bridge)
from the COMPLETE qtype artifact (QTypeLR, rounded to 6 decimals as the original export, train_qtype_classifier.py:134).
Afterwards — verification only — the live files are compared with the historical prediction files.
Output (fail-if-exists): outputs/vselect_e2e_v1/live_inputs/{sf_fold0.jsonl, sf_fold1.jsonl, qtype.jsonl, compare.json, meta.json}
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rag_hotpotqa.utils.config import load_config, load_settings  # noqa: E402
from rag_hotpotqa.data.eval_parts import PARTS, select_eval_subset



def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=PARTS, default="v_select")
    ap.add_argument("--ids-file", default=None)
    ap.add_argument("--authorisation", default=None)
    ap.add_argument("--out-root", default="outputs/final_eval_v1")
    args = ap.parse_args()
    OUT = ROOT / args.out_root / args.part / "live_inputs"
    if OUT.exists():
        raise SystemExit(f"{OUT} exists — refusing to overwrite")
    st = load_settings()
    import torch
    from transformers import AutoModel, AutoTokenizer

    from rag_hotpotqa.data.eval_parts import load_eval_part
    from rag_hotpotqa.data.documents import build_question
    from rag_hotpotqa.release.integrity import model_dir
    from rag_hotpotqa.training.qtype_artifact import QTypeLR
    from rag_hotpotqa.training.sf import article_sentences, score_pairs
    from rag_hotpotqa.utils.io import sha256_file
    from rag_hotpotqa.utils.runtime import git_commit, peak_rss_mb

    torch.set_num_threads(8)
    rcfg = load_config("retrieval", st)
    ids, recs = load_eval_part(st, args.part, args.authorisation)
    ids, recs = select_eval_subset(ids, recs, args.ids_file)
    qs = [build_question(r, rcfg["document"]["template"]) for r in recs]
    OUT.mkdir(parents=True)
    meta, cmp = {"git_commit": git_commit(ROOT), "utc": datetime.now(timezone.utc).isoformat(), "seconds": {}}, {}
    for f in (0, 1):
        run = model_dir(f"sf_selector_fold{f}")  # models/sf_selector_fold{f} (was outputs/sf_selector_v2/sf_cf_v1/fold{f}/final)
        enc, tok = AutoModel.from_pretrained(run), AutoTokenizer.from_pretrained(run)
        head = torch.nn.Linear(enc.config.hidden_size, 1)
        from safetensors.torch import load_file  # head converted from head.pt; tensors verified bit-identical
        head.load_state_dict(load_file(str(run / "head.safetensors")))
        enc.eval()
        meta[f"sf_fold{f}_weights_sha256"] = {"encoder": sha256_file(run / "model.safetensors"), "head": sha256_file(run / "head.safetensors")}
        t = time.perf_counter()
        with open(OUT / f"sf_fold{f}.jsonl", "w", encoding="utf-8") as fh:
            for q in qs:
                for idx in range(len(q.docs)):
                    ss = article_sentences(q, idx)
                    sc = score_pairs(enc, head, tok, [q.question] * len(ss), [s for _, s in ss])
                    fh.write(json.dumps({"qid": q.qid, "doc": idx, "sent_ids": [i for i, _ in ss], "scores": [round(float(x), 5) for x in sc]}) + "\n")
        meta["seconds"][f"sf_fold{f}"] = round(time.perf_counter() - t, 1)
        del enc
    clf = QTypeLR.load(model_dir("qtype_frozen_lr"), st)
    t = time.perf_counter()
    p = clf.p_bridge([r["question"] for r in recs])
    meta["seconds"]["qtype"] = round(time.perf_counter() - t, 1)
    with open(OUT / "qtype.jsonl", "w", encoding="utf-8") as fh:
        for r, pb in zip(recs, p):
            fh.write(json.dumps({"qid": r["_id"], "p_bridge": round(float(pb), 6)}) + "\n")
    meta["peak_rss_mb"] = round(peak_rss_mb(), 1)
    meta["files_sha256"] = {p_.name: sha256_file(p_) for p_ in sorted(OUT.glob("*.jsonl"))}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    meta["part"] = args.part
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
