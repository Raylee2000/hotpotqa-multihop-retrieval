"""Load check of retrained models through the public inference loaders (no evaluation data is read).

With RAG_HOTPOTQA_MODELS_MANIFEST set to the given manifest, exactly as `run_eval.py --models-manifest` does:
  1. verify_models(): every file of every model matches the manifest (names / file sets as configs/models.json);
  2. General Encoder and CSH: SentenceTransformer(dir) as scripts/final_eval/gen_candidates.py load_model;
  3. SF folds: AutoModel / AutoTokenizer + head.safetensors and score_pairs as scripts/final_eval/live_inputs.py;
  4. QType: QTypeLR.load (COMPLETE artifact) as live_inputs.py;
  5. R5: feature schema order == the candidate FEATURES of gen_candidates.py, R5Trees.load as scripts/rank_public.py.
Every model scores a few Training questions (finite outputs, expected shapes). The full pipeline compatibility check is
`python run.py evaluate --models-manifest <manifest> --shards 1`.
Usage: python scripts/check_trained_models.py --manifest outputs/trained_models/<profile>/models.json
Output: LOAD_CHECK.json next to the manifest (written only on PASS).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_hotpotqa.release.integrity import MODELS_MANIFEST_ENV  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="repository-relative trained model manifest")
    ap.add_argument("--n-questions", type=int, default=3)
    args = ap.parse_args()
    os.environ[MODELS_MANIFEST_ENV] = args.manifest
    from rag_hotpotqa.data.access import load_split_ids, load_split_records
    from rag_hotpotqa.data.documents import build_question
    from rag_hotpotqa.experiments.runner import select_subset
    from rag_hotpotqa.release.integrity import model_dir, models_manifest_path, verify_models
    from rag_hotpotqa.training.qtype_artifact import QTypeLR
    from rag_hotpotqa.training.sf import article_sentences, score_pairs
    from rag_hotpotqa.utils.config import load_config, load_settings

    st = load_settings()
    rcfg = load_config("retrieval", st)
    report = {"manifest": models_manifest_path(), "utc": datetime.now(timezone.utc).isoformat(), "models": {}}
    verify_models()
    report["files_verified"] = "PASS"
    import torch
    from safetensors.torch import load_file
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(8)
    ids = select_subset(load_split_ids(st, "training", "train"), args.n_questions)
    qs = [build_question(r, rcfg["document"]["template"]) for r in load_split_records(st, "training", "train", ids)]
    for name in ("general_encoder", "csh"):
        m = SentenceTransformer(str(model_dir(name)), device="cpu")
        m.max_seq_length = rcfg["dense"]["max_seq_length"]
        e = m.encode([q.question for q in qs] + [d.text for d in qs[0].docs], normalize_embeddings=True)
        assert e.shape[1] == 384 and np.isfinite(e).all() and np.allclose(np.linalg.norm(e, axis=1), 1, atol=1e-4)
        report["models"][name] = {"loader": "SentenceTransformer", "embeddings": list(e.shape)}
    for f in (0, 1):
        run = model_dir(f"sf_selector_fold{f}")
        enc, tok = AutoModel.from_pretrained(run), AutoTokenizer.from_pretrained(run)
        head = torch.nn.Linear(enc.config.hidden_size, 1)
        head.load_state_dict(load_file(str(run / "head.safetensors")))
        enc.eval()
        ss = article_sentences(qs[0], 0)
        sc = score_pairs(enc, head, tok, [qs[0].question] * len(ss), [s for _, s in ss])
        assert len(sc) == len(ss) and np.isfinite(sc).all() and ((sc >= 0) & (sc <= 1)).all()
        report["models"][f"sf_selector_fold{f}"] = {"loader": "AutoModel + head.safetensors", "sentences_scored": len(sc)}
    clf = QTypeLR.load(model_dir("qtype_frozen_lr"), st)
    p = clf.p_bridge([q.question for q in qs])
    assert p.shape == (len(qs),) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    report["models"]["qtype_frozen_lr"] = {"loader": "QTypeLR (COMPLETE artifact)", "p_bridge": [round(float(x), 6) for x in p]}
    spec = importlib.util.spec_from_file_location("gen_candidates_public", ROOT / "scripts/final_eval/gen_candidates.py")
    gc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gc)
    from rag_hotpotqa.ranking.r5_trees import R5Trees

    r5 = model_dir("r5_full")
    schema = json.loads((r5 / "feature_schema.json").read_text(encoding="utf-8"))
    assert schema["candidate_feature_order"] == gc.FEATURES, "R5 candidate feature order differs from gen_candidates.py"
    model = R5Trees.load(r5 / "r5_trees.npz")
    assert model.features == schema["features"]
    s = model.predict(np.random.default_rng(0).normal(size=(16, len(model.features))))
    assert s.shape == (16,) and np.isfinite(s).all()
    report["models"]["r5_full"] = {"loader": "R5Trees (npz, allow_pickle=False)", "n_trees": model.n_trees}
    report["status"] = "PASS"
    out = (ROOT / args.manifest).parent / "LOAD_CHECK.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
