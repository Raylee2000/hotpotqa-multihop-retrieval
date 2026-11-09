"""Assemble the retrained Full-method models in the released layout and write their model manifest.

For every model of configs/models.json the exact released file set is copied from the training outputs
(the selected checkpoint of each run) to outputs/trained_models/<profile>/models/<name>/, and
outputs/trained_models/<profile>/models.json is written in the schema of configs/models.json (target_dir, empty_dirs,
per-file sha256 / bytes; no Release asset fields). The public inference reads it with
`python run.py evaluate --models-manifest outputs/trained_models/<profile>/models.json` and verifies every file.

Sources (configs/training_full.yaml run names):
  general_encoder     outputs/training_v1/<general>/result.json "best" snapshot (pre-registered selection score)
  csh                 outputs/csh_v1/<csh>/result.json "best" snapshot (QV2 conditional R@1)
  sf_selector_fold<f> outputs/sf_selector_v2/<sf>/fold<f>/final (final step; head.safetensors)
  qtype_frozen_lr     <qtype_artifact> (COMPLETE artifact: manifest + independent reload check)
  r5_full             outputs/path_reranker_v1/<r5>/{r5_trees.npz, feature_schema.json}
License / provenance documents of the released file set (LICENSE_FILES: LICENSE, NOTICE, DATA_SOURCES.md,
MODIFICATIONS.md, PROVENANCE.json, UPSTREAM_MODEL_CARD.md, MINILM_MIT_LICENSE.txt) are never produced by training.
They are taken from the tracked release_assets/legal/<name>/ and must match configs/models.json byte for byte;
LICENSE, NOTICE, DATA_SOURCES.md, UPSTREAM_MODEL_CARD.md and MINILM_MIT_LICENSE.txt are copied unchanged.
PROVENANCE.json and MODIFICATIONS.md describe the released checkpoint, so for retrained weights they are rewritten:
the new PROVENANCE.json records the retrained files (source run, sha256) and keeps the released license / upstream /
training-data records; MODIFICATIONS.md states the retraining and keeps the released notice verbatim below it.
A missing file is fatal; files outside the released set (e.g. the auto-generated sentence-transformers README.md)
are not copied. Output (fail-if-exists): outputs/trained_models/<profile>/{models/, models.json}
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_hotpotqa.release.integrity import MODELS_MANIFEST, read_json, sha256_bytes, sha256_path  # noqa: E402
from rag_hotpotqa.utils.artifacts import is_complete  # noqa: E402
from rag_hotpotqa.utils.config import load_yaml  # noqa: E402

LEGAL_DIR = "release_assets/legal"
LICENSE_FILES = ("LICENSE", "NOTICE", "DATA_SOURCES.md", "MODIFICATIONS.md", "PROVENANCE.json", "UPSTREAM_MODEL_CARD.md",
                 "MINILM_MIT_LICENSE.txt")
REWRITTEN = ("PROVENANCE.json", "MODIFICATIONS.md")


def released_legal(name: str, spec: dict) -> dict[str, bytes]:
    """The released license / provenance documents of model `name`, verified against configs/models.json."""
    out = {}
    for rel in sorted(set(spec["files"]) & set(LICENSE_FILES)):
        p = ROOT / LEGAL_DIR / name / rel
        if not p.is_file():
            raise SystemExit(f"{name}: {LEGAL_DIR}/{name}/{rel} missing (license documents are not training outputs)")
        b = p.read_bytes()
        if sha256_bytes(b) != spec["files"][rel]["sha256"]:
            raise SystemExit(f"{name}: {LEGAL_DIR}/{name}/{rel} differs from {MODELS_MANIFEST}")
        out[rel] = b
    return out


def retrained_legal(name: str, legal: dict[str, bytes], profile: str, source_run: str, trained: dict) -> dict[str, bytes]:
    """License documents for retrained weights: unchanged texts, PROVENANCE.json / MODIFICATIONS.md rewritten."""
    out = {k: v for k, v in legal.items() if k not in REWRITTEN}
    if "PROVENANCE.json" in legal:
        rel_prov = json.loads(legal["PROVENANCE.json"].decode("utf-8"))
        prov = {"model": name, "retrained": True, "profile": profile,
                "note": "Retrained by python run.py train (configs/training_full.yaml); not the released weights.",
                "source_run": source_run, "trained_files": trained,
                "released_provenance_sha256": sha256_bytes(legal["PROVENANCE.json"])}
        prov.update({k: rel_prov[k] for k in ("license", "license_source", "license_text_sha256", "training_data", "upstream")
                     if k in rel_prov})
        out["PROVENANCE.json"] = (json.dumps(prov, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if "MODIFICATIONS.md" in legal:
        head = (f"# Modification notice — {name} (retrained)\n\n"
                f"These weights were retrained locally by `python run.py train --profile {profile}` from the pinned base model "
                f"and the fixed HotpotQA splits (source run: {source_run}); they are not the released v1 weights. "
                "The training method is the one described in the released notice below; the retrained files are "
                "listed with their SHA256 in PROVENANCE.json.\n\n## Released v1 modification notice (verbatim)\n\n")
        out["MODIFICATIONS.md"] = head.encode("utf-8") + legal["MODIFICATIONS.md"]
    return out


def sources(runs: dict) -> tuple[dict, dict]:
    gen = json.loads((ROOT / "outputs/training_v1" / runs["general"] / "result.json").read_text(encoding="utf-8"))
    csh = json.loads((ROOT / "outputs/csh_v1" / runs["csh"] / "result.json").read_text(encoding="utf-8"))
    r5 = json.loads((ROOT / "outputs/path_reranker_v1" / runs["r5"] / "result.json").read_text(encoding="utf-8"))
    src = {"general_encoder": gen["best"]["snapshot"], "csh": csh["best"]["snapshot"],
           "sf_selector_fold0": f"outputs/sf_selector_v2/{runs['sf']}/fold0/final",
           "sf_selector_fold1": f"outputs/sf_selector_v2/{runs['sf']}/fold1/final",
           "qtype_frozen_lr": runs["qtype_artifact"], "r5_full": f"outputs/path_reranker_v1/{runs['r5']}"}
    selection = {"general_encoder": {"step": gen["best"]["step"], "metrics": gen["best"]["metrics"],
                                     "history": [{"step": h["step"], "score": h["metrics"]["score"]} for h in gen["selection_history"]]},
                 "csh": {"step": csh["best"]["step"], "metrics": csh["best"]["metrics"],
                         "history": [{"step": h["step"], "score": h["metrics"]["score"]} for h in csh["selection_history"]]},
                 "r5_full": {"hyper": r5["hyper"], "grid_v_select_pr1": r5["grid"], "v_select": r5["v_select"]}}
    return src, selection


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    args = ap.parse_args()
    cfg = load_yaml(ROOT / "configs/training_full.yaml")
    runs = cfg["profiles"][args.profile]["runs"]
    base = ROOT / "outputs/trained_models" / args.profile
    dest_root, man_path = base / "models", base / "models.json"
    if dest_root.exists() or man_path.exists():
        raise SystemExit(f"{dest_root.relative_to(ROOT)} exists — refusing to overwrite")
    if not is_complete(ROOT / runs["qtype_artifact"]):
        raise SystemExit(f"{runs['qtype_artifact']} is not a COMPLETE artifact")
    released = read_json(MODELS_MANIFEST)["models"]
    src, selection = sources(runs)
    tmp = base / "models.partial"
    if tmp.exists():
        shutil.rmtree(tmp)
    out = {"version": "trained-v1", "profile": args.profile,
           "note": "Retrained by python run.py train (configs/training_full.yaml); not the released weights. "
                   "Same model names, directory layout and file sets as configs/models.json.",
           "created_utc": datetime.now(timezone.utc).isoformat(), "selection": selection, "models": {}}
    legal = {name: released_legal(name, spec) for name, spec in released.items()}   # checked before anything is copied
    for name, spec in released.items():
        s = ROOT / src[name]
        d = tmp / name
        files = {}
        for rel in sorted(set(spec["files"]) - set(LICENSE_FILES)):
            p = s / rel
            if not p.is_file():
                raise SystemExit(f"{name}: {src[name]}/{rel} missing (trained file set: "
                                 f"{sorted(set(spec['files']) - set(LICENSE_FILES))})")
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, d / rel)
            files[rel] = {"sha256": sha256_path(d / rel), "bytes": (d / rel).stat().st_size}
        for rel, b in retrained_legal(name, legal[name], args.profile, src[name], dict(files)).items():
            (d / rel).write_bytes(b)
            files[rel] = {"sha256": sha256_bytes(b), "bytes": len(b)}
        files = dict(sorted(files.items()))
        if set(files) != set(spec["files"]):
            raise SystemExit(f"{name}: exported file set differs from {MODELS_MANIFEST}")
        for e in spec.get("empty_dirs", []):
            (d / e).mkdir(parents=True, exist_ok=True)
        out["models"][name] = {"target_dir": f"outputs/trained_models/{args.profile}/models/{name}", "role": spec["role"],
                               "source_run": src[name], "empty_dirs": spec.get("empty_dirs", []), "files": files}
    tmp.rename(dest_root)
    man_path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"PASS exported {len(out['models'])} models -> {man_path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
