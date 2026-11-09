"""MAINTAINER-ONLY: build the model Release assets and configs/models.json from the research repository.

Not part of the public workflow. Reads only safe formats (safetensors, JSON, tokenizer text, numpy without
pickle) from an explicit allowlist, plus the two pre-converted files (r5_trees.npz, SF head safetensors), plus the
license / provenance documents of each model from --legal (default release_assets/legal/<name>/: LICENSE, NOTICE,
MODIFICATIONS.md, DATA_SOURCES.md, PROVENANCE.json and, for the MiniLM-derived models, UPSTREAM_MODEL_CARD.md and
MINILM_MIT_LICENSE.txt). A model without its required license documents is refused. Zips are deterministic (sorted
entries, fixed timestamps). Usage:

    python tools/maintainer/package_assets.py --research-root <research repo> --converted release_assets/staging
        [--legal release_assets/legal] [--out release_assets] [--manifest configs/models.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION = "v1"
ASSET_REVISION = "license-metadata-1"
LICENSE = "Apache-2.0"
LEGAL_REQUIRED = ("LICENSE", "NOTICE", "MODIFICATIONS.md", "DATA_SOURCES.md", "PROVENANCE.json")
ST_FILES = ["config.json", "config_sentence_transformers.json", "model.safetensors", "modules.json",
            "sentence_bert_config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
            "vocab.txt", "1_Pooling/config.json"]
SF_FILES = ["config.json", "model.safetensors", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt"]
QT_FILES = ["artifact_manifest.json", "COMPLETE.json", "feature_schema.json", "inmemory_p_bridge_train_crossfit.npy",
            "inmemory_p_bridge_v_select.npy", "lr_fold0.npz", "lr_fold1.npz", "lr_full.npz", "reload_check.json"]

MODELS = {
    "general_encoder": {"src": "outputs/training_v1/hardneg_v1/snapshot_step01869", "files": ST_FILES, "dirs": ["2_Normalize"],
                        "role": "General Encoder：第一跳、General 第二跳與 General baseline（hardneg 全參數微調，step 1869）"},
    "csh": {"src": "outputs/csh_v1/csh_main_v1/snapshot_step01500", "files": ST_FILES, "dirs": ["2_Normalize"],
            "role": "Conditional Second-Hop Encoder（main，step 1500）"},
    "sf_selector_fold0": {"src": "outputs/sf_selector_v2/sf_cf_v1/fold0/final", "files": SF_FILES,
                          "extra": {"head.safetensors": "sf_head_fold0.safetensors"},
                          "role": "Supporting Fact Selector cross-encoder，cross-fit fold 0（head 由 head.pt 轉為 safetensors）"},
    "sf_selector_fold1": {"src": "outputs/sf_selector_v2/sf_cf_v1/fold1/final", "files": SF_FILES,
                          "extra": {"head.safetensors": "sf_head_fold1.safetensors"},
                          "role": "Supporting Fact Selector cross-encoder，cross-fit fold 1"},
    "qtype_frozen_lr": {"src": "outputs/qtype_artifacts_v1/qtype_v2_frozen_lr", "files": QT_FILES,
                        "role": "QType frozen-LR（凍結 MiniLM 句向量 + Logistic Regression）；原 qtype_v2 權重未保存，此為同條件重新擬合的替代品"},
    "r5_full": {"src": "outputs/path_reranker_refit_v1/general_csh/R5_lambdamart__full", "files": ["feature_schema.json"],
                "extra": {"r5_trees.npz": "r5_trees.npz"},
                "role": "R5_full 學習式排序器（200 棵回歸樹，depth 4，lr 0.1）；由 model.joblib 轉為無 pickle 的 npz"},
}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--research-root", type=Path, required=True)
    ap.add_argument("--converted", type=Path, required=True)
    ap.add_argument("--legal", type=Path, default=ROOT / "release_assets" / "legal")
    ap.add_argument("--out", type=Path, default=ROOT / "release_assets")
    ap.add_argument("--manifest", type=Path, default=ROOT / "configs" / "models.json")
    args = ap.parse_args()
    manifest = {"version": VERSION,
                "note": "Release assets include Apache-2.0 license, notices, modification statements and provenance. "
                        "All pre-existing archive entries retain their original bytes. ZIPs and every extracted file are SHA256 verified.",
                "models": {}}
    for name, spec in MODELS.items():
        src = args.research_root / spec["src"]
        entries = {rel: (src / rel).read_bytes() for rel in spec["files"]}
        for rel, conv in spec.get("extra", {}).items():
            entries[rel] = (args.converted / conv).read_bytes()
        legal_dir = args.legal / name
        missing = [f for f in LEGAL_REQUIRED if not (legal_dir / f).is_file()]
        if missing:
            raise SystemExit(f"{name}: license documents missing in {legal_dir}: {missing}")
        for f in sorted(q for q in legal_dir.iterdir() if q.is_file()):
            if f.name in entries:
                raise SystemExit(f"{name}: {f.name} is both a model file and a license document")
            entries[f.name] = f.read_bytes()
        asset = f"hotpotqa-{name.replace('_', '-')}-{VERSION}.zip"
        path = args.out / asset
        if path.exists():
            raise SystemExit(f"{path} exists")
        with zipfile.ZipFile(path, "x", zipfile.ZIP_DEFLATED) as z:
            for d in sorted(spec.get("dirs", [])):
                z.writestr(zipfile.ZipInfo(d + "/", (1980, 1, 1, 0, 0, 0)), b"")
            for rel in sorted(entries):
                info = zipfile.ZipInfo(rel, (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                z.writestr(info, entries[rel])
        data = path.read_bytes()
        manifest["models"][name] = {"asset": asset, "asset_sha256": sha(data), "asset_bytes": len(data), "target_dir": f"models/{name}",
                                    "role": spec["role"], "source_run": spec["src"], "empty_dirs": spec.get("dirs", []),
                                    "files": {rel: {"sha256": sha(b), "bytes": len(b)} for rel, b in sorted(entries.items())},
                                    "license": LICENSE, "license_file": "LICENSE", "provenance_file": "PROVENANCE.json"}
        print(asset, len(data), sha(data))
    manifest["asset_revision"] = ASSET_REVISION
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    lines = [f"{m['asset_sha256']}  {m['asset']}" for m in manifest["models"].values()]
    (args.out / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
