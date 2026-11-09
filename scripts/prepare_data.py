"""Prepare the fixed HotpotQA Distractor data and verify every split. Never re-samples and never edits a manifest.

Steps (each verified; any mismatch stops):
  1. obtain the three parquet files of hotpotqa/hotpot_qa@<pinned revision> (Hugging Face, or a local copy);
  2. SHA256 of each parquet file must equal data/manifests/release_data_lock.json;
  3. canonical JSONL (original HotpotQA schema) is written with the research code
     (rag_hotpotqa.data.loader) and must be byte-identical to the locked SHA256 — this holds for the Dev file too,
     whose locked version was produced from the official hotpot_dev_distractor_v1.json;
  4. split ID files are written from the full ID lists stored in data/manifests/splits_manifest.json and must match
     the locked file SHA256 and ids_sha256;
  5. V-select / V-report (sha256-ordered Validation), B/C holdout and Official Dev IDs are recomputed and checked,
     together with type counts and pairwise disjointness.
Usage:
  python scripts/prepare_data.py --source hf
  python scripts/prepare_data.py --source local --parquet-dir <dir containing the three .parquet files>
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_hotpotqa.release.integrity import DATA_LOCK, IntegrityError, read_json, sha256_path  # noqa: E402

RAW = ROOT / "data/raw/hf_hotpot_qa"


def obtain(lock: dict, source: str, parquet_dir: Path | None) -> None:
    for rel, e in lock["hf"]["files"].items():
        dst = RAW / rel
        if dst.is_file() and sha256_path(dst) == e["sha256"]:
            print(f"ok      {rel} (already present)")
            continue
        if dst.exists():
            raise IntegrityError(f"{dst.relative_to(ROOT)} exists but differs from the lock; remove it manually")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if source == "hf":
            from huggingface_hub import hf_hub_download

            # Public dataset: never send a stored token.
            hf_hub_download(lock["hf"]["repo_id"], rel, repo_type="dataset", revision=lock["hf"]["revision"],
                            local_dir=str(RAW), token=False)
        else:
            cands = [parquet_dir / rel, parquet_dir / Path(rel).name]
            src = next((c for c in cands if c.is_file()), None)
            if src is None:
                raise IntegrityError(f"{Path(rel).name} not found under {parquet_dir}")
            shutil.copyfile(src, dst)
        if dst.stat().st_size != e["bytes"] or sha256_path(dst) != e["sha256"]:
            raise IntegrityError(f"{rel}: SHA256 differs from the pinned Hugging Face revision")
        print(f"ok      {rel}")


def canonicalize(lock: dict) -> None:
    from rag_hotpotqa.data.loader import iter_parquet, write_canonical_jsonl

    jobs = {"data/raw/canonical/dev_distractor.jsonl": ["distractor/validation-00000-of-00001.parquet"],
            "data/raw/canonical/train.jsonl": ["distractor/train-00000-of-00002.parquet", "distractor/train-00001-of-00002.parquet"]}
    for rel, parts in jobs.items():
        e, p = lock["canonical"][rel], ROOT / rel
        if not (p.is_file() and sha256_path(p) == e["sha256"]):
            if p.exists():
                raise IntegrityError(f"{rel} exists but differs from the lock; remove it manually")
            write_canonical_jsonl(iter_parquet([RAW / x for x in parts]), p, ROOT)
        if p.stat().st_size != e["bytes"] or sha256_path(p) != e["sha256"]:
            raise IntegrityError(f"{rel}: canonical file differs from the locked research version")
        print(f"ok      {rel}")


def write_splits(lock: dict) -> dict:
    from rag_hotpotqa.data.splits import ids_hash
    from rag_hotpotqa.utils.io import atomic_write_text

    man = read_json("data/manifests/splits_manifest.json")
    for rel, e in lock["split_files"].items():
        name = Path(rel).name.removesuffix("_ids.txt")
        ids = man["split_ids"][name]
        if len(ids) != e["n"] or ids_hash(ids) != e["ids_sha256"] or man["splits"][name]["ids_sha256"] != e["ids_sha256"]:
            raise IntegrityError(f"{name}: manifest IDs differ from the lock")
        p = ROOT / rel
        if not p.exists():
            atomic_write_text(p, "\n".join(ids) + "\n", ROOT)
        if sha256_path(p) != e["sha256"]:
            raise IntegrityError(f"{rel} differs from the locked split file")
        print(f"ok      {rel} ({len(ids)} IDs)")
    return man


def check_parts(lock: dict, man: dict) -> dict:
    from rag_hotpotqa.data.access import partition_validation
    from rag_hotpotqa.data.splits import ids_hash

    def stream(rel):  # (id, type) per record, without keeping the records in memory
        with open(ROOT / rel, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                yield r["_id"], r["type"]

    parts = {k: list(v) for k, v in man["split_ids"].items()}
    parts.update(partition_validation(parts["validation"]))
    parts["bc_holdout"] = read_json("data/manifests/qtype_balanced_holdout_v1.json")["ids"]
    dev = list(stream("data/raw/canonical/dev_distractor.jsonl"))
    parts["official_dev_distractor"] = [i for i, _ in dev]
    types = dict(dev)
    train_ids = set()
    for i, t in stream("data/raw/canonical/train.jsonl"):
        types[i] = t
        train_ids.add(i)
    if set(parts["official_dev_distractor"]) & train_ids:
        raise IntegrityError("Official Dev IDs occur in Train")
    report = {}
    for name, e in lock["parts"].items():
        ids = parts[name]
        if len(ids) != e["n"] or len(set(ids)) != len(ids) or ids_hash(ids) != e["ids_sha256"]:
            raise IntegrityError(f"{name}: IDs differ from the lock")
        missing = [i for i in ids if i not in types]
        if missing:
            raise IntegrityError(f"{name}: {len(missing)} IDs not found in the canonical data")
        report[name] = {"n": len(ids), "ids_sha256": e["ids_sha256"], "types": dict(Counter(types[i] for i in ids))}
        print(f"ok      {name}: {len(ids)} questions {report[name]['types']}")
    disjoint = ["exploration", "training", "validation", "bc_holdout", "official_dev_distractor"]
    for i, a in enumerate(disjoint):
        for b in disjoint[i + 1:]:
            if set(parts[a]) & set(parts[b]):
                raise IntegrityError(f"{a} and {b} overlap")
    if not set(parts["v_select"]) | set(parts["v_report"]) == set(parts["validation"]) or set(parts["v_select"]) & set(parts["v_report"]):
        raise IntegrityError("V-select / V-report do not partition Validation")
    print("ok      all evaluation parts are disjoint; V-select and V-report partition Validation")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["hf", "local"], required=True)
    ap.add_argument("--parquet-dir", type=Path)
    args = ap.parse_args()
    if args.source == "local" and not args.parquet_dir:
        ap.error("--parquet-dir is required with --source local")
    try:
        lock = read_json(DATA_LOCK)
        obtain(lock, args.source, args.parquet_dir)
        canonicalize(lock)
        man = write_splits(lock)
        report = check_parts(lock, man)
    except IntegrityError as e:
        print(f"STOP: {e}", file=sys.stderr)
        return 2
    out = ROOT / "outputs/data_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"status": "PASS", "parts": report}, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("PASS data preparation and verification")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
