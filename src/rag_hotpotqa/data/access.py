"""Single entry point for reading question splits, with purpose-based access control.

| purpose | allowed split   | use                                                     |
|---------|-----------------|---------------------------------------------------------|
| oer     | exploration     | O/E/R method exploration (no weight updates)            |
| train   | training        | fine-tuning, positive / negative construction           |
| select  | validation      | checkpoint / hyper-parameter selection (no gradients)   |
| final   | official_dev    | final evaluation, only after configs/final_frozen.yaml  |

Every load also checks the split's ID file against the SHA256 recorded in
data/manifests/splits_manifest.json, so a modified or re-drawn split is fatal.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..utils.config import Settings, load_config, load_yaml
from .loader import load_canonical_jsonl
from .splits import ids_hash

SPLIT_POLICY: dict[str, frozenset[str]] = {
    "oer": frozenset({"exploration"}),
    "train": frozenset({"training"}),
    "select": frozenset({"validation"}),
    "final": frozenset({"official_dev"}),
}
FINAL_FROZEN_CONFIG = "configs/final_frozen.yaml"


class SplitPolicyError(RuntimeError):
    """A component tried to read a split its purpose does not allow (fatal)."""


class SplitIntegrityError(RuntimeError):
    """Split IDs do not match the frozen manifest (fatal)."""


def check_split_allowed(split: str, purpose: str) -> None:
    allowed = SPLIT_POLICY.get(purpose)
    if allowed is None:
        raise SplitPolicyError(f"unknown purpose {purpose!r}")
    if split not in allowed:
        raise SplitPolicyError(f"purpose {purpose!r} may only read {sorted(allowed)}, got {split!r}")


def _check_final_frozen(root: Path) -> None:
    path = root / FINAL_FROZEN_CONFIG
    if not path.exists() or load_yaml(path).get("status") != "FROZEN":
        raise SplitPolicyError("official_dev is locked until configs/final_frozen.yaml has status: FROZEN")


def load_split_ids(settings: Settings, split: str, purpose: str) -> list[str]:
    check_split_allowed(split, purpose)
    root = settings.project_root
    manifest = json.loads((root / load_config("data", settings)["splits"]["manifest_path"]).read_text(encoding="utf-8"))
    if split == "official_dev":
        _check_final_frozen(root)
        dev = load_canonical_jsonl(root / load_config("data", settings)["canonical"]["dev_path"])
        ids = [r["_id"] for r in dev]
    else:
        ids = (settings.splits_dir / f"{split}_ids.txt").read_text(encoding="utf-8").split()
    expected = manifest["splits"][split]["ids_sha256"]
    if ids_hash(ids) != expected:
        raise SplitIntegrityError(f"{split} IDs hash {ids_hash(ids)} != manifest {expected}")
    return ids


def load_split_records(settings: Settings, split: str, purpose: str, ids: list[str] | None = None) -> list[dict[str, Any]]:
    """Records for ``split`` (optionally a subset ``ids`` of it), in ``ids`` order."""
    all_ids = load_split_ids(settings, split, purpose)
    wanted = all_ids if ids is None else ids
    extra = set(wanted) - set(all_ids)
    if extra:
        raise SplitPolicyError(f"{len(extra)} requested IDs are not in split {split!r}")
    dcfg = load_config("data", settings)
    path = dcfg["canonical"]["dev_path"] if split == "official_dev" else dcfg["canonical"]["train_path"]
    recs = {r["_id"]: r for r in load_canonical_jsonl(settings.project_root / path, set(wanted))}
    return [recs[i] for i in wanted]


# ---- Validation selection / reporting partition (Phase 3 design §13.6) -------------------------------------
# Fixed once: the first V_SELECT_N Validation IDs in sha256(id) order (the order `select_subset` uses) form
# V-select (all model / hyper-parameter selection); the rest form V-report (read once, after every choice is
# frozen, only with final_report=True; each such read is logged). The split files themselves are unchanged.
VALIDATION_PARTITION = "data/manifests/validation_partition.json"
V_SELECT_N = 2000
V_REPORT_ACCESS_LOG = "outputs/v_report_access.jsonl"


def partition_validation(ids: list[str], n_select: int = V_SELECT_N) -> dict[str, list[str]]:
    import hashlib

    order = sorted(ids, key=lambda i: hashlib.sha256(i.encode()).hexdigest())
    return {"v_select": order[:n_select], "v_report": order[n_select:]}


def check_part_access(part: str, final_report: bool) -> None:
    if part not in ("v_select", "v_report"):
        raise SplitPolicyError(f"unknown validation part {part!r}")
    if part == "v_report" and not final_report:
        raise SplitPolicyError("V-report is read only for the final, frozen evaluation (final_report=True)")


def build_validation_partition(settings: Settings) -> dict[str, Any]:
    """Write the partition manifest if absent; otherwise verify it. Returns the manifest."""
    from datetime import datetime, timezone

    root = settings.project_root
    parts = partition_validation(load_split_ids(settings, "validation", "select"))
    current = {k: {"n": len(v), "ids_sha256": ids_hash(v)} for k, v in parts.items()}
    path = root / VALIDATION_PARTITION
    if path.exists():
        man = json.loads(path.read_text(encoding="utf-8"))
        if man["parts"] != current:
            raise SplitIntegrityError(f"{path} does not match the Validation split")
        return man
    man = {"rule": f"Validation IDs sorted by sha256(id); first {V_SELECT_N} = v_select, rest = v_report",
           "created_utc": datetime.now(timezone.utc).isoformat(), "parts": current}
    path.write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8")
    return man


def load_validation_part(settings: Settings, part: str, final_report: bool = False, reason: str = "") -> list[str]:
    """IDs of V-select or V-report, verified against the frozen partition manifest."""
    check_part_access(part, final_report)
    root = settings.project_root
    path = root / VALIDATION_PARTITION
    if not path.exists():
        raise SplitIntegrityError(f"{path} missing; run build_validation_partition once")
    man = json.loads(path.read_text(encoding="utf-8"))
    ids = partition_validation(load_split_ids(settings, "validation", "select"))[part]
    if ids_hash(ids) != man["parts"][part]["ids_sha256"]:
        raise SplitIntegrityError(f"{part} IDs do not match {path}")
    if part == "v_report":
        from datetime import datetime, timezone

        log = root / V_REPORT_ACCESS_LOG
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "reason": reason}) + "\n")
    return ids
