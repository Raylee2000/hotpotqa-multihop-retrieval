"""Evaluation datasets for the final evaluation entry points (scripts/final_eval/), with an explicit authorisation gate.

  v_select    Validation V-select 2K — model-selection data, always readable (purpose "select")
  v_report    Validation V-report 8K — read ONLY with an authorisation file (see below); every read is logged by
              access.load_validation_part(final_report=True)
  bc_holdout  qtype_balanced_holdout_v1 (official Train questions never allocated to any split; B/C balanced) — read ONLY
              with an authorisation file (purpose "holdout_eval")

Authorisation file (JSON), created by a human after the method is frozen:
  {"status": "FROZEN", "approved_parts": ["v_report", ...], "approved_by": "...", "approved_utc": "...",
   "freeze_manifest": "<path>", "freeze_manifest_sha256": "<sha256 of that file>"}
Without it, v_report / bc_holdout raise SplitPolicyError before any ID is read.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..utils.config import Settings, load_config
from ..utils.io import sha256_file
from .access import SplitPolicyError, load_split_ids, load_split_records, load_validation_part
from .holdout import HOLDOUT_PURPOSE, load_holdout_ids
from .loader import load_canonical_jsonl

PARTS = ("v_select", "v_report", "bc_holdout", "official_dev_distractor")
HOLDOUT_NAME = "qtype_balanced_holdout_v1"


def check_authorisation(settings: Settings, part: str, authorisation: str | None) -> dict | None:
    if part == "v_select":
        return None
    if part not in PARTS:
        raise SplitPolicyError(f"unknown evaluation part {part!r}")
    if not authorisation:
        raise SplitPolicyError(f"{part} requires an authorisation file (status FROZEN, approved_parts including {part!r})")
    path = Path(authorisation)
    path = path if path.is_absolute() else settings.project_root / path
    if not path.exists():
        raise SplitPolicyError(f"authorisation file {path} not found")
    auth = json.loads(path.read_text(encoding="utf-8"))
    if auth.get("status") != "FROZEN" or part not in auth.get("approved_parts", []):
        raise SplitPolicyError(f"authorisation {path} does not approve {part!r} (status {auth.get('status')!r})")
    fm = settings.project_root / auth.get("freeze_manifest", "")
    if not fm.is_file() or sha256_file(fm) != auth.get("freeze_manifest_sha256"):
        raise SplitPolicyError("freeze manifest missing or its sha256 differs from the authorisation")
    return auth


def load_eval_part(settings: Settings, part: str, authorisation: str | None = None) -> tuple[list[str], list[dict[str, Any]]]:
    """(ids, records) in the canonical order of the part; records are raw HotpotQA dicts (same format for all parts)."""
    check_authorisation(settings, part, authorisation)
    if part == "v_select":
        ids = load_validation_part(settings, "v_select")
        return ids, load_split_records(settings, "validation", "select", ids)
    if part == "v_report":
        ids = load_validation_part(settings, "v_report", final_report=True, reason=f"final evaluation; authorisation {authorisation}")
        return ids, load_split_records(settings, "validation", "select", ids)
    if part == "official_dev_distractor":
        ids = load_split_ids(settings, "official_dev", "final")
        return ids, load_split_records(settings, "official_dev", "final", ids)
    ids = load_holdout_ids(settings, HOLDOUT_NAME, HOLDOUT_PURPOSE)
    train = settings.project_root / load_config("data", settings)["canonical"]["train_path"]
    recs = {r["_id"]: r for r in load_canonical_jsonl(train, set(ids))}
    return ids, [recs[i] for i in ids]


def select_eval_subset(ids, recs, ids_file=None):
    """A fixed shard must be a unique, ordered subsequence of its approved part."""
    if ids_file is None:
        return ids, recs
    wanted = Path(ids_file).read_text(encoding="utf-8").split()
    if not wanted or len(wanted) != len(set(wanted)):
        raise SplitPolicyError("empty or duplicate shard IDs")
    selected = set(wanted)
    if [i for i in ids if i in selected] != wanted:
        raise SplitPolicyError("shard IDs not in canonical partition order or outside approved partition")
    by_id = dict(zip(ids,recs))
    return wanted, [by_id[i] for i in wanted]
