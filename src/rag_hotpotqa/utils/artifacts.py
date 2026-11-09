"""Model Artifact Contract.

A training run is COMPLETE only when (1) every fitted component is saved, (2) artifact_manifest.json lists every
file with its sha256, the feature schema and provenance, and (3) an independent process reloaded the artifact,
re-predicted a fixed verification set and matched the in-memory predictions — recorded in COMPLETE.json.
result.json alone never counts as completion.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .io import sha256_file

MANIFEST, COMPLETE = "artifact_manifest.json", "COMPLETE.json"


class ArtifactError(RuntimeError):
    """Missing, corrupted or inconsistent model artifact (fatal)."""


def write_manifest(art_dir: Path, files: dict[str, Path], **info: Any) -> dict:
    """files: logical name -> path inside art_dir. Extra info (model class, params, feature schema, data, provenance)."""
    entries = {}
    for name, p in files.items():
        p = Path(p)
        if not p.exists():
            raise ArtifactError(f"cannot write manifest: {name} ({p}) does not exist")
        entries[name] = {"path": p.relative_to(art_dir).as_posix(), "sha256": sha256_file(p), "bytes": p.stat().st_size}
    man = {"contract": "model-artifact-v1", "files": entries, "written_utc": datetime.now(timezone.utc).isoformat(), **info}
    (art_dir / MANIFEST).write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    return man


def verify_manifest(art_dir: Path) -> dict:
    """Manifest present, every listed file present with the recorded sha256; returns the manifest."""
    mp = Path(art_dir) / MANIFEST
    if not mp.exists():
        raise ArtifactError(f"{mp} missing")
    man = json.loads(mp.read_text(encoding="utf-8"))
    for name, e in man["files"].items():
        p = Path(art_dir) / e["path"]
        if not p.exists():
            raise ArtifactError(f"artifact file {name} missing: {p}")
        if sha256_file(p) != e["sha256"]:
            raise ArtifactError(f"artifact file {name} corrupted: sha256 mismatch ({p})")
    return man


def check_feature_schema(expected: Sequence[str], actual: Sequence[str]) -> None:
    if list(expected) != list(actual):
        raise ArtifactError(f"feature order mismatch: expected {list(expected)[:5]}..., got {list(actual)[:5]}...")


def compare_predictions(reference: np.ndarray, reloaded: np.ndarray, atol: float) -> float:
    reference, reloaded = np.asarray(reference, float), np.asarray(reloaded, float)
    if reference.shape != reloaded.shape:
        raise ArtifactError(f"reload prediction shape {reloaded.shape} != {reference.shape}")
    diff = float(np.max(np.abs(reference - reloaded))) if reference.size else 0.0
    if not diff <= atol:
        raise ArtifactError(f"reload predictions differ: max |diff| {diff:.3g} > {atol}")
    return diff


def mark_complete(art_dir: Path, reload_check: dict) -> None:
    if reload_check.get("status") != "PASS":
        raise ArtifactError("COMPLETE requires reload_check status PASS")
    man = verify_manifest(art_dir)
    (Path(art_dir) / COMPLETE).write_text(json.dumps({"status": "COMPLETE", "manifest_sha256": sha256_file(Path(art_dir) / MANIFEST),
                                                      "reload_check": reload_check, "n_files": len(man["files"]),
                                                      "utc": datetime.now(timezone.utc).isoformat()}, indent=1), encoding="utf-8")


def is_complete(art_dir: Path) -> bool:
    """True only if COMPLETE.json exists, matches the current manifest, and every file still verifies."""
    cp = Path(art_dir) / COMPLETE
    if not cp.exists():
        return False
    try:
        verify_manifest(art_dir)
    except ArtifactError:
        return False
    c = json.loads(cp.read_text(encoding="utf-8"))
    return c.get("status") == "COMPLETE" and c.get("manifest_sha256") == sha256_file(Path(art_dir) / MANIFEST) \
        and c.get("reload_check", {}).get("status") == "PASS"
