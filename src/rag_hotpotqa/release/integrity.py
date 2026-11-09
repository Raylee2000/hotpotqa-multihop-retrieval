"""Integrity checks for the public release: file hashes, the release manifest, model files and fixed data.

Every check raises ``IntegrityError``; callers stop on it. Nothing here relaxes a check to make a run succeed.
All paths are relative to the repository root (``ROOT``).
"""
from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
RELEASE_MANIFEST = "configs/release_manifest.json"
MODELS_MANIFEST = "configs/models.json"
# Optional: a repository-relative model manifest written by scripts/train_full.py for retrained weights under outputs/
# (same schema and file sets as configs/models.json). Unset = the released weights in models/.
MODELS_MANIFEST_ENV = "RAG_HOTPOTQA_MODELS_MANIFEST"
DATA_LOCK = "data/manifests/release_data_lock.json"
TEXT_SUFFIXES = {".py", ".md", ".json", ".yaml", ".yml", ".txt", ".sh", ".ps1", ".toml", ".cfg", ".example", ""}


class IntegrityError(RuntimeError):
    pass


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def text_sha256(path: Path) -> str:
    """Hash of a text file with CRLF normalised to LF, so a Windows checkout verifies like a Linux one."""
    return sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


def file_sha256(path: Path) -> str:
    """Release-manifest hash: line-ending-normalised for text files, raw bytes otherwise (e.g. .gz)."""
    return text_sha256(path) if path.suffix in TEXT_SUFFIXES else sha256_path(path)


def read_json(rel: str) -> Any:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def verify_release_manifest() -> dict:
    """Code/config files listed in configs/release_manifest.json must match (line endings normalised)."""
    man = read_json(RELEASE_MANIFEST)
    bad = []
    for rel, h in man["files"].items():
        p = ROOT / rel
        if not p.is_file():
            bad.append(f"missing {rel}")
        elif file_sha256(p) != h:
            bad.append(f"changed {rel}")
    if bad:
        raise IntegrityError("release manifest check failed: " + "; ".join(bad[:20]) + (" …" if len(bad) > 20 else ""))
    return man


def models_manifest_path() -> str:
    """Repository-relative path of the active model manifest (configs/models.json unless overridden)."""
    rel = os.environ.get(MODELS_MANIFEST_ENV, "").strip()
    if not rel:
        return MODELS_MANIFEST
    p = (ROOT / rel).resolve()
    root = ROOT.resolve()
    if root not in p.parents or not p.is_file():
        raise IntegrityError(f"{MODELS_MANIFEST_ENV}={rel!r}: not a manifest file inside the repository")
    return p.relative_to(root).as_posix()


def _inside_root(rel: str) -> Path:
    p = (ROOT / rel).resolve()
    if ROOT.resolve() not in p.parents:
        raise IntegrityError(f"model directory {rel!r} is outside the repository")
    return p


def model_dir(name: str) -> Path:
    """Directory of model `name` according to the active manifest (default: models/<name>)."""
    return ROOT / _inside_root(read_json(models_manifest_path())["models"][name]["target_dir"]).relative_to(ROOT.resolve())


def model_rel(name: str) -> str:
    return model_dir(name).relative_to(ROOT).as_posix()


def verify_model_dir(name: str, spec: dict) -> None:
    target = _inside_root(spec["target_dir"])
    if not target.is_dir():
        raise IntegrityError(f"model {name} missing: {spec['target_dir']} (run: python run.py models ...)")
    expected = set(spec["files"])
    present = {p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()}
    if present != expected:
        raise IntegrityError(f"model {name}: unexpected file set (extra {sorted(present - expected)}, missing {sorted(expected - present)})")
    for d in spec.get("empty_dirs", []):
        if not (target / d).is_dir():
            raise IntegrityError(f"model {name}: directory {d} missing")
    for rel, e in spec["files"].items():
        if sha256_path(target / rel) != e["sha256"]:
            raise IntegrityError(f"model {name}: {rel} sha256 mismatch")


def verify_models(names: list[str] | None = None) -> dict:
    man = read_json(models_manifest_path())
    if set(man["models"]) != set(read_json(MODELS_MANIFEST)["models"]):
        raise IntegrityError(f"{models_manifest_path()}: model names differ from {MODELS_MANIFEST}")
    for name, spec in man["models"].items():
        if names is None or name in names:
            verify_model_dir(name, spec)
    return man


def safe_extract(zip_path: Path, spec: dict, dest: Path) -> None:
    """Extract a model zip into an empty directory: only the listed relative files and directories are allowed."""
    allowed = set(spec["files"]) | {d + "/" for d in spec.get("empty_dirs", [])}
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        if set(names) != allowed or len(names) != len(set(names)):
            raise IntegrityError(f"{zip_path.name}: archive entries differ from the manifest")
        for n in names:
            p = PurePosixPath(n)
            if p.is_absolute() or ".." in p.parts or "\\" in n or ":" in n:
                raise IntegrityError(f"{zip_path.name}: unsafe entry {n!r}")
        dest.mkdir(parents=True, exist_ok=False)
        for info in z.infolist():
            out = dest / info.filename
            if info.is_dir():
                out.mkdir(parents=True, exist_ok=True)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            data = z.read(info)
            if sha256_bytes(data) != spec["files"][info.filename]["sha256"]:
                raise IntegrityError(f"{zip_path.name}: {info.filename} sha256 mismatch")
            out.write_bytes(data)


def verify_base_model(settings) -> dict:
    """The pinned MiniLM in the project-local Hugging Face cache (weights SHA256 from configs/retrieval.yaml)."""
    from ..retrieval.factory import check_hf_cache_inside_project, verify_dense_weights
    from ..utils.config import load_config

    d = load_config("retrieval", settings)["dense"]
    check_hf_cache_inside_project(settings, True)
    try:
        return verify_dense_weights(d["model"], d["revision"], d["weights_sha256"])
    except Exception as e:  # noqa: BLE001 - re-raise as an integrity failure with the cause
        raise IntegrityError(f"base model {d['model']}@{d['revision']} not available or not verified: {e}") from e


def verify_data(full: bool = True) -> dict:
    """Canonical data and split ID files must match the data lock byte for byte."""
    lock = read_json(DATA_LOCK)
    files = dict(lock["split_files"])
    if full:
        files.update(lock["canonical"])
    for rel, e in files.items():
        p = ROOT / rel
        if not p.is_file():
            raise IntegrityError(f"data file missing: {rel} (run: python run.py data ...)")
        if p.stat().st_size != e["bytes"] or sha256_path(p) != e["sha256"]:
            raise IntegrityError(f"data file {rel} differs from the locked version")
    return lock
