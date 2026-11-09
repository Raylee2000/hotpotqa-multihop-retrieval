"""Centralised settings: .env for machine-specific paths/secrets, YAML for research config.

All filesystem writes in this project go through :func:`safe_path`, which refuses
paths outside PROJECT_ROOT.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_PACKAGE_ROOT = Path(__file__).resolve().parents[3]


class PathSafetyError(RuntimeError):
    """Raised when a write target resolves outside the project root."""


def _truthy(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_root: Path
    output_root: Path
    cache_root: Path
    model_cache_root: Path
    hf_home: Path
    hf_token: str | None
    random_seed: int
    device: str
    max_workers: int
    allow_remote_model_download: bool
    original_dev_json: Path | None

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"

    @property
    def splits_dir(self) -> Path:
        return self.data_root / "splits"

    @property
    def manifests_dir(self) -> Path:
        return self.data_root / "manifests"


def _resolve(root: Path, value: str | None, default: str) -> Path:
    raw = value.strip() if value and value.strip() else default
    p = Path(raw).expanduser()
    return (p if p.is_absolute() else root / p).resolve()


def load_settings(env_file: Path | None = None) -> Settings:
    env_path = env_file or (_PACKAGE_ROOT / ".env")
    if env_path.exists():
        load_dotenv(env_path, override=False)
    root_env = os.getenv("PROJECT_ROOT", "").strip()
    root = Path(root_env).expanduser().resolve() if root_env else _PACKAGE_ROOT
    original_dev = os.getenv("ORIGINAL_DEV_JSON", "").strip()
    settings = Settings(
        project_root=root,
        data_root=_resolve(root, os.getenv("DATA_ROOT"), "data"),
        output_root=_resolve(root, os.getenv("OUTPUT_ROOT"), "outputs"),
        cache_root=_resolve(root, os.getenv("CACHE_ROOT"), ".cache"),
        model_cache_root=_resolve(root, os.getenv("MODEL_CACHE_ROOT"), ".cache/models"),
        hf_home=_resolve(root, os.getenv("HF_HOME"), ".cache/huggingface"),
        hf_token=os.getenv("HF_TOKEN") or None,
        random_seed=int(os.getenv("RANDOM_SEED", "42")),
        device=os.getenv("DEVICE", "auto"),
        max_workers=int(os.getenv("MAX_WORKERS", "2")),
        allow_remote_model_download=_truthy(os.getenv("ALLOW_REMOTE_MODEL_DOWNLOAD", "false")),
        original_dev_json=_resolve(root, original_dev, "") if original_dev else None,
    )
    # Keep Hugging Face caches inside the project.
    os.environ["HF_HOME"] = str(settings.hf_home)
    if not settings.allow_remote_model_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return settings


def safe_path(path: Path | str, root: Path) -> Path:
    """Resolve ``path`` and assert it is inside ``root`` (the project root)."""
    resolved = Path(path).resolve()
    root_resolved = Path(root).resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise PathSafetyError(f"Refusing to write outside project root: {resolved}") from exc
    return resolved


def load_yaml(path: Path | str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(name: str, settings: Settings | None = None) -> dict[str, Any]:
    s = settings or load_settings()
    return load_yaml(s.project_root / "configs" / f"{name}.yaml")
