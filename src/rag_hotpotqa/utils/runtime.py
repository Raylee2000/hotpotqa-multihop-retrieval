"""Seeding, logging, device detection and resource measurement."""
from __future__ import annotations

import logging
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import psutil


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def get_logger(name: str = "rag_hotpotqa", log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def resolve_device(requested: str = "auto") -> str:
    """Return 'cuda', 'xpu' or 'cpu'. Never assumes CUDA exists; falls back to CPU."""
    try:
        import torch
    except ImportError:
        return "cpu"
    req = (requested or "auto").lower()
    available = {"cpu": True, "cuda": torch.cuda.is_available()}
    available["xpu"] = bool(getattr(torch, "xpu", None) and torch.xpu.is_available())
    if req == "auto":
        for dev in ("cuda", "xpu"):
            if available[dev]:
                return dev
        return "cpu"
    if not available.get(req, False):
        logging.getLogger("rag_hotpotqa").warning("Device %s unavailable; falling back to cpu", req)
        return "cpu"
    return req


def peak_rss_mb() -> float:
    """Peak resident memory of this process (Windows: peak working set)."""
    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)
    if peak is None:
        try:
            import resource

            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        except ImportError:  # pragma: no cover
            peak = info.rss
    return peak / (1024 * 1024)


def current_rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


@contextmanager
def timed() -> Iterator[dict[str, float]]:
    out: dict[str, float] = {}
    t0 = time.perf_counter()
    try:
        yield out
    finally:
        out["seconds"] = time.perf_counter() - t0


def check_resources(path: Path, min_free_gb: float, min_free_ram_gb: float) -> dict[str, Any]:
    """Fail fast before long jobs when disk or RAM is insufficient."""
    free_disk = shutil.disk_usage(path).free / 1024**3
    free_ram = psutil.virtual_memory().available / 1024**3
    if free_disk < min_free_gb:
        raise RuntimeError(f"Insufficient disk: {free_disk:.1f} GB free < {min_free_gb} GB required")
    if free_ram < min_free_ram_gb:
        raise RuntimeError(f"Insufficient RAM: {free_ram:.1f} GB free < {min_free_ram_gb} GB required")
    return {"free_disk_gb": round(free_disk, 2), "free_ram_gb": round(free_ram, 2)}


def git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True).stdout
        return out.stdout.strip() + ("-dirty" if dirty.strip() else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "logical_cpus": psutil.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 2),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["xpu_available"] = bool(getattr(torch, "xpu", None) and torch.xpu.is_available())
        info["torch_threads"] = torch.get_num_threads()
    except ImportError:
        pass
    for pkg in ("sentence_transformers", "transformers", "numpy", "pyarrow"):
        try:
            mod = __import__(pkg)
            info[pkg] = getattr(mod, "__version__", "?")
        except ImportError:
            pass
    return info
