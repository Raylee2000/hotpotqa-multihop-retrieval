"""Read-only git provenance for training logs (works without git or outside a git repository)."""
from __future__ import annotations

import subprocess
from pathlib import Path


def git_dirty(root: Path) -> list[str]:
    """`git status --porcelain` lines, or [] when git is missing or root is not a repository (read-only)."""
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
    except (FileNotFoundError, OSError):
        return []
    return [x for x in out.stdout.splitlines() if x.strip()] if out.returncode == 0 else []
