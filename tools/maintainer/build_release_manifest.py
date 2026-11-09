"""MAINTAINER-ONLY: write configs/release_manifest.json (hashes of the code, configuration, data manifests and
formal reference results that the public evaluation depends on). Run after every intended change, then review
the diff. The public runner refuses to evaluate when any listed file differs.
Usage: python tools/maintainer/build_release_manifest.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rag_hotpotqa.release.integrity import RELEASE_MANIFEST, file_sha256  # noqa: E402

INCLUDE = ["run.py", "pyproject.toml", "requirements-lock.txt", "scripts/**/*.py", "src/**/*.py", "configs/*",
           "data/manifests/*", "results/reference/*"]


def main() -> int:
    files = sorted({p for pat in INCLUDE for p in ROOT.glob(pat) if p.is_file() and "__pycache__" not in p.parts})
    files = [p for p in files if p.relative_to(ROOT).as_posix() != RELEASE_MANIFEST]
    man = {"description": "SHA256 of every file the public evaluation depends on (text files hashed with LF line endings). "
                          "Replaces the research repository's 285-file freeze as the run gate; see README.md §08.",
           "files": {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in files}}
    (ROOT / RELEASE_MANIFEST).write_text(json.dumps(man, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
