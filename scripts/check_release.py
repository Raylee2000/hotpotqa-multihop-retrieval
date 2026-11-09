"""Pre-run check: code/config manifest, model files, base model and data lock. Runs no model.

Each item reports PASS / MISSING / FAIL; the exit code is non-zero unless everything required is PASS.
Usage: python scripts/check_release.py [--code-only]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_hotpotqa.release.integrity import (IntegrityError, read_json, verify_base_model, verify_data,  # noqa: E402
                                            verify_model_dir, verify_release_manifest)


def item(label: str, fn) -> bool:
    try:
        fn()
        print(f"PASS     {label}")
        return True
    except IntegrityError as e:
        print(f"{'MISSING' if 'missing' in str(e) else 'FAIL':8s} {label}: {e}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-only", action="store_true", help="only the release manifest (no data / models needed)")
    args = ap.parse_args()
    ok = item("code and configuration (configs/release_manifest.json)", verify_release_manifest)
    if not args.code_only:
        for name, spec in read_json("configs/models.json")["models"].items():
            ok &= item(f"model {name}", lambda n=name, s=spec: verify_model_dir(n, s))

        def base():
            from rag_hotpotqa.utils.config import load_settings

            verify_base_model(load_settings())

        ok &= item("base model sentence-transformers/all-MiniLM-L6-v2 (pinned revision)", base)
        ok &= item("data (canonical JSONL and split ID files)", lambda: verify_data(full=True))
    print("READY" if ok else "NOT READY")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
