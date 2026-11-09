"""Obtain and verify the released model weights and the pinned base model.

Released weights (configs/models.json): each asset is a zip whose SHA256 is checked before extraction; extraction
only accepts the listed files, and every extracted file is checked again. Sources:
  --from-dir DIR      assets already downloaded into DIR (e.g. from the GitHub Release page)
  --base-url URL      download "<URL>/<asset name>" over HTTPS (the release maintainer publishes the URL;
                      this repository ships no URL because none exists before publication)
Base model (configs/base_model.json, not redistributed):
  --base-model hf                 download the pinned revision from Hugging Face into .cache/huggingface
  --base-model-cache DIR          copy models--sentence-transformers--all-MiniLM-L6-v2 from an existing HF hub cache
  --base-model-only               only the base model (what `python run.py train` needs)
Only safe formats are installed: safetensors, JSON, tokenizer text files and numpy arrays without pickle.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_hotpotqa.release.integrity import (IntegrityError, read_json, safe_extract, sha256_path,  # noqa: E402
                                            verify_model_dir)

ASSETS = ROOT / "release_assets"
BASE_REPO_DIR = "models--sentence-transformers--all-MiniLM-L6-v2"


def fetch_asset(name: str, spec: dict, from_dir: Path | None, base_url: str | None) -> Path:
    if from_dir is not None:
        path = from_dir / spec["asset"]
        if not path.is_file():
            raise IntegrityError(f"{spec['asset']} not found in {from_dir}")
    else:
        if not base_url.startswith("https://"):
            raise IntegrityError("--base-url must be an https:// URL")
        ASSETS.mkdir(exist_ok=True)
        path = ASSETS / spec["asset"]
        if not path.exists():
            url = base_url.rstrip("/") + "/" + spec["asset"]
            print(f"download {url}")
            tmp = path.with_suffix(".partial")
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:  # noqa: S310 - https enforced above
                shutil.copyfileobj(r, f)
            tmp.replace(path)
    if path.stat().st_size != spec["asset_bytes"] or sha256_path(path) != spec["asset_sha256"]:
        raise IntegrityError(f"{spec['asset']}: SHA256 differs from configs/models.json")
    return path


def install_models(from_dir: Path | None, base_url: str | None) -> None:
    man = read_json("configs/models.json")
    for name, spec in man["models"].items():
        target = ROOT / spec["target_dir"]
        if target.exists():
            verify_model_dir(name, spec)
            print(f"ok      {name} (already installed, verified)")
            continue
        path = fetch_asset(name, spec, from_dir, base_url)
        tmp = Path(tempfile.mkdtemp(dir=target.parent if target.parent.exists() else ROOT, prefix=".extract_"))
        try:
            safe_extract(path, spec, tmp / "m")
            (tmp / "m").replace(target)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        verify_model_dir(name, spec)
        print(f"ok      {name} -> {spec['target_dir']}")


def install_base_model(mode: str, cache_dir: Path | None) -> None:
    cfg = read_json("configs/base_model.json")
    hub = ROOT / ".cache/huggingface/hub"
    snap = hub / BASE_REPO_DIR / "snapshots" / cfg["revision"]
    if not snap.exists():
        if mode == "hf":
            os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
            from huggingface_hub import snapshot_download

            snapshot_download(cfg["repo_id"], revision=cfg["revision"], allow_patterns=sorted(cfg["files"]),
                              cache_dir=str(hub), token=False)
        else:
            src = cache_dir / BASE_REPO_DIR
            if not (src / "snapshots" / cfg["revision"]).is_dir():
                raise IntegrityError(f"{src} does not contain revision {cfg['revision']}")
            hub.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copytree(src, hub / BASE_REPO_DIR, dirs_exist_ok=False)
            except (shutil.Error, OSError) as e:
                shutil.rmtree(hub / BASE_REPO_DIR, ignore_errors=True)
                raise IntegrityError(f"copying the base model failed ({type(e).__name__}); on Windows, a repository "
                                     "path that is too long (260-character limit) causes this: use a shorter path "
                                     "or enable Windows long paths") from None
    for rel, e in cfg["files"].items():
        p = snap / rel
        if not p.is_file() or sha256_path(p) != e["sha256"]:
            raise IntegrityError(f"base model file {rel} missing or SHA256 mismatch")
    print(f"ok      base model {cfg['repo_id']}@{cfg['revision'][:12]} ({len(cfg['files'])} files verified)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--from-dir", type=Path)
    src.add_argument("--base-url")
    ap.add_argument("--base-model", choices=["hf", "cache", "skip"], default="hf")
    ap.add_argument("--base-model-cache", type=Path, help="existing Hugging Face hub cache directory (with --base-model cache)")
    ap.add_argument("--base-model-only", action="store_true",
                    help="install only the pinned base model (enough for python run.py train; no released weights)")
    args = ap.parse_args()
    if args.base_model_only:
        if args.from_dir is not None or args.base_url is not None or args.base_model == "skip":
            ap.error("--base-model-only takes no --from-dir / --base-url and needs --base-model hf|cache")
        if args.base_model == "cache" and not args.base_model_cache:
            ap.error("--base-model cache requires --base-model-cache")
        try:
            install_base_model(args.base_model, args.base_model_cache)
        except IntegrityError as e:
            print(f"STOP: {e}", file=sys.stderr)
            return 2
        print("PASS base model installed and verified")
        return 0
    if args.from_dir is None and args.base_url is None:
        ap.error("give --from-dir (downloaded Release assets) or --base-url (published Release download URL)")
    if args.base_model == "cache" and not args.base_model_cache:
        ap.error("--base-model cache requires --base-model-cache")
    try:
        (ROOT / "models").mkdir(exist_ok=True)
        install_models(args.from_dir, args.base_url)
        if args.base_model != "skip":
            install_base_model(args.base_model, args.base_model_cache)
    except IntegrityError as e:
        print(f"STOP: {e}", file=sys.stderr)
        return 2
    print("PASS models installed and verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
