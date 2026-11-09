"""Single entry point. Run from anywhere; all paths are relative to this repository.

  python run.py check                       verify code/config manifest, installed models and data (no model run)
  python run.py test                        unit tests (no data, models or network)
  python run.py data --source hf            download the pinned HotpotQA parquet, canonicalise, verify all splits
  python run.py data --source local --parquet-dir DIR
  python run.py models --from-dir DIR       install released weights from downloaded Release assets (+ base model)
  python run.py models --base-url URL       download released weights from the published Release URL
  python run.py smoke                       first 128-question shard of Official Dev, then compare with formal
  python run.py baselines [--part P ...]    the six formal baselines only
  python run.py evaluate [--part P ...]     baselines + final pipeline (R5_full, R0); default part: Official Dev
  python run.py compare --run RUN           compare a run with the formal results
  python run.py train --profile full        retrain every Full-method model from the base model + fixed splits
  python run.py train --profile smoke       tiny-subset training smoke test (data, training, checkpoints, export, loading)
  python run.py evaluate --models-manifest outputs/trained_models/<profile>/models.json   evaluate retrained models
Long runs (CPU, research hardware): Official Dev ~4.5 h, V-report ~5.5 h, B/C ~12.6 h.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = [sys.executable, "-X", "utf8"]


def call(args: list[str]) -> int:
    return subprocess.run([*PY, *args], cwd=ROOT).returncode


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "check":
        return call(["scripts/check_release.py", *rest])
    if cmd == "test":
        return call(["-m", "pytest", "-q", "-p", "no:cacheprovider", *rest])
    if cmd == "data":
        return call(["scripts/prepare_data.py", *rest])
    if cmd == "models":
        return call(["scripts/fetch_models.py", *rest])
    if cmd == "smoke":
        name = "smoke_official_dev_shard0"
        rc = call(["scripts/run_eval.py", "--part", "official_dev_distractor", "--shards", "1", "--run-name", name, *rest])
        return rc or call(["scripts/compare_results.py", "--run", name])
    if cmd in ("baselines", "evaluate"):
        if "--part" not in rest:
            rest = ["--part", "official_dev_distractor", *rest]
        return call(["scripts/run_eval.py", "--stages", "baselines" if cmd == "baselines" else "all", *rest])
    if cmd == "compare":
        return call(["scripts/compare_results.py", *rest])
    if cmd == "train":
        return call(["scripts/train_full.py", *rest])
    print(f"unknown command {cmd!r}\n{__doc__}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
