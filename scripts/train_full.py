"""One-click training of every model of the Full method, in dependency order (configs/training_full.yaml).

  python run.py train --profile full      formal configuration (CPU, many hours; see the notes below and README.md §07)
  python run.py train --profile smoke     tiny subsets: data -> training -> checkpoints -> selection -> export -> load
  python run.py train --profile smoke --dry-run      print the resolved commands only

Prerequisites: python run.py data ... (fixed splits) and the pinned base model
(python run.py models --base-model-only --base-model hf). Released weights are NOT needed and never used.
Steps (each a research training script run as a subprocess, offline, 8 threads):
  general_encoder -> qtype -> qtype_artifact -> sf_fold0 -> sf_fold1 -> csh_mine -> csh -> candidates (V-select,
  Training) -> r5 -> export (outputs/trained_models/<profile>/models + models.json) -> load_check.
Resume: re-run the same command. Finished steps (marker present) are skipped; interrupted training resumes from its
checkpoint (each script refuses a changed configuration). Logs: outputs/trained_models/<profile>/logs/<step>.log
Use the result: python run.py evaluate --models-manifest outputs/trained_models/<profile>/models.json [...]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_hotpotqa.utils.config import load_yaml  # noqa: E402

CONFIG = "configs/training_full.yaml"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_plan(profile: str) -> tuple[dict, list[dict]]:
    cfg = load_yaml(ROOT / CONFIG)
    if profile not in cfg["profiles"]:
        raise SystemExit(f"unknown profile {profile!r} (configs/training_full.yaml: {sorted(cfg['profiles'])})")
    names = [s["name"] for s in cfg["steps"]]
    seen = set()
    for s in cfg["steps"]:
        missing = [n for n in s["needs"] if n not in seen]
        if missing:
            raise SystemExit(f"{CONFIG}: step {s['name']} needs {missing}, which do not run before it")
        seen.add(s["name"])
    extra = cfg["profiles"][profile].get("extra_args") or {}
    unknown = sorted(set(extra) - set(names))
    if unknown:
        raise SystemExit(f"{CONFIG}: extra_args for unknown steps {unknown}")
    return cfg, cfg["steps"]


def best_snapshot(result: Path) -> str:
    if not result.is_file():
        raise SystemExit(f"{result.relative_to(ROOT).as_posix()} missing (the upstream training has not finished)")
    snap = json.loads(result.read_text(encoding="utf-8"))["best"]["snapshot"]
    if not (ROOT / snap / "model.safetensors").is_file():
        raise SystemExit(f"selected snapshot {snap} (from {result.relative_to(ROOT)}) is missing")
    return snap


def resolve(text: str, profile: str, runs: dict) -> str:
    def rep(m):
        key = m.group(1)
        if key == "profile":
            return profile
        if key.startswith("runs."):
            return str(runs[key[5:]])
        if key == "best.general":
            return best_snapshot(ROOT / "outputs/training_v1" / runs["general"] / "result.json")
        if key == "best.csh":
            return best_snapshot(ROOT / "outputs/csh_v1" / runs["csh"] / "result.json")
        raise SystemExit(f"unknown placeholder {{{key}}}")
    return re.sub(r"\{([a-z0-9_.]+)\}", rep, text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True, help="full | smoke (configs/training_full.yaml)")
    ap.add_argument("--dry-run", action="store_true", help="print the resolved steps; run nothing")
    ap.add_argument("--until", default=None, help="stop after this step (e.g. general_encoder)")
    args = ap.parse_args()
    cfg, steps = load_plan(args.profile)
    if args.until and args.until not in [s["name"] for s in steps]:
        raise SystemExit(f"--until: unknown step {args.until!r}")
    runs = cfg["profiles"][args.profile]["runs"]
    extra = cfg["profiles"][args.profile].get("extra_args") or {}
    env = dict(os.environ, **{k: str(v) for k, v in cfg["env"].items()})
    env.pop("RAG_HOTPOTQA_MODELS_MANIFEST", None)   # training never reads released / other model manifests
    base = ROOT / "outputs" / "trained_models" / args.profile
    logs = base / "logs"
    if not args.dry_run:
        from rag_hotpotqa.release.integrity import IntegrityError, verify_base_model, verify_data
        from rag_hotpotqa.utils.config import load_settings

        try:
            verify_data(full=True)
            verify_base_model(load_settings())
        except IntegrityError as e:
            print(f"STOP prerequisites: {e}\n  (python run.py data ...; python run.py models --base-model-only --base-model hf)",
                  file=sys.stderr)
            return 2
        logs.mkdir(parents=True, exist_ok=True)
    history = base / "train_full_history.jsonl"
    for s in steps:
        name = s["name"]
        marker = ROOT / resolve(s["marker"], args.profile, runs)
        if marker.exists():
            print(f"{now()} SKIP {name} ({marker.relative_to(ROOT).as_posix()} exists)", flush=True)
        elif args.dry_run:
            try:
                cmd = [resolve(c, args.profile, runs) for c in s["cmd"]] + list(extra.get(name, []))
            except SystemExit:  # {best.*} is known only after the upstream training ran
                cmd = [c for c in s["cmd"]] + list(extra.get(name, []))
            print(f"[dry-run] {name}: python {' '.join(cmd)}")
        else:
            cmd = [resolve(c, args.profile, runs) for c in s["cmd"]] + list(extra.get(name, []))
            log = logs / f"{name}.log"
            print(f"{now()} START {name}: python {' '.join(cmd)}  (log {log.relative_to(ROOT).as_posix()})", flush=True)
            t = time.perf_counter()
            with open(log, "ab") as fh:
                fh.write(f"\n===== {now()} {' '.join(cmd)}\n".encode())
                fh.flush()
                rc = subprocess.run([sys.executable, "-X", "utf8", *cmd], cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
            sec = round(time.perf_counter() - t, 1)
            with open(history, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"utc": now(), "step": name, "cmd": cmd, "exit_code": rc, "seconds": sec}) + "\n")
            if rc:
                print(f"{now()} FAILED {name} (exit {rc}, {sec} s); see {log.relative_to(ROOT).as_posix()}. "
                      "Re-run the same command to resume.", file=sys.stderr)
                return 3
            if name == "csh_mine":   # the mining script has no completion file of its own
                marker.write_text(json.dumps({"utc": now(), "cmd": cmd}) + "\n", encoding="utf-8")
            if not marker.exists():
                print(f"{now()} FAILED {name}: exit 0 but marker {marker.relative_to(ROOT).as_posix()} missing", file=sys.stderr)
                return 3
            print(f"{now()} OK {name} ({sec} s)", flush=True)
        if args.until == name:
            print(f"stopped after {name} (--until)")
            return 0
    if not args.dry_run:
        print(f"PASS training profile {args.profile}: outputs/trained_models/{args.profile}/models.json\n"
              f"  evaluate: python run.py evaluate --models-manifest outputs/trained_models/{args.profile}/models.json [--part P] [--shards N]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
