#!/usr/bin/env bash
# Create .venv with Python 3.11 and install the locked dependencies (CPU PyTorch wheel index).
# Usage: bash scripts/setup_env.sh   (set PYTHON to a Python 3.11 executable if `python3.11` is not on PATH)
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3.11}
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version' || { echo "Python 3.11 required" >&2; exit 1; }
[ -e .venv ] && { echo ".venv exists; remove it or activate it" >&2; exit 1; }
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade "pip<26"
.venv/bin/python -m pip install --require-virtualenv -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
echo "done: source .venv/bin/activate && python run.py check --code-only"
