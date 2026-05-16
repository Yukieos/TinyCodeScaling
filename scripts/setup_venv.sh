#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${1:-python3.11}"

"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements.txt
