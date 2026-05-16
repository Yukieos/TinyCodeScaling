#!/usr/bin/env bash
set -euo pipefail

python -m tinycodescaling run --config configs/experiments/v01_humaneval_qwen15b.yaml "$@"

