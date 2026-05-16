#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <summary.json>" >&2
  exit 1
fi

python -m tinycodescaling report --summary "$1"

