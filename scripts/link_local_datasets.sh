#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGISTRY="${DSLAM_DATA_REGISTRY:-$ROOT/data/datasets.json}"
OUTPUT="${1:-$ROOT/data/local}"

python3 "$ROOT/scripts/dslam_data.py" --registry "$REGISTRY" link --output "$OUTPUT" --force
