#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGISTRY="${DSLAM_DATA_REGISTRY:-$ROOT/data/datasets.json}"
DATASET_ID="${1:-frontend_raw_wxyz}"
OUT_DIR="${2:-$ROOT/runs/frontend_${DATASET_ID}_$(date +%Y%m%d_%H%M%S)}"
MAX_FRAMES="${3:-30}"
CONFIG="${4:-$ROOT/frontend/basic_model_based_SLAM/config/world_sam_pipeline_foundation_panoptic_person_v2_milddilate_local.json}"
FILTER_MODE="${5:-filtered}"

SEQUENCE_ROOT="$(
  python3 "$ROOT/scripts/dslam_data.py" --registry "$REGISTRY" get "$DATASET_ID" sequence_root
)"

mkdir -p "$OUT_DIR"

export PYTHONPATH="$ROOT/frontend/basic_model_based_SLAM/scripts:${PYTHONPATH:-}"

python3 "$ROOT/frontend/basic_model_based_SLAM/scripts/run_rgbd_slam_benchmark.py" \
  --sequence-root "$SEQUENCE_ROOT" \
  --output-dir "$OUT_DIR" \
  --filter-mode "$FILTER_MODE" \
  --config "$CONFIG" \
  --max-frames "$MAX_FRAMES"

echo "Frontend output: $OUT_DIR"
