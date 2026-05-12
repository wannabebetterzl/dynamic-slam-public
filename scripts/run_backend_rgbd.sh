#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGISTRY="${DSLAM_DATA_REGISTRY:-$ROOT/data/datasets.json}"
DATASET_ID="${1:-backend_maskonly_smoke30_wxyz}"
PROFILE="${2:-semantic_only}"
OUT_DIR="${3:-$ROOT/runs/${DATASET_ID}_${PROFILE}_$(date +%Y%m%d_%H%M%S)}"

get_dataset() {
  python3 "$ROOT/scripts/dslam_data.py" --registry "$REGISTRY" get "$DATASET_ID" "$1"
}

get_tool() {
  python3 "$ROOT/scripts/dslam_data.py" --registry "$REGISTRY" tool "$1"
}

SEQUENCE_ROOT="$(get_dataset sequence_root)"
ASSOCIATIONS="$(get_dataset associations)"
MASK_ROOT="$(get_dataset mask_root)"
GROUND_TRUTH="$(get_dataset ground_truth)"
RGBD_TUM="${DSLAM_RGBD_TUM:-$(get_tool rgbd_tum)}"
ORB_VOCAB="${DSLAM_ORB_VOCAB:-$(get_tool orb_vocab)}"
ORB_CONFIG="${DSLAM_ORB_CONFIG:-$(get_tool rgbd_config)}"

mkdir -p "$OUT_DIR"

export STSLAM_USE_VIEWER="${STSLAM_USE_VIEWER:-0}"
export STSLAM_DISABLE_FRAME_SLEEP="${STSLAM_DISABLE_FRAME_SLEEP:-1}"
export ORB_SLAM3_MASK_MODE="${ORB_SLAM3_MASK_MODE:-postfilter}"
export STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES="${STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES:-1}"
export STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURE_STAGES="${STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURE_STAGES:-*}"
export STSLAM_RGBD_DYNAMIC_FRONTEND_SPLIT="${STSLAM_RGBD_DYNAMIC_FRONTEND_SPLIT:-0}"
export STSLAM_SEMANTIC_GEOMETRIC_VERIFICATION="${STSLAM_SEMANTIC_GEOMETRIC_VERIFICATION:-0}"
export STSLAM_SEMANTIC_CANDIDATE_GEOMETRY_GATE="${STSLAM_SEMANTIC_CANDIDATE_GEOMETRY_GATE:-0}"
export STSLAM_SEMANTIC_CANDIDATE_SPARSE_FLOW_GATE="${STSLAM_SEMANTIC_CANDIDATE_SPARSE_FLOW_GATE:-0}"
export STSLAM_GEOMETRIC_DYNAMIC_REJECTION="${STSLAM_GEOMETRIC_DYNAMIC_REJECTION:-0}"
export STSLAM_SEMANTIC_CONSERVATIVE_DYNAMIC_DELETE="${STSLAM_SEMANTIC_CONSERVATIVE_DYNAMIC_DELETE:-0}"
export STSLAM_SEMANTIC_STRICT_STATIC_KEEP="${STSLAM_SEMANTIC_STRICT_STATIC_KEEP:-0}"

case "$PROFILE" in
  semantic_only)
    ;;
  geom_dynamic_reject)
    export STSLAM_GEOMETRIC_DYNAMIC_REJECTION="${STSLAM_GEOMETRIC_DYNAMIC_REJECTION:-1}"
    export STSLAM_GEOMETRIC_DYNAMIC_REJECTION_STAGES="${STSLAM_GEOMETRIC_DYNAMIC_REJECTION_STAGES:-track_local_map_pre_pose}"
    export STSLAM_GEOMETRIC_DYNAMIC_REJECTION_MIN_STATIC_MAP_OBSERVATIONS="${STSLAM_GEOMETRIC_DYNAMIC_REJECTION_MIN_STATIC_MAP_OBSERVATIONS:-1}"
    export STSLAM_GEOMETRIC_DYNAMIC_REJECTION_MAX_REPROJ_ERROR_PX="${STSLAM_GEOMETRIC_DYNAMIC_REJECTION_MAX_REPROJ_ERROR_PX:-5.0}"
    export STSLAM_GEOMETRIC_DYNAMIC_REJECTION_MAX_DEPTH_ERROR_M="${STSLAM_GEOMETRIC_DYNAMIC_REJECTION_MAX_DEPTH_ERROR_M:-0.10}"
    ;;
  candidate_sparseflow_tracklocal)
    export STSLAM_SEMANTIC_GEOMETRIC_VERIFICATION="${STSLAM_SEMANTIC_GEOMETRIC_VERIFICATION:-1}"
    export STSLAM_SEMANTIC_GEOMETRIC_RESCUE_STAGES="${STSLAM_SEMANTIC_GEOMETRIC_RESCUE_STAGES:-*}"
    export STSLAM_SEMANTIC_CANDIDATE_GEOMETRY_GATE="${STSLAM_SEMANTIC_CANDIDATE_GEOMETRY_GATE:-1}"
    export STSLAM_SEMANTIC_CANDIDATE_SPARSE_FLOW_GATE="${STSLAM_SEMANTIC_CANDIDATE_SPARSE_FLOW_GATE:-1}"
    export STSLAM_SEMANTIC_SPARSE_FLOW_GATE_STAGES="${STSLAM_SEMANTIC_SPARSE_FLOW_GATE_STAGES:-track_local_map_pre_pose}"
    export STSLAM_SEMANTIC_FLOW_MAX_DYNAMIC_REJECT_RATIO="${STSLAM_SEMANTIC_FLOW_MAX_DYNAMIC_REJECT_RATIO:-0.15}"
    ;;
  strict_static_keep_sparseflow)
    export STSLAM_SEMANTIC_GEOMETRIC_VERIFICATION="${STSLAM_SEMANTIC_GEOMETRIC_VERIFICATION:-1}"
    export STSLAM_SEMANTIC_GEOMETRIC_RESCUE_STAGES="${STSLAM_SEMANTIC_GEOMETRIC_RESCUE_STAGES:-*}"
    export STSLAM_SEMANTIC_CANDIDATE_GEOMETRY_GATE="${STSLAM_SEMANTIC_CANDIDATE_GEOMETRY_GATE:-1}"
    export STSLAM_SEMANTIC_CANDIDATE_SPARSE_FLOW_GATE="${STSLAM_SEMANTIC_CANDIDATE_SPARSE_FLOW_GATE:-1}"
    export STSLAM_SEMANTIC_SPARSE_FLOW_GATE_STAGES="${STSLAM_SEMANTIC_SPARSE_FLOW_GATE_STAGES:-track_local_map_pre_pose}"
    export STSLAM_SEMANTIC_STRICT_STATIC_KEEP="${STSLAM_SEMANTIC_STRICT_STATIC_KEEP:-1}"
    ;;
  *)
    echo "Unknown profile: $PROFILE" >&2
    echo "Known profiles: semantic_only, geom_dynamic_reject, candidate_sparseflow_tracklocal, strict_static_keep_sparseflow" >&2
    exit 2
    ;;
esac

{
  echo "dataset_id=$DATASET_ID"
  echo "profile=$PROFILE"
  echo "sequence_root=$SEQUENCE_ROOT"
  echo "associations=$ASSOCIATIONS"
  echo "mask_root=$MASK_ROOT"
  echo "ground_truth=$GROUND_TRUTH"
  echo "rgbd_tum=$RGBD_TUM"
  echo "orb_vocab=$ORB_VOCAB"
  echo "orb_config=$ORB_CONFIG"
  env | sort | grep -E '^(ORB_SLAM3|STSLAM)_' || true
} > "$OUT_DIR/run_manifest.txt"

set +e
(
  cd "$OUT_DIR"
  "$RGBD_TUM" "$ORB_VOCAB" "$ORB_CONFIG" "$SEQUENCE_ROOT" "$ASSOCIATIONS" "$MASK_ROOT"
) > "$OUT_DIR/stdout.log" 2> "$OUT_DIR/stderr.log"
RC=$?
set -e
echo "$RC" > "$OUT_DIR/returncode.txt"

TRAJECTORY="$OUT_DIR/CameraTrajectory.txt"
if [[ -s "$TRAJECTORY" && -f "$GROUND_TRUTH" ]]; then
  python3 "$ROOT/tools/evaluate_trajectory_ate.py" \
    --ground-truth "$GROUND_TRUTH" \
    --estimated "$TRAJECTORY" \
    --alignment all \
    --json-out "$OUT_DIR/eval_unified_all.json" \
    --text-out "$OUT_DIR/eval_unified_all.txt"
else
  echo "No trajectory evaluation was produced. Check $OUT_DIR/stdout.log and $OUT_DIR/stderr.log." >&2
fi

echo "Run directory: $OUT_DIR"
exit "$RC"
