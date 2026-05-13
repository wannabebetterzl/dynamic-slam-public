#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ID="${DATASET_ID:-backend_maskonly_full_wxyz}"
REPEATS="${REPEATS:-3}"
OUT_ROOT="${1:-$ROOT/runs/full_repeat_ablation_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUT_ROOT"

run_case() {
  local case_name="$1"
  local profile="$2"
  shift 2

  for repeat in $(seq 1 "$REPEATS"); do
    local out_dir="$OUT_ROOT/$case_name/r${repeat}"
    echo "[repeat-matrix] case=$case_name repeat=$repeat/$REPEATS out=$out_dir"
    env "$@" bash "$ROOT/scripts/run_backend_rgbd.sh" \
      "$DATASET_ID" "$profile" "$out_dir"
  done
}

COMMON_ENV=(
  ORB_SLAM3_MASK_MODE=postfilter
  STSLAM_RGBD_DYNAMIC_FRONTEND_SPLIT=0
  STSLAM_SEMANTIC_CANDIDATE_GEOMETRY_GATE=0
  STSLAM_SEMANTIC_CANDIDATE_SPARSE_FLOW_GATE=0
  STSLAM_SEMANTIC_CONSERVATIVE_DYNAMIC_DELETE=0
  STSLAM_SEMANTIC_GEOMETRIC_VERIFICATION=0
  STSLAM_SEMANTIC_STRICT_STATIC_KEEP=0
)

run_case noop_metadata_only semantic_only \
  "${COMMON_ENV[@]}" \
  STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=0 \
  STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURE_STAGES=none \
  STSLAM_GEOMETRIC_DYNAMIC_REJECTION=0

run_case geom_riskonly_cap010 geom_dynamic_reject \
  "${COMMON_ENV[@]}" \
  STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=0 \
  STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURE_STAGES=none \
  STSLAM_GEOMETRIC_DYNAMIC_REJECTION=1 \
  STSLAM_GEOMETRIC_DYNAMIC_REJECTION_ACTION=risk_only \
  STSLAM_GEOMETRIC_DYNAMIC_REJECTION_MAX_REJECT_RATIO=0.10 \
  STSLAM_GEOMETRIC_DYNAMIC_REJECTION_PROTECT_MIN_INLIERS=45

run_case geom_hard_cap010_protect45 geom_dynamic_reject \
  "${COMMON_ENV[@]}" \
  STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=0 \
  STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURE_STAGES=none \
  STSLAM_GEOMETRIC_DYNAMIC_REJECTION=1 \
  STSLAM_GEOMETRIC_DYNAMIC_REJECTION_ACTION=hard_delete \
  STSLAM_GEOMETRIC_DYNAMIC_REJECTION_MAX_REJECT_RATIO=0.10 \
  STSLAM_GEOMETRIC_DYNAMIC_REJECTION_PROTECT_MIN_INLIERS=45

run_case geom_soft_cap010_w025 geom_dynamic_reject \
  "${COMMON_ENV[@]}" \
  STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=0 \
  STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURE_STAGES=none \
  STSLAM_GEOMETRIC_DYNAMIC_REJECTION=1 \
  STSLAM_GEOMETRIC_DYNAMIC_REJECTION_ACTION=soft_weight \
  STSLAM_GEOMETRIC_DYNAMIC_REJECTION_SOFT_WEIGHT=0.25 \
  STSLAM_GEOMETRIC_DYNAMIC_REJECTION_MAX_REJECT_RATIO=0.10 \
  STSLAM_GEOMETRIC_DYNAMIC_REJECTION_PROTECT_MIN_INLIERS=45

run_case geom_soft_cap005_w050 geom_dynamic_reject \
  "${COMMON_ENV[@]}" \
  STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=0 \
  STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURE_STAGES=none \
  STSLAM_GEOMETRIC_DYNAMIC_REJECTION=1 \
  STSLAM_GEOMETRIC_DYNAMIC_REJECTION_ACTION=soft_weight \
  STSLAM_GEOMETRIC_DYNAMIC_REJECTION_SOFT_WEIGHT=0.50 \
  STSLAM_GEOMETRIC_DYNAMIC_REJECTION_MAX_REJECT_RATIO=0.05 \
  STSLAM_GEOMETRIC_DYNAMIC_REJECTION_PROTECT_MIN_INLIERS=45

python3 - "$OUT_ROOT" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])


def load_metrics(run_dir):
    path = run_dir / "eval_unified_all.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    by_alignment = {item["alignment"]: item for item in data.get("results", [])}
    se3 = by_alignment.get("se3")
    sim3 = by_alignment.get("sim3")
    if not se3 or not sim3:
        return None
    return {
        "run": str(run_dir),
        "matched": se3["matched_poses"],
        "ate_se3": se3["ate_rmse_m"],
        "rpet_se3": se3["rpet_rmse_m"],
        "rper_se3": se3["rper_rmse_deg"],
        "ate_sim3": sim3["ate_rmse_m"],
        "sim3_scale": sim3["alignment_scale"],
    }


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"median": None, "mean": None, "min": None, "max": None}
    return {
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
    }


summary_rows = []
raw_rows = []
for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    metrics = []
    for run_dir in sorted(p for p in case_dir.iterdir() if p.is_dir()):
        row = load_metrics(run_dir)
        if row:
            row["case"] = case_dir.name
            metrics.append(row)
            raw_rows.append(row)
    if not metrics:
        continue
    summary = {"case": case_dir.name, "n": len(metrics)}
    for key in ["matched", "ate_se3", "ate_sim3", "sim3_scale", "rpet_se3", "rper_se3"]:
        for stat_name, value in stats([m[key] for m in metrics]).items():
            summary[f"{key}_{stat_name}"] = value
    summary_rows.append(summary)

raw_path = root / "summary_raw.csv"
summary_path = root / "summary_stats.csv"
with raw_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=[
        "case", "run", "matched", "ate_se3", "ate_sim3",
        "sim3_scale", "rpet_se3", "rper_se3",
    ])
    writer.writeheader()
    writer.writerows(raw_rows)

with summary_path.open("w", newline="", encoding="utf-8") as handle:
    fieldnames = ["case", "n"] + [
        f"{key}_{stat}"
        for key in ["matched", "ate_se3", "ate_sim3", "sim3_scale", "rpet_se3", "rper_se3"]
        for stat in ["median", "mean", "min", "max"]
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"summary_raw={raw_path}")
print(f"summary_stats={summary_path}")
print("| case | n | ATE-SE3 median | ATE-SE3 range | ATE-Sim3 median | scale median | RPEt median | RPER median |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
for row in summary_rows:
    print(
        f"| {row['case']} | {row['n']} | "
        f"{row['ate_se3_median']:.6f} | "
        f"{row['ate_se3_min']:.6f}-{row['ate_se3_max']:.6f} | "
        f"{row['ate_sim3_median']:.6f} | "
        f"{row['sim3_scale_median']:.6f} | "
        f"{row['rpet_se3_median']:.6f} | "
        f"{row['rper_se3_median']:.6f} |"
    )
PY

echo "Repeat matrix finished: $OUT_ROOT"
