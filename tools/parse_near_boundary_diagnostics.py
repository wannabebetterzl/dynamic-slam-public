#!/usr/bin/env python3
"""Parse near-boundary map admission, culling, and pose-use diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


EVENT_RE = re.compile(r"(\w+)=([^\s]+)")


def parse_case(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("case must be NAME=/path/to/run_dir")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("case name cannot be empty")
    return name, Path(path)


def to_float(value: object, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: object, default: int = 0) -> int:
    return int(to_float(value, float(default)))


def read_eval(run_dir: Path) -> Dict[str, object]:
    path = run_dir / "eval_unified_all.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    by_alignment = {item.get("alignment"): item for item in data.get("results", [])}
    se3 = by_alignment.get("se3", {})
    sim3 = by_alignment.get("sim3", {})
    return {
        "matched": se3.get("matched_poses", ""),
        "coverage": se3.get("trajectory_coverage", ""),
        "ate_se3": se3.get("ate_rmse_m", ""),
        "ate_sim3": sim3.get("ate_rmse_m", ""),
        "scale": sim3.get("alignment_scale", ""),
        "rpet": se3.get("rpet_rmse_m", ""),
        "rper": se3.get("rper_rmse_deg", ""),
    }


def read_last_observability(run_dir: Path) -> Dict[str, object]:
    path = run_dir / "observability_frame_stats.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    last = rows[-1]
    return {
        "final_keyframes": last.get("num_keyframes", ""),
        "final_mappoints": last.get("num_mappoints", ""),
        "estimated_accum_path_m": last.get("estimated_accum_path_m", ""),
        "keyframe_created_count": sum(1 for row in rows if row.get("is_keyframe_created") == "1"),
    }


def add_value(
    totals: Dict[str, float],
    per_frame: Dict[int, Dict[str, float]],
    values: Dict[str, str],
    output_key: str,
    source_key: str,
) -> None:
    if source_key not in values:
        return
    amount = to_float(values[source_key])
    totals[output_key] += amount
    frame = values.get("frame")
    if frame not in (None, ""):
        per_frame[to_int(frame)][output_key] += amount


def add_weighted_mean(
    totals: Dict[str, float],
    per_frame: Dict[int, Dict[str, float]],
    values: Dict[str, str],
    mean_key: str,
    weight_key: str,
    output_key: str,
) -> None:
    if mean_key not in values or weight_key not in values:
        return
    weight = to_float(values[weight_key])
    if weight <= 0:
        return
    weighted = to_float(values[mean_key]) * weight
    totals[f"{output_key}_weighted_sum"] += weighted
    totals[f"{output_key}_weight"] += weight
    frame = values.get("frame")
    if frame not in (None, ""):
        row = per_frame[to_int(frame)]
        row[f"{output_key}_weighted_sum"] += weighted
        row[f"{output_key}_weight"] += weight


def finalize_weighted_means(row: Dict[str, object], keys: Iterable[str]) -> None:
    for key in keys:
        weight = to_float(row.pop(f"{key}_weight", 0))
        weighted_sum = to_float(row.pop(f"{key}_weighted_sum", 0))
        row[key] = weighted_sum / weight if weight > 0 else ""


def safe_ratio(num: object, den: object) -> object:
    denominator = to_float(den)
    if denominator <= 0:
        return ""
    return to_float(num) / denominator


ADMISSION_FIELDS = [
    "created_near_boundary_new_points",
    "created_clean_static_new_points",
    "created_direct_dynamic_new_points",
    "existing_near_boundary_candidates",
    "accepted_depth_candidates_pre_boundary",
    "triangulated_points",
    "created_map_points",
]

CULLING_FIELDS = [
    "recent_points",
    "remaining_recent_points",
    "recent_near_boundary",
    "recent_clean_static",
    "recent_direct_dynamic",
    "near_prebad",
    "near_culled_found_ratio",
    "near_culled_low_obs",
    "near_survived",
    "near_matured",
    "clean_culled_found_ratio",
    "clean_culled_low_obs",
    "clean_survived",
    "clean_matured",
]

POSE_COUNT_FIELDS = [
    "all_edges",
    "all_inliers",
    "all_outliers",
    "near_edges",
    "near_inliers",
    "near_outliers",
    "clean_edges",
    "clean_inliers",
    "clean_outliers",
    "direct_dynamic_edges",
    "direct_dynamic_inliers",
    "direct_dynamic_outliers",
    "admission_near_edges",
    "admission_near_inliers",
    "admission_near_outliers",
]

POSE_MEAN_SPECS = [
    ("near_chi2_mean", "near_edges", "pose_near_chi2_mean_weighted"),
    ("near_inlier_chi2_mean", "near_inliers", "pose_near_inlier_chi2_mean_weighted"),
    ("clean_chi2_mean", "clean_edges", "pose_clean_chi2_mean_weighted"),
    ("clean_inlier_chi2_mean", "clean_inliers", "pose_clean_inlier_chi2_mean_weighted"),
    ("admission_near_chi2_mean", "admission_near_edges", "pose_admission_near_chi2_mean_weighted"),
]

WEIGHTED_MEAN_KEYS = [spec[2] for spec in POSE_MEAN_SPECS]


def parse_stdout_events(run_dir: Path) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
    totals: Dict[str, float] = defaultdict(float)
    per_frame: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    path = run_dir / "stdout.log"
    if not path.exists():
        return totals, per_frame

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("[STSLAM_NEAR_BOUNDARY_"):
            continue
        tag = line.split("]", 1)[0].strip("[")
        values = dict(EVENT_RE.findall(line))
        stage = values.get("stage", "unknown")

        if tag == "STSLAM_NEAR_BOUNDARY_ADMISSION":
            prefix = "ckf" if stage == "create_new_keyframe" else "lm"
            for field in ADMISSION_FIELDS:
                add_value(totals, per_frame, values, f"{prefix}_{field}", field)
        elif tag == "STSLAM_NEAR_BOUNDARY_CULLING":
            for field in CULLING_FIELDS:
                add_value(totals, per_frame, values, f"culling_{field}", field)
        elif tag == "STSLAM_NEAR_BOUNDARY_POSE_USE":
            for field in POSE_COUNT_FIELDS:
                add_value(totals, per_frame, values, f"pose_{field}", field)
            for mean_key, weight_key, output_key in POSE_MEAN_SPECS:
                add_weighted_mean(totals, per_frame, values, mean_key, weight_key, output_key)

    return totals, per_frame


def derive_metrics(row: Dict[str, object]) -> None:
    row["ckf_created_near_share"] = safe_ratio(
        row.get("ckf_created_near_boundary_new_points", 0),
        to_float(row.get("ckf_created_near_boundary_new_points", 0))
        + to_float(row.get("ckf_created_clean_static_new_points", 0)),
    )
    row["lm_created_near_share"] = safe_ratio(
        row.get("lm_created_near_boundary_new_points", 0),
        to_float(row.get("lm_created_near_boundary_new_points", 0))
        + to_float(row.get("lm_created_clean_static_new_points", 0)),
    )
    row["pose_near_outlier_rate"] = safe_ratio(row.get("pose_near_outliers", 0), row.get("pose_near_edges", 0))
    row["pose_clean_outlier_rate"] = safe_ratio(row.get("pose_clean_outliers", 0), row.get("pose_clean_edges", 0))
    row["pose_direct_dynamic_outlier_rate"] = safe_ratio(
        row.get("pose_direct_dynamic_outliers", 0), row.get("pose_direct_dynamic_edges", 0)
    )
    row["pose_admission_near_outlier_rate"] = safe_ratio(
        row.get("pose_admission_near_outliers", 0), row.get("pose_admission_near_edges", 0)
    )
    row["pose_near_to_clean_chi2_ratio"] = safe_ratio(
        row.get("pose_near_chi2_mean_weighted", 0), row.get("pose_clean_chi2_mean_weighted", 0)
    )
    row["culling_near_removed_events"] = (
        to_float(row.get("culling_near_prebad", 0))
        + to_float(row.get("culling_near_culled_found_ratio", 0))
        + to_float(row.get("culling_near_culled_low_obs", 0))
    )
    row["culling_clean_removed_events"] = (
        to_float(row.get("culling_clean_culled_found_ratio", 0))
        + to_float(row.get("culling_clean_culled_low_obs", 0))
    )
    row["culling_near_removed_event_rate"] = safe_ratio(
        row.get("culling_near_removed_events", 0), row.get("culling_recent_near_boundary", 0)
    )
    row["culling_clean_removed_event_rate"] = safe_ratio(
        row.get("culling_clean_removed_events", 0), row.get("culling_recent_clean_static", 0)
    )


def collect_case(name: str, run_dir: Path) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    totals, per_frame_events = parse_stdout_events(run_dir)
    eval_metrics = read_eval(run_dir)
    observability = read_last_observability(run_dir)
    summary: Dict[str, object] = {
        "case": name,
        **eval_metrics,
        **observability,
        "run_dir": str(run_dir),
    }
    summary.update(dict(totals))
    finalize_weighted_means(summary, WEIGHTED_MEAN_KEYS)
    derive_metrics(summary)

    frame_rows: List[Dict[str, object]] = []
    for frame_id, values in sorted(per_frame_events.items()):
        row: Dict[str, object] = {"case": name, "frame_id": frame_id}
        row.update(dict(values))
        finalize_weighted_means(row, WEIGHTED_MEAN_KEYS)
        derive_metrics(row)
        frame_rows.append(row)

    return summary, frame_rows


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fields: List[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        fieldnames = fields
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=parse_case, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--per-frame-out", type=Path, required=True)
    args = parser.parse_args()

    summaries: List[Dict[str, object]] = []
    frame_rows: List[Dict[str, object]] = []
    for name, run_dir in args.case:
        summary, rows = collect_case(name, run_dir)
        summaries.append(summary)
        frame_rows.extend(rows)

    write_csv(args.summary_out, summaries)
    write_csv(args.per_frame_out, frame_rows)

    print(f"summary={args.summary_out}")
    print(f"per_frame={args.per_frame_out}")
    print("| case | ATE-SE3 | scale | CKF near/clean | LM near/clean | pose near/clean outlier | near/clean chi2 | cull near/clean removed |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summaries:
        def fmt(value: object) -> str:
            if value in ("", None):
                return ""
            try:
                return f"{float(value):.6f}"
            except (TypeError, ValueError):
                return str(value)

        print(
            f"| {row['case']} | {fmt(row.get('ate_se3'))} | {fmt(row.get('scale'))} | "
            f"{int(to_float(row.get('ckf_created_near_boundary_new_points', 0)))}/"
            f"{int(to_float(row.get('ckf_created_clean_static_new_points', 0)))} | "
            f"{int(to_float(row.get('lm_created_near_boundary_new_points', 0)))}/"
            f"{int(to_float(row.get('lm_created_clean_static_new_points', 0)))} | "
            f"{fmt(row.get('pose_near_outlier_rate'))}/{fmt(row.get('pose_clean_outlier_rate'))} | "
            f"{fmt(row.get('pose_near_chi2_mean_weighted'))}/{fmt(row.get('pose_clean_chi2_mean_weighted'))} | "
            f"{int(to_float(row.get('culling_near_removed_events', 0)))}/"
            f"{int(to_float(row.get('culling_clean_removed_events', 0)))} |"
        )


if __name__ == "__main__":
    main()
