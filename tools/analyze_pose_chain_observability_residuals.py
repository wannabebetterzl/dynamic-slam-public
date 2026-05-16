#!/usr/bin/env python3
"""Join pose-chain observability logs with aligned trajectory residuals."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from evaluate_trajectory_ate import align_positions, load_tum_trajectory


def parse_range(value: str) -> Tuple[str, int, int]:
    if "-" not in value:
        raise argparse.ArgumentTypeError("range must be START-END")
    start_s, end_s = value.split("-", 1)
    start = int(start_s)
    end = int(end_s)
    if start < 0 or end < start:
        raise argparse.ArgumentTypeError("range must satisfy 0 <= START <= END")
    return f"{start}-{end}", start, end


def associate_indices(gt: Dict[str, np.ndarray], est: Dict[str, np.ndarray], max_diff: float) -> Tuple[np.ndarray, np.ndarray]:
    gt_i = 0
    est_i = 0
    gt_indices: List[int] = []
    est_indices: List[int] = []
    while gt_i < len(gt["times"]) and est_i < len(est["times"]):
        diff = float(est["times"][est_i] - gt["times"][gt_i])
        if abs(diff) <= max_diff:
            gt_indices.append(gt_i)
            est_indices.append(est_i)
            gt_i += 1
            est_i += 1
        elif diff > 0:
            gt_i += 1
        else:
            est_i += 1
    if not gt_indices:
        raise RuntimeError("No matched trajectory pairs were found.")
    return np.asarray(gt_indices, dtype=np.int64), np.asarray(est_indices, dtype=np.int64)


def to_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(str(value))
    except (TypeError, ValueError):
        return default


def to_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def read_observability(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            converted: Dict[str, object] = dict(row)
            converted["frame_id"] = to_int(row.get("frame_id"), -1)
            converted["timestamp"] = to_float(row.get("timestamp"), math.nan)
            rows.append(converted)
    rows.sort(key=lambda item: to_float(item.get("timestamp"), math.inf))
    return rows


def nearest_observability_row(rows: Sequence[Dict[str, object]], timestamp: float, max_diff: float) -> Dict[str, object] | None:
    if not rows:
        return None
    times = np.asarray([to_float(row.get("timestamp"), math.nan) for row in rows], dtype=np.float64)
    idx = int(np.nanargmin(np.abs(times - timestamp)))
    if not math.isfinite(float(times[idx])) or abs(float(times[idx]) - timestamp) > max_diff:
        return None
    return rows[idx]


def path_length(xyz: np.ndarray) -> float:
    if len(xyz) < 2:
        return float("nan")
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())


def rmse(values: Sequence[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(math.sqrt(np.mean(arr * arr)))


def mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def fmt(value: object) -> object:
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.9f}"
    return value


def build_frame_rows(
    gt_path: Path,
    trajectory_path: Path,
    observability_path: Path,
    max_diff: float,
) -> List[Dict[str, object]]:
    gt = load_tum_trajectory(gt_path)
    est = load_tum_trajectory(trajectory_path)
    gt_idx, est_idx = associate_indices(gt, est, max_diff)
    gt_xyz = gt["xyz"][gt_idx]
    est_xyz = est["xyz"][est_idx]
    est_times = est["times"][est_idx]
    se3_aligned, _, _, _, _ = align_positions(gt_xyz, est_xyz, "se3")
    sim3_aligned, _, _, sim3_scale, _ = align_positions(gt_xyz, est_xyz, "sim3")
    se3_error = np.linalg.norm(gt_xyz - se3_aligned, axis=1)
    sim3_error = np.linalg.norm(gt_xyz - sim3_aligned, axis=1)
    observability_rows = read_observability(observability_path)

    frame_rows: List[Dict[str, object]] = []
    prev_est: np.ndarray | None = None
    prev_gt: np.ndarray | None = None
    for match_index, timestamp in enumerate(est_times):
        obs = nearest_observability_row(observability_rows, float(timestamp), max_diff)
        if obs is None:
            obs = {}
        est_step = float(np.linalg.norm(est_xyz[match_index] - prev_est)) if prev_est is not None else 0.0
        gt_step = float(np.linalg.norm(gt_xyz[match_index] - prev_gt)) if prev_gt is not None else 0.0
        frame_rows.append(
            {
                "match_index": match_index,
                "frame_id": to_int(obs.get("frame_id"), match_index),
                "timestamp": float(timestamp),
                "se3_error_m": float(se3_error[match_index]),
                "sim3_error_m": float(sim3_error[match_index]),
                "global_sim3_scale": float(sim3_scale),
                "est_step_m": est_step,
                "gt_step_m": gt_step,
                "est_gt_step_ratio": est_step / gt_step if gt_step > 1e-12 else 0.0,
                "is_keyframe_created": to_int(obs.get("is_keyframe_created"), 0),
                "tracked_map_points": to_int(obs.get("tracked_map_points"), 0),
                "tracked_static_map_points": to_int(obs.get("tracked_static_map_points"), 0),
                "inlier_map_matches_after_pose": to_int(obs.get("inlier_map_matches_after_pose"), 0),
                "local_map_matches_before_pose": to_int(obs.get("local_map_matches_before_pose"), 0),
                "v14_static_inlier_count": to_int(obs.get("v14_static_inlier_count"), 0),
                "v14_static_inlier_grid_coverage": to_float(obs.get("v14_static_inlier_grid_coverage"), 0.0),
                "v14_boundary_inlier_frac": to_float(obs.get("v14_boundary_inlier_frac"), 0.0),
                "v14_step_ratio_proxy": to_float(obs.get("v14_step_ratio_proxy"), 0.0),
                "v14_recent_keyframe_rate": to_float(obs.get("v14_recent_keyframe_rate"), 0.0),
                "v14_recent_keyframes_per_m": to_float(obs.get("v14_recent_keyframes_per_m"), 0.0),
                "v14_support_low": to_int(obs.get("v14_support_low"), 0),
                "v14_motion_pressure": to_int(obs.get("v14_motion_pressure"), 0),
                "v14_keyframe_pressure": to_int(obs.get("v14_keyframe_pressure"), 0),
                "v14_boundary_pressure": to_int(obs.get("v14_boundary_pressure"), 0),
                "v14_pose_chain_risk": to_int(obs.get("v14_pose_chain_risk"), 0),
            }
        )
        prev_est = est_xyz[match_index]
        prev_gt = gt_xyz[match_index]
    return frame_rows


def segment_for_frame(frame: int, ranges: Sequence[Tuple[str, int, int]]) -> str:
    for label, start, end in ranges:
        if start <= frame <= end:
            return label
    return ""


def summarize_segment(
    label: str,
    rows: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    se3 = [to_float(row.get("se3_error_m")) for row in rows]
    sim3 = [to_float(row.get("sim3_error_m")) for row in rows]
    est_path = sum(to_float(row.get("est_step_m")) for row in rows)
    gt_path = sum(to_float(row.get("gt_step_m")) for row in rows)
    keyframes = sum(to_int(row.get("is_keyframe_created")) for row in rows)
    support_low = sum(to_int(row.get("v14_support_low")) for row in rows)
    motion_pressure = sum(to_int(row.get("v14_motion_pressure")) for row in rows)
    keyframe_pressure = sum(to_int(row.get("v14_keyframe_pressure")) for row in rows)
    boundary_pressure = sum(to_int(row.get("v14_boundary_pressure")) for row in rows)
    strict_risk = sum(to_int(row.get("v14_pose_chain_risk")) for row in rows)
    relaxed_risk = sum(
        1
        for row in rows
        if to_int(row.get("v14_keyframe_pressure"))
        and (to_int(row.get("v14_motion_pressure")) or to_int(row.get("v14_support_low")))
    )
    step_spikes = sum(1 for row in rows if to_float(row.get("v14_step_ratio_proxy")) >= 2.0)
    return {
        "segment": label,
        "frames": len(rows),
        "frame_start": min((to_int(row.get("frame_id")) for row in rows), default=-1),
        "frame_end": max((to_int(row.get("frame_id")) for row in rows), default=-1),
        "se3_rmse_m": rmse(se3),
        "sim3_rmse_m": rmse(sim3),
        "est_path_m": est_path,
        "gt_path_m": gt_path,
        "path_ratio": est_path / gt_path if gt_path > 1e-12 else float("nan"),
        "keyframes": keyframes,
        "keyframe_rate": keyframes / len(rows) if rows else float("nan"),
        "support_low_frames": support_low,
        "motion_pressure_frames": motion_pressure,
        "keyframe_pressure_frames": keyframe_pressure,
        "boundary_pressure_frames": boundary_pressure,
        "strict_risk_frames": strict_risk,
        "relaxed_risk_frames": relaxed_risk,
        "step_spike_frames": step_spikes,
        "mean_static_inliers": mean([to_float(row.get("v14_static_inlier_count")) for row in rows]),
        "min_static_inliers": min((to_float(row.get("v14_static_inlier_count")) for row in rows), default=float("nan")),
        "mean_grid_coverage": mean([to_float(row.get("v14_static_inlier_grid_coverage")) for row in rows]),
        "min_grid_coverage": min((to_float(row.get("v14_static_inlier_grid_coverage")) for row in rows), default=float("nan")),
        "mean_step_ratio_proxy": mean([to_float(row.get("v14_step_ratio_proxy")) for row in rows]),
        "max_step_ratio_proxy": max((to_float(row.get("v14_step_ratio_proxy")) for row in rows), default=float("nan")),
        "mean_boundary_frac": mean([to_float(row.get("v14_boundary_inlier_frac")) for row in rows]),
        "max_boundary_frac": max((to_float(row.get("v14_boundary_inlier_frac")) for row in rows), default=float("nan")),
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(value) for key, value in row.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--range", type=parse_range, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-diff", type=float, default=0.03)
    args = parser.parse_args()

    frame_rows = build_frame_rows(
        args.ground_truth,
        args.run_dir / "CameraTrajectory.txt",
        args.run_dir / "observability_frame_stats.csv",
        args.max_diff,
    )
    for row in frame_rows:
        row["segment"] = segment_for_frame(to_int(row.get("frame_id"), -1), args.range)
    segment_rows = []
    for label, _start, _end in args.range:
        segment_rows.append(summarize_segment(label, [row for row in frame_rows if row.get("segment") == label]))
    write_csv(args.out_dir / "pose_chain_observability_residual_frames.csv", frame_rows)
    write_csv(args.out_dir / "pose_chain_observability_residual_segments.csv", segment_rows)
    print(args.out_dir / "pose_chain_observability_residual_segments.csv")


if __name__ == "__main__":
    main()
