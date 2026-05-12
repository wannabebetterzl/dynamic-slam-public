#!/usr/bin/env python3
# coding=utf-8

import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_tum_xyz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    return data[:, 0], data[:, 1:4]


def align_points_rigid(est, gt):
    est_mean = np.mean(est, axis=0)
    gt_mean = np.mean(gt, axis=0)
    est_centered = est - est_mean
    gt_centered = gt - gt_mean
    covariance = est_centered.T @ gt_centered / max(len(est), 1)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = gt_mean - rotation @ est_mean
    aligned = (rotation @ est.T).T + translation
    return aligned, rotation, translation


def associate_trajectories_with_indices(gt_times, gt_xyz, est_times, est_xyz, max_diff):
    pairs = []
    gt_idx = 0
    est_idx = 0
    while gt_idx < len(gt_times) and est_idx < len(est_times):
        diff = est_times[est_idx] - gt_times[gt_idx]
        if abs(diff) <= max_diff:
            pairs.append(
                {
                    "gt_idx": gt_idx,
                    "est_idx": est_idx,
                    "gt_time": float(gt_times[gt_idx]),
                    "est_time": float(est_times[est_idx]),
                    "gt_xyz": gt_xyz[gt_idx],
                    "est_xyz": est_xyz[est_idx],
                }
            )
            gt_idx += 1
            est_idx += 1
        elif diff > 0:
            gt_idx += 1
        else:
            est_idx += 1
    return pairs


def longest_streak(flags: List[bool]) -> Tuple[int, int]:
    best_len = 0
    best_start = -1
    current_len = 0
    current_start = -1
    for idx, flag in enumerate(flags):
        if flag:
            if current_len == 0:
                current_start = idx
            current_len += 1
            if current_len > best_len:
                best_len = current_len
                best_start = current_start
        else:
            current_len = 0
            current_start = -1
    return best_start, best_len


def first_sustained_crossing(values: List[float], threshold: float, min_len: int) -> int:
    flags = [v > threshold for v in values]
    for start in range(0, len(flags) - min_len + 1):
        if all(flags[start:start + min_len]):
            return start
    return -1


def moving_average(values: List[float], radius: int) -> List[float]:
    out = []
    for i in range(len(values)):
        lo = max(0, i - radius)
        hi = min(len(values), i + radius + 1)
        out.append(float(np.mean(values[lo:hi])))
    return out


def analyze_experiment(args, experiment_name: str) -> Dict[str, object]:
    exp_dir = os.path.join(args.experiments_root, experiment_name)
    runtime_dir = os.path.join(args.runtime_root, experiment_name, "orb_runtime")
    feature_dir = os.path.join(args.feature_root, experiment_name)

    gt_path = os.path.join(exp_dir, "sequence", "groundtruth.txt")
    est_path = os.path.join(runtime_dir, "CameraTrajectory.txt")
    frame_stats_path = os.path.join(exp_dir, "frame_stats.csv")
    feature_path = os.path.join(feature_dir, "feature_support.csv")
    orb_stats_path = os.path.join(runtime_dir, "orb_frame_stats.csv")

    gt_times, gt_xyz = load_tum_xyz(gt_path)
    est_times, est_xyz = load_tum_xyz(est_path)
    frame_rows = load_csv(frame_stats_path)
    feature_rows = load_csv(feature_path)
    orb_rows = load_csv(orb_stats_path)

    pairs = associate_trajectories_with_indices(gt_times, gt_xyz, est_times, est_xyz, args.max_diff)
    if not pairs:
        raise RuntimeError(f"No trajectory matches for {experiment_name}")

    gt = np.array([p["gt_xyz"] for p in pairs], dtype=np.float64)
    est = np.array([p["est_xyz"] for p in pairs], dtype=np.float64)
    aligned_est, rotation, translation = align_points_rigid(est, gt)
    errors = np.linalg.norm(gt - aligned_est, axis=1)
    smoothed_errors = moving_average(errors.tolist(), args.smooth_radius)

    joined_rows = []
    for pair, aligned_xyz, err, smooth_err in zip(pairs, aligned_est, errors, smoothed_errors):
        est_idx = pair["est_idx"]
        frame_row = frame_rows[est_idx]
        feature_row = feature_rows[est_idx]
        orb_row = orb_rows[est_idx]
        joined_rows.append(
            {
                "match_index": len(joined_rows) + 1,
                "frame_index": int(orb_row["frame_index"]),
                "rgb_timestamp": float(frame_row["rgb_timestamp"]),
                "gt_timestamp": float(pair["gt_time"]),
                "est_timestamp": float(pair["est_time"]),
                "aligned_error_m": float(err),
                "smoothed_error_m": float(smooth_err),
                "mask_ratio": float(frame_row["mask_ratio"]),
                "filtered_detections": int(frame_row["filtered_detections"]),
                "static_orb_before": float(feature_row["static_orb_before"]),
                "dynamic_orb_before": float(feature_row["dynamic_orb_before"]),
                "boundary_risk_orb_before": float(feature_row["boundary_risk_orb_before"]),
                "filtered_dynamic_residual_orb_after": float(feature_row["filtered_dynamic_residual_orb_after"]),
                "matches_inliers": int(orb_row["matches_inliers"]),
                "tracked_map_points": int(orb_row["tracked_map_points"]),
                "tracking_state": int(orb_row["tracking_state"]),
            }
        )

    thresholds = [0.03, 0.05, 0.10]
    crossings = {}
    error_list = [row["smoothed_error_m"] for row in joined_rows]
    for thr in thresholds:
        idx = first_sustained_crossing(error_list, thr, args.min_sustain)
        crossings[f"first_sustained_crossing_gt_{thr:.2f}m"] = int(joined_rows[idx]["frame_index"]) if idx >= 0 else -1

    peak_start, peak_len = longest_streak([v > args.peak_threshold for v in error_list])
    peak_window = {}
    if peak_start >= 0:
        segment = joined_rows[peak_start:peak_start + peak_len]
        peak_window = {
            "start_frame_index": int(segment[0]["frame_index"]),
            "end_frame_index": int(segment[-1]["frame_index"]),
            "length": int(peak_len),
            "mean_smoothed_error_m": float(np.mean([r["smoothed_error_m"] for r in segment])),
            "mean_mask_ratio": float(np.mean([r["mask_ratio"] for r in segment])),
            "mean_static_orb_before": float(np.mean([r["static_orb_before"] for r in segment])),
            "mean_dynamic_orb_before": float(np.mean([r["dynamic_orb_before"] for r in segment])),
            "mean_filtered_dynamic_residual_orb_after": float(np.mean([r["filtered_dynamic_residual_orb_after"] for r in segment])),
            "mean_matches_inliers": float(np.mean([r["matches_inliers"] for r in segment])),
            "mean_tracked_map_points": float(np.mean([r["tracked_map_points"] for r in segment])),
        }

    result = {
        "experiment_name": experiment_name,
        "matched_frames": len(joined_rows),
        "alignment_rotation": rotation.tolist(),
        "alignment_translation": translation.tolist(),
        "mean_error_m": float(np.mean(errors)),
        "rmse_error_m": float(np.sqrt(np.mean(errors ** 2))),
        "max_error_m": float(np.max(errors)),
        "crossings": crossings,
        "peak_window": peak_window,
    }

    exp_out_dir = os.path.join(args.output_dir, experiment_name)
    ensure_dir(exp_out_dir)
    with open(os.path.join(exp_out_dir, "segment_error_rows.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(joined_rows[0].keys()))
        writer.writeheader()
        writer.writerows(joined_rows)
    with open(os.path.join(exp_out_dir, "segment_error_summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def main():
    parser = argparse.ArgumentParser(description="Analyze per-frame trajectory divergence against frontend and ORB stats.")
    parser.add_argument("--experiment-names", nargs="+", required=True)
    parser.add_argument("--experiments-root", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-diff", type=float, default=0.03)
    parser.add_argument("--smooth-radius", type=int, default=5)
    parser.add_argument("--min-sustain", type=int, default=10)
    parser.add_argument("--peak-threshold", type=float, default=0.10)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    results = [analyze_experiment(args, name) for name in args.experiment_names]
    with open(os.path.join(args.output_dir, "segment_error_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps({"output_dir": os.path.abspath(args.output_dir), "experiments": len(results)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
