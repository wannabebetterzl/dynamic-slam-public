#!/usr/bin/env python3
# coding=utf-8

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
from typing import Dict, List, Tuple

import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_tum_xyz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    return data[:, 0], data[:, 1:4]


def associate_trajectories(gt_times, gt_xyz, est_times, est_xyz, max_diff):
    pairs = []
    gt_idx = 0
    est_idx = 0
    while gt_idx < len(gt_times) and est_idx < len(est_times):
        diff = est_times[est_idx] - gt_times[gt_idx]
        if abs(diff) <= max_diff:
            pairs.append((gt_xyz[gt_idx], est_xyz[est_idx]))
            gt_idx += 1
            est_idx += 1
        elif diff > 0:
            gt_idx += 1
        else:
            est_idx += 1
    return pairs


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
    return aligned


def compute_trajectory_metrics(gt_path: str, est_path: str, max_diff: float):
    gt_times, gt_xyz = load_tum_xyz(gt_path)
    est_times, est_xyz = load_tum_xyz(est_path)
    pairs = associate_trajectories(gt_times, gt_xyz, est_times, est_xyz, max_diff)
    if not pairs:
        raise RuntimeError("No matched trajectory pairs were found.")

    gt = np.array([item[0] for item in pairs], dtype=np.float64)
    est = np.array([item[1] for item in pairs], dtype=np.float64)
    aligned_est = align_points_rigid(est, gt)
    errors = np.linalg.norm(gt - aligned_est, axis=1)

    ate_rmse = float(np.sqrt(np.mean(errors ** 2)))
    ate_mean = float(np.mean(errors))
    if len(gt) < 2:
        rpe_rmse = 0.0
    else:
        gt_delta = gt[1:] - gt[:-1]
        est_delta = aligned_est[1:] - aligned_est[:-1]
        rpe = np.linalg.norm(gt_delta - est_delta, axis=1)
        rpe_rmse = float(np.sqrt(np.mean(rpe ** 2)))

    return {
        "matched_poses": int(len(pairs)),
        "ground_truth_poses": int(len(gt_times)),
        "estimated_poses": int(len(est_times)),
        "trajectory_coverage": float(len(pairs) / max(len(gt_times), 1)),
        "ate_rmse_m": ate_rmse,
        "ate_mean_m": ate_mean,
        "rpe_rmse_m": rpe_rmse,
    }


def trajectory_path_from_run(run_dir: str) -> str:
    for candidate in ("CameraTrajectory.txt", "KeyFrameTrajectory.txt"):
        path = os.path.join(run_dir, candidate)
        if os.path.isfile(path):
            return path
    return ""


def longest_streak(values: List[bool]) -> int:
    best = 0
    current = 0
    for item in values:
        if item:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def summarize_runtime_stats(stats_rows: List[Dict[str, str]]) -> Dict[str, float]:
    if not stats_rows:
        raise RuntimeError("Empty ORB runtime stats.")

    states = [int(row["tracking_state"]) for row in stats_rows]
    inliers = [int(row["matches_inliers"]) for row in stats_rows]
    map_points = [int(row["tracked_map_points"]) for row in stats_rows]
    keypoints = [int(row["tracked_keypoints"]) for row in stats_rows]
    track_time = [float(row["track_time_sec"]) for row in stats_rows]

    ok_flags = [state == 2 for state in states]
    lost_flags = [state == 4 for state in states]
    low_inlier_30 = [value < 30 for value in inliers]
    low_inlier_50 = [value < 50 for value in inliers]

    return {
        "frames_logged": int(len(stats_rows)),
        "mean_matches_inliers": float(np.mean(inliers)),
        "median_matches_inliers": float(np.median(inliers)),
        "min_matches_inliers": int(np.min(inliers)),
        "mean_tracked_map_points": float(np.mean(map_points)),
        "median_tracked_map_points": float(np.median(map_points)),
        "mean_tracked_keypoints": float(np.mean(keypoints)),
        "ok_ratio": float(np.mean(ok_flags)),
        "lost_ratio": float(np.mean(lost_flags)),
        "low_inlier_ratio_lt30": float(np.mean(low_inlier_30)),
        "low_inlier_ratio_lt50": float(np.mean(low_inlier_50)),
        "longest_low_inlier_streak_lt30": int(longest_streak(low_inlier_30)),
        "longest_low_inlier_streak_lt50": int(longest_streak(low_inlier_50)),
        "mean_track_time_sec": float(np.mean(track_time)),
        "median_track_time_sec": float(np.median(track_time)),
    }


def run_one_experiment(args, experiment_dir: str) -> Dict[str, object]:
    experiment_dir = os.path.abspath(experiment_dir)
    summary_path = os.path.join(experiment_dir, "benchmark_summary.json")
    result_path = os.path.join(experiment_dir, "result.json")
    if os.path.isfile(summary_path):
        summary = load_json(summary_path)
    elif os.path.isfile(result_path):
        summary = load_json(result_path)
    else:
        raise RuntimeError(f"Missing benchmark_summary.json/result.json in {experiment_dir}")

    export_root = summary["export_root"]
    gt_path = os.path.join(export_root, "groundtruth.txt")
    experiment_name = os.path.basename(experiment_dir.rstrip("/"))
    run_dir = os.path.join(args.output_dir, experiment_name, "orb_runtime")
    ensure_dir(run_dir)

    stats_csv = os.path.join(run_dir, "orb_frame_stats.csv")
    cmd = [
        args.orb_exec,
        args.orb_vocab,
        args.orb_config,
        export_root,
        os.path.join(export_root, "associations.txt"),
        stats_csv,
    ]

    proc = subprocess.run(
        cmd,
        cwd=run_dir,
        text=True,
        capture_output=True,
        timeout=args.orb_timeout,
        check=False,
    )

    with open(os.path.join(run_dir, "stdout.log"), "w", encoding="utf-8") as f:
        f.write(proc.stdout or "")
    with open(os.path.join(run_dir, "stderr.log"), "w", encoding="utf-8") as f:
        f.write(proc.stderr or "")

    traj_path = trajectory_path_from_run(run_dir)
    runtime_rows = load_csv(stats_csv) if os.path.isfile(stats_csv) else []

    metrics = {}
    if traj_path and os.path.isfile(gt_path):
        metrics = compute_trajectory_metrics(gt_path, traj_path, args.eval_max_diff)

    runtime_summary = summarize_runtime_stats(runtime_rows) if runtime_rows else {}
    result = {
        "experiment_name": experiment_name,
        "experiment_dir": experiment_dir,
        "export_root": export_root,
        "returncode": int(proc.returncode),
        "trajectory_path": traj_path,
        "stats_csv": stats_csv if os.path.isfile(stats_csv) else "",
        "trajectory_metrics": metrics,
        "runtime_summary": runtime_summary,
    }

    with open(os.path.join(run_dir, "runtime_summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def main():
    parser = argparse.ArgumentParser(description="Run ORB-SLAM3 on exported experiment sequences and collect per-frame runtime stats.")
    parser.add_argument("--experiments", nargs="+", required=True, help="Experiment directories with result.json or benchmark_summary.json.")
    parser.add_argument("--output-dir", required=True, help="Output directory for ORB runtime suites.")
    parser.add_argument("--orb-exec", required=True, help="Path to ORB-SLAM3 rgbd_tum executable.")
    parser.add_argument("--orb-vocab", required=True, help="Path to ORB vocabulary.")
    parser.add_argument("--orb-config", required=True, help="Path to ORB RGB-D config yaml.")
    parser.add_argument("--orb-timeout", type=int, default=1800, help="Timeout for one ORB run in seconds.")
    parser.add_argument("--eval-max-diff", type=float, default=0.03, help="Max timestamp difference used in trajectory evaluation.")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    summaries = []
    for experiment_dir in args.experiments:
        summaries.append(run_one_experiment(args, experiment_dir))

    flat_rows = []
    for item in summaries:
        row = {
            "experiment_name": item["experiment_name"],
            "returncode": item["returncode"],
        }
        row.update(item.get("trajectory_metrics", {}))
        row.update(item.get("runtime_summary", {}))
        flat_rows.append(row)

    csv_path = os.path.join(args.output_dir, "orb_runtime_comparison.csv")
    if flat_rows:
        fieldnames = list(flat_rows[0].keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)

    with open(os.path.join(args.output_dir, "orb_runtime_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    print(json.dumps({"output_dir": os.path.abspath(args.output_dir), "experiments": len(summaries)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
