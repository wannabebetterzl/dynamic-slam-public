#!/usr/bin/env python3
# coding=utf-8

import argparse
import csv
import json
import os
import re
from typing import Dict, List

import numpy as np


WINDOWS = (30, 60, 120)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def median(values: List[float]) -> float:
    return float(np.median(values)) if values else 0.0


def longest_streak(flags: List[bool]) -> int:
    best = 0
    current = 0
    for flag in flags:
        if flag:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def parse_initial_map_points(stdout_path: str) -> int:
    if not os.path.isfile(stdout_path):
        return 0
    text = open(stdout_path, "r", encoding="utf-8").read()
    match = re.search(r"New Map created with (\d+) points", text)
    return int(match.group(1)) if match else 0


def summarize_window(feature_rows, runtime_rows, n: int) -> Dict[str, float]:
    feat = feature_rows[:n]
    run = runtime_rows[:n]

    static_before = [float(r["static_orb_before"]) for r in feat]
    dynamic_before = [float(r["dynamic_orb_before"]) for r in feat]
    boundary_before = [float(r["boundary_risk_orb_before"]) for r in feat]
    mask_ratio = [float(r["mask_ratio"]) for r in feat]
    filtered_static_after = [float(r["filtered_static_orb_after"]) for r in feat]
    filtered_dynamic_after = [float(r["filtered_dynamic_residual_orb_after"]) for r in feat]
    grid_after = [float(r["filtered_grid_coverage_after"]) for r in feat]

    inliers = [int(r["matches_inliers"]) for r in run]
    map_points = [int(r["tracked_map_points"]) for r in run]
    states = [int(r["tracking_state"]) for r in run]

    low_100 = [v < 100 for v in inliers]
    low_150 = [v < 150 for v in inliers]
    zero_inlier = [v == 0 for v in inliers]
    not_ok = [v != 2 for v in states]

    return {
        "frames": int(n),
        "mean_static_orb_before": mean(static_before),
        "mean_dynamic_orb_before": mean(dynamic_before),
        "mean_boundary_risk_orb_before": mean(boundary_before),
        "mean_mask_ratio": mean(mask_ratio),
        "mean_filtered_static_orb_after": mean(filtered_static_after),
        "mean_filtered_dynamic_residual_after": mean(filtered_dynamic_after),
        "mean_filtered_grid_coverage_after": mean(grid_after),
        "mean_matches_inliers": mean(inliers),
        "median_matches_inliers": median(inliers),
        "min_matches_inliers": int(min(inliers)) if inliers else 0,
        "mean_tracked_map_points": mean(map_points),
        "median_tracked_map_points": median(map_points),
        "low_inlier_ratio_lt100": mean([1.0 if x else 0.0 for x in low_100]),
        "low_inlier_ratio_lt150": mean([1.0 if x else 0.0 for x in low_150]),
        "zero_inlier_ratio": mean([1.0 if x else 0.0 for x in zero_inlier]),
        "not_ok_ratio": mean([1.0 if x else 0.0 for x in not_ok]),
        "longest_low_inlier_streak_lt100": int(longest_streak(low_100)),
        "longest_low_inlier_streak_lt150": int(longest_streak(low_150)),
        "longest_zero_inlier_streak": int(longest_streak(zero_inlier)),
    }


def analyze_experiment(experiment_name: str, feature_root: str, runtime_root: str) -> Dict[str, object]:
    feature_csv = os.path.join(feature_root, experiment_name, "feature_support.csv")
    runtime_dir = os.path.join(runtime_root, experiment_name, "orb_runtime")
    runtime_csv = os.path.join(runtime_dir, "orb_frame_stats.csv")
    runtime_summary_json = os.path.join(runtime_dir, "runtime_summary.json")
    stdout_log = os.path.join(runtime_dir, "stdout.log")

    feature_rows = load_csv(feature_csv)
    runtime_rows = load_csv(runtime_csv)
    runtime_summary = load_json(runtime_summary_json)

    if len(feature_rows) != len(runtime_rows):
        raise RuntimeError(f"Frame count mismatch in {experiment_name}: {len(feature_rows)} vs {len(runtime_rows)}")

    result = {
        "experiment_name": experiment_name,
        "ate_rmse_m": float(runtime_summary["trajectory_metrics"]["ate_rmse_m"]),
        "rpe_rmse_m": float(runtime_summary["trajectory_metrics"]["rpe_rmse_m"]),
        "trajectory_coverage": float(runtime_summary["trajectory_metrics"]["trajectory_coverage"]),
        "initial_map_points": int(parse_initial_map_points(stdout_log)),
        "windows": {},
    }

    for n in WINDOWS:
        result["windows"][f"first_{n}"] = summarize_window(feature_rows, runtime_rows, n)

    return result


def flatten_result(result: Dict[str, object]) -> Dict[str, object]:
    row = {
        "experiment_name": result["experiment_name"],
        "ate_rmse_m": result["ate_rmse_m"],
        "rpe_rmse_m": result["rpe_rmse_m"],
        "trajectory_coverage": result["trajectory_coverage"],
        "initial_map_points": result["initial_map_points"],
    }
    for window_name, metrics in result["windows"].items():
        prefix = window_name
        for key, value in metrics.items():
            row[f"{prefix}_{key}"] = value
    return row


def main():
    parser = argparse.ArgumentParser(description="Analyze initialization-stage static support and ORB runtime behavior.")
    parser.add_argument("--experiment-names", nargs="+", required=True, help="Experiment directory names.")
    parser.add_argument("--feature-root", required=True, help="Root directory containing per-experiment feature_support.csv folders.")
    parser.add_argument("--runtime-root", required=True, help="Root directory containing per-experiment orb_runtime folders.")
    parser.add_argument("--output-dir", required=True, help="Output directory for initialization analysis.")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    results = [analyze_experiment(name, args.feature_root, args.runtime_root) for name in args.experiment_names]
    rows = [flatten_result(item) for item in results]

    csv_path = os.path.join(args.output_dir, "initialization_window_comparison.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(args.output_dir, "initialization_window_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(json.dumps({"output_dir": os.path.abspath(args.output_dir), "experiments": len(results)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
