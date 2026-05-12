#!/usr/bin/env python3
# coding=utf-8

import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np


ORB_DEFAULTS = {
    "nfeatures": 1000,
    "scaleFactor": 1.2,
    "nlevels": 8,
    "fastThreshold": 20,
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def compute_orb_keypoints(image_bgr: np.ndarray, nfeatures: int):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=nfeatures, **{k: v for k, v in ORB_DEFAULTS.items() if k != "nfeatures"})
    keypoints = orb.detect(gray, None)
    return keypoints or []


def split_keypoints_by_mask(keypoints, mask: np.ndarray):
    inside = []
    outside = []
    h, w = mask.shape[:2]
    for kp in keypoints:
        x = int(round(kp.pt[0]))
        y = int(round(kp.pt[1]))
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        if mask[y, x] > 0:
            inside.append(kp)
        else:
            outside.append(kp)
    return outside, inside


def dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    pixels = max(int(pixels), 1)
    kernel = np.ones((pixels, pixels), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)


def count_keypoints_in_mask(keypoints, mask: np.ndarray) -> int:
    _, inside = split_keypoints_by_mask(keypoints, mask)
    return len(inside)


def grid_coverage_ratio(keypoints, image_shape: Tuple[int, int], grid_cols: int, grid_rows: int) -> float:
    h, w = image_shape[:2]
    occupied = set()
    for kp in keypoints:
        x = int(round(kp.pt[0]))
        y = int(round(kp.pt[1]))
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        col = min(grid_cols - 1, int(x * grid_cols / max(w, 1)))
        row = min(grid_rows - 1, int(y * grid_rows / max(h, 1)))
        occupied.add((row, col))
    total_cells = max(grid_cols * grid_rows, 1)
    return float(len(occupied) / total_cells)


def safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def mean(items: List[float]) -> float:
    return float(np.mean(items)) if items else 0.0


def analyze_experiment(experiment_dir: str, output_dir: str, nfeatures: int, ring_width: int, grid_cols: int, grid_rows: int):
    summary_path = os.path.join(experiment_dir, "benchmark_summary.json")
    result_path = os.path.join(experiment_dir, "result.json")
    frame_stats_path = os.path.join(experiment_dir, "frame_stats.csv")
    if not os.path.isfile(frame_stats_path):
        raise RuntimeError(f"Missing frame_stats.csv in {experiment_dir}")

    if os.path.isfile(summary_path):
        summary = load_json(summary_path)
    elif os.path.isfile(result_path):
        summary = load_json(result_path)
    else:
        raise RuntimeError(f"Missing benchmark_summary.json/result.json in {experiment_dir}")

    frame_stats = load_csv(frame_stats_path)
    export_root = summary["export_root"]
    experiment_name = os.path.basename(os.path.abspath(experiment_dir))

    rows = []
    for row in frame_stats:
        rgb_rel = row["rgb_path"]
        rgb_name = os.path.basename(rgb_rel)
        source_rgb = os.path.join(summary["sequence_root"], rgb_rel)
        filtered_rgb = os.path.join(export_root, rgb_rel)
        mask_path = os.path.join(export_root, "mask", rgb_name)

        source_image = cv2.imread(source_rgb, cv2.IMREAD_COLOR)
        filtered_image = cv2.imread(filtered_rgb, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if source_image is None or filtered_image is None or mask is None:
            continue

        mask = (mask > 0).astype(np.uint8)
        ring = np.clip(dilate_mask(mask, ring_width) - mask, 0, 1).astype(np.uint8)

        source_kp = compute_orb_keypoints(source_image, nfeatures)
        filtered_kp = compute_orb_keypoints(filtered_image, nfeatures)

        static_before, dynamic_before = split_keypoints_by_mask(source_kp, mask)
        filtered_outside, filtered_inside = split_keypoints_by_mask(filtered_kp, mask)
        boundary_before = count_keypoints_in_mask(source_kp, ring)
        boundary_after = count_keypoints_in_mask(filtered_kp, ring)

        rows.append(
            {
                "frame_index": int(row["frame_index"]),
                "original_orb_count": int(len(source_kp)),
                "static_orb_before": int(len(static_before)),
                "dynamic_orb_before": int(len(dynamic_before)),
                "filtered_orb_count": int(len(filtered_kp)),
                "filtered_static_orb_after": int(len(filtered_outside)),
                "filtered_dynamic_residual_orb_after": int(len(filtered_inside)),
                "boundary_risk_orb_before": int(boundary_before),
                "boundary_risk_orb_after": int(boundary_after),
                "mask_ratio": float(row.get("mask_ratio", 0.0) or 0.0),
                "static_support_ratio_before": safe_ratio(len(static_before), len(source_kp)),
                "dynamic_pollution_ratio_before": safe_ratio(len(dynamic_before), len(source_kp)),
                "filtered_static_retention_ratio": safe_ratio(len(filtered_outside), len(static_before)),
                "filtered_dynamic_residual_ratio": safe_ratio(len(filtered_inside), len(filtered_kp)),
                "static_grid_coverage_before": grid_coverage_ratio(static_before, source_image.shape, grid_cols, grid_rows),
                "filtered_grid_coverage_after": grid_coverage_ratio(filtered_outside, filtered_image.shape, grid_cols, grid_rows),
                "filtered_total_grid_coverage_after": grid_coverage_ratio(filtered_kp, filtered_image.shape, grid_cols, grid_rows),
            }
        )

    if not rows:
        raise RuntimeError(f"No valid analysis frames found in {experiment_dir}")

    exp_output_dir = os.path.join(output_dir, experiment_name)
    ensure_dir(exp_output_dir)

    csv_path = os.path.join(exp_output_dir, "feature_support.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_json = {
        "experiment_dir": experiment_dir,
        "experiment_name": experiment_name,
        "frames_analyzed": len(rows),
        "mean_original_orb_count": mean([r["original_orb_count"] for r in rows]),
        "mean_static_orb_before": mean([r["static_orb_before"] for r in rows]),
        "mean_dynamic_orb_before": mean([r["dynamic_orb_before"] for r in rows]),
        "mean_filtered_orb_count": mean([r["filtered_orb_count"] for r in rows]),
        "mean_filtered_static_orb_after": mean([r["filtered_static_orb_after"] for r in rows]),
        "mean_filtered_dynamic_residual_orb_after": mean([r["filtered_dynamic_residual_orb_after"] for r in rows]),
        "mean_boundary_risk_orb_before": mean([r["boundary_risk_orb_before"] for r in rows]),
        "mean_boundary_risk_orb_after": mean([r["boundary_risk_orb_after"] for r in rows]),
        "mean_mask_ratio": mean([r["mask_ratio"] for r in rows]),
        "mean_static_support_ratio_before": mean([r["static_support_ratio_before"] for r in rows]),
        "mean_dynamic_pollution_ratio_before": mean([r["dynamic_pollution_ratio_before"] for r in rows]),
        "mean_filtered_static_retention_ratio": mean([r["filtered_static_retention_ratio"] for r in rows]),
        "mean_filtered_dynamic_residual_ratio": mean([r["filtered_dynamic_residual_ratio"] for r in rows]),
        "mean_static_grid_coverage_before": mean([r["static_grid_coverage_before"] for r in rows]),
        "mean_filtered_grid_coverage_after": mean([r["filtered_grid_coverage_after"] for r in rows]),
        "mean_filtered_total_grid_coverage_after": mean([r["filtered_total_grid_coverage_after"] for r in rows]),
    }

    with open(os.path.join(exp_output_dir, "feature_support_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)

    return summary_json


def main():
    parser = argparse.ArgumentParser(description="Analyze static feature support and dynamic residual risk from exported SLAM experiments.")
    parser.add_argument("--experiments", nargs="+", required=True, help="Experiment directories containing benchmark_summary.json and frame_stats.csv.")
    parser.add_argument("--output-dir", required=True, help="Directory to store per-experiment and aggregate analysis outputs.")
    parser.add_argument("--orb-nfeatures", type=int, default=1000, help="ORB feature count for offline analysis.")
    parser.add_argument("--ring-width", type=int, default=7, help="Boundary ring width in pixels for risk analysis.")
    parser.add_argument("--grid-cols", type=int, default=8, help="Grid columns used for spatial coverage.")
    parser.add_argument("--grid-rows", type=int, default=6, help="Grid rows used for spatial coverage.")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    summaries = []
    for experiment_dir in args.experiments:
        summaries.append(
            analyze_experiment(
                experiment_dir=os.path.abspath(experiment_dir),
                output_dir=os.path.abspath(args.output_dir),
                nfeatures=args.orb_nfeatures,
                ring_width=args.ring_width,
                grid_cols=args.grid_cols,
                grid_rows=args.grid_rows,
            )
        )

    aggregate_csv = os.path.join(args.output_dir, "feature_support_comparison.csv")
    if summaries:
        with open(aggregate_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)

    aggregate_json = os.path.join(args.output_dir, "feature_support_comparison.json")
    with open(aggregate_json, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    print(json.dumps({"output_dir": os.path.abspath(args.output_dir), "experiments_analyzed": len(summaries)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
