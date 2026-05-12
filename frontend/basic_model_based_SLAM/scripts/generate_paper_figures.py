#!/usr/bin/env python3
# coding=utf-8

import argparse
import csv
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from rflysim_slam_nav.world_sam_pipeline import WorldSamFilterPipeline


ORB_DEFAULTS = {
    "nfeatures": 1000,
    "scaleFactor": 1.2,
    "nlevels": 8,
    "fastThreshold": 20,
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_tum_xyz(path):
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
            pairs.append((gt_xyz[gt_idx], est_xyz[est_idx], gt_idx, est_idx))
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


def build_aligned_trajectory(gt_path, est_path, max_diff):
    gt_times, gt_xyz = load_tum_xyz(gt_path)
    est_times, est_xyz = load_tum_xyz(est_path)
    pairs = associate_trajectories(gt_times, gt_xyz, est_times, est_xyz, max_diff)
    if not pairs:
        raise RuntimeError(f"No matched trajectory pairs found for {est_path}")
    gt = np.array([item[0] for item in pairs], dtype=np.float64)
    est = np.array([item[1] for item in pairs], dtype=np.float64)
    aligned_est = align_points_rigid(est, gt)
    return gt, aligned_est


def pick_frame(frame_stats, frame_index):
    if frame_index > 0:
        idx = max(0, min(frame_index - 1, len(frame_stats) - 1))
        return frame_stats[idx]

    ranked = sorted(
        frame_stats,
        key=lambda row: (
            float(row.get("mask_ratio", 0.0) or 0.0),
            float(row.get("mean_dynamic_memory", 0.0) or 0.0),
            float(row.get("filtered_detections", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return ranked[0]


def to_rgb(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def compute_orb_keypoints(image_bgr, nfeatures):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=nfeatures, **{k: v for k, v in ORB_DEFAULTS.items() if k != "nfeatures"})
    keypoints = orb.detect(gray, None)
    return keypoints or []


def draw_keypoints(image_bgr, keypoints, color=(0, 255, 0), box_half_size=3, thickness=1):
    canvas = image_bgr.copy()
    for kp in keypoints:
        x = int(round(kp.pt[0]))
        y = int(round(kp.pt[1]))
        cv2.rectangle(
            canvas,
            (x - box_half_size, y - box_half_size),
            (x + box_half_size, y + box_half_size),
            color,
            thickness,
        )
    return canvas


def split_keypoints_by_mask(keypoints, mask):
    dynamic_points = []
    static_points = []
    h, w = mask.shape[:2]
    for kp in keypoints:
        x = int(round(kp.pt[0]))
        y = int(round(kp.pt[1]))
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        if mask[y, x] > 0:
            dynamic_points.append(kp)
        else:
            static_points.append(kp)
    return static_points, dynamic_points


def load_frame_triplet(filtered_experiment, frame_index, config_path):
    summary = load_json(os.path.join(filtered_experiment, "benchmark_summary.json"))
    frame_stats = load_csv(os.path.join(filtered_experiment, "frame_stats.csv"))
    selected = pick_frame(frame_stats, frame_index)

    rgb_rel = selected["rgb_path"]
    depth_rel = selected["depth_path"]
    source_rgb = os.path.join(summary["sequence_root"], rgb_rel)
    source_depth = os.path.join(summary["sequence_root"], depth_rel)
    filtered_rgb = os.path.join(summary["export_root"], rgb_rel)

    image = cv2.imread(source_rgb, cv2.IMREAD_COLOR)
    depth = cv2.imread(source_depth, cv2.IMREAD_UNCHANGED)
    filtered = cv2.imread(filtered_rgb, cv2.IMREAD_COLOR)
    if image is None or depth is None or filtered is None:
        raise RuntimeError("Failed to load one or more selected frame assets.")

    pipeline = WorldSamFilterPipeline(config_path or summary.get("config_path") or "")
    result = pipeline.process(image, depth_mm=depth)

    mask = result["mask"].astype(np.uint8) * 255
    return selected, image, result["overlay"], mask, filtered


def save_frame_panel(output_dir, filtered_experiment, frame_index, config_path, dpi, orb_nfeatures):
    selected, image, overlay, mask, filtered = load_frame_triplet(
        filtered_experiment, frame_index, config_path
    )

    ensure_dir(output_dir)

    original_kp = compute_orb_keypoints(image, orb_nfeatures)
    filtered_kp = compute_orb_keypoints(filtered, orb_nfeatures)
    static_kp, dynamic_kp = split_keypoints_by_mask(original_kp, mask)

    original_vis = draw_keypoints(image, original_kp, color=(0, 255, 0))
    filtered_vis = draw_keypoints(filtered, filtered_kp, color=(0, 255, 0))
    dynamic_vis = image.copy()
    dynamic_vis = draw_keypoints(dynamic_vis, static_kp, color=(0, 255, 0))
    dynamic_vis = draw_keypoints(dynamic_vis, dynamic_kp, color=(0, 0, 255))

    cv2.imwrite(os.path.join(output_dir, "selected_frame_orb_original.png"), original_vis)
    cv2.imwrite(os.path.join(output_dir, "selected_frame_orb_filtered.png"), filtered_vis)
    cv2.imwrite(os.path.join(output_dir, "selected_frame_orb_mask_split.png"), dynamic_vis)

    feature_summary = {
        "original_keypoints": int(len(original_kp)),
        "filtered_keypoints": int(len(filtered_kp)),
        "dynamic_region_keypoints_before": int(len(dynamic_kp)),
        "static_region_keypoints_before": int(len(static_kp)),
        "keypoint_delta": int(len(filtered_kp) - len(original_kp)),
        "dynamic_region_ratio_before": float(len(dynamic_kp) / max(len(original_kp), 1)),
    }
    with open(os.path.join(output_dir, "orb_feature_summary.json"), "w", encoding="utf-8") as f:
        json.dump(feature_summary, f, indent=2, ensure_ascii=False)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    panels = [
        (f"ORB Before ({len(original_kp)})", to_rgb(original_vis)),
        (f"Dynamic-region ORB ({len(dynamic_kp)})", to_rgb(dynamic_vis)),
        (f"ORB After ({len(filtered_kp)})", to_rgb(filtered_vis)),
    ]
    for ax, (title, image_data) in zip(axes[:3], panels):
        ax.imshow(image_data)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    bar_ax = axes[3]
    names = ["before", "dynamic", "after"]
    values = [len(original_kp), len(dynamic_kp), len(filtered_kp)]
    colors = ["#4daf4a", "#e41a1c", "#4daf4a"]
    bar_ax.bar(names, values, color=colors, width=0.6)
    bar_ax.set_title("Feature Count", fontsize=11)
    bar_ax.set_ylabel("Keypoints")
    bar_ax.grid(axis="y", linestyle="--", alpha=0.3)
    for idx, value in enumerate(values):
        bar_ax.text(idx, value + max(values) * 0.02, str(value), ha="center", va="bottom", fontsize=10)

    stats_text = (
        f"frame={selected['frame_index']}  "
        f"mask_ratio={float(selected.get('mask_ratio', 0.0) or 0.0):.3f}  "
        f"delta={len(filtered_kp) - len(original_kp)}  "
        f"dynamic_ratio={len(dynamic_kp) / max(len(original_kp), 1):.3f}"
    )
    fig.suptitle(stats_text, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "orb_feature_comparison.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "selected_frame_index": int(selected["frame_index"]),
        "selected_rgb_path": selected["rgb_path"],
        "mask_ratio": float(selected.get("mask_ratio", 0.0) or 0.0),
        "mean_dynamic_memory": float(selected.get("mean_dynamic_memory", 0.0) or 0.0),
        "runtime_ms": float(selected.get("runtime_ms", 0.0) or 0.0),
        **feature_summary,
    }


def save_trajectory_plot(raw_experiment, compare_experiments, labels, output_dir, max_diff, dpi):
    ensure_dir(output_dir)
    raw_summary = load_json(os.path.join(raw_experiment, "benchmark_summary.json"))
    gt_path = os.path.join(raw_summary["export_root"], "groundtruth.txt")
    raw_traj = raw_summary["orb_slam3"]["trajectory_path"]

    gt_points, raw_points = build_aligned_trajectory(gt_path, raw_traj, max_diff)

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    ax.plot(gt_points[:, 0], gt_points[:, 2], color="black", linewidth=2.2, label="Ground Truth")
    ax.plot(raw_points[:, 0], raw_points[:, 2], color="#d95f02", linewidth=1.8, label=labels[0])

    metric_rows = []
    for experiment, label in zip(compare_experiments, labels[1:]):
        summary = load_json(os.path.join(experiment, "benchmark_summary.json"))
        traj_path = summary["orb_slam3"]["trajectory_path"]
        _, aligned_points = build_aligned_trajectory(gt_path, traj_path, max_diff)
        ax.plot(aligned_points[:, 0], aligned_points[:, 2], linewidth=1.8, label=label)
        metrics = summary.get("trajectory_metrics", {})
        metric_rows.append(
            {
                "label": label,
                "ate_rmse_m": float(metrics.get("ate_rmse_m", 0.0)),
                "rpe_rmse_m": float(metrics.get("rpe_rmse_m", 0.0)),
                "coverage": float(metrics.get("trajectory_coverage", 0.0)),
            }
        )

    raw_metrics = raw_summary.get("trajectory_metrics", {})
    metric_rows.insert(
        0,
        {
            "label": labels[0],
            "ate_rmse_m": float(raw_metrics.get("ate_rmse_m", 0.0)),
            "rpe_rmse_m": float(raw_metrics.get("rpe_rmse_m", 0.0)),
            "coverage": float(raw_metrics.get("trajectory_coverage", 0.0)),
        },
    )

    ax.set_xlabel("X / m")
    ax.set_ylabel("Z / m")
    ax.set_title("Trajectory Comparison")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "trajectory_comparison.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return metric_rows


def main():
    parser = argparse.ArgumentParser(description="Generate paper-ready SLAM figures from benchmark outputs.")
    parser.add_argument("--raw-experiment", required=True, help="Raw baseline benchmark directory.")
    parser.add_argument(
        "--compare-experiment",
        action="append",
        required=True,
        help="Filtered benchmark directory. Repeatable.",
    )
    parser.add_argument(
        "--label",
        action="append",
        required=True,
        help="Legend labels. The first label is for the raw baseline, then one per compare experiment.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to save paper figures.")
    parser.add_argument("--frame-index", type=int, default=0, help="1-based frame index. Use 0 to auto-pick the strongest dynamic frame.")
    parser.add_argument("--config", default="", help="Optional config path for re-generating overlay and mask.")
    parser.add_argument("--max-diff", type=float, default=0.03, help="Timestamp matching threshold for trajectory alignment.")
    parser.add_argument("--dpi", type=int, default=220, help="Figure export DPI.")
    parser.add_argument("--orb-nfeatures", type=int, default=1000, help="ORB keypoint count used for qualitative visualization.")
    args = parser.parse_args()

    if len(args.label) != len(args.compare_experiment) + 1:
        raise RuntimeError("--label count must equal 1 + number of --compare-experiment entries.")

    ensure_dir(args.output_dir)
    metric_rows = save_trajectory_plot(
        args.raw_experiment,
        args.compare_experiment,
        args.label,
        args.output_dir,
        args.max_diff,
        args.dpi,
    )
    frame_summary = save_frame_panel(
        args.output_dir,
        args.compare_experiment[-1],
        args.frame_index,
        args.config,
        args.dpi,
        args.orb_nfeatures,
    )

    summary = {
        "raw_experiment": args.raw_experiment,
        "compare_experiments": args.compare_experiment,
        "labels": args.label,
        "trajectory_metrics": metric_rows,
        "frame_summary": frame_summary,
    }
    with open(os.path.join(args.output_dir, "figure_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
