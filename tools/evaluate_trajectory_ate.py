#!/usr/bin/env python3
# coding=utf-8
"""Unified trajectory evaluation for this project.

The default ATE report uses rigid SE(3) Umeyama alignment, matching the
standard TUM RGB-D / ORB-SLAM reporting convention.  Origin-only and Sim(3)
alignment are kept as explicit diagnostic modes so experiments cannot silently
mix evaluation protocols.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


Trajectory = Dict[str, np.ndarray]


def load_tum_trajectory(path: Path) -> Trajectory:
    rows: List[List[float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 4:
            continue
        rows.append([float(value) for value in parts[:8]])

    if not rows:
        raise RuntimeError(f"No trajectory poses were found in {path}")

    max_len = max(len(row) for row in rows)
    data = np.full((len(rows), max_len), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        data[i, : len(row)] = row

    result: Trajectory = {
        "times": data[:, 0],
        "xyz": data[:, 1:4],
    }
    if data.shape[1] >= 8 and np.all(np.isfinite(data[:, 4:8])):
        result["quat_xyzw"] = normalize_quaternions(data[:, 4:8])
    return result


def normalize_quaternions(quats: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quats, axis=1)
    safe = norms > 0
    normalized = quats.copy()
    normalized[safe] /= norms[safe, None]
    return normalized


def associate_by_timestamp(
    gt: Trajectory, est: Trajectory, max_diff: float
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    gt_times = gt["times"]
    est_times = est["times"]
    gt_xyz = gt["xyz"]
    est_xyz = est["xyz"]
    gt_quat = gt.get("quat_xyzw")
    est_quat = est.get("quat_xyzw")

    gt_idx = 0
    est_idx = 0
    matched_gt: List[np.ndarray] = []
    matched_est: List[np.ndarray] = []
    matched_gt_quat: List[np.ndarray] = []
    matched_est_quat: List[np.ndarray] = []

    while gt_idx < len(gt_times) and est_idx < len(est_times):
        diff = est_times[est_idx] - gt_times[gt_idx]
        if abs(diff) <= max_diff:
            matched_gt.append(gt_xyz[gt_idx])
            matched_est.append(est_xyz[est_idx])
            if gt_quat is not None and est_quat is not None:
                matched_gt_quat.append(gt_quat[gt_idx])
                matched_est_quat.append(est_quat[est_idx])
            gt_idx += 1
            est_idx += 1
        elif diff > 0:
            gt_idx += 1
        else:
            est_idx += 1

    if not matched_gt:
        raise RuntimeError("No matched trajectory pairs were found. Check timestamps or max-diff.")

    gt_q = np.array(matched_gt_quat, dtype=np.float64) if matched_gt_quat else None
    est_q = np.array(matched_est_quat, dtype=np.float64) if matched_est_quat else None
    return (
        np.array(matched_gt, dtype=np.float64),
        np.array(matched_est, dtype=np.float64),
        gt_q,
        est_q,
    )


def umeyama_align(
    est_xyz: np.ndarray, gt_xyz: np.ndarray, with_scale: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    est_mean = np.mean(est_xyz, axis=0)
    gt_mean = np.mean(gt_xyz, axis=0)
    est_centered = est_xyz - est_mean
    gt_centered = gt_xyz - gt_mean

    covariance = est_centered.T @ gt_centered / max(len(est_xyz), 1)
    u, singular_values, vt = np.linalg.svd(covariance)

    correction = np.eye(3)
    if np.linalg.det(vt.T @ u.T) < 0:
        correction[2, 2] = -1.0

    rotation = vt.T @ correction @ u.T
    if with_scale:
        variance = np.mean(np.sum(est_centered * est_centered, axis=1))
        scale = float(np.sum(singular_values * np.diag(correction)) / variance) if variance > 0 else 1.0
    else:
        scale = 1.0

    translation = gt_mean - scale * (rotation @ est_mean)
    aligned = scale * (rotation @ est_xyz.T).T + translation
    return aligned, rotation, translation, scale


def align_positions(
    gt_xyz: np.ndarray, est_xyz: np.ndarray, alignment: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    if alignment == "origin":
        aligned = est_xyz - est_xyz[0] + gt_xyz[0]
        return aligned, np.eye(3), gt_xyz[0] - est_xyz[0], 1.0, "origin_translation"
    if alignment == "none":
        return est_xyz.copy(), np.eye(3), np.zeros(3), 1.0, "none"
    if alignment == "se3":
        aligned, rotation, translation, scale = umeyama_align(est_xyz, gt_xyz, with_scale=False)
        return aligned, rotation, translation, scale, "rigid_umeyama_se3"
    if alignment == "sim3":
        aligned, rotation, translation, scale = umeyama_align(est_xyz, gt_xyz, with_scale=True)
        return aligned, rotation, translation, scale, "similarity_umeyama_sim3"
    raise ValueError(f"Unsupported alignment: {alignment}")


def compute_translation_stats(errors: np.ndarray, prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}_rmse_m": float(math.sqrt(np.mean(errors * errors))),
        f"{prefix}_mean_m": float(np.mean(errors)),
        f"{prefix}_median_m": float(np.median(errors)),
        f"{prefix}_std_m": float(np.std(errors)),
        f"{prefix}_min_m": float(np.min(errors)),
        f"{prefix}_max_m": float(np.max(errors)),
    }


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def rotation_angle_deg(rotation: np.ndarray) -> float:
    value = (np.trace(rotation) - 1.0) / 2.0
    value = min(1.0, max(-1.0, float(value)))
    return math.degrees(math.acos(value))


def compute_rpe(
    gt_xyz: np.ndarray,
    aligned_est_xyz: np.ndarray,
    gt_quat: Optional[np.ndarray],
    est_quat: Optional[np.ndarray],
    delta: int,
) -> Dict[str, float]:
    if len(gt_xyz) <= delta:
        return {}

    gt_delta = gt_xyz[delta:] - gt_xyz[:-delta]
    est_delta = aligned_est_xyz[delta:] - aligned_est_xyz[:-delta]
    trans_errors = np.linalg.norm(gt_delta - est_delta, axis=1)
    metrics = compute_translation_stats(trans_errors, "rpet")
    metrics["rpe_delta"] = int(delta)

    if gt_quat is not None and est_quat is not None and len(gt_quat) > delta:
        rot_errors: List[float] = []
        gt_rot = [quat_xyzw_to_matrix(q) for q in gt_quat]
        est_rot = [quat_xyzw_to_matrix(q) for q in est_quat]
        for i in range(len(gt_rot) - delta):
            gt_rel = gt_rot[i].T @ gt_rot[i + delta]
            est_rel = est_rot[i].T @ est_rot[i + delta]
            error_rot = gt_rel.T @ est_rel
            rot_errors.append(rotation_angle_deg(error_rot))
        rot = np.array(rot_errors, dtype=np.float64)
        metrics.update(
            {
                "rper_rmse_deg": float(math.sqrt(np.mean(rot * rot))),
                "rper_mean_deg": float(np.mean(rot)),
                "rper_median_deg": float(np.median(rot)),
                "rper_max_deg": float(np.max(rot)),
            }
        )
    return metrics


def evaluate(
    gt_path: Path,
    est_path: Path,
    alignment: str,
    max_diff: float,
    rpe_delta: int,
) -> Dict[str, object]:
    gt_traj = load_tum_trajectory(gt_path)
    est_traj = load_tum_trajectory(est_path)
    gt_xyz, est_xyz, gt_quat, est_quat = associate_by_timestamp(gt_traj, est_traj, max_diff)
    aligned_est, rotation, translation, scale, alignment_method = align_positions(
        gt_xyz, est_xyz, alignment
    )
    ate_errors = np.linalg.norm(gt_xyz - aligned_est, axis=1)

    metrics: Dict[str, object] = {
        "ground_truth": str(gt_path),
        "estimate": str(est_path),
        "ground_truth_poses": int(len(gt_traj["times"])),
        "estimated_poses": int(len(est_traj["times"])),
        "matched_poses": int(len(gt_xyz)),
        "trajectory_coverage": float(len(gt_xyz) / max(len(gt_traj["times"]), 1)),
        "alignment": alignment,
        "alignment_method": alignment_method,
        "alignment_scale": float(scale),
        "alignment_rotation": rotation.tolist(),
        "alignment_translation": translation.tolist(),
    }
    metrics.update(compute_translation_stats(ate_errors, "ate"))
    metrics.update(compute_rpe(gt_xyz, aligned_est, gt_quat, est_quat, rpe_delta))
    return metrics


def print_text(metrics: Dict[str, object]) -> None:
    print(f"groundtruth: {metrics['ground_truth']}")
    print(f"estimate:    {metrics['estimate']}")
    print(
        "matched poses: "
        f"{metrics['matched_poses']} / {metrics['ground_truth_poses']} "
        f"(coverage={metrics['trajectory_coverage']:.4f})"
    )
    print(
        f"alignment: {metrics['alignment_method']} "
        f"(scale={metrics['alignment_scale']:.9f})"
    )
    print(f"ATE RMSE:  {metrics['ate_rmse_m']:.6f} m")
    print(f"ATE Mean:  {metrics['ate_mean_m']:.6f} m")
    print(f"ATE Median:{metrics['ate_median_m']:.6f} m")
    if "rpet_rmse_m" in metrics:
        print(f"RPEt RMSE: {metrics['rpet_rmse_m']:.6f} m")
    if "rper_rmse_deg" in metrics:
        print(f"RPER RMSE: {metrics['rper_rmse_deg']:.6f} deg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified ATE/RPE evaluation for TUM trajectories.")
    parser.add_argument("--ground-truth", required=True, type=Path, help="Ground-truth TUM trajectory.")
    parser.add_argument("--estimated", required=True, type=Path, help="Estimated TUM trajectory.")
    parser.add_argument(
        "--alignment",
        choices=("se3", "sim3", "origin", "none", "all"),
        default="se3",
        help="Trajectory alignment protocol. Default: se3.",
    )
    parser.add_argument("--max-diff", type=float, default=0.03, help="Timestamp matching threshold in seconds.")
    parser.add_argument("--rpe-delta", type=int, default=1, help="Pose step for RPE.")
    parser.add_argument("--json-out", type=Path, help="Optional JSON output path.")
    parser.add_argument("--text-out", type=Path, help="Optional text output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alignments: Iterable[str] = ("se3", "sim3", "origin") if args.alignment == "all" else (args.alignment,)
    results = [
        evaluate(args.ground_truth, args.estimated, alignment, args.max_diff, args.rpe_delta)
        for alignment in alignments
    ]

    text_blocks: List[str] = []
    for idx, metrics in enumerate(results):
        if idx:
            text_blocks.append("")
        lines: List[str] = []
        # Capture text without adding another dependency.
        lines.append(f"groundtruth: {metrics['ground_truth']}")
        lines.append(f"estimate:    {metrics['estimate']}")
        lines.append(
            "matched poses: "
            f"{metrics['matched_poses']} / {metrics['ground_truth_poses']} "
            f"(coverage={metrics['trajectory_coverage']:.4f})"
        )
        lines.append(
            f"alignment: {metrics['alignment_method']} "
            f"(scale={metrics['alignment_scale']:.9f})"
        )
        lines.append(f"ATE RMSE:  {metrics['ate_rmse_m']:.6f} m")
        lines.append(f"ATE Mean:  {metrics['ate_mean_m']:.6f} m")
        lines.append(f"ATE Median:{metrics['ate_median_m']:.6f} m")
        if "rpet_rmse_m" in metrics:
            lines.append(f"RPEt RMSE: {metrics['rpet_rmse_m']:.6f} m")
        if "rper_rmse_deg" in metrics:
            lines.append(f"RPER RMSE: {metrics['rper_rmse_deg']:.6f} deg")
        text_blocks.append("\n".join(lines))

    text = "\n".join(text_blocks)
    print(text)

    if args.text_out:
        args.text_out.parent.mkdir(parents=True, exist_ok=True)
        args.text_out.write_text(text + "\n", encoding="utf-8")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload: object = results[0] if len(results) == 1 else {"results": results}
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
