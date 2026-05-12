#!/usr/bin/env python3
# coding=utf-8

import argparse

import numpy as np


def load_tum(path):
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    return data[:, 0], data[:, 1:4]


def associate(gt_times, gt_xyz, est_times, est_xyz, max_diff):
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


def compute_metrics(pairs):
    gt = np.array([p[0] for p in pairs], dtype=np.float64)
    est = np.array([p[1] for p in pairs], dtype=np.float64)

    gt = gt - gt[0]
    est = est - est[0]
    errors = np.linalg.norm(gt - est, axis=1)

    ate_rmse = float(np.sqrt(np.mean(errors**2)))
    ate_mean = float(np.mean(errors))

    if len(gt) < 2:
        return ate_rmse, ate_mean, 0.0

    gt_delta = gt[1:] - gt[:-1]
    est_delta = est[1:] - est[:-1]
    rpe = np.linalg.norm(gt_delta - est_delta, axis=1)
    rpe_rmse = float(np.sqrt(np.mean(rpe**2)))
    return ate_rmse, ate_mean, rpe_rmse


def main():
    parser = argparse.ArgumentParser(description="Evaluate SLAM trajectory files in TUM format.")
    parser.add_argument("--ground-truth", required=True, help="Ground truth TUM trajectory path.")
    parser.add_argument("--estimated", required=True, help="Estimated TUM trajectory path.")
    parser.add_argument("--max-diff", type=float, default=0.03, help="Maximum timestamp difference in seconds.")
    args = parser.parse_args()

    gt_times, gt_xyz = load_tum(args.ground_truth)
    est_times, est_xyz = load_tum(args.estimated)
    pairs = associate(gt_times, gt_xyz, est_times, est_xyz, args.max_diff)
    if not pairs:
        raise RuntimeError("No matched trajectory pairs were found. Check timestamps or max-diff.")

    ate_rmse, ate_mean, rpe_rmse = compute_metrics(pairs)
    print(f"Matched poses: {len(pairs)}")
    print(f"ATE RMSE: {ate_rmse:.6f} m")
    print(f"ATE Mean: {ate_mean:.6f} m")
    print(f"RPE RMSE: {rpe_rmse:.6f} m")


if __name__ == "__main__":
    main()
