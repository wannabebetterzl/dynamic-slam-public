#!/usr/bin/env python3
"""Offline trajectory residual comparison between two backend runs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from evaluate_trajectory_ate import (
    align_positions,
    associate_by_timestamp,
    load_tum_trajectory,
)


def parse_case(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("case must be NAME=/path/to/run_dir")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("case name cannot be empty")
    return name, Path(path)


def cumulative_path(xyz: np.ndarray) -> np.ndarray:
    result = np.zeros(len(xyz), dtype=np.float64)
    if len(xyz) > 1:
        result[1:] = np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))
    return result


def rmse(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    return float(math.sqrt(np.mean(values * values)))


def mean(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def path_ratio(est_xyz: np.ndarray, gt_xyz: np.ndarray) -> float:
    if len(est_xyz) < 2 or len(gt_xyz) < 2:
        return float("nan")
    est_len = float(np.sum(np.linalg.norm(np.diff(est_xyz, axis=0), axis=1)))
    gt_len = float(np.sum(np.linalg.norm(np.diff(gt_xyz, axis=0), axis=1)))
    if abs(gt_len) <= 1e-12:
        return float("nan")
    return est_len / gt_len


def load_aligned(run_dir: Path, gt_path: Path, max_diff: float) -> Dict[str, np.ndarray | float]:
    gt = load_tum_trajectory(gt_path)
    est = load_tum_trajectory(run_dir / "CameraTrajectory.txt")
    gt_xyz, est_xyz, _, _ = associate_by_timestamp(gt, est, max_diff)
    se3_aligned, _, _, _, _ = align_positions(gt_xyz, est_xyz, "se3")
    sim3_aligned, _, _, sim3_scale, _ = align_positions(gt_xyz, est_xyz, "sim3")
    return {
        "gt_xyz": gt_xyz,
        "est_xyz": est_xyz,
        "se3_aligned": se3_aligned,
        "sim3_aligned": sim3_aligned,
        "se3_error": np.linalg.norm(gt_xyz - se3_aligned, axis=1),
        "sim3_error": np.linalg.norm(gt_xyz - sim3_aligned, axis=1),
        "est_cum": cumulative_path(est_xyz),
        "gt_cum": cumulative_path(gt_xyz),
        "sim3_scale": float(sim3_scale),
    }


def segment_ratio(data: Dict[str, np.ndarray | float], start: int, end: int) -> float:
    est = data["est_xyz"]
    gt = data["gt_xyz"]
    assert isinstance(est, np.ndarray)
    assert isinstance(gt, np.ndarray)
    return path_ratio(est[start : end + 1], gt[start : end + 1])


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_frame_rows(
    baseline: Dict[str, np.ndarray | float],
    probe: Dict[str, np.ndarray | float],
) -> List[Dict[str, object]]:
    b_se3 = baseline["se3_error"]
    p_se3 = probe["se3_error"]
    b_sim3 = baseline["sim3_error"]
    p_sim3 = probe["sim3_error"]
    assert isinstance(b_se3, np.ndarray)
    assert isinstance(p_se3, np.ndarray)
    assert isinstance(b_sim3, np.ndarray)
    assert isinstance(p_sim3, np.ndarray)
    n = min(len(b_se3), len(p_se3))
    rows: List[Dict[str, object]] = []
    for i in range(n):
        rows.append(
            {
                "match_index": i,
                "baseline_se3_error_m": f"{b_se3[i]:.9f}",
                "probe_se3_error_m": f"{p_se3[i]:.9f}",
                "se3_delta_probe_minus_baseline_m": f"{p_se3[i] - b_se3[i]:.9f}",
                "baseline_sim3_error_m": f"{b_sim3[i]:.9f}",
                "probe_sim3_error_m": f"{p_sim3[i]:.9f}",
                "sim3_delta_probe_minus_baseline_m": f"{p_sim3[i] - b_sim3[i]:.9f}",
            }
        )
    return rows


def build_bin_rows(
    baseline: Dict[str, np.ndarray | float],
    probe: Dict[str, np.ndarray | float],
    bin_size: int,
) -> List[Dict[str, object]]:
    b_se3 = baseline["se3_error"]
    p_se3 = probe["se3_error"]
    b_sim3 = baseline["sim3_error"]
    p_sim3 = probe["sim3_error"]
    assert isinstance(b_se3, np.ndarray)
    assert isinstance(p_se3, np.ndarray)
    assert isinstance(b_sim3, np.ndarray)
    assert isinstance(p_sim3, np.ndarray)
    n = min(len(b_se3), len(p_se3))
    rows: List[Dict[str, object]] = []
    for start in range(0, n, bin_size):
        end = min(n - 1, start + bin_size - 1)
        b_se3_seg = b_se3[start : end + 1]
        p_se3_seg = p_se3[start : end + 1]
        b_sim3_seg = b_sim3[start : end + 1]
        p_sim3_seg = p_sim3[start : end + 1]
        b_ratio = segment_ratio(baseline, start, end)
        p_ratio = segment_ratio(probe, start, end)
        rows.append(
            {
                "match_start": start,
                "match_end": end,
                "count": end - start + 1,
                "baseline_se3_rmse_m": f"{rmse(b_se3_seg):.9f}",
                "probe_se3_rmse_m": f"{rmse(p_se3_seg):.9f}",
                "se3_rmse_delta_probe_minus_baseline_m": f"{rmse(p_se3_seg) - rmse(b_se3_seg):.9f}",
                "baseline_sim3_rmse_m": f"{rmse(b_sim3_seg):.9f}",
                "probe_sim3_rmse_m": f"{rmse(p_sim3_seg):.9f}",
                "sim3_rmse_delta_probe_minus_baseline_m": f"{rmse(p_sim3_seg) - rmse(b_sim3_seg):.9f}",
                "baseline_se3_mean_m": f"{mean(b_se3_seg):.9f}",
                "probe_se3_mean_m": f"{mean(p_se3_seg):.9f}",
                "baseline_path_ratio": f"{b_ratio:.9f}",
                "probe_path_ratio": f"{p_ratio:.9f}",
                "path_ratio_delta_probe_minus_baseline": f"{p_ratio - b_ratio:.9f}",
            }
        )
    return rows


def render_summary(
    baseline_name: str,
    probe_name: str,
    baseline: Dict[str, np.ndarray | float],
    probe: Dict[str, np.ndarray | float],
    bin_rows: List[Dict[str, object]],
) -> str:
    b_se3 = baseline["se3_error"]
    p_se3 = probe["se3_error"]
    b_sim3 = baseline["sim3_error"]
    p_sim3 = probe["sim3_error"]
    assert isinstance(b_se3, np.ndarray)
    assert isinstance(p_se3, np.ndarray)
    assert isinstance(b_sim3, np.ndarray)
    assert isinstance(p_sim3, np.ndarray)
    worst_se3 = sorted(
        bin_rows,
        key=lambda row: float(row["se3_rmse_delta_probe_minus_baseline_m"]),
        reverse=True,
    )[:5]
    best_sim3 = sorted(
        bin_rows,
        key=lambda row: float(row["sim3_rmse_delta_probe_minus_baseline_m"]),
    )[:5]
    lines = [
        "Offline trajectory residual comparison",
        f"baseline={baseline_name}",
        f"probe={probe_name}",
        "",
        "Global:",
        f"baseline_se3_rmse={rmse(b_se3):.9f} probe_se3_rmse={rmse(p_se3):.9f} delta={rmse(p_se3)-rmse(b_se3):.9f}",
        f"baseline_sim3_rmse={rmse(b_sim3):.9f} probe_sim3_rmse={rmse(p_sim3):.9f} delta={rmse(p_sim3)-rmse(b_sim3):.9f}",
        f"baseline_scale={baseline['sim3_scale']:.9f} probe_scale={probe['sim3_scale']:.9f} delta={probe['sim3_scale']-baseline['sim3_scale']:.9f}",
        "",
        "Worst SE3 bins for probe:",
    ]
    for row in worst_se3:
        lines.append(
            "matches={match_start}-{match_end} se3_delta={se3_rmse_delta_probe_minus_baseline_m} "
            "sim3_delta={sim3_rmse_delta_probe_minus_baseline_m} path_ratio_delta={path_ratio_delta_probe_minus_baseline}".format(**row)
        )
    lines.append("")
    lines.append("Best Sim3 bins for probe:")
    for row in best_sim3:
        lines.append(
            "matches={match_start}-{match_end} sim3_delta={sim3_rmse_delta_probe_minus_baseline_m} "
            "se3_delta={se3_rmse_delta_probe_minus_baseline_m} path_ratio_delta={path_ratio_delta_probe_minus_baseline}".format(**row)
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--baseline", type=parse_case, required=True)
    parser.add_argument("--probe", type=parse_case, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="trajectory_residual_compare")
    parser.add_argument("--bin-size", type=int, default=50)
    parser.add_argument("--max-diff", type=float, default=0.03)
    args = parser.parse_args()

    baseline_name, baseline_run = args.baseline
    probe_name, probe_run = args.probe
    baseline = load_aligned(baseline_run, args.ground_truth, args.max_diff)
    probe = load_aligned(probe_run, args.ground_truth, args.max_diff)
    frame_rows = build_frame_rows(baseline, probe)
    bin_rows = build_bin_rows(baseline, probe, args.bin_size)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.out_dir / f"{args.prefix}_frames.csv",
        frame_rows,
        list(frame_rows[0].keys()) if frame_rows else [],
    )
    write_csv(
        args.out_dir / f"{args.prefix}_bins.csv",
        bin_rows,
        list(bin_rows[0].keys()) if bin_rows else [],
    )
    summary = render_summary(baseline_name, probe_name, baseline, probe, bin_rows)
    (args.out_dir / f"{args.prefix}_summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()

