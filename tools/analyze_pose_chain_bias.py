#!/usr/bin/env python3
"""Decompose scale and pose-chain bias by matched trajectory segments."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from evaluate_trajectory_ate import align_positions, load_tum_trajectory


def parse_case(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("case must be NAME=/path/to/CameraTrajectory.txt or run_dir")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("case name cannot be empty")
    p = Path(path)
    if p.is_dir():
        p = p / "CameraTrajectory.txt"
    return name, p


def parse_range(value: str) -> Tuple[int, int]:
    if "-" not in value:
        raise argparse.ArgumentTypeError("range must be START-END")
    start_s, end_s = value.split("-", 1)
    start = int(start_s)
    end = int(end_s)
    if start < 0 or end < start:
        raise argparse.ArgumentTypeError("range must satisfy 0 <= START <= END")
    return start, end


def associate(gt: Dict[str, np.ndarray], est: Dict[str, np.ndarray], max_diff: float) -> Tuple[np.ndarray, np.ndarray]:
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


def rmse(values: Sequence[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(math.sqrt(np.mean(arr * arr)))


def path_length(xyz: np.ndarray) -> float:
    if len(xyz) < 2:
        return float("nan")
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())


def chord_length(xyz: np.ndarray) -> float:
    if len(xyz) < 2:
        return float("nan")
    return float(np.linalg.norm(xyz[-1] - xyz[0]))


def safe_ratio(num: float, den: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or abs(den) <= 1e-12:
        return float("nan")
    return num / den


def fmt(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.9f}"


def load_case(gt: Dict[str, np.ndarray], trajectory_path: Path, max_diff: float) -> Dict[str, np.ndarray | float]:
    est = load_tum_trajectory(trajectory_path)
    gt_idx, est_idx = associate(gt, est, max_diff)
    gt_xyz = gt["xyz"][gt_idx]
    est_xyz = est["xyz"][est_idx]
    se3_aligned, _, _, _, _ = align_positions(gt_xyz, est_xyz, "se3")
    sim3_aligned, _, _, sim3_scale, _ = align_positions(gt_xyz, est_xyz, "sim3")
    return {
        "gt_xyz": gt_xyz,
        "est_xyz": est_xyz,
        "se3_error": np.linalg.norm(gt_xyz - se3_aligned, axis=1),
        "sim3_error": np.linalg.norm(gt_xyz - sim3_aligned, axis=1),
        "sim3_scale": float(sim3_scale),
        "matched": float(len(est_idx)),
        "trajectory_path": str(trajectory_path),
    }


def build_rows(cases: Dict[str, Dict[str, np.ndarray | float]], ranges: Sequence[Tuple[int, int]]) -> List[Dict[str, str]]:
    first = next(iter(cases.values()))
    total_n = int(first["matched"])
    rows: List[Dict[str, str]] = []

    all_ranges = [("global", 0, total_n - 1)]
    all_ranges.extend((f"{start}-{min(end, total_n - 1)}", start, min(end, total_n - 1)) for start, end in ranges)

    for label, start, end in all_ranges:
        if start >= total_n or end < start:
            continue
        gt_xyz = first["gt_xyz"]
        assert isinstance(gt_xyz, np.ndarray)
        gt_seg = gt_xyz[start : end + 1]
        gt_path = path_length(gt_seg)
        gt_chord = chord_length(gt_seg)
        row: Dict[str, str] = {
            "segment": label,
            "match_start": str(start),
            "match_end": str(end),
            "count": str(end - start + 1),
            "gt_path_m": fmt(gt_path),
            "gt_chord_m": fmt(gt_chord),
        }
        for name, data in cases.items():
            est_xyz = data["est_xyz"]
            se3_error = data["se3_error"]
            sim3_error = data["sim3_error"]
            assert isinstance(est_xyz, np.ndarray)
            assert isinstance(se3_error, np.ndarray)
            assert isinstance(sim3_error, np.ndarray)
            est_seg = est_xyz[start : end + 1]
            est_path = path_length(est_seg)
            est_chord = chord_length(est_seg)
            row[f"{name}_se3_rmse_m"] = fmt(rmse(se3_error[start : end + 1]))
            row[f"{name}_sim3_rmse_m"] = fmt(rmse(sim3_error[start : end + 1]))
            row[f"{name}_global_sim3_scale"] = fmt(float(data["sim3_scale"]))
            row[f"{name}_est_path_m"] = fmt(est_path)
            row[f"{name}_path_ratio"] = fmt(safe_ratio(est_path, gt_path))
            row[f"{name}_est_chord_m"] = fmt(est_chord)
            row[f"{name}_chord_ratio"] = fmt(safe_ratio(est_chord, gt_chord))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--case", type=parse_case, action="append", required=True)
    parser.add_argument("--range", type=parse_range, action="append", required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--max-diff", type=float, default=0.03)
    args = parser.parse_args()

    gt = load_tum_trajectory(args.ground_truth)
    cases = {name: load_case(gt, path, args.max_diff) for name, path in args.case}
    rows = build_rows(cases, args.range)
    write_csv(args.out_csv, rows)
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
