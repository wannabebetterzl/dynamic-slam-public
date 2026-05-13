#!/usr/bin/env python3
"""Compute path diagnostics using the same timestamp association as ATE eval.

This script intentionally separates global Sim(3) scale from trajectory arc
length.  Arc length is useful for jitter/tortuosity diagnosis, but it should not
be treated as a direct replacement for Umeyama scale.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from evaluate_trajectory_ate import (
    align_positions,
    associate_by_timestamp,
    load_tum_trajectory,
)


def parse_manifest(path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    if not path.exists():
        return result
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        result[key] = value
    return result


def path_length(xyz: np.ndarray) -> float:
    if len(xyz) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))


def endpoint_displacement(xyz: np.ndarray) -> float:
    if len(xyz) < 2:
        return 0.0
    return float(np.linalg.norm(xyz[-1] - xyz[0]))


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if abs(denominator) <= 1e-12:
        return None
    return float(numerator / denominator)


def percentile(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.quantile(values, q))


def segment_ratio_stats(gt_xyz: np.ndarray, est_xyz: np.ndarray) -> Dict[str, float | int | None]:
    if len(gt_xyz) < 2 or len(est_xyz) < 2:
        return {
            "segment_count": 0,
            "segment_ratio_mean": None,
            "segment_ratio_median": None,
            "segment_ratio_p10": None,
            "segment_ratio_p90": None,
        }

    gt_steps = np.linalg.norm(np.diff(gt_xyz, axis=0), axis=1)
    est_steps = np.linalg.norm(np.diff(est_xyz, axis=0), axis=1)
    valid = gt_steps > 1e-9
    ratios = est_steps[valid] / gt_steps[valid]
    if ratios.size == 0:
        return {
            "segment_count": 0,
            "segment_ratio_mean": None,
            "segment_ratio_median": None,
            "segment_ratio_p10": None,
            "segment_ratio_p90": None,
        }
    return {
        "segment_count": int(ratios.size),
        "segment_ratio_mean": float(np.mean(ratios)),
        "segment_ratio_median": percentile(ratios, 0.50),
        "segment_ratio_p10": percentile(ratios, 0.10),
        "segment_ratio_p90": percentile(ratios, 0.90),
    }


def build_diagnostics(
    ground_truth: Path,
    estimated: Path,
    max_diff: float,
    alignments: Iterable[str],
) -> Dict[str, object]:
    gt_traj = load_tum_trajectory(ground_truth)
    est_traj = load_tum_trajectory(estimated)
    gt_xyz, est_xyz, _, _ = associate_by_timestamp(gt_traj, est_traj, max_diff)

    gt_arc = path_length(gt_xyz)
    est_arc = path_length(est_xyz)
    gt_endpoint = endpoint_displacement(gt_xyz)
    est_endpoint = endpoint_displacement(est_xyz)

    payload: Dict[str, object] = {
        "ground_truth": str(ground_truth),
        "estimated": str(estimated),
        "max_diff": max_diff,
        "ground_truth_poses": int(len(gt_traj["times"])),
        "estimated_poses": int(len(est_traj["times"])),
        "matched_poses": int(len(gt_xyz)),
        "raw": {
            "gt_arc_length_m": gt_arc,
            "est_arc_length_m": est_arc,
            "est_gt_arc_length_ratio": safe_ratio(est_arc, gt_arc),
            "gt_endpoint_displacement_m": gt_endpoint,
            "est_endpoint_displacement_m": est_endpoint,
            "est_gt_endpoint_displacement_ratio": safe_ratio(est_endpoint, gt_endpoint),
            **segment_ratio_stats(gt_xyz, est_xyz),
        },
        "alignments": {},
    }

    alignment_payload: Dict[str, object] = {}
    for alignment in alignments:
        aligned_est, _, _, scale, method = align_positions(gt_xyz, est_xyz, alignment)
        aligned_arc = path_length(aligned_est)
        aligned_endpoint = endpoint_displacement(aligned_est)
        alignment_payload[alignment] = {
            "alignment_method": method,
            "alignment_scale": float(scale),
            "aligned_est_arc_length_m": aligned_arc,
            "aligned_est_gt_arc_length_ratio": safe_ratio(aligned_arc, gt_arc),
            "aligned_est_endpoint_displacement_m": aligned_endpoint,
            "aligned_est_gt_endpoint_displacement_ratio": safe_ratio(
                aligned_endpoint, gt_endpoint
            ),
            **segment_ratio_stats(gt_xyz, aligned_est),
        }
    payload["alignments"] = alignment_payload
    return payload


def format_optional(value: object, digits: int = 6) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_text(payload: Dict[str, object]) -> str:
    raw = payload["raw"]
    assert isinstance(raw, dict)
    lines = [
        f"groundtruth: {payload['ground_truth']}",
        f"estimate:    {payload['estimated']}",
        f"matched poses: {payload['matched_poses']} / {payload['ground_truth_poses']}",
        "raw matched path:",
        f"  gt_arc={format_optional(raw['gt_arc_length_m'])} m",
        f"  est_arc={format_optional(raw['est_arc_length_m'])} m",
        f"  est/gt_arc={format_optional(raw['est_gt_arc_length_ratio'])}",
        f"  est/gt_endpoint={format_optional(raw['est_gt_endpoint_displacement_ratio'])}",
        f"  segment_ratio_median={format_optional(raw['segment_ratio_median'])}",
        f"  segment_ratio_p10_p90={format_optional(raw['segment_ratio_p10'])}-{format_optional(raw['segment_ratio_p90'])}",
        "aligned path:",
    ]
    alignments = payload["alignments"]
    assert isinstance(alignments, dict)
    for name, value in alignments.items():
        assert isinstance(value, dict)
        lines.append(
            f"  {name}: scale={format_optional(value['alignment_scale'])} "
            f"arc_ratio={format_optional(value['aligned_est_gt_arc_length_ratio'])} "
            f"endpoint_ratio={format_optional(value['aligned_est_gt_endpoint_displacement_ratio'])} "
            f"segment_median={format_optional(value['segment_ratio_median'])}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Run directory containing run_manifest.txt and CameraTrajectory.txt.")
    parser.add_argument("--ground-truth", type=Path, help="Ground-truth TUM trajectory.")
    parser.add_argument("--estimated", type=Path, help="Estimated TUM trajectory.")
    parser.add_argument("--max-diff", type=float, default=0.03)
    parser.add_argument(
        "--alignment",
        choices=("se3", "sim3", "origin", "none", "all"),
        default="all",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--text-out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ground_truth = args.ground_truth
    estimated = args.estimated

    if args.run_dir:
        manifest = parse_manifest(args.run_dir / "run_manifest.txt")
        ground_truth = ground_truth or Path(manifest["ground_truth"])
        estimated = estimated or args.run_dir / "CameraTrajectory.txt"

    if not ground_truth or not estimated:
        raise SystemExit("Provide --run-dir or both --ground-truth and --estimated.")

    alignments: Tuple[str, ...] = (
        ("se3", "sim3", "origin") if args.alignment == "all" else (args.alignment,)
    )
    payload = build_diagnostics(ground_truth, estimated, args.max_diff, alignments)
    text = render_text(payload)
    print(text)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.text_out:
        args.text_out.parent.mkdir(parents=True, exist_ok=True)
        args.text_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
