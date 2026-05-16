#!/usr/bin/env python3
"""Summarize V14 pose-chain quality observability logs by frame segment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple


DEFAULT_SEGMENTS: Tuple[Tuple[str, int, int], ...] = (
    ("000-149", 0, 149),
    ("150-299", 150, 299),
    ("300-549", 300, 549),
    ("550-599", 550, 599),
    ("600-799", 600, 799),
    ("800-908", 800, 908),
)


def to_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def to_int(row: Dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_segments(values: Iterable[str]) -> List[Tuple[str, int, int]]:
    if not values:
        return list(DEFAULT_SEGMENTS)
    parsed: List[Tuple[str, int, int]] = []
    for item in values:
        name, span = item.split(":", 1) if ":" in item else (item, item)
        start_s, end_s = span.split("-", 1)
        parsed.append((name, int(start_s), int(end_s)))
    return parsed


def segment_rows(rows: List[Dict[str, str]], start: int, end: int) -> List[Dict[str, str]]:
    return [
        row
        for row in rows
        if start <= to_int(row, "frame_id", -1) <= end
    ]


def safe_mean(values: List[float]) -> float:
    return mean(values) if values else 0.0


def summarize_segment(name: str, start: int, end: int, rows: List[Dict[str, str]]) -> Dict[str, object]:
    if not rows:
        return {
            "segment": name,
            "start": start,
            "end": end,
            "frames": 0,
        }

    risk_rows = [row for row in rows if to_int(row, "v14_pose_chain_risk") == 1]
    keyframe_rows = [row for row in rows if to_int(row, "is_keyframe_created") == 1]
    frame_steps = [to_float(row, "estimated_frame_step_m") for row in rows]
    step_ratios = [to_float(row, "v14_step_ratio_proxy") for row in rows]
    static_inliers = [to_float(row, "v14_static_inlier_count") for row in rows]
    static_coverage = [to_float(row, "v14_static_inlier_grid_coverage") for row in rows]
    boundary_frac = [to_float(row, "v14_boundary_inlier_frac") for row in rows]
    accum_start = to_float(rows[0], "estimated_accum_path_m")
    accum_end = to_float(rows[-1], "estimated_accum_path_m")

    return {
        "segment": name,
        "start": start,
        "end": end,
        "frames": len(rows),
        "risk_frames": len(risk_rows),
        "risk_rate": len(risk_rows) / max(1, len(rows)),
        "support_low_frames": sum(to_int(row, "v14_support_low") for row in rows),
        "motion_pressure_frames": sum(to_int(row, "v14_motion_pressure") for row in rows),
        "keyframe_pressure_frames": sum(to_int(row, "v14_keyframe_pressure") for row in rows),
        "boundary_pressure_frames": sum(to_int(row, "v14_boundary_pressure") for row in rows),
        "created_keyframes": len(keyframe_rows),
        "path_delta_m": accum_end - accum_start,
        "mean_frame_step_m": safe_mean(frame_steps),
        "max_frame_step_m": max(frame_steps) if frame_steps else 0.0,
        "mean_step_ratio_proxy": safe_mean(step_ratios),
        "max_step_ratio_proxy": max(step_ratios) if step_ratios else 0.0,
        "mean_static_inlier_count": safe_mean(static_inliers),
        "min_static_inlier_count": min(static_inliers) if static_inliers else 0.0,
        "mean_static_inlier_grid_coverage": safe_mean(static_coverage),
        "min_static_inlier_grid_coverage": min(static_coverage) if static_coverage else 0.0,
        "mean_boundary_inlier_frac": safe_mean(boundary_frac),
        "max_boundary_inlier_frac": max(boundary_frac) if boundary_frac else 0.0,
        "first_risk_frame": to_int(risk_rows[0], "frame_id", -1) if risk_rows else -1,
        "last_risk_frame": to_int(risk_rows[-1], "frame_id", -1) if risk_rows else -1,
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--segment",
        action="append",
        default=[],
        help="Segment as name:start-end or start-end. Can be repeated.",
    )
    args = parser.parse_args()

    obs_path = args.run_dir / "observability_frame_stats.csv"
    rows = read_rows(obs_path)
    segments = parse_segments(args.segment)

    summary_rows = [
        summarize_segment(name, start, end, segment_rows(rows, start, end))
        for name, start, end in segments
    ]
    risk_rows = [
        row
        for row in rows
        if to_int(row, "v14_pose_chain_risk") == 1
    ]

    write_csv(args.out_dir / "pose_chain_quality_segment_summary.csv", summary_rows)
    write_csv(args.out_dir / "pose_chain_quality_risk_frames.csv", risk_rows)


if __name__ == "__main__":
    main()
