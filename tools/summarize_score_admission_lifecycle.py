#!/usr/bin/env python3
"""Summarize unique score-admitted MapPoint lifecycle diagnostics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple


def as_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, "") or default)
    except ValueError:
        return default


def as_int(row: Dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, "") or default))
    except ValueError:
        return default


def resolve_lifecycle_csv(path: Path) -> Path:
    if path.is_dir():
        return path / "score_admission_lifecycle.csv"
    return path


def parse_case(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        path = Path(spec)
        return path.parent.name or path.stem, path
    name, path = spec.split("=", 1)
    return name, Path(path)


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return mean(values) if values else 0.0


def summarize_case(case: str, csv_path: Path) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    csv_path = resolve_lifecycle_csv(csv_path)
    if not csv_path.exists():
        raise SystemExit(f"Missing lifecycle CSV for {case}: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    per_point: List[Dict[str, object]] = []
    for row in rows:
        out: Dict[str, object] = {"case": case}
        out.update(row)
        per_point.append(out)

    total = len(rows)
    sums = {
        "observations": sum(as_int(r, "observations") for r in rows),
        "pose_use_edges": sum(as_int(r, "pose_use_edges") for r in rows),
        "pose_use_inliers": sum(as_int(r, "pose_use_inliers") for r in rows),
        "lba_edges": sum(as_int(r, "lba_edges") for r in rows),
        "lba_inliers": sum(as_int(r, "lba_inliers") for r in rows),
        "lba_local_edges": sum(as_int(r, "lba_local_edges") for r in rows),
        "lba_fixed_edges": sum(as_int(r, "lba_fixed_edges") for r in rows),
    }

    weighted_lba_chi2_sum = sum(
        as_float(r, "lba_chi2_mean") * as_int(r, "lba_edges") for r in rows
    )
    weighted_pose_chi2_sum = sum(
        as_float(r, "pose_use_chi2_mean") * as_int(r, "pose_use_edges") for r in rows
    )

    summary: Dict[str, object] = {
        "case": case,
        "source": str(csv_path),
        "unique_points": total,
        "bad_points": sum(as_int(r, "is_bad") == 1 for r in rows),
        "alive_points": sum(as_int(r, "is_bad") == 0 for r in rows),
        "obs_ge2_points": sum(as_int(r, "observations") >= 2 for r in rows),
        "obs_ge3_points": sum(as_int(r, "observations") >= 3 for r in rows),
        "matured_points": sum(as_int(r, "lifecycle_matured") > 0 for r in rows),
        "prebad_points": sum(as_int(r, "lifecycle_prebad") > 0 for r in rows),
        "found_ratio_culled_points": sum(
            as_int(r, "lifecycle_found_ratio_cull") > 0 for r in rows
        ),
        "low_obs_culled_points": sum(as_int(r, "lifecycle_low_obs_cull") > 0 for r in rows),
        "v7_residual_culled_points": sum(
            as_int(r, "lifecycle_v7_residual_cull") > 0 for r in rows
        ),
        "v7_low_use_culled_points": sum(
            as_int(r, "lifecycle_v7_low_use_cull") > 0 for r in rows
        ),
        "survived_event_points": sum(as_int(r, "lifecycle_survived") > 0 for r in rows),
        "pose_use_points": sum(as_int(r, "pose_use_edges") > 0 for r in rows),
        "lba_window_points": sum(as_int(r, "lba_windows") > 0 for r in rows),
        "lba_edge_points": sum(as_int(r, "lba_edges") > 0 for r in rows),
        "observations_mean": safe_mean(as_float(r, "observations") for r in rows),
        "found_ratio_mean": safe_mean(as_float(r, "found_ratio") for r in rows),
        "pose_inlier_rate_global": (
            sums["pose_use_inliers"] / sums["pose_use_edges"]
            if sums["pose_use_edges"]
            else 0.0
        ),
        "pose_chi2_mean_weighted": (
            weighted_pose_chi2_sum / sums["pose_use_edges"] if sums["pose_use_edges"] else 0.0
        ),
        "lba_edges_per_point": sums["lba_edges"] / total if total else 0.0,
        "lba_inlier_rate_global": (
            sums["lba_inliers"] / sums["lba_edges"] if sums["lba_edges"] else 0.0
        ),
        "lba_local_edge_rate_global": (
            sums["lba_local_edges"] / sums["lba_edges"] if sums["lba_edges"] else 0.0
        ),
        "lba_fixed_edge_rate_global": (
            sums["lba_fixed_edges"] / sums["lba_edges"] if sums["lba_edges"] else 0.0
        ),
        "lba_chi2_mean_weighted": (
            weighted_lba_chi2_sum / sums["lba_edges"] if sums["lba_edges"] else 0.0
        ),
        "score_total_mean": safe_mean(as_float(r, "total_score") for r in rows),
        "geom_baseline_mean": safe_mean(as_float(r, "geom_baseline") for r in rows),
        "geom_cos_parallax_mean": safe_mean(as_float(r, "geom_cos_parallax") for r in rows),
        "geom_parallax_score_mean": safe_mean(
            as_float(r, "geom_parallax_score") for r in rows
        ),
        "geom_reproj_ratio1_mean": safe_mean(
            as_float(r, "geom_reproj_ratio1") for r in rows
        ),
        "geom_reproj_ratio2_mean": safe_mean(
            as_float(r, "geom_reproj_ratio2") for r in rows
        ),
        "geom_scale_score_mean": safe_mean(as_float(r, "geom_scale_score") for r in rows),
        "ref_distance_mean": safe_mean(as_float(r, "ref_distance") for r in rows),
    }
    summary.update({f"sum_{key}": value for key, value in sums.items()})
    return summary, per_point


def write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help="Case as name=/path/to/run_dir_or_score_admission_lifecycle.csv",
    )
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--per-point-out")
    args = parser.parse_args()

    summaries: List[Dict[str, object]] = []
    points: List[Dict[str, object]] = []
    for spec in args.case:
        case, path = parse_case(spec)
        summary, per_point = summarize_case(case, path)
        summaries.append(summary)
        points.extend(per_point)

    write_rows(Path(args.summary_out), summaries)
    if args.per_point_out:
        write_rows(Path(args.per_point_out), points)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
