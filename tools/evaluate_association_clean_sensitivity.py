#!/usr/bin/env python3
"""Evaluate trajectories on association-clean timestamp subsets."""

from __future__ import annotations

import argparse
import csv
import json
from bisect import bisect_left
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from evaluate_trajectory_ate import evaluate


def parse_case(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("case must be NAME=/path/to/run_dir")
    name, path = value.split("=", 1)
    return name, Path(path)


def parse_variant(value: str) -> Tuple[str, float, float]:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("variant must be NAME:RGB_DEPTH_MAX:GT_MAX")
    return parts[0], float(parts[1]), float(parts[2])


def parse_manifest(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def read_timestamps(path: Path) -> List[float]:
    times: List[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if parts:
            times.append(float(parts[0]))
    return times


def nearest_diff(sorted_times: Sequence[float], value: float) -> Optional[float]:
    if not sorted_times:
        return None
    idx = bisect_left(sorted_times, value)
    candidates: List[float] = []
    if idx < len(sorted_times):
        candidates.append(abs(sorted_times[idx] - value))
    if idx > 0:
        candidates.append(abs(sorted_times[idx - 1] - value))
    return min(candidates) if candidates else None


def clean_association_timestamps(
    associations: Path,
    ground_truth: Path,
    rgb_depth_max: float,
    gt_max: float,
) -> Tuple[set[float], Dict[str, object]]:
    gt_times = sorted(read_timestamps(ground_truth))
    clean: set[float] = set()
    total = 0
    rgb_depth_bad = 0
    gt_bad = 0
    max_rgb_depth_diff = 0.0
    max_gt_diff = 0.0

    for line in associations.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 4:
            continue
        total += 1
        rgb_ts = float(parts[0])
        depth_ts = float(parts[2])
        rgb_depth_diff = abs(rgb_ts - depth_ts)
        gt_diff = nearest_diff(gt_times, rgb_ts)
        max_rgb_depth_diff = max(max_rgb_depth_diff, rgb_depth_diff)
        if gt_diff is not None:
            max_gt_diff = max(max_gt_diff, gt_diff)
        if rgb_depth_diff > rgb_depth_max:
            rgb_depth_bad += 1
            continue
        if gt_diff is None or gt_diff > gt_max:
            gt_bad += 1
            continue
        clean.add(round(rgb_ts, 6))

    return clean, {
        "association_rows": total,
        "clean_association_rows": len(clean),
        "rgb_depth_bad_rows": rgb_depth_bad,
        "gt_bad_rows": gt_bad,
        "max_rgb_depth_diff": max_rgb_depth_diff,
        "max_gt_diff": max_gt_diff,
        "rgb_depth_threshold": rgb_depth_max,
        "gt_threshold": gt_max,
    }


def filter_estimate_by_timestamps(est_path: Path, clean_timestamps: set[float], out_path: Path) -> Dict[str, int]:
    kept: List[str] = []
    total = 0
    for line in est_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        total += 1
        parts = stripped.split()
        if not parts:
            continue
        timestamp = round(float(parts[0]), 6)
        if timestamp in clean_timestamps:
            kept.append(stripped)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return {"estimate_poses": total, "filtered_estimate_poses": len(kept)}


def eval_all(gt_path: Path, est_path: Path, max_diff: float) -> Dict[str, object]:
    results = [evaluate(gt_path, est_path, alignment, max_diff, 1) for alignment in ("se3", "sim3", "origin")]
    by = {item["alignment"]: item for item in results}
    se3 = by["se3"]
    sim3 = by["sim3"]
    return {
        "matched_poses": se3.get("matched_poses", ""),
        "coverage": se3.get("trajectory_coverage", ""),
        "ate_se3_rmse_m": se3.get("ate_rmse_m", ""),
        "ate_sim3_rmse_m": sim3.get("ate_rmse_m", ""),
        "sim3_scale": sim3.get("alignment_scale", ""),
        "rpet_rmse_m": se3.get("rpet_rmse_m", ""),
        "rper_rmse_deg": se3.get("rper_rmse_deg", ""),
        "results": results,
    }


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=parse_case, required=True)
    parser.add_argument(
        "--variant",
        action="append",
        type=parse_variant,
        default=[],
        help="Subset variant as NAME:RGB_DEPTH_MAX:GT_MAX",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--max-diff", type=float, default=0.03)
    args = parser.parse_args()

    variants = [("current", None, None)] + args.variant
    rows: List[Dict[str, object]] = []
    for case_name, run_dir in args.case:
        manifest = parse_manifest(run_dir / "run_manifest.txt")
        associations = Path(manifest["associations"])
        ground_truth = Path(manifest["ground_truth"])
        estimate = run_dir / "CameraTrajectory.txt"
        for variant_name, rgb_depth_max, gt_max in variants:
            variant_dir = args.out_dir / case_name / variant_name
            if rgb_depth_max is None or gt_max is None:
                est_for_eval = estimate
                subset_stats: Dict[str, object] = {
                    "association_rows": "",
                    "clean_association_rows": "",
                    "rgb_depth_bad_rows": "",
                    "gt_bad_rows": "",
                    "max_rgb_depth_diff": "",
                    "max_gt_diff": "",
                    "rgb_depth_threshold": "",
                    "gt_threshold": "",
                    "estimate_poses": "",
                    "filtered_estimate_poses": "",
                }
            else:
                clean_ts, subset_stats = clean_association_timestamps(
                    associations, ground_truth, rgb_depth_max, gt_max
                )
                est_for_eval = variant_dir / "CameraTrajectory.filtered.txt"
                subset_stats.update(filter_estimate_by_timestamps(estimate, clean_ts, est_for_eval))

            metrics = eval_all(ground_truth, est_for_eval, args.max_diff)
            payload = metrics.pop("results")
            variant_dir.mkdir(parents=True, exist_ok=True)
            (variant_dir / "eval_unified_all.json").write_text(
                json.dumps({"results": payload}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            row: Dict[str, object] = {
                "case": case_name,
                "variant": variant_name,
                "dataset_id": manifest.get("dataset_id", ""),
                "run_dir": str(run_dir),
                "estimate": str(est_for_eval),
            }
            row.update(subset_stats)
            row.update(metrics)
            rows.append(row)

    fieldnames = [
        "case",
        "variant",
        "dataset_id",
        "association_rows",
        "clean_association_rows",
        "rgb_depth_bad_rows",
        "gt_bad_rows",
        "max_rgb_depth_diff",
        "max_gt_diff",
        "rgb_depth_threshold",
        "gt_threshold",
        "estimate_poses",
        "filtered_estimate_poses",
        "matched_poses",
        "coverage",
        "ate_se3_rmse_m",
        "ate_sim3_rmse_m",
        "sim3_scale",
        "rpet_rmse_m",
        "rper_rmse_deg",
        "run_dir",
        "estimate",
    ]
    write_csv(args.summary_out, rows, fieldnames)

    print(f"summary={args.summary_out}")
    print("| case | variant | matched | ATE-SE3 | ATE-Sim3 | scale | filtered/full |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for row in rows:
        filtered = row.get("filtered_estimate_poses", "")
        total = row.get("estimate_poses", "")
        if filtered == "":
            filtered_text = "full"
        else:
            filtered_text = f"{filtered}/{total}"
        print(
            f"| {row['case']} | {row['variant']} | {row['matched_poses']} | "
            f"{float(row['ate_se3_rmse_m']):.6f} | "
            f"{float(row['ate_sim3_rmse_m']):.6f} | "
            f"{float(row['sim3_scale']):.6f} | {filtered_text} |"
        )


if __name__ == "__main__":
    main()
