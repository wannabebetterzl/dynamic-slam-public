#!/usr/bin/env python3
"""Check RGB-D association, mask, and ground-truth integrity for SLAM runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional local dependency
    cv2 = None
    np = None


def parse_associations(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"{path}:{line_no}: expected 4 columns, got {len(parts)}")
        rows.append(
            {
                "line": line_no,
                "rgb_time": float(parts[0]),
                "rgb_rel": parts[1],
                "depth_time": float(parts[2]),
                "depth_rel": parts[3],
            }
        )
    return rows


def parse_timestamps(path: Path | None) -> list[float]:
    if path is None or not path.exists():
        return []
    values: list[float] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            values.append(float(line.split()[0]))
        except (IndexError, ValueError):
            continue
    return sorted(values)


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "median": None, "max": None}
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    median = ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    return {
        "count": n,
        "min": ordered[0],
        "mean": sum(ordered) / n,
        "median": median,
        "max": ordered[-1],
    }


def nearest_distances(source: list[float], target: list[float]) -> list[float]:
    if not source or not target:
        return []
    distances: list[float] = []
    j = 0
    for value in source:
        while j + 1 < len(target) and abs(target[j + 1] - value) <= abs(target[j] - value):
            j += 1
        distances.append(abs(target[j] - value))
    return distances


def read_image(path: Path):
    if cv2 is None:
        return None
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def image_ratios(depth_path: Path, mask_path: Path | None) -> dict[str, float | None]:
    if cv2 is None or np is None:
        return {}

    depth = read_image(depth_path)
    if depth is None:
        return {}

    valid_depth = depth > 0
    ratios: dict[str, float | None] = {
        "depth_valid_ratio": float(np.mean(valid_depth)) if valid_depth.size else None,
    }

    if mask_path is None or not mask_path.exists():
        return ratios

    mask = read_image(mask_path)
    if mask is None:
        return ratios
    if mask.ndim == 3:
        dynamic = np.any(mask > 0, axis=2)
    else:
        dynamic = mask > 0

    if dynamic.shape != valid_depth.shape:
        ratios["mask_shape_matches_depth"] = 0.0
        return ratios

    ratios["mask_shape_matches_depth"] = 1.0
    ratios["mask_dynamic_ratio"] = float(np.mean(dynamic)) if dynamic.size else None
    ratios["mask_inside_depth_valid_ratio"] = (
        float(np.mean(valid_depth[dynamic])) if np.any(dynamic) else None
    )
    ratios["mask_outside_depth_valid_ratio"] = (
        float(np.mean(valid_depth[~dynamic])) if np.any(~dynamic) else None
    )
    return ratios


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    sequence_root = Path(args.sequence_root).expanduser()
    associations = Path(args.associations).expanduser()
    mask_root = Path(args.mask_root).expanduser() if args.mask_root else None
    groundtruth = Path(args.groundtruth).expanduser() if args.groundtruth else None

    rows = parse_associations(associations)
    rgb_depth_diffs: list[float] = []
    depth_valid_ratios: list[float] = []
    mask_dynamic_ratios: list[float] = []
    mask_inside_depth_valid_ratios: list[float] = []
    mask_outside_depth_valid_ratios: list[float] = []
    mask_shape_mismatches = 0
    missing_rgb: list[str] = []
    missing_depth: list[str] = []
    missing_mask: list[str] = []
    excessive_time_diff: list[dict[str, Any]] = []

    for row in rows:
        rgb_path = sequence_root / row["rgb_rel"]
        depth_path = sequence_root / row["depth_rel"]
        mask_path = mask_root / Path(row["rgb_rel"]).name if mask_root else None
        diff = abs(row["rgb_time"] - row["depth_time"])
        rgb_depth_diffs.append(diff)

        if not rgb_path.exists():
            missing_rgb.append(str(rgb_path))
        if not depth_path.exists():
            missing_depth.append(str(depth_path))
        if mask_path is not None and not mask_path.exists():
            missing_mask.append(str(mask_path))
        if diff > args.max_time_diff:
            excessive_time_diff.append(
                {
                    "line": row["line"],
                    "rgb_time": row["rgb_time"],
                    "depth_time": row["depth_time"],
                    "diff": diff,
                }
            )

        if depth_path.exists():
            ratios = image_ratios(depth_path, mask_path)
            if isinstance(ratios.get("depth_valid_ratio"), float):
                depth_valid_ratios.append(ratios["depth_valid_ratio"])  # type: ignore[arg-type]
            if isinstance(ratios.get("mask_dynamic_ratio"), float):
                mask_dynamic_ratios.append(ratios["mask_dynamic_ratio"])  # type: ignore[arg-type]
            if isinstance(ratios.get("mask_inside_depth_valid_ratio"), float):
                mask_inside_depth_valid_ratios.append(
                    ratios["mask_inside_depth_valid_ratio"]  # type: ignore[arg-type]
                )
            if isinstance(ratios.get("mask_outside_depth_valid_ratio"), float):
                mask_outside_depth_valid_ratios.append(
                    ratios["mask_outside_depth_valid_ratio"]  # type: ignore[arg-type]
                )
            if ratios.get("mask_shape_matches_depth") == 0.0:
                mask_shape_mismatches += 1

    assoc_times = sorted(row["rgb_time"] for row in rows)
    gt_times = parse_timestamps(groundtruth)
    gt_diffs = nearest_distances(assoc_times, gt_times)
    gt_matched = sum(1 for diff in gt_diffs if diff <= args.max_time_diff)

    samples = args.sample_limit
    return {
        "sequence_root": str(sequence_root),
        "associations": str(associations),
        "mask_root": str(mask_root) if mask_root else None,
        "groundtruth": str(groundtruth) if groundtruth else None,
        "max_time_diff": args.max_time_diff,
        "opencv_available": cv2 is not None,
        "counts": {
            "associations": len(rows),
            "missing_rgb": len(missing_rgb),
            "missing_depth": len(missing_depth),
            "missing_mask": len(missing_mask),
            "rgb_depth_time_diff_exceeds_threshold": len(excessive_time_diff),
            "mask_shape_mismatches": mask_shape_mismatches,
            "groundtruth_timestamps": len(gt_times),
            "associations_matched_to_groundtruth": gt_matched,
        },
        "ratios": {
            "groundtruth_coverage_of_associations": gt_matched / len(rows) if rows else math.nan,
            "association_coverage_of_groundtruth": gt_matched / len(gt_times) if gt_times else math.nan,
        },
        "stats": {
            "rgb_depth_time_diff": stats(rgb_depth_diffs),
            "nearest_groundtruth_time_diff": stats(gt_diffs),
            "depth_valid_ratio": stats(depth_valid_ratios),
            "mask_dynamic_ratio": stats(mask_dynamic_ratios),
            "mask_inside_depth_valid_ratio": stats(mask_inside_depth_valid_ratios),
            "mask_outside_depth_valid_ratio": stats(mask_outside_depth_valid_ratios),
        },
        "samples": {
            "missing_rgb": missing_rgb[:samples],
            "missing_depth": missing_depth[:samples],
            "missing_mask": missing_mask[:samples],
            "rgb_depth_time_diff_exceeds_threshold": excessive_time_diff[:samples],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-root", required=True)
    parser.add_argument("--associations", required=True)
    parser.add_argument("--mask-root")
    parser.add_argument("--groundtruth")
    parser.add_argument("--max-time-diff", type=float, default=0.03)
    parser.add_argument("--sample-limit", type=int, default=20)
    parser.add_argument("--out")
    args = parser.parse_args()

    report = build_report(args)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    counts = report["counts"]
    ratios = report["ratios"]
    time_stats = report["stats"]["rgb_depth_time_diff"]
    print(f"associations={counts['associations']}")
    print(
        "missing="
        f"rgb:{counts['missing_rgb']} "
        f"depth:{counts['missing_depth']} "
        f"mask:{counts['missing_mask']}"
    )
    print(
        "rgb_depth_time_diff="
        f"max:{time_stats['max']} "
        f"exceeds:{counts['rgb_depth_time_diff_exceeds_threshold']}"
    )
    print(
        "groundtruth_coverage_of_associations="
        f"{ratios['groundtruth_coverage_of_associations']:.6f}"
    )
    if args.out:
        print(f"report={args.out}")

    failed = any(
        counts[key] > 0
        for key in (
            "missing_rgb",
            "missing_depth",
            "missing_mask",
            "rgb_depth_time_diff_exceeds_threshold",
            "mask_shape_mismatches",
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
