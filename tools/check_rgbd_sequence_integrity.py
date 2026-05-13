#!/usr/bin/env python3
"""Check RGB-D association, mask side-channel, and GT timestamp integrity."""

from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-root", required=True, type=Path)
    parser.add_argument("--associations", required=True, type=Path)
    parser.add_argument("--mask-root", type=Path)
    parser.add_argument("--groundtruth", type=Path)
    parser.add_argument("--max-time-diff", type=float, default=0.03)
    parser.add_argument("--json-out", "--out", dest="json_out", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--skip-depth-stats",
        action="store_true",
        help="Skip reading depth images for valid-pixel statistics.",
    )
    return parser.parse_args()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_associations(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            rows.append({"line_no": line_no, "parse_error": raw_line})
            continue
        try:
            rgb_t = float(parts[0])
            depth_t = float(parts[2])
        except ValueError:
            rows.append({"line_no": line_no, "parse_error": raw_line})
            continue
        rows.append(
            {
                "line_no": line_no,
                "rgb_t": rgb_t,
                "rgb_path": parts[1],
                "depth_t": depth_t,
                "depth_path": parts[3],
            }
        )
    return rows


def read_timestamps(path: Path | None) -> list[float]:
    if not path or not path.exists():
        return []
    timestamps: list[float] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            timestamps.append(float(line.split()[0]))
        except (ValueError, IndexError):
            continue
    return sorted(timestamps)


def nearest_abs_diff(sorted_values: list[float], value: float) -> float | None:
    if not sorted_values:
        return None
    pos = bisect.bisect_left(sorted_values, value)
    candidates = []
    if pos < len(sorted_values):
        candidates.append(abs(sorted_values[pos] - value))
    if pos > 0:
        candidates.append(abs(sorted_values[pos - 1] - value))
    return min(candidates) if candidates else None


def numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(clean),
        "min": min(clean),
        "max": max(clean),
        "mean": mean(clean),
        "median": median(clean),
    }


def depth_valid_ratio(path: Path) -> float | None:
    try:
        import cv2  # type: ignore
    except ImportError:
        return None

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        return None
    return float((image > 0).sum()) / float(image.size)


def summarize(args: argparse.Namespace) -> tuple[dict[str, object], list[str]]:
    rows = read_associations(args.associations)
    parse_errors = [r for r in rows if "parse_error" in r]
    valid_rows = [r for r in rows if "parse_error" not in r]
    sequence_root = args.sequence_root

    rgb_paths = [resolve_path(sequence_root, str(r["rgb_path"])) for r in valid_rows]
    depth_paths = [resolve_path(sequence_root, str(r["depth_path"])) for r in valid_rows]
    missing_rgb = [str(p) for p in rgb_paths if not p.exists()]
    missing_depth = [str(p) for p in depth_paths if not p.exists()]
    rgb_depth_diffs = [abs(float(r["rgb_t"]) - float(r["depth_t"])) for r in valid_rows]
    large_rgb_depth_diffs = [v for v in rgb_depth_diffs if v > args.max_time_diff]

    mask_missing: list[str] = []
    if args.mask_root:
        for rgb_path in rgb_paths:
            expected = args.mask_root / f"{rgb_path.stem}.png"
            if not expected.exists():
                mask_missing.append(str(expected))

    gt_timestamps = read_timestamps(args.groundtruth)
    gt_diffs = [
        nearest_abs_diff(gt_timestamps, float(r["rgb_t"]))
        for r in valid_rows
        if gt_timestamps
    ]
    gt_diffs_clean = [v for v in gt_diffs if v is not None]
    large_gt_diffs = [v for v in gt_diffs_clean if v > args.max_time_diff]

    depth_ratios: list[float] = []
    depth_ratio_unreadable = 0
    if not args.skip_depth_stats:
        for path in sorted(set(depth_paths)):
            if not path.exists():
                continue
            ratio = depth_valid_ratio(path)
            if ratio is None:
                depth_ratio_unreadable += 1
            else:
                depth_ratios.append(ratio)

    summary: dict[str, object] = {
        "sequence_root": str(sequence_root),
        "associations": str(args.associations),
        "association_rows": len(rows),
        "valid_association_rows": len(valid_rows),
        "parse_error_rows": len(parse_errors),
        "parse_error_examples": parse_errors[:5],
        "rgb": {
            "unique_paths": len(set(rgb_paths)),
            "missing_count": len(missing_rgb),
            "missing_examples": missing_rgb[:10],
        },
        "depth": {
            "unique_paths": len(set(depth_paths)),
            "duplicate_reuse_count": len(depth_paths) - len(set(depth_paths)),
            "missing_count": len(missing_depth),
            "missing_examples": missing_depth[:10],
            "valid_ratio": numeric_summary(depth_ratios),
            "valid_ratio_unreadable_count": depth_ratio_unreadable,
        },
        "mask": {
            "mask_root": str(args.mask_root) if args.mask_root else None,
            "expected_count": len(rgb_paths) if args.mask_root else None,
            "missing_count": len(mask_missing) if args.mask_root else None,
            "missing_examples": mask_missing[:10],
        },
        "timestamp": {
            "rgb_depth_abs_diff": numeric_summary(rgb_depth_diffs),
            "rgb_depth_over_threshold_count": len(large_rgb_depth_diffs),
            "rgb_depth_over_threshold_examples": large_rgb_depth_diffs[:10],
            "groundtruth_abs_diff": numeric_summary(gt_diffs_clean),
            "groundtruth_over_threshold_count": len(large_gt_diffs),
            "groundtruth_over_threshold_examples": large_gt_diffs[:10],
        },
    }

    issues: list[str] = []
    if parse_errors:
        issues.append(f"parse_errors={len(parse_errors)}")
    if missing_rgb:
        issues.append(f"missing_rgb={len(missing_rgb)}")
    if missing_depth:
        issues.append(f"missing_depth={len(missing_depth)}")
    if mask_missing:
        issues.append(f"missing_masks={len(mask_missing)}")
    if large_rgb_depth_diffs:
        issues.append(f"rgb_depth_time_diff>{args.max_time_diff}={len(large_rgb_depth_diffs)}")
    if large_gt_diffs:
        issues.append(f"gt_time_diff>{args.max_time_diff}={len(large_gt_diffs)}")
    summary["issues"] = issues
    return summary, issues


def main() -> int:
    args = parse_args()
    summary, issues = summarize(args)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if issues:
        print("Integrity issues:", ", ".join(issues))
        return 1 if args.strict else 0
    print("Integrity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
