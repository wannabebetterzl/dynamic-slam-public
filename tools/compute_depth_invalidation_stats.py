#!/usr/bin/env python3
"""Measure how filtered depth differs from raw depth around dynamic masks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Iterable

import cv2  # type: ignore
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-sequence", required=True, type=Path)
    parser.add_argument("--filtered-sequence", required=True, type=Path)
    parser.add_argument("--associations", required=True, type=Path)
    parser.add_argument("--mask-root", required=True, type=Path)
    parser.add_argument("--boundary-radius", type=int, default=5)
    parser.add_argument("--csv-out", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    return parser.parse_args()


def read_associations(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            raise SystemExit(f"Bad association row {line_no}: {raw_line}")
        rows.append(
            {
                "frame_id": str(len(rows)),
                "rgb_t": parts[0],
                "rgb_path": parts[1],
                "depth_t": parts[2],
                "depth_path": parts[3],
            }
        )
    return rows


def read_image(path: Path, flags: int) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise SystemExit(f"Could not read image: {path}")
    return image


def binary_mask(path: Path) -> np.ndarray:
    mask = read_image(path, cv2.IMREAD_UNCHANGED)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask > 0


def mask_boundary(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not mask.any():
        return np.zeros_like(mask, dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    mask_u8 = mask.astype(np.uint8)
    dilated = cv2.dilate(mask_u8, kernel) > 0
    eroded = cv2.erode(mask_u8, kernel) > 0
    return np.logical_and(dilated, np.logical_not(eroded))


def safe_ratio(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else 0.0


def stats(values: Iterable[float]) -> dict[str, float | int]:
    clean = [float(v) for v in values]
    if not clean:
        return {"count": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(clean),
        "mean": mean(clean),
        "median": median(clean),
        "std": pstdev(clean) if len(clean) > 1 else 0.0,
        "min": min(clean),
        "max": max(clean),
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    metric_names = [
        "mask_area_ratio",
        "boundary_area_ratio",
        "raw_valid_ratio",
        "filtered_valid_ratio",
        "raw_to_invalid_ratio_all_pixels",
        "raw_to_invalid_ratio_raw_valid",
        "raw_to_invalid_inside_mask_ratio_mask_raw_valid",
        "raw_to_invalid_outside_mask_ratio_outside_raw_valid",
        "raw_to_invalid_boundary_ratio_boundary_raw_valid",
        "raw_to_changed_nonzero_ratio_raw_valid",
        "invalidated_pixel_fraction_inside_mask",
        "invalidated_pixel_fraction_boundary",
        "invalidated_pixel_fraction_outside_mask",
    ]
    summary: dict[str, object] = {
        "frames": len(rows),
        "unique_depth_paths": len({str(r["depth_path"]) for r in rows}),
        "metrics": {name: stats(float(r[name]) for r in rows) for name in metric_names},
        "totals": {},
    }
    total_keys = [
        "pixels",
        "mask_pixels",
        "boundary_pixels",
        "raw_valid_pixels",
        "filtered_valid_pixels",
        "raw_to_invalid_pixels",
        "raw_to_invalid_inside_mask_pixels",
        "raw_to_invalid_outside_mask_pixels",
        "raw_to_invalid_boundary_pixels",
        "raw_to_changed_nonzero_pixels",
    ]
    totals = {key: int(sum(int(r[key]) for r in rows)) for key in total_keys}
    totals["raw_to_invalid_ratio_all_pixels"] = safe_ratio(
        totals["raw_to_invalid_pixels"], totals["pixels"]
    )
    totals["raw_to_invalid_ratio_raw_valid"] = safe_ratio(
        totals["raw_to_invalid_pixels"], totals["raw_valid_pixels"]
    )
    totals["inside_mask_share_of_invalidated"] = safe_ratio(
        totals["raw_to_invalid_inside_mask_pixels"], totals["raw_to_invalid_pixels"]
    )
    totals["boundary_share_of_invalidated"] = safe_ratio(
        totals["raw_to_invalid_boundary_pixels"], totals["raw_to_invalid_pixels"]
    )
    totals["outside_mask_share_of_invalidated"] = safe_ratio(
        totals["raw_to_invalid_outside_mask_pixels"], totals["raw_to_invalid_pixels"]
    )
    summary["totals"] = totals
    return summary


def main() -> int:
    args = parse_args()
    rows = read_associations(args.associations)
    output_rows: list[dict[str, object]] = []

    for row in rows:
        raw_depth_path = args.raw_sequence / row["depth_path"]
        filtered_depth_path = args.filtered_sequence / row["depth_path"]
        mask_path = args.mask_root / f"{Path(row['rgb_path']).stem}.png"

        raw_depth = read_image(raw_depth_path, cv2.IMREAD_UNCHANGED)
        filtered_depth = read_image(filtered_depth_path, cv2.IMREAD_UNCHANGED)
        mask = binary_mask(mask_path)
        if raw_depth.shape != filtered_depth.shape or raw_depth.shape != mask.shape:
            raise SystemExit(
                "Shape mismatch: "
                f"raw={raw_depth_path} {raw_depth.shape}, "
                f"filtered={filtered_depth_path} {filtered_depth.shape}, "
                f"mask={mask_path} {mask.shape}"
            )

        boundary = mask_boundary(mask, args.boundary_radius)
        raw_valid = raw_depth > 0
        filtered_valid = filtered_depth > 0
        raw_to_invalid = np.logical_and(raw_valid, np.logical_not(filtered_valid))
        raw_to_changed_nonzero = np.logical_and(
            np.logical_and(raw_valid, filtered_valid), raw_depth != filtered_depth
        )
        outside_mask = np.logical_not(mask)

        pixels = int(raw_depth.size)
        mask_pixels = int(mask.sum())
        boundary_pixels = int(boundary.sum())
        raw_valid_pixels = int(raw_valid.sum())
        filtered_valid_pixels = int(filtered_valid.sum())
        raw_to_invalid_pixels = int(raw_to_invalid.sum())
        raw_to_invalid_inside_mask_pixels = int(np.logical_and(raw_to_invalid, mask).sum())
        raw_to_invalid_outside_mask_pixels = int(
            np.logical_and(raw_to_invalid, outside_mask).sum()
        )
        raw_to_invalid_boundary_pixels = int(np.logical_and(raw_to_invalid, boundary).sum())
        raw_to_changed_nonzero_pixels = int(raw_to_changed_nonzero.sum())

        mask_raw_valid = int(np.logical_and(raw_valid, mask).sum())
        outside_raw_valid = int(np.logical_and(raw_valid, outside_mask).sum())
        boundary_raw_valid = int(np.logical_and(raw_valid, boundary).sum())

        output_rows.append(
            {
                "frame_id": row["frame_id"],
                "rgb_t": row["rgb_t"],
                "depth_t": row["depth_t"],
                "rgb_path": row["rgb_path"],
                "depth_path": row["depth_path"],
                "pixels": pixels,
                "mask_pixels": mask_pixels,
                "boundary_pixels": boundary_pixels,
                "raw_valid_pixels": raw_valid_pixels,
                "filtered_valid_pixels": filtered_valid_pixels,
                "raw_to_invalid_pixels": raw_to_invalid_pixels,
                "raw_to_invalid_inside_mask_pixels": raw_to_invalid_inside_mask_pixels,
                "raw_to_invalid_outside_mask_pixels": raw_to_invalid_outside_mask_pixels,
                "raw_to_invalid_boundary_pixels": raw_to_invalid_boundary_pixels,
                "raw_to_changed_nonzero_pixels": raw_to_changed_nonzero_pixels,
                "mask_area_ratio": safe_ratio(mask_pixels, pixels),
                "boundary_area_ratio": safe_ratio(boundary_pixels, pixels),
                "raw_valid_ratio": safe_ratio(raw_valid_pixels, pixels),
                "filtered_valid_ratio": safe_ratio(filtered_valid_pixels, pixels),
                "raw_to_invalid_ratio_all_pixels": safe_ratio(raw_to_invalid_pixels, pixels),
                "raw_to_invalid_ratio_raw_valid": safe_ratio(
                    raw_to_invalid_pixels, raw_valid_pixels
                ),
                "raw_to_invalid_inside_mask_ratio_mask_raw_valid": safe_ratio(
                    raw_to_invalid_inside_mask_pixels, mask_raw_valid
                ),
                "raw_to_invalid_outside_mask_ratio_outside_raw_valid": safe_ratio(
                    raw_to_invalid_outside_mask_pixels, outside_raw_valid
                ),
                "raw_to_invalid_boundary_ratio_boundary_raw_valid": safe_ratio(
                    raw_to_invalid_boundary_pixels, boundary_raw_valid
                ),
                "raw_to_changed_nonzero_ratio_raw_valid": safe_ratio(
                    raw_to_changed_nonzero_pixels, raw_valid_pixels
                ),
                "invalidated_pixel_fraction_inside_mask": safe_ratio(
                    raw_to_invalid_inside_mask_pixels, raw_to_invalid_pixels
                ),
                "invalidated_pixel_fraction_boundary": safe_ratio(
                    raw_to_invalid_boundary_pixels, raw_to_invalid_pixels
                ),
                "invalidated_pixel_fraction_outside_mask": safe_ratio(
                    raw_to_invalid_outside_mask_pixels, raw_to_invalid_pixels
                ),
            }
        )

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "raw_sequence": str(args.raw_sequence),
        "filtered_sequence": str(args.filtered_sequence),
        "associations": str(args.associations),
        "mask_root": str(args.mask_root),
        "boundary_radius": args.boundary_radius,
        **summarize(output_rows),
    }
    args.json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    totals = summary["totals"]
    metrics = summary["metrics"]
    print(f"frames={summary['frames']} unique_depth_paths={summary['unique_depth_paths']}")
    print(f"mask_area_mean={metrics['mask_area_ratio']['mean']:.6f}")
    print(f"raw_valid_mean={metrics['raw_valid_ratio']['mean']:.6f}")
    print(f"filtered_valid_mean={metrics['filtered_valid_ratio']['mean']:.6f}")
    print(f"raw_to_invalid/raw_valid={totals['raw_to_invalid_ratio_raw_valid']:.6f}")
    print(f"invalidated_inside_mask_share={totals['inside_mask_share_of_invalidated']:.6f}")
    print(f"invalidated_boundary_share={totals['boundary_share_of_invalidated']:.6f}")
    print(f"invalidated_outside_mask_share={totals['outside_mask_share_of_invalidated']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
