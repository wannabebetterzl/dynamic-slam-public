#!/usr/bin/env python3
"""Build random/static depth-dropout controls matching filtered-depth invalidation counts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2  # type: ignore
import numpy as np


DEFAULT_RAW = Path(
    "/home/lj/dynamic_SLAM/results/20260505_yoloe_sam3_maskonly_wxyz/sequence"
)
DEFAULT_FILTERED = Path(
    "/home/lj/d-drive/CODEX/basic_model_based_SLAM/experiments/"
    "20260504_yoloe_sam3_boxfallback_wxyz/sequence"
)
DEFAULT_OUTPUT = Path(
    "/home/lj/dynamic-slam-public/data/depth_dropout_controls_20260512"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-sequence", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--filtered-sequence", type=Path, default=DEFAULT_FILTERED)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260512)
    return parser.parse_args()


def read_associations(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            raise SystemExit(f"Bad association row: {raw_line}")
        rows.append(
            {
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


def read_mask(path: Path) -> np.ndarray:
    image = read_image(path, cv2.IMREAD_UNCHANGED)
    if image.ndim == 3:
        image = image[:, :, 0]
    return image > 0


def safe_symlink(src: Path, dst: Path) -> None:
    if not src.exists():
        raise SystemExit(f"Missing symlink source: {src}")
    if dst.is_symlink():
        if Path(os.readlink(dst)) == src:
            return
        dst.unlink()
    elif dst.exists():
        raise SystemExit(f"Refusing to replace non-symlink path: {dst}")
    os.symlink(src, dst)


def choose_pixels(
    rng: np.random.Generator,
    candidates: np.ndarray,
    target_count: int,
) -> tuple[np.ndarray, int]:
    selected = np.zeros(candidates.shape, dtype=bool)
    ys, xs = np.nonzero(candidates)
    available = len(xs)
    count = min(target_count, available)
    if count > 0:
        chosen = rng.choice(available, size=count, replace=False)
        selected[ys[chosen], xs[chosen]] = True
    return selected, available


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise SystemExit(f"Failed to write image: {path}")


def main() -> int:
    args = parse_args()
    raw_sequence = args.raw_sequence.resolve()
    filtered_sequence = args.filtered_sequence.resolve()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    associations = filtered_sequence / "associations.txt"
    rows = read_associations(associations)
    depth_to_rgb: dict[str, list[str]] = {}
    for row in rows:
        depth_to_rgb.setdefault(row["depth_path"], []).append(row["rgb_path"])

    variants = {
        "C_random_same_count": {
            "description": "raw RGB + raw depth with random invalidation count matched to filtered depth",
            "candidate": "raw_valid_all_pixels",
        },
        "C_static_same_count": {
            "description": "raw RGB + raw depth with same-count invalidation outside dynamic-mask union",
            "candidate": "raw_valid_outside_dynamic_union",
        },
    }

    manifest: dict[str, object] = {
        "raw_sequence": str(raw_sequence),
        "filtered_sequence": str(filtered_sequence),
        "seed": args.seed,
        "associations": str(associations),
        "unique_depth_paths": len(depth_to_rgb),
        "variants": {},
    }

    for variant, spec in variants.items():
        sequence_root = output_root / variant / "sequence"
        depth_root = sequence_root / "depth"
        sequence_root.mkdir(parents=True, exist_ok=True)
        depth_root.mkdir(parents=True, exist_ok=True)

        safe_symlink(raw_sequence / "rgb", sequence_root / "rgb")
        safe_symlink(filtered_sequence / "mask", sequence_root / "mask")
        safe_symlink(filtered_sequence / "meta", sequence_root / "meta")
        for name, src in {
            "associations.txt": associations,
            "groundtruth.txt": filtered_sequence / "groundtruth.txt",
            "rgb.txt": raw_sequence / "rgb.txt",
            "depth.txt": filtered_sequence / "depth.txt",
        }.items():
            safe_symlink(src, sequence_root / name)

        rng = np.random.default_rng(args.seed + (0 if variant.startswith("C_random") else 1000003))
        rows_out: list[dict[str, object]] = []
        total_target = 0
        total_selected = 0
        total_available = 0

        for depth_rel, rgb_rels in sorted(depth_to_rgb.items()):
            raw_depth = read_image(raw_sequence / depth_rel, cv2.IMREAD_UNCHANGED)
            filtered_depth = read_image(filtered_sequence / depth_rel, cv2.IMREAD_UNCHANGED)
            if raw_depth.shape != filtered_depth.shape:
                raise SystemExit(f"Depth shape mismatch for {depth_rel}")
            raw_valid = raw_depth > 0
            filtered_valid = filtered_depth > 0
            target = np.logical_and(raw_valid, np.logical_not(filtered_valid))
            target_count = int(target.sum())

            dynamic_union = np.zeros(raw_depth.shape, dtype=bool)
            for rgb_rel in rgb_rels:
                mask_path = filtered_sequence / "mask" / f"{Path(rgb_rel).stem}.png"
                dynamic_union |= read_mask(mask_path)

            if spec["candidate"] == "raw_valid_all_pixels":
                candidates = raw_valid
            elif spec["candidate"] == "raw_valid_outside_dynamic_union":
                candidates = np.logical_and(raw_valid, np.logical_not(dynamic_union))
            else:
                raise SystemExit(f"Unknown candidate mode: {spec['candidate']}")

            selected, available = choose_pixels(rng, candidates, target_count)
            control_depth = raw_depth.copy()
            control_depth[selected] = 0
            write_png(sequence_root / depth_rel, control_depth)

            selected_count = int(selected.sum())
            total_target += target_count
            total_selected += selected_count
            total_available += available
            rows_out.append(
                {
                    "depth_path": depth_rel,
                    "associated_rgb_count": len(rgb_rels),
                    "target_invalidated_pixels": target_count,
                    "candidate_pixels": available,
                    "selected_invalidated_pixels": selected_count,
                    "selected_inside_dynamic_union_pixels": int(np.logical_and(selected, dynamic_union).sum()),
                    "selected_outside_dynamic_union_pixels": int(
                        np.logical_and(selected, np.logical_not(dynamic_union)).sum()
                    ),
                    "selected_target_overlap_pixels": int(np.logical_and(selected, target).sum()),
                }
            )

        depth_stats_path = sequence_root / "dropout_control_depth_stats.csv"
        with depth_stats_path.open("w", encoding="utf-8") as f:
            headers = list(rows_out[0].keys())
            f.write(",".join(headers) + "\n")
            for row in rows_out:
                f.write(",".join(str(row[h]) for h in headers) + "\n")

        variant_manifest = {
            "variant": variant,
            "description": spec["description"],
            "candidate": spec["candidate"],
            "sequence_root": str(sequence_root),
            "target_invalidated_pixels": total_target,
            "selected_invalidated_pixels": total_selected,
            "candidate_pixels": total_available,
            "target_match_ratio": float(total_selected / total_target) if total_target else 0.0,
            "depth_stats": str(depth_stats_path),
        }
        (sequence_root / "variant_manifest.json").write_text(
            json.dumps(variant_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest["variants"][variant] = variant_manifest
        print(
            f"{variant}: target={total_target} selected={total_selected} "
            f"candidate={total_available} sequence={sequence_root}"
        )

    (output_root / "dropout_controls_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Output root: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
