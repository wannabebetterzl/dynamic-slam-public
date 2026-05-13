#!/usr/bin/env python3
"""Prepare a raw RGB-D sequence with frontend masks as a side-channel.

The frontend "filtered" export modifies RGB/depth images. D²MA needs the
opposite contract: keep raw RGB/depth untouched, but expose mask/meta files
that share the RGB timestamp basename used by ORB-SLAM3.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-sequence", required=True, type=Path)
    parser.add_argument("--mask-donor-sequence", required=True, type=Path)
    parser.add_argument("--output-sequence", required=True, type=Path)
    parser.add_argument(
        "--description",
        default="raw RGB/depth with frontend mask/meta side-channel",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing symlinks/files created by this tool.",
    )
    return parser.parse_args()


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")


def remove_if_allowed(dst: Path, force: bool) -> None:
    if not dst.exists() and not dst.is_symlink():
        return
    if not force:
        raise SystemExit(f"Refusing to replace existing path without --force: {dst}")
    if dst.is_dir() and not dst.is_symlink():
        raise SystemExit(f"Refusing to replace real directory: {dst}")
    dst.unlink()


def safe_symlink(src: Path, dst: Path, force: bool) -> None:
    require_path(src, "source")
    src = src.resolve()
    if dst.is_symlink():
        if Path(os.readlink(dst)) == src:
            return
        remove_if_allowed(dst, force)
    elif dst.exists():
        remove_if_allowed(dst, force)
    os.symlink(src, dst)


def read_associations(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            raise SystemExit(f"Bad association row {line_no} in {path}: {raw_line}")
        rows.append(
            {
                "rgb_time": parts[0],
                "rgb_rel": parts[1],
                "depth_time": parts[2],
                "depth_rel": parts[3],
            }
        )
    return rows


def validate_sequence(sequence_root: Path, donor: Path) -> dict[str, object]:
    rows = read_associations(sequence_root / "associations.txt")
    missing_rgb: list[str] = []
    missing_depth: list[str] = []
    missing_mask: list[str] = []
    missing_meta: list[str] = []
    mask_root = sequence_root / "mask"
    meta_root = sequence_root / "meta"

    rgb_paths: set[str] = set()
    depth_paths: set[str] = set()
    for row in rows:
        rgb_rel = row["rgb_rel"]
        depth_rel = row["depth_rel"]
        rgb_paths.add(rgb_rel)
        depth_paths.add(depth_rel)

        rgb_path = sequence_root / rgb_rel
        depth_path = sequence_root / depth_rel
        if not rgb_path.exists():
            missing_rgb.append(str(rgb_path))
        if not depth_path.exists():
            missing_depth.append(str(depth_path))

        rgb_name = Path(rgb_rel).name
        mask_path = mask_root / f"{Path(rgb_rel).stem}.png"
        if not mask_path.exists():
            missing_mask.append(str(mask_path))

        # Current frontend writes meta files as "<rgb filename>.txt".
        meta_candidates = [meta_root / f"{rgb_name}.txt", meta_root / f"{Path(rgb_rel).stem}.txt"]
        if meta_root.exists() and not any(path.exists() for path in meta_candidates):
            missing_meta.append(str(meta_candidates[0]))

    return {
        "association_rows": len(rows),
        "unique_rgb_paths": len(rgb_paths),
        "unique_depth_paths": len(depth_paths),
        "missing_rgb_count": len(missing_rgb),
        "missing_depth_count": len(missing_depth),
        "missing_mask_count": len(missing_mask),
        "missing_meta_count": len(missing_meta) if meta_root.exists() else None,
        "missing_examples": {
            "rgb": missing_rgb[:5],
            "depth": missing_depth[:5],
            "mask": missing_mask[:5],
            "meta": missing_meta[:5],
        },
        "donor_sequence": str(donor),
    }


def main() -> int:
    args = parse_args()
    raw = args.raw_sequence.resolve()
    donor = args.mask_donor_sequence.resolve()
    output = args.output_sequence

    require_path(raw / "rgb", "raw rgb directory")
    require_path(raw / "depth", "raw depth directory")
    require_path(donor / "associations.txt", "donor associations")
    require_path(donor / "groundtruth.txt", "donor groundtruth")
    require_path(donor / "mask", "donor mask directory")

    output.mkdir(parents=True, exist_ok=True)
    safe_symlink(raw / "rgb", output / "rgb", args.force)
    safe_symlink(raw / "depth", output / "depth", args.force)
    safe_symlink(donor / "associations.txt", output / "associations.txt", args.force)
    safe_symlink(donor / "groundtruth.txt", output / "groundtruth.txt", args.force)
    safe_symlink(donor / "mask", output / "mask", args.force)

    optional_links = {
        "meta": donor / "meta",
        "rgb.txt": donor / "rgb.txt",
        "depth.txt": donor / "depth.txt",
    }
    for dst_name, src in optional_links.items():
        if src.exists():
            safe_symlink(src, output / dst_name, args.force)

    counts = validate_sequence(output, donor)
    if any(counts[key] for key in ["missing_rgb_count", "missing_depth_count", "missing_mask_count"]):
        raise SystemExit(json.dumps(counts, ensure_ascii=False, indent=2))

    manifest = {
        "description": args.description,
        "raw_sequence": str(raw),
        "mask_donor_sequence": str(donor),
        "output_sequence": str(output.resolve()),
        "policy": {
            "rgb": "raw",
            "depth": "raw",
            "mask_meta": "frontend side-channel only",
        },
        "counts": counts,
    }
    (output / "sidechannel_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
