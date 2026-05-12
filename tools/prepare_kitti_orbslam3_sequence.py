#!/usr/bin/env python3
"""Prepare an ORB-SLAM3 stereo_kitti sequence from KITTI raw/processed inputs.

This creates a lightweight sequence folder with:

  - times.txt
  - image_0/000000.png ...    (left images; can be raw or filtered)
  - image_1/000000.png ...    (right images; always from raw KITTI right camera)

The output is directly compatible with ORB-SLAM3 Examples/Stereo/stereo_kitti.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seq-root",
        default="/home/lj/dynamic_SLAM/datasets/kitti_tracking/0004",
        help="KITTI tracking sequence root",
    )
    parser.add_argument(
        "--processed-name",
        default="data_gmflow_gmstereo",
        help="Processed KITTI dir that provides times.txt and optional filtered left images",
    )
    parser.add_argument(
        "--left-source",
        choices=["raw", "processed"],
        default="raw",
        help="Use raw left images or processed image_0 as ORB left input",
    )
    parser.add_argument(
        "--out-name",
        default="orbslam3_raw_stereo",
        help="Output folder name under seq-root",
    )
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--link-mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def materialize(src: Path, dst: Path, mode: str) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def load_times(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    seq_root = Path(args.seq_root).resolve()
    processed = seq_root / args.processed_name
    raw_left = seq_root / "raw" / "training" / "image_02" / seq_root.name
    raw_right = seq_root / "raw" / "training" / "image_03" / seq_root.name
    processed_left = processed / "image_0"
    processed_motion = processed / "motion"
    times_path = processed / "times.txt"

    if not processed.exists():
        raise RuntimeError(f"Processed KITTI dir not found: {processed}")
    if not raw_left.exists():
        raise RuntimeError(f"Raw left dir not found: {raw_left}")
    if not raw_right.exists():
        raise RuntimeError(f"Raw right dir not found: {raw_right}")
    if not times_path.exists():
        raise RuntimeError(f"times.txt not found: {times_path}")
    if args.left_source == "processed" and not processed_left.exists():
        raise RuntimeError(f"Processed left dir not found: {processed_left}")

    out_root = seq_root / args.out_name
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    ensure_dir(out_root / "image_0")
    ensure_dir(out_root / "image_1")
    ensure_dir(out_root / "motion")

    times = load_times(times_path)
    n = len(times) if args.max_frames < 0 else min(len(times), args.max_frames)

    left_dir = processed_left if args.left_source == "processed" else raw_left

    for idx in range(n):
        name = f"{idx:06d}.png"
        left_src = left_dir / name
        right_src = raw_right / name
        if not left_src.exists():
            raise RuntimeError(f"Missing left image: {left_src}")
        if not right_src.exists():
            raise RuntimeError(f"Missing right image: {right_src}")
        materialize(left_src, out_root / "image_0" / name, args.link_mode)
        materialize(right_src, out_root / "image_1" / name, args.link_mode)
        motion_src = processed_motion / f"{idx:06d}.txt"
        if motion_src.exists():
            materialize(motion_src, out_root / "motion" / f"{idx:06d}.txt", args.link_mode)

    with (out_root / "times.txt").open("w", encoding="utf-8") as f:
        for line in times[:n]:
            f.write(line + "\n")

    summary = {
        "seq_root": str(seq_root),
        "processed_name": args.processed_name,
        "left_source": args.left_source,
        "out_root": str(out_root),
        "frames": n,
        "link_mode": args.link_mode,
    }
    print(summary)


if __name__ == "__main__":
    main()
