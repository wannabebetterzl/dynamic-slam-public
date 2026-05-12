#!/usr/bin/env python3
"""Render filtered KITTI left images by applying DynoSAM motion masks to image_0."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-root", default="/home/lj/dynamic_SLAM/datasets/kitti_tracking/0004")
    parser.add_argument("--src-name", required=True, help="Processed KITTI dataset containing image_0/motion.")
    parser.add_argument("--dst-name", required=True, help="Output dataset name under seq-root.")
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fill", choices=["black", "gray"], default="black")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def materialize_non_image0(src: Path, dst: Path, overwrite: bool) -> None:
    if overwrite and dst.exists():
        shutil.rmtree(dst)
    ensure_dir(dst)
    for name in ("depth", "flow", "motion"):
        ensure_dir(dst / name)
        for item in sorted((src / name).iterdir()):
            if item.is_file():
                link_or_copy(item, dst / name / item.name)
    for name in ("times.txt", "pose_gt.txt", "object_pose.txt"):
        link_or_copy(src / name, dst / name)
    ensure_dir(dst / "image_0")


def render_filtered_image(image_rgb: np.ndarray, motion: np.ndarray, fill: str) -> np.ndarray:
    out = image_rgb.copy()
    dynamic = motion > 0
    if fill == "gray":
        out[dynamic] = 127
    else:
        out[dynamic] = 0
    return out


def main() -> None:
    args = parse_args()
    seq_root = Path(args.seq_root).resolve()
    src = seq_root / args.src_name
    dst = seq_root / args.dst_name
    if not src.exists():
        raise RuntimeError(f"Source dataset not found: {src}")

    materialize_non_image0(src, dst, args.overwrite)

    frames = sorted((src / "image_0").glob("*.png"))
    n = len(frames) if args.max_frames < 0 else min(args.max_frames, len(frames))
    changed = 0
    changed_ratios: list[float] = []

    for frame in range(n):
        image_path = src / "image_0" / f"{frame:06d}.png"
        motion_path = src / "motion" / f"{frame:06d}.txt"
        image_rgb = np.array(Image.open(image_path).convert("RGB"))
        if motion_path.exists():
            motion = np.loadtxt(motion_path, dtype=np.int32)
        else:
            motion = np.zeros(image_rgb.shape[:2], dtype=np.int32)
        if motion.shape != image_rgb.shape[:2]:
            raise RuntimeError(
                f"Motion/image shape mismatch at frame {frame:06d}: {motion.shape} vs {image_rgb.shape[:2]}"
            )
        filtered = render_filtered_image(image_rgb, motion, args.fill)
        dynamic_pixels = int(np.count_nonzero(motion))
        if dynamic_pixels > 0:
            changed += 1
            changed_ratios.append(dynamic_pixels / motion.size)
        Image.fromarray(filtered).save(dst / "image_0" / f"{frame:06d}.png")
        if (frame + 1) % 25 == 0 or frame + 1 == n:
            print(f"[render] {frame + 1}/{n}")

    summary = {
        "src": str(src),
        "dst": str(dst),
        "frames": n,
        "fill": args.fill,
        "changed_frames": changed,
        "mean_changed_ratio": float(np.mean(changed_ratios)) if changed_ratios else 0.0,
        "max_changed_ratio": float(np.max(changed_ratios)) if changed_ratios else 0.0,
    }
    (dst / "filtered_image0_stats.json").write_text(
        __import__("json").dumps(summary, indent=2), encoding="utf-8"
    )
    print(summary)


if __name__ == "__main__":
    main()
