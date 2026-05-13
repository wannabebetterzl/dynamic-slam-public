#!/usr/bin/env python3
"""Prepare RGB/depth crossed sequences for early-intervention ablation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DEFAULT_RAW = Path(
    "/home/lj/dynamic_SLAM/results/20260505_yoloe_sam3_maskonly_wxyz/sequence"
)
DEFAULT_FILTERED = Path(
    "/home/lj/d-drive/CODEX/basic_model_based_SLAM/experiments/"
    "20260504_yoloe_sam3_boxfallback_wxyz/sequence"
)
DEFAULT_OUTPUT = Path(
    "/home/lj/dynamic-slam-public/data/early_intervention_ablation_20260512"
)


VARIANTS = {
    "A_raw_rgb_raw_depth": {
        "rgb": "raw",
        "depth": "raw",
        "description": "A: raw RGB + raw depth",
    },
    "B_filtered_rgb_raw_depth": {
        "rgb": "filtered",
        "depth": "raw",
        "description": "B: image-level filtered RGB + raw depth",
    },
    "C_raw_rgb_filtered_depth": {
        "rgb": "raw",
        "depth": "filtered",
        "description": "C: raw RGB + image-level filtered depth",
    },
    "D_filtered_rgb_filtered_depth": {
        "rgb": "filtered",
        "depth": "filtered",
        "description": "D: image-level filtered RGB + image-level filtered depth",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-sequence", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--filtered-sequence", type=Path, default=DEFAULT_FILTERED)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")


def source_root(name: str, raw: Path, filtered: Path) -> Path:
    return raw if name == "raw" else filtered


def safe_symlink(src: Path, dst: Path) -> None:
    require_path(src, "source")
    if dst.is_symlink():
        if Path(os.readlink(dst)) == src:
            return
        dst.unlink()
    elif dst.exists():
        raise SystemExit(f"Refusing to replace non-symlink path: {dst}")
    os.symlink(src, dst)


def read_associations(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            raise SystemExit(f"Bad association row in {path}: {raw_line}")
        rows.append((parts[1], parts[3]))
    return rows


def count_unique_paths(sequence_root: Path, associations: Path) -> dict[str, int]:
    rows = read_associations(associations)
    missing_rgb = 0
    missing_depth = 0
    rgb_paths = set()
    depth_paths = set()
    for rgb_rel, depth_rel in rows:
        rgb_path = sequence_root / rgb_rel
        depth_path = sequence_root / depth_rel
        rgb_paths.add(rgb_rel)
        depth_paths.add(depth_rel)
        missing_rgb += 0 if rgb_path.exists() else 1
        missing_depth += 0 if depth_path.exists() else 1
    return {
        "association_rows": len(rows),
        "unique_rgb": len(rgb_paths),
        "unique_depth": len(depth_paths),
        "missing_rgb": missing_rgb,
        "missing_depth": missing_depth,
    }


def main() -> int:
    args = parse_args()
    raw = args.raw_sequence.resolve()
    filtered = args.filtered_sequence.resolve()
    output_root = args.output_root

    for root, label in [(raw, "raw sequence"), (filtered, "filtered sequence")]:
        require_path(root / "rgb", f"{label} rgb")
        require_path(root / "depth", f"{label} depth")
        require_path(root / "associations.txt", f"{label} associations")
        require_path(root / "groundtruth.txt", f"{label} groundtruth")

    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "raw_sequence": str(raw),
        "filtered_sequence": str(filtered),
        "variants": {},
    }

    for name, spec in VARIANTS.items():
        sequence_root = output_root / name / "sequence"
        sequence_root.mkdir(parents=True, exist_ok=True)
        rgb_root = source_root(spec["rgb"], raw, filtered)
        depth_root = source_root(spec["depth"], raw, filtered)

        safe_symlink(rgb_root / "rgb", sequence_root / "rgb")
        safe_symlink(depth_root / "depth", sequence_root / "depth")

        index_links = {
            "associations.txt": filtered / "associations.txt",
            "groundtruth.txt": filtered / "groundtruth.txt",
            "rgb.txt": rgb_root / "rgb.txt",
            "depth.txt": depth_root / "depth.txt",
        }
        for dst_name, src in index_links.items():
            safe_symlink(src, sequence_root / dst_name)

        optional_links = {
            "mask": filtered / "mask",
            "meta": filtered / "meta",
        }
        for dst_name, src in optional_links.items():
            if src.exists():
                safe_symlink(src, sequence_root / dst_name)

        counts = count_unique_paths(sequence_root, sequence_root / "associations.txt")
        manifest = {
            "variant": name,
            "description": spec["description"],
            "rgb_source": str(rgb_root),
            "depth_source": str(depth_root),
            "associations_source": str(filtered / "associations.txt"),
            "groundtruth_source": str(filtered / "groundtruth.txt"),
            "counts": counts,
            "backend_policy": {
                "profile": "hybrid_sequential_semantic_only",
                "pass_mask_arg": 0,
                "mask_mode": "off",
                "force_filter_detected_dynamic_features": 0,
            },
        }
        (sequence_root / "variant_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary["variants"][name] = manifest
        print(f"{name}: {counts}")

    (output_root / "ablation_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Output root: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
