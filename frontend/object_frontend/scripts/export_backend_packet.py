#!/usr/bin/env python3
"""Export a minimal dynamic-SLAM backend packet."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

from dynamic_object_frontend.backend_packet import build_backend_packet


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a backend-neutral dynamic object SLAM packet.")
    parser.add_argument("--sequence", required=True, help="Exported RGB-D sequence root containing object_associations.txt.")
    parser.add_argument("--observations", required=True, help="Object observation export root.")
    parser.add_argument("--camera-trajectory", required=True, help="TUM-format camera trajectory, usually CameraTrajectory.txt.")
    parser.add_argument("--out", required=True, help="Output packet directory.")
    parser.add_argument("--max-pose-time-diff", type=float, default=0.03, help="Max timestamp difference for frame-pose matching.")
    args = parser.parse_args()

    manifest = build_backend_packet(
        sequence_root=Path(args.sequence),
        observations_root=Path(args.observations),
        camera_trajectory=Path(args.camera_trajectory),
        output_root=Path(args.out),
        max_pose_time_diff=args.max_pose_time_diff,
    )
    print(json.dumps(manifest["validation"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

