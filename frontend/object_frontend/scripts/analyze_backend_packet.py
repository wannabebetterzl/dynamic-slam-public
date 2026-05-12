#!/usr/bin/env python3
"""Analyze a dynamic-SLAM backend packet before backend integration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Check backend packet track continuity and file integrity.")
    parser.add_argument("--packet", required=True, help="Backend packet directory.")
    parser.add_argument("--out", default="", help="Optional JSON output path.")
    parser.add_argument("--jump-threshold-m", type=float, default=0.20, help="World-centroid step threshold.")
    args = parser.parse_args()

    packet = Path(args.packet)
    observations = json.loads((packet / "object_observations.json").read_text(encoding="utf-8"))
    frames = json.loads((packet / "frames.json").read_text(encoding="utf-8"))
    tracks: dict[int, list[dict]] = {}
    missing_clouds = []
    for item in observations:
        object_id = int(item["object_id"])
        tracks.setdefault(object_id, []).append(item)
        cloud_path = item.get("point_cloud_path")
        if cloud_path and not Path(cloud_path).is_file():
            missing_clouds.append(cloud_path)

    track_reports = []
    for object_id, records in sorted(tracks.items()):
        records = sorted(records, key=lambda item: int(item["frame_id"]))
        worlds = [
            np.asarray(item["centroid_world"], dtype=np.float64)
            for item in records
            if item.get("centroid_world") is not None
        ]
        if len(worlds) > 1:
            world_arr = np.asarray(worlds, dtype=np.float64)
            steps = np.linalg.norm(world_arr[1:] - world_arr[:-1], axis=1)
        else:
            steps = np.asarray([], dtype=np.float64)
        jumps = []
        for i, step in enumerate(steps):
            if float(step) >= args.jump_threshold_m:
                jumps.append(
                    {
                        "from_frame": int(records[i]["frame_id"]),
                        "to_frame": int(records[i + 1]["frame_id"]),
                        "step_m": float(step),
                    }
                )
        track_reports.append(
            {
                "object_id": object_id,
                "semantic_label": str(records[0].get("semantic_label", "")) if records else "",
                "frames": len(records),
                "observations_with_depth": int(sum(1 for item in records if int(item.get("num_depth_pixels", 0)) > 0)),
                "observations_with_pose": int(sum(1 for item in records if item.get("camera_pose_available"))),
                "mean_world_step_m": float(np.mean(steps)) if steps.size else None,
                "p90_world_step_m": float(np.percentile(steps, 90)) if steps.size else None,
                "max_world_step_m": float(np.max(steps)) if steps.size else None,
                "large_jumps": jumps,
            }
        )

    report = {
        "packet": str(packet.resolve()),
        "frames": len(frames),
        "objects": len(observations),
        "tracks": len(track_reports),
        "missing_cloud_paths": len(missing_clouds),
        "jump_threshold_m": float(args.jump_threshold_m),
        "tracks_detail": track_reports,
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

