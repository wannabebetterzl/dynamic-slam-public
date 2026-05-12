#!/usr/bin/env python3
"""Export backend-ready object observations from a basic_frontend sequence."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from dynamic_object_frontend import CameraIntrinsics, DetectionRecord, build_object_observations


DEFAULT_PERSON_SEMANTIC_ID = 11


def load_associations(path: Path) -> list[tuple[float, str, float, str]]:
    rows: list[tuple[float, str, float, str]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 4:
            rows.append((float(parts[0]), parts[1], float(parts[2]), parts[3]))
    return rows


def load_object_associations(path: Path) -> dict[str, dict[str, str | None]]:
    mapping: dict[str, dict[str, str | None]] = {}
    if not path.exists():
        return mapping
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 7:
            rgb_rel = parts[1]
            raw_depth_rel = parts[4]
            instance_mask_rel = parts[7] if len(parts) >= 8 and parts[7] else None
            mapping[rgb_rel] = {"raw_depth": raw_depth_rel, "instance_mask": instance_mask_rel}
    return mapping


def parse_meta(path: Path, semantic_id: int) -> list[DetectionRecord]:
    records: list[DetectionRecord] = []
    if not path.exists():
        return records
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 9:
            continue
        try:
            track_id = int(parts[0])
            if track_id <= 0:
                continue
            dynamic_score = float(parts[5])
            temporal_consistency = float(parts[6])
            geometry_dynamic_score = float(parts[7])
            extra = [float(value) for value in parts[9:19]]
            while len(extra) < 10:
                extra.append(0.0)
            held_track = bool(int(float(parts[19]))) if len(parts) > 19 else False
            hold_misses = int(float(parts[20])) if len(parts) > 20 else 0
            records.append(
                DetectionRecord(
                    object_id=track_id,
                    semantic_id=semantic_id,
                    semantic_label="person",
                    bbox_2d=(int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])),
                    dynamic_score=dynamic_score,
                    temporal_consistency=temporal_consistency,
                    geometry_dynamic_score=geometry_dynamic_score,
                    filter_out=bool(int(parts[8])),
                    confidence=max(0.0, min(1.0, 0.5 * dynamic_score + 0.5 * temporal_consistency)),
                    match_score=extra[0],
                    association_bbox_iou=extra[1],
                    association_mask_iou=extra[2],
                    association_appearance=extra[3],
                    association_depth=extra[4],
                    association_id_match=extra[5],
                    temporal_fusion_score=extra[6],
                    temporal_id_consistency=extra[7],
                    temporal_mask_agreement=extra[8],
                    temporal_box_agreement=extra[9],
                    held_track=held_track,
                    hold_misses=hold_misses,
                )
            )
        except ValueError:
            continue
    return records


def default_tum3_intrinsics(depth_scale: float) -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=535.4,
        fy=539.2,
        cx=320.1,
        cy=247.6,
        depth_scale=depth_scale,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True, type=Path)
    parser.add_argument(
        "--dynamic-depth-sequence",
        type=Path,
        default=None,
        help="Optional raw RGB-D sequence root used only for object depth.",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--semantic-id", type=int, default=DEFAULT_PERSON_SEMANTIC_ID)
    parser.add_argument("--depth-scale", type=float, default=5000.0)
    parser.add_argument("--max-depth-m", type=float, default=8.0)
    parser.add_argument("--max-points-per-object", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    association_path = args.sequence / "associations.txt"
    rows = load_associations(association_path)
    object_inputs_by_rgb = load_object_associations(args.sequence / "object_associations.txt")
    if args.limit > 0:
        rows = rows[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    frames_dir = args.out / "frames"
    clouds_dir = args.out / "clouds"
    frames_dir.mkdir(parents=True, exist_ok=True)
    clouds_dir.mkdir(parents=True, exist_ok=True)
    intrinsics = default_tum3_intrinsics(args.depth_scale)

    all_summary = []
    for frame_id, (rgb_ts, rgb_rel, _depth_ts, depth_rel) in enumerate(rows):
        rgb_name = Path(rgb_rel).name
        if args.dynamic_depth_sequence:
            depth_root = args.dynamic_depth_sequence
            object_depth_rel = depth_rel
        else:
            depth_root = args.sequence
            object_inputs = object_inputs_by_rgb.get(rgb_rel, {})
            object_depth_rel = object_inputs.get("raw_depth") or depth_rel
        depth = cv2.imread(str(depth_root / object_depth_rel), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(args.sequence / "mask" / rgb_name), cv2.IMREAD_UNCHANGED)
        instance_mask = None
        instance_mask_rel = object_inputs_by_rgb.get(rgb_rel, {}).get("instance_mask")
        if instance_mask_rel:
            instance_mask = cv2.imread(str(args.sequence / instance_mask_rel), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(depth_root / object_depth_rel)
        if mask is None:
            mask = np.zeros(depth.shape[:2], dtype=np.uint8)
        detections = parse_meta(args.sequence / "meta" / f"{rgb_name}.txt", args.semantic_id)
        prefix = f"{frame_id:06d}_{Path(rgb_name).stem}"
        observations, clouds = build_object_observations(
            frame_id=frame_id,
            timestamp=rgb_ts,
            depth=depth,
            binary_mask=mask,
            detections=detections,
            intrinsics=intrinsics,
            instance_mask=instance_mask,
            max_depth_m=args.max_depth_m,
            max_points_per_object=args.max_points_per_object,
            point_cloud_prefix=f"clouds/{prefix}",
        )
        for obs in observations:
            if obs.point_cloud_file and obs.object_id in clouds:
                np.save(args.out / obs.point_cloud_file, clouds[obs.object_id])
        frame_payload = {
            "frame_id": frame_id,
            "timestamp": rgb_ts,
            "rgb": rgb_rel,
                "depth": depth_rel,
                "dynamic_depth_root": str(depth_root),
                "object_depth": object_depth_rel,
            "intrinsics": asdict(intrinsics),
            "instance_mask": instance_mask_rel,
            "objects": [obs.to_json_dict() for obs in observations],
        }
        (frames_dir / f"{prefix}.json").write_text(json.dumps(frame_payload, indent=2), encoding="utf-8")
        all_summary.append(
            {
                "frame_id": frame_id,
                "timestamp": rgb_ts,
                "objects": len(observations),
                "objects_with_depth": sum(1 for obs in observations if obs.num_depth_pixels > 0),
                "total_depth_points": sum(obs.num_depth_pixels for obs in observations),
            }
        )
    summary = {
        "sequence": str(args.sequence),
        "dynamic_depth_sequence": str(args.dynamic_depth_sequence) if args.dynamic_depth_sequence else str(args.sequence),
        "frames": len(all_summary),
        "objects": sum(item["objects"] for item in all_summary),
        "objects_with_depth": sum(item["objects_with_depth"] for item in all_summary),
        "total_depth_points": sum(item["total_depth_points"] for item in all_summary),
        "frames_detail": all_summary,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "frames_detail"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
