"""Build a minimal dynamic-backend packet from frontend observations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math
import shutil

import numpy as np


@dataclass(frozen=True)
class TumPose:
    timestamp: float
    translation: np.ndarray
    quaternion_xyzw: np.ndarray

    @property
    def rotation(self) -> np.ndarray:
        return quaternion_xyzw_to_matrix(self.quaternion_xyzw)

    def transform_camera_to_world(self, point_camera: np.ndarray) -> np.ndarray:
        return self.rotation @ point_camera.reshape(3) + self.translation.reshape(3)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "timestamp": float(self.timestamp),
            "translation": [float(x) for x in self.translation.reshape(3)],
            "quaternion_xyzw": [float(x) for x in self.quaternion_xyzw.reshape(4)],
        }


def quaternion_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in q.reshape(4)]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        raise ValueError("zero-norm quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def load_tum_trajectory(path: Path) -> list[TumPose]:
    poses: list[TumPose] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            values = [float(v) for v in parts[:8]]
            poses.append(
                TumPose(
                    timestamp=values[0],
                    translation=np.asarray(values[1:4], dtype=np.float64),
                    quaternion_xyzw=np.asarray(values[4:8], dtype=np.float64),
                )
            )
    return sorted(poses, key=lambda pose: pose.timestamp)


def match_pose(timestamp: float, poses: list[TumPose], max_diff: float) -> tuple[TumPose | None, float | None]:
    if not poses:
        return None, None
    times = np.asarray([pose.timestamp for pose in poses], dtype=np.float64)
    idx = int(np.argmin(np.abs(times - float(timestamp))))
    diff = abs(float(times[idx]) - float(timestamp))
    if diff > max_diff:
        return None, diff
    return poses[idx], diff


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_object_associations(sequence_root: Path) -> list[dict[str, Any]]:
    path = sequence_root / "object_associations.txt"
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for frame_id, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                raise ValueError(f"Malformed object association line: {line}")
            rows.append(
                {
                    "frame_id": len(rows),
                    "timestamp": float(parts[0]),
                    "rgb": parts[1],
                    "filtered_depth_timestamp": float(parts[2]),
                    "filtered_depth": parts[3],
                    "raw_depth": parts[4],
                    "mask": parts[5],
                    "meta": parts[6],
                    "instance_mask": parts[7] if len(parts) >= 8 and parts[7] else None,
                }
            )
    return rows


def _as_point(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _world_bbox(camera_min: Any, camera_max: Any, pose: TumPose | None) -> tuple[list[float] | None, list[float] | None]:
    if pose is None:
        return None, None
    pmin = _as_point(camera_min)
    pmax = _as_point(camera_max)
    if pmin is None or pmax is None:
        return None, None
    corners = []
    for x in (pmin[0], pmax[0]):
        for y in (pmin[1], pmax[1]):
            for z in (pmin[2], pmax[2]):
                corners.append(pose.transform_camera_to_world(np.asarray([x, y, z], dtype=np.float64)))
    arr = np.asarray(corners, dtype=np.float64)
    return arr.min(axis=0).tolist(), arr.max(axis=0).tolist()


def _copy_if_exists(src: Path, dst: Path) -> str | None:
    if not src.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst.name


def build_backend_packet(
    *,
    sequence_root: Path,
    observations_root: Path,
    camera_trajectory: Path,
    output_root: Path,
    max_pose_time_diff: float = 0.03,
    copy_camera_trajectory: bool = True,
) -> dict[str, Any]:
    """Create a backend-neutral packet for dynamic object SLAM integration.

    The packet intentionally keeps static-SLAM imagery/depth and dynamic-object
    raw-depth observations separate. Object point clouds are referenced instead
    of copied to keep smoke packets small.
    """

    sequence_root = sequence_root.resolve()
    observations_root = observations_root.resolve()
    camera_trajectory = camera_trajectory.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    poses = load_tum_trajectory(camera_trajectory)
    associations = load_object_associations(sequence_root)
    frame_payloads = [load_json(path) for path in sorted((observations_root / "frames").glob("*.json"))]
    frame_by_id = {int(payload["frame_id"]): payload for payload in frame_payloads}

    frames: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    tracks: dict[int, list[dict[str, Any]]] = {}
    missing_pose_frames = 0
    missing_depth_observations = 0
    max_pose_time_error = 0.0

    for assoc in associations:
        frame_id = int(assoc["frame_id"])
        timestamp = float(assoc["timestamp"])
        pose, pose_dt = match_pose(timestamp, poses, max_pose_time_diff)
        if pose is None:
            missing_pose_frames += 1
        if pose_dt is not None:
            max_pose_time_error = max(max_pose_time_error, float(pose_dt))
        payload = frame_by_id.get(frame_id, {"objects": []})
        frame_record = {
            **assoc,
            "camera_pose_twc": pose.to_json_dict() if pose is not None else None,
            "camera_pose_time_error_sec": float(pose_dt) if pose_dt is not None else None,
            "num_objects": len(payload.get("objects", [])),
        }
        frames.append(frame_record)

        for obj in payload.get("objects", []):
            centroid_camera = _as_point(obj.get("centroid_camera"))
            centroid_world = None
            if centroid_camera is not None and pose is not None:
                centroid_world = pose.transform_camera_to_world(centroid_camera).tolist()
            if int(obj.get("num_depth_pixels", 0)) <= 0:
                missing_depth_observations += 1
            bbox_world_min, bbox_world_max = _world_bbox(
                obj.get("bbox_3d_camera_min"),
                obj.get("bbox_3d_camera_max"),
                pose,
            )
            cloud_rel = obj.get("point_cloud_file")
            cloud_path = str((observations_root / cloud_rel).resolve()) if cloud_rel else None
            record = {
                **obj,
                "camera_pose_available": pose is not None,
                "centroid_world": [float(x) for x in centroid_world] if centroid_world is not None else None,
                "bbox_3d_world_min": [float(x) for x in bbox_world_min] if bbox_world_min is not None else None,
                "bbox_3d_world_max": [float(x) for x in bbox_world_max] if bbox_world_max is not None else None,
                "point_cloud_path": cloud_path,
                "source_raw_depth": assoc["raw_depth"],
                "source_filtered_depth": assoc["filtered_depth"],
                "source_mask": assoc["mask"],
                "source_instance_mask": assoc.get("instance_mask"),
            }
            observations.append(record)
            tracks.setdefault(int(obj["object_id"]), []).append(record)

    track_records: list[dict[str, Any]] = []
    for object_id, records in sorted(tracks.items()):
        records = sorted(records, key=lambda item: int(item["frame_id"]))
        world_centroids = [
            np.asarray(item["centroid_world"], dtype=np.float64)
            for item in records
            if item.get("centroid_world") is not None
        ]
        camera_centroids = [
            np.asarray(item["centroid_camera"], dtype=np.float64)
            for item in records
            if item.get("centroid_camera") is not None
        ]
        world_steps = [
            float(np.linalg.norm(world_centroids[i] - world_centroids[i - 1]))
            for i in range(1, len(world_centroids))
        ]
        camera_steps = [
            float(np.linalg.norm(camera_centroids[i] - camera_centroids[i - 1]))
            for i in range(1, len(camera_centroids))
        ]
        first = records[0]
        track_records.append(
            {
                "object_id": object_id,
                "semantic_id": int(first.get("semantic_id", 0)),
                "semantic_label": str(first.get("semantic_label", "")),
                "frames": len(records),
                "first_frame": int(records[0]["frame_id"]),
                "last_frame": int(records[-1]["frame_id"]),
                "observations_with_depth": int(sum(1 for item in records if int(item.get("num_depth_pixels", 0)) > 0)),
                "observations_with_camera_pose": int(sum(1 for item in records if item.get("camera_pose_available"))),
                "mean_depth_points": float(np.mean([int(item.get("num_depth_pixels", 0)) for item in records])),
                "mean_camera_centroid_step_m": float(np.mean(camera_steps)) if camera_steps else None,
                "mean_world_centroid_step_m": float(np.mean(world_steps)) if world_steps else None,
                "max_world_centroid_step_m": float(np.max(world_steps)) if world_steps else None,
                "first_centroid_world": records[0].get("centroid_world"),
                "last_centroid_world": records[-1].get("centroid_world"),
                "motion_hint": "backend_unconfirmed",
            }
        )

    source_track_summary = observations_root / "track_summary.json"
    source_observation_summary = observations_root / "summary.json"
    copied_camera = None
    if copy_camera_trajectory:
        copied_camera = _copy_if_exists(camera_trajectory, output_root / camera_trajectory.name)

    validation = {
        "frames": len(frames),
        "camera_poses": len(poses),
        "frames_with_camera_pose": len(frames) - missing_pose_frames,
        "missing_pose_frames": missing_pose_frames,
        "max_pose_time_error_sec": max_pose_time_error,
        "objects": len(observations),
        "objects_with_depth": len(observations) - missing_depth_observations,
        "missing_depth_observations": missing_depth_observations,
        "tracks": len(track_records),
        "all_frames_have_pose": missing_pose_frames == 0,
        "all_objects_have_depth": missing_depth_observations == 0,
    }
    manifest = {
        "schema": "dynamic_slam_backend_packet_v0",
        "pose_convention": "camera_pose_twc maps camera-frame points into world frame",
        "sequence_root": str(sequence_root),
        "observations_root": str(observations_root),
        "camera_trajectory": str(camera_trajectory),
        "copied_camera_trajectory": copied_camera,
        "source_track_summary": str(source_track_summary) if source_track_summary.is_file() else None,
        "source_observation_summary": str(source_observation_summary) if source_observation_summary.is_file() else None,
        "validation": validation,
        "files": {
            "frames": "frames.json",
            "object_observations": "object_observations.json",
            "tracks": "tracks.json",
            "validation": "validation.json",
        },
    }

    write_json(output_root / "frames.json", frames)
    write_json(output_root / "object_observations.json", observations)
    write_json(output_root / "tracks.json", track_records)
    write_json(output_root / "validation.json", validation)
    write_json(output_root / "manifest.json", manifest)
    return manifest
