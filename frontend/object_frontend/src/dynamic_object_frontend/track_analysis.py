"""Cross-frame object observation analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np


@dataclass
class ObjectTrackSummary:
    object_id: int
    semantic_id: int
    semantic_label: str
    frames: int
    first_frame: int
    last_frame: int
    observations_with_depth: int
    mean_depth_points: float
    mean_step_translation_m: float | None
    max_step_translation_m: float | None
    mean_speed_mps: float | None
    max_speed_mps: float | None
    motion_state: str
    confidence: float

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_frame_observation_files(frames_dir: Path) -> list[dict[str, Any]]:
    payloads = []
    for path in sorted(frames_dir.glob("*.json")):
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    return payloads


def _centroid_array(obj: dict[str, Any]) -> np.ndarray | None:
    value = obj.get("centroid_camera")
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _classify_motion(
    translations: list[float],
    speeds: list[float],
    *,
    static_translation_m: float,
    moving_translation_m: float,
    static_speed_mps: float,
    moving_speed_mps: float,
    min_steps: int,
) -> tuple[str, float]:
    if len(translations) < min_steps:
        return "uncertain", 0.25
    mean_translation = float(np.mean(translations))
    mean_speed = float(np.mean(speeds)) if speeds else 0.0
    max_translation = float(np.max(translations))
    max_speed = float(np.max(speeds)) if speeds else 0.0
    if max_translation <= static_translation_m and max_speed <= static_speed_mps:
        return "static", 0.85
    if mean_translation >= moving_translation_m or mean_speed >= moving_speed_mps:
        return "moving", 0.8
    return "uncertain", 0.5


def analyze_object_tracks(
    observations_root: Path,
    *,
    static_translation_m: float = 0.03,
    moving_translation_m: float = 0.10,
    static_speed_mps: float = 0.20,
    moving_speed_mps: float = 0.75,
    min_depth_observations: int = 2,
    min_motion_steps: int = 2,
) -> list[ObjectTrackSummary]:
    """Analyze object tracks from exported ObjectObservation JSON files."""

    frames_dir = observations_root / "frames"
    frames = _load_frame_observation_files(frames_dir)
    tracks: dict[int, list[tuple[int, float, dict[str, Any]]]] = {}
    for frame in frames:
        frame_id = int(frame["frame_id"])
        timestamp = float(frame["timestamp"])
        for obj in frame.get("objects", []):
            object_id = int(obj["object_id"])
            tracks.setdefault(object_id, []).append((frame_id, timestamp, obj))

    summaries: list[ObjectTrackSummary] = []
    for object_id, records in sorted(tracks.items()):
        records = sorted(records, key=lambda item: item[0])
        first_obj = records[0][2]
        centroids = []
        translations = []
        speeds = []
        depth_counts = []
        last_centroid = None
        last_timestamp = None
        observations_with_depth = 0
        for _frame_id, timestamp, obj in records:
            depth_count = int(obj.get("num_depth_pixels", 0))
            depth_counts.append(depth_count)
            centroid = _centroid_array(obj)
            if centroid is not None and depth_count > 0:
                observations_with_depth += 1
                centroids.append(centroid)
                if last_centroid is not None and last_timestamp is not None:
                    step = float(np.linalg.norm(centroid - last_centroid))
                    dt = max(1e-6, float(timestamp) - float(last_timestamp))
                    translations.append(step)
                    speeds.append(step / dt)
                last_centroid = centroid
                last_timestamp = timestamp
        if observations_with_depth < min_depth_observations:
            state = "uncertain"
            confidence = 0.2
        else:
            state, confidence = _classify_motion(
                translations,
                speeds,
                static_translation_m=static_translation_m,
                moving_translation_m=moving_translation_m,
                static_speed_mps=static_speed_mps,
                moving_speed_mps=moving_speed_mps,
                min_steps=min_motion_steps,
            )
        summaries.append(
            ObjectTrackSummary(
                object_id=object_id,
                semantic_id=int(first_obj.get("semantic_id", 0)),
                semantic_label=str(first_obj.get("semantic_label", "")),
                frames=len(records),
                first_frame=int(records[0][0]),
                last_frame=int(records[-1][0]),
                observations_with_depth=observations_with_depth,
                mean_depth_points=float(np.mean(depth_counts)) if depth_counts else 0.0,
                mean_step_translation_m=float(np.mean(translations)) if translations else None,
                max_step_translation_m=float(np.max(translations)) if translations else None,
                mean_speed_mps=float(np.mean(speeds)) if speeds else None,
                max_speed_mps=float(np.max(speeds)) if speeds else None,
                motion_state=state,
                confidence=confidence,
            )
        )
    return summaries

