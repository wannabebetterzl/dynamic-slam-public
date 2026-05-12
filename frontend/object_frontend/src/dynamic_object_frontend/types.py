"""Typed records for object-level measurements."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float = 5000.0


@dataclass(frozen=True)
class DetectionRecord:
    object_id: int
    semantic_id: int
    semantic_label: str
    bbox_2d: tuple[int, int, int, int]
    dynamic_score: float
    temporal_consistency: float
    geometry_dynamic_score: float
    filter_out: bool
    confidence: float = 1.0
    match_score: float = 0.0
    association_bbox_iou: float = 0.0
    association_mask_iou: float = 0.0
    association_appearance: float = 0.0
    association_depth: float = 0.0
    association_id_match: float = 0.0
    temporal_fusion_score: float = 0.0
    temporal_id_consistency: float = 0.0
    temporal_mask_agreement: float = 0.0
    temporal_box_agreement: float = 0.0
    held_track: bool = False
    hold_misses: int = 0


@dataclass
class ObjectObservation:
    frame_id: int
    timestamp: float
    object_id: int
    semantic_id: int
    semantic_label: str
    bbox_2d: tuple[int, int, int, int]
    dynamic_score: float
    temporal_consistency: float
    geometry_dynamic_score: float
    confidence: float
    num_mask_pixels: int
    num_depth_pixels: int
    centroid_camera: tuple[float, float, float] | None
    bbox_3d_camera_min: tuple[float, float, float] | None
    bbox_3d_camera_max: tuple[float, float, float] | None
    point_cloud_file: str | None
    match_score: float = 0.0
    association_bbox_iou: float = 0.0
    association_mask_iou: float = 0.0
    association_appearance: float = 0.0
    association_depth: float = 0.0
    association_id_match: float = 0.0
    temporal_fusion_score: float = 0.0
    temporal_id_consistency: float = 0.0
    temporal_mask_agreement: float = 0.0
    temporal_box_agreement: float = 0.0
    held_track: bool = False
    hold_misses: int = 0

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def points_to_tuple(value: np.ndarray | None) -> tuple[float, float, float] | None:
    if value is None:
        return None
    return tuple(float(x) for x in value.reshape(3))
