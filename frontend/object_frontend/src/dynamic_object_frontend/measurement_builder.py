"""Convert instance masks and RGB-D depth into object observations."""

from __future__ import annotations

import numpy as np

from .types import CameraIntrinsics, DetectionRecord, ObjectObservation, points_to_tuple


def _clamp_bbox(bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(0, min(width - 1, int(x2)))
    y2 = max(0, min(height - 1, int(y2)))
    if x2 < x1 or y2 < y1:
        return None
    return x1, y1, x2, y2


def depth_to_camera_points(
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    max_depth_m: float = 8.0,
    max_points: int = 4096,
) -> np.ndarray:
    """Back-project masked depth pixels to camera-frame 3D points."""

    if depth.ndim != 2:
        raise ValueError("depth must be single-channel")
    if mask.shape[:2] != depth.shape[:2]:
        raise ValueError("mask/depth shape mismatch")

    valid = mask.astype(bool) & (depth > 0)
    if intrinsics.depth_scale > 0:
        z = depth.astype(np.float32) / float(intrinsics.depth_scale)
    else:
        z = depth.astype(np.float32)
    valid &= z > 0
    valid &= z <= float(max_depth_m)
    ys, xs = np.nonzero(valid)
    if xs.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if xs.size > max_points:
        step = max(1, xs.size // max_points)
        xs = xs[::step][:max_points]
        ys = ys[::step][:max_points]
    zs = z[ys, xs]
    x3 = (xs.astype(np.float32) - float(intrinsics.cx)) * zs / float(intrinsics.fx)
    y3 = (ys.astype(np.float32) - float(intrinsics.cy)) * zs / float(intrinsics.fy)
    return np.stack([x3, y3, zs], axis=1).astype(np.float32)


def build_object_observations(
    *,
    frame_id: int,
    timestamp: float,
    depth: np.ndarray,
    binary_mask: np.ndarray,
    detections: list[DetectionRecord],
    intrinsics: CameraIntrinsics,
    instance_mask: np.ndarray | None = None,
    semantic_id_to_label: dict[int, str] | None = None,
    max_depth_m: float = 8.0,
    max_points_per_object: int = 4096,
    point_cloud_prefix: str | None = None,
) -> tuple[list[ObjectObservation], dict[int, np.ndarray]]:
    """Build object observations and per-object camera-frame point clouds.

    When a dense instance-id mask is supplied it is used as the primary object
    mask. Otherwise, pixels are assigned from the binary dynamic foreground mask
    by detection boxes, greedily by confidence and dynamic score.
    """

    if binary_mask.ndim == 3:
        binary_mask = binary_mask[:, :, 0]
    fg = binary_mask > 0
    if instance_mask is not None:
        if instance_mask.ndim == 3:
            instance_mask = instance_mask[:, :, 0]
        if instance_mask.shape[:2] != fg.shape:
            raise ValueError("instance_mask/depth shape mismatch")
    height, width = fg.shape
    assigned = np.zeros_like(fg, dtype=bool)
    clouds: dict[int, np.ndarray] = {}
    observations: list[ObjectObservation] = []
    semantic_id_to_label = semantic_id_to_label or {}

    ordered = sorted(
        detections,
        key=lambda det: (det.confidence, det.dynamic_score, det.temporal_consistency),
        reverse=True,
    )
    for det in ordered:
        bbox = _clamp_bbox(det.bbox_2d, width, height)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        object_mask = np.zeros_like(fg, dtype=bool)
        if instance_mask is not None:
            roi = instance_mask[y1 : y2 + 1, x1 : x2 + 1] == det.object_id
        else:
            roi = fg[y1 : y2 + 1, x1 : x2 + 1] & ~assigned[y1 : y2 + 1, x1 : x2 + 1]
        object_mask[y1 : y2 + 1, x1 : x2 + 1] = roi
        if instance_mask is None:
            assigned[y1 : y2 + 1, x1 : x2 + 1] |= roi
        mask_pixels = int(np.count_nonzero(object_mask))
        points = depth_to_camera_points(
            depth,
            object_mask,
            intrinsics,
            max_depth_m=max_depth_m,
            max_points=max_points_per_object,
        )
        clouds[det.object_id] = points
        centroid = None
        bbox_min = None
        bbox_max = None
        if points.size > 0:
            centroid = points.mean(axis=0)
            bbox_min = points.min(axis=0)
            bbox_max = points.max(axis=0)
        label = det.semantic_label or semantic_id_to_label.get(det.semantic_id, "")
        point_cloud_file = f"{point_cloud_prefix}_object_{det.object_id}.npy" if point_cloud_prefix else None
        observations.append(
            ObjectObservation(
                frame_id=frame_id,
                timestamp=timestamp,
                object_id=det.object_id,
                semantic_id=det.semantic_id,
                semantic_label=label,
                bbox_2d=bbox,
                dynamic_score=det.dynamic_score,
                temporal_consistency=det.temporal_consistency,
                geometry_dynamic_score=det.geometry_dynamic_score,
                confidence=det.confidence,
                num_mask_pixels=mask_pixels,
                num_depth_pixels=int(points.shape[0]),
                centroid_camera=points_to_tuple(centroid),
                bbox_3d_camera_min=points_to_tuple(bbox_min),
                bbox_3d_camera_max=points_to_tuple(bbox_max),
                point_cloud_file=point_cloud_file,
                match_score=det.match_score,
                association_bbox_iou=det.association_bbox_iou,
                association_mask_iou=det.association_mask_iou,
                association_appearance=det.association_appearance,
                association_depth=det.association_depth,
                association_id_match=det.association_id_match,
                temporal_fusion_score=det.temporal_fusion_score,
                temporal_id_consistency=det.temporal_id_consistency,
                temporal_mask_agreement=det.temporal_mask_agreement,
                temporal_box_agreement=det.temporal_box_agreement,
                held_track=det.held_track,
                hold_misses=det.hold_misses,
            )
        )
    return observations, clouds
