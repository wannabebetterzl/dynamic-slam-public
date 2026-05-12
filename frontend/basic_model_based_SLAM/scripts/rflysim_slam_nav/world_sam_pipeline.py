#!/usr/bin/env python3
# coding=utf-8

import json
import os
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass

import cv2
import numpy as np


PERSON_ALIASES = {
    "person",
    "pedestrian",
    "worker",
    "adult",
    "human",
    "human body",
    "head",
    "arm",
    "hand",
    "leg",
    "thigh",
}
BOX_ALIASES = {"box", "package", "parcel", "carton", "suitcase", "bag"}
BALLOON_ALIASES = {"balloon", "sports ball", "ball"}


def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def canonical_class_name(name):
    lower = str(name).strip().lower()
    if lower in PERSON_ALIASES:
        return "person"
    if lower in BOX_ALIASES:
        return "box"
    if lower in BALLOON_ALIASES:
        return "balloon"
    return lower


def resolve_model_path(path, config_path=None):
    if not path:
        return path

    expanded = os.path.expandvars(os.path.expanduser(str(path)))
    if os.path.isabs(expanded):
        return expanded

    candidates = []
    if config_path:
        config_dir = os.path.dirname(os.path.abspath(config_path))
        candidates.append(os.path.join(config_dir, expanded))

    root_dir = project_root()
    candidates.append(os.path.join(root_dir, expanded))
    candidates.append(os.path.abspath(expanded))

    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    if os.path.sep not in expanded and not expanded.startswith("."):
        return expanded
    return os.path.abspath(candidates[0])


def resolve_config_paths(config, config_path=None):
    resolved = deep_update({}, config)
    detectors = resolved.get("detectors", {})
    for detector_cfg in detectors.values():
        if isinstance(detector_cfg, dict) and detector_cfg.get("model"):
            detector_cfg["model"] = resolve_model_path(detector_cfg["model"], config_path=config_path)

    segmenter_cfg = resolved.get("segmenter", {})
    if isinstance(segmenter_cfg, dict) and segmenter_cfg.get("checkpoint"):
        segmenter_cfg["checkpoint"] = resolve_model_path(
            segmenter_cfg["checkpoint"], config_path=config_path
        )
    return resolved


# 统一加载主流程配置。
# 注意：这里保留了 drone_detector 这个历史兼容入口，
# 但在当前论文主线配置 world_sam_pipeline_foundation_panoptic_person_v2_local.json 中它默认是关闭的。
# 也就是说，当前基础模型增强 SLAM 主实验默认只启用 open_vocab_detector。
def load_pipeline_config(config_path=None):
    default_config = {
        "detectors": {
            "drone_detector": {
                "enabled": True,
                "type": "onnx_yolo",
                "model": "weights/best.onnx",
                "device": "auto",
                "imgsz": 640,
                "conf": 0.25,
                "iou": 0.45,
                "class_names": ["person", "drone"],
                "target_classes": ["drone"],
                "source_name": "drone_detector",
            },
            "open_vocab_detector": {
                "enabled": True,
                "type": "yolo_world",
                "model": "weights/yolov8s-world.pt",
                "device": "auto",
                "imgsz": 960,
                "conf": 0.20,
                "iou": 0.45,
                "classes": ["person", "pedestrian", "worker", "car", "truck", "bus", "bicycle", "motorcycle"],
                "source_name": "open_vocab_detector",
            },
        },
        "fusion": {
            "enabled": True,
            "iou": 0.50,
            "class_agnostic": False,
            "source_priority": {
                "drone_detector": 3.0,
                "open_vocab_detector": 1.0,
            },
        },
        "segmenter": {
            "type": "sam",
            "checkpoint": "weights/sam_vit_b_01ec64.pth",
            "model_type": "vit_b",
            "device": "auto",
            "multimask": True,
            "box_multimask_output": False,
            "allow_box_fallback": True,
            "temporal_tracking": {
                "enabled": False,
                "max_frames": 4,
                "history_max_frames": 12,
                "prompt_state_history_max_frames": 24,
                "key_prompt_history_max_frames": 48,
                "max_prompt_records_per_object": 4,
                "max_key_prompt_records_per_object": 2,
                "key_prompt_min_quality": 0.64,
                "key_prompt_min_track_confirmation": 0.58,
                "key_prompt_min_dynamic_score": 0.50,
                "key_prompt_min_foundation_score": 0.50,
                "memory_min_quality": 0.64,
                "memory_min_foundation_score": 0.50,
                "memory_min_track_confirmation": 0.52,
                "memory_bootstrap_min_quality": 0.56,
                "memory_bootstrap_min_foundation_score": 0.46,
                "memory_require_filter_out": True,
                "key_prompt_min_temporal_support": 0.28,
                "key_require_consistent_track_id": False,
                "memory_min_temporal_support": 0.18,
                "memory_require_consistent_track_id": True,
                "rebase_max_frames": 48,
                "object_max_idle_frames": 9,
                "mask_threshold": 0.0,
                "min_mask_pixels": 96,
                "slow_propagation_sec": 8.0,
                "slow_propagation_streak": 1,
                "slow_propagation_min_frame_idx": 12,
                "force_refresh_on_slow_propagation": True,
                "reset_on_slow_propagation": True,
                "jpeg_quality": 95,
                "offload_video_to_cpu": True,
                "offload_state_to_cpu": False,
                "async_loading_frames": False,
                "vos_optimized": False,
                "refresh_fusion": {
                    "enabled": True,
                    "class_agnostic": False,
                    "min_mask_iou": 0.12,
                    "min_box_iou": 0.08,
                    "match_score_threshold": 0.34,
                    "strong_match_score": 0.62,
                    "max_area_ratio": 2.4,
                    "mask_expand_pixels": 5,
                    "weights": {
                        "mask_iou": 0.46,
                        "box_iou": 0.20,
                        "score_agreement": 0.18,
                        "class_match": 0.16,
                    },
                },
            },
        },
        "runtime": {
            "detector_interval": 3,
            "dilate_pixels": 5,
            "mask_blur_kernel": 21,
            "rgb_fill_mode": "blur",
            "max_inpaint_area_ratio": 0.12,
        },
        "mask_schedule": {
            "enabled": False,
            "sam_frames": 30,
            "post_init_mode": "box",
            "post_init_box_dilate_pixels": 0,
        },
        "depth_filter": {
            "enabled": True,
            "dilate_pixels": 12,
            "depth_margin_mm": 250.0,
            "min_valid_pixels": 40,
            "min_keep_ratio": 0.18,
        },
        "mask_boundary_refine": {
            "enabled": False,
            "edge_width": 3,
            "min_boundary_pixels": 24,
            "min_keep_ratio": 0.58,
            "depth_margin_mm": 120.0,
            "image_grad_threshold": 18.0,
        },
        "geometry_consistency": {
            "enabled": True,
            "max_corners": 400,
            "quality_level": 0.01,
            "min_distance": 7,
            "block_size": 7,
            "lk_win_size": [21, 21],
            "lk_max_level": 3,
            "ransac_reproj_threshold": 3.0,
            "residual_norm_px": 4.0,
            "inlier_residual_px": 2.0,
            "min_global_points": 24,
            "min_mask_points": 6,
            "verification_min_support_points": 8,
            "dynamic_confirm_threshold": 0.58,
            "static_veto_threshold": 0.70,
            "static_veto_min_hits": 4,
            "static_veto_min_confirmation": 0.55,
            "weak_scene_context_threshold": 0.22,
            "dynamic_margin": 0.10,
            "motion_margin": 0.06,
            "tube_margin": 0.08,
            "motion_weight": 0.32,
            "static_weight": 0.24,
            "dominance_margin": 0.08,
            "ring_dilate_pixels": 9,
        },
        "task_relevance": {
            "enabled": True,
            "min_score": 0.45,
            "default_semantic_weight": 0.55,
            "semantic_weights": {
                "person": 0.95,
                "pedestrian": 0.95,
                "worker": 0.90,
                "drone": 0.85,
                "box": 0.78,
                "package": 0.78,
                "parcel": 0.78,
                "balloon": 0.72,
                "sports ball": 0.72,
                "bicycle": 0.70,
                "motorcycle": 0.70,
                "car": 0.60,
                "truck": 0.55,
                "bus": 0.55,
            },
            "weights": {
                "semantic": 0.28,
                "area": 0.12,
                "center": 0.14,
                "depth": 0.12,
                "confidence": 0.12,
                "sam": 0.10,
                "foundation": 0.12,
            },
            "area_norm_ratio": 0.08,
            "depth_norm_mm": 1500.0,
        },
        "foundation_reliability": {
            "enabled": True,
            "min_score": 0.32,
            "min_mask_pixels": 96,
            "min_mask_ratio": 0.00018,
            "border_margin_pixels": 14,
            "boundary_quality": {
                "enabled": False,
                "edge_width": 3,
                "min_boundary_pixels": 24,
                "image_grad_norm": 36.0,
                "depth_norm_mm": 180.0,
            },
            "weights": {
                "detector_confidence": 0.30,
                "segment_quality": 0.25,
                "mask_area": 0.15,
                "compactness": 0.15,
                "border": 0.15,
            },
        },
        "panoptic_memory": {
            "enabled": True,
            "match_iou": 0.35,
            "max_age": 4,
            "min_hits": 2,
            "confirm_threshold": 0.58,
            "motion_filter_threshold": 0.18,
            "static_protection_threshold": 0.72,
            "static_protection_scene_context_ceiling": 1.01,
            "high_confidence_override": 0.90,
            "ema_momentum": 0.65,
            "center_motion_norm": 0.040,
            "depth_motion_norm_mm": 600.0,
            "appearance_bins": [8, 8],
            "association_weights": {
                "bbox_iou": 0.28,
                "mask_iou": 0.22,
                "appearance": 0.30,
                "depth": 0.20
            },
            "tube_motion_threshold": 0.28,
            "tube_min_hits": 3,
            "tube_displacement_norm_ratio": 0.12,
            "tube_support_classes": ["box", "balloon"],
            "weak_dynamic_guard": {
                "enabled": False,
                "classes": ["person", "pedestrian", "worker"],
                "scene_context_threshold": 0.14,
                "motion_threshold": 0.22,
                "dynamic_memory_threshold": 0.78,
                "static_score_floor": 0.62,
            },
            "confirmed_dynamic_track": {
                "enabled": False,
                "required_for_dynamic_memory": False,
                "classes": ["person", "pedestrian", "worker"],
                "min_hits": 2,
                "min_streak": 3,
                "min_match_score": 0.42,
                "stable_match_score": 0.56,
                "min_component_votes": 2,
                "min_bbox_iou": 0.18,
                "min_mask_iou": 0.10,
                "min_appearance": 0.18,
                "min_depth_similarity": 0.12,
                "min_stable_match_streak": 2,
                "min_geometry_streak": 2,
                "min_geometry_support_points": 8,
                "min_geometry_dynamic_score": 0.32,
                "confirmation_floor": 0.56,
                "geometry_margin": 0.06,
                "motion_floor": 0.12,
                "tube_motion_floor": 0.18,
                "temporal_score_momentum": 0.65,
                "activation_score": 0.56,
                "release_score": 0.40,
                "static_reset_floor": 0.74,
                "uncertain_decay": 1,
                "unconfirmed_memory_cap": 0.60,
            },
        },
        "dynamic_memory": {
            "enabled": True,
            "min_score": 0.60,
            "min_hits": 2,
            "foundation_floor": 0.36,
            "static_ceiling": 0.92,
            "ema_momentum": 0.72,
            "propagation_decay": 0.94,
            "high_confidence_detection_score": 0.75,
            "high_confidence_boost": 0.10,
            "high_confidence_motion_floor": 0.12,
            "min_motion_score_for_activation": 0.08,
            "min_tube_motion_score_for_activation": 0.16,
            "min_dynamic_evidence": 0.16,
            "adaptive_scene_gate_enabled": False,
            "scene_context_threshold": 0.185,
            "scene_context_momentum": 0.70,
            "default_semantic_prior": 0.10,
            "semantic_priors": {
                "person": 0.78,
                "box": 0.40,
                "balloon": 0.52,
                "drone": 0.65
            },
            "weights": {
                "motion": 0.30,
                "tube_motion": 0.18,
                "track_confirmation": 0.20,
                "task": 0.12,
                "foundation": 0.10,
                "anti_static": 0.10
            }
        },
    }
    if not config_path:
        default_mainline_config = os.path.join(
            project_root(),
            "config",
            "world_sam_pipeline_foundation_panoptic_person_v2_local.json",
        )
        if os.path.exists(default_mainline_config):
            config_path = default_mainline_config
        else:
            return resolve_config_paths(default_config)
    with open(config_path, "r", encoding="utf-8") as f:
        user_config = json.load(f)
    merged = deep_update(default_config, user_config)
    upgraded = upgrade_legacy_config(merged)
    return resolve_config_paths(upgraded, config_path=config_path)


def deep_update(base, incoming):
    result = dict(base)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def upgrade_legacy_config(config):
    if "detectors" in config:
        config.setdefault(
            "foundation_reliability",
            {
                "enabled": True,
                "min_score": 0.32,
                "min_mask_pixels": 96,
                "min_mask_ratio": 0.00018,
                "border_margin_pixels": 14,
                "weights": {
                    "detector_confidence": 0.30,
                    "segment_quality": 0.25,
                    "mask_area": 0.15,
                    "compactness": 0.15,
                    "border": 0.15,
                },
            },
        )
        config.setdefault(
            "panoptic_memory",
            {
                "enabled": True,
                "match_iou": 0.35,
                "max_age": 4,
                "min_hits": 2,
                "confirm_threshold": 0.58,
                "motion_filter_threshold": 0.18,
                "static_protection_threshold": 0.72,
                "static_protection_scene_context_ceiling": 1.01,
                "high_confidence_override": 0.90,
                "ema_momentum": 0.65,
                "center_motion_norm": 0.040,
                "depth_motion_norm_mm": 600.0,
                "appearance_bins": [8, 8],
                "association_weights": {
                    "bbox_iou": 0.28,
                    "mask_iou": 0.22,
                    "appearance": 0.30,
                    "depth": 0.20
                },
                "tube_motion_threshold": 0.28,
                "tube_min_hits": 3,
                "tube_displacement_norm_ratio": 0.12,
                "tube_support_classes": ["box", "balloon"],
                "weak_dynamic_guard": {
                    "enabled": False,
                    "classes": ["person", "pedestrian", "worker"],
                    "scene_context_threshold": 0.14,
                    "motion_threshold": 0.22,
                    "dynamic_memory_threshold": 0.78,
                    "static_score_floor": 0.62,
                },
                "confirmed_dynamic_track": {
                    "enabled": False,
                    "required_for_dynamic_memory": False,
                    "classes": ["person", "pedestrian", "worker"],
                    "min_hits": 2,
                    "min_streak": 3,
                    "min_match_score": 0.42,
                    "stable_match_score": 0.56,
                    "min_component_votes": 2,
                    "min_bbox_iou": 0.18,
                    "min_mask_iou": 0.10,
                    "min_appearance": 0.18,
                    "min_depth_similarity": 0.12,
                    "min_stable_match_streak": 2,
                    "min_geometry_streak": 2,
                    "min_geometry_support_points": 8,
                    "min_geometry_dynamic_score": 0.32,
                    "confirmation_floor": 0.56,
                    "geometry_margin": 0.06,
                    "motion_floor": 0.12,
                    "tube_motion_floor": 0.18,
                    "temporal_score_momentum": 0.65,
                    "activation_score": 0.56,
                    "release_score": 0.40,
                    "static_reset_floor": 0.74,
                    "uncertain_decay": 1,
                    "unconfirmed_memory_cap": 0.60,
                },
            },
        )
        return config

    legacy_detector = config.get("detector", {})
    config["detectors"] = {
        "drone_detector": {
            "enabled": False,
            "type": "onnx_yolo",
            "model": "weights/best.onnx",
            "device": "auto",
            "imgsz": 640,
            "conf": 0.25,
            "iou": 0.45,
            "class_names": ["person", "drone"],
            "target_classes": ["drone"],
            "source_name": "drone_detector",
        },
        "open_vocab_detector": {
            "enabled": True,
            "type": legacy_detector.get("type", "yolo_world"),
            "model": legacy_detector.get("model", "yolov8s-world.pt"),
            "device": legacy_detector.get("device", "auto"),
            "imgsz": legacy_detector.get("imgsz", 640),
            "conf": legacy_detector.get("conf", 0.25),
            "iou": legacy_detector.get("iou", 0.45),
            "classes": legacy_detector.get("classes", ["person", "pedestrian"]),
            "source_name": "open_vocab_detector",
        },
    }
    config.setdefault(
        "fusion",
        {"enabled": True, "iou": 0.50, "class_agnostic": False, "source_priority": {"drone_detector": 3.0, "open_vocab_detector": 1.0}},
    )
    config.setdefault(
        "task_relevance",
        {
            "enabled": True,
            "min_score": 0.45,
            "default_semantic_weight": 0.55,
            "semantic_weights": {"person": 0.95, "pedestrian": 0.95, "worker": 0.90, "drone": 0.85},
            "weights": {"semantic": 0.28, "area": 0.12, "center": 0.14, "depth": 0.12, "confidence": 0.12, "sam": 0.10, "foundation": 0.12},
            "area_norm_ratio": 0.08,
            "depth_norm_mm": 1500.0,
        },
    )
    config.setdefault(
        "foundation_reliability",
        {
            "enabled": True,
            "min_score": 0.32,
            "min_mask_pixels": 96,
            "min_mask_ratio": 0.00018,
            "border_margin_pixels": 14,
            "weights": {
                "detector_confidence": 0.30,
                "segment_quality": 0.25,
                "mask_area": 0.15,
                "compactness": 0.15,
                "border": 0.15,
            },
        },
    )
    config.setdefault(
        "panoptic_memory",
        {
            "enabled": True,
            "match_iou": 0.35,
            "max_age": 4,
            "min_hits": 2,
            "confirm_threshold": 0.58,
            "motion_filter_threshold": 0.18,
            "static_protection_threshold": 0.72,
            "static_protection_scene_context_ceiling": 1.01,
            "high_confidence_override": 0.90,
            "ema_momentum": 0.65,
            "center_motion_norm": 0.040,
            "depth_motion_norm_mm": 600.0,
        },
    )
    config.setdefault(
        "dynamic_memory",
        {
            "enabled": True,
            "min_score": 0.60,
            "min_hits": 2,
            "foundation_floor": 0.36,
            "static_ceiling": 0.92,
            "ema_momentum": 0.72,
            "propagation_decay": 0.94,
            "high_confidence_detection_score": 0.75,
            "high_confidence_boost": 0.10,
            "scene_context_threshold": 0.185,
            "scene_context_momentum": 0.70,
            "default_semantic_prior": 0.10,
            "semantic_priors": {
                "person": 0.78,
                "box": 0.40,
                "balloon": 0.52,
                "drone": 0.65,
            },
            "weights": {
                "motion": 0.30,
                "tube_motion": 0.18,
                "track_confirmation": 0.20,
                "task": 0.12,
                "foundation": 0.10,
                "anti_static": 0.10,
            },
        },
    )
    return config


def resolve_device(device_name):
    if device_name != "auto":
        return device_name
    try:
        import torch  # pylint: disable=import-error

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def score_with_priority(score, source_name, fusion_cfg):
    priority = float(fusion_cfg.get("source_priority", {}).get(source_name, 1.0))
    return float(score) * priority


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    class_name: str
    class_id: int = -1
    source_name: str = "unknown"
    track_id: int = -1

    @property
    def area(self):
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    @property
    def canonical_name(self):
        return canonical_class_name(self.class_name)


@dataclass
class TrackState:
    track_id: int
    class_name: str
    bbox: tuple
    mask: np.ndarray
    center: tuple
    hits: int = 1
    misses: int = 0
    age: int = 1
    ema_confidence: float = 0.0
    ema_relevance: float = 0.0
    ema_reliability: float = 0.0
    confirmation_score: float = 0.0
    motion_score: float = 0.0
    static_score: float = 0.0
    last_depth: float = 0.0
    appearance: np.ndarray = None
    start_center: tuple = None
    cumulative_displacement: float = 0.0
    max_center_distance: float = 0.0
    tube_motion_score: float = 0.0
    max_motion_score: float = 0.0
    dynamic_memory_score: float = 0.0
    dynamic_confirmation_streak: int = 0
    stable_match_streak: int = 0
    geometry_confirmation_streak: int = 0
    temporal_dynamic_score: float = 0.0
    geometry_dynamic_score: float = 0.0
    geometry_static_score: float = 0.0
    geometry_support: int = 0
    temporal_fusion_score: float = 0.0
    temporal_id_consistency: float = 0.0
    temporal_mask_agreement: float = 0.0
    temporal_box_agreement: float = 0.0


def compute_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = np.maximum(area_a + area_b - inter, 1e-6)
    return inter / union


def compute_box_iou(box_a, box_b):
    boxes = np.array([box_b], dtype=np.float32)
    return float(compute_iou(np.array(box_a, dtype=np.float32), boxes)[0])


def compute_mask_iou(mask_a, mask_b):
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    union = float(np.count_nonzero(a | b))
    if union <= 0:
        return 0.0
    inter = float(np.count_nonzero(a & b))
    return inter / union


def mask_center(mask, detection):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.5 * float(detection.x1 + detection.x2), 0.5 * float(detection.y1 + detection.y2)
    return float(np.mean(xs)), float(np.mean(ys))


def median_mask_depth(depth_mm, mask):
    if depth_mm is None:
        return 0.0
    values = depth_mm[(mask > 0) & (depth_mm > 0)]
    if len(values) < 20:
        return 0.0
    return float(np.median(values))


def mask_compactness(mask, detection):
    bbox_area = max(1.0, float(detection.area))
    return float(np.clip(float(np.count_nonzero(mask)) / bbox_area, 0.0, 1.0))


def box_border_score(detection, image_shape, margin_pixels):
    image_h, image_w = image_shape
    margin = min(detection.x1, detection.y1, image_w - detection.x2, image_h - detection.y2)
    denom = max(float(margin_pixels), 1.0)
    return float(np.clip(margin / denom, 0.25, 1.0))


def erode_mask(mask, pixels):
    pixels = max(int(pixels), 1)
    kernel = np.ones((pixels, pixels), dtype=np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1)


def dilate_mask(mask, pixels):
    pixels = max(int(pixels), 1)
    kernel = np.ones((pixels, pixels), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)


def mask_boundary(mask, edge_width):
    if np.count_nonzero(mask) == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    core = erode_mask(mask, edge_width)
    return np.clip(mask.astype(np.uint8) - core, 0, 1)


def mask_outer_ring(mask, edge_width):
    outer = dilate_mask(mask, edge_width)
    return np.clip(outer.astype(np.uint8) - mask.astype(np.uint8), 0, 1)


def extract_instance_appearance(image_bgr, mask, bins=(8, 8)):
    bins = tuple(int(v) for v in bins)
    if mask is None or np.count_nonzero(mask) < 16:
        return np.zeros((bins[0] * bins[1],), dtype=np.float32)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask_u8 = (mask > 0).astype(np.uint8)
    hist = cv2.calcHist([hsv], [0, 1], mask_u8, bins, [0, 180, 0, 256])
    hist = hist.astype(np.float32).reshape(-1)
    norm = float(np.sum(hist))
    if norm > 0:
        hist /= norm
    return hist


def appearance_similarity(hist_a, hist_b):
    if hist_a is None or hist_b is None:
        return 0.0
    if len(hist_a) == 0 or len(hist_b) == 0:
        return 0.0
    return float(np.clip(np.sum(np.minimum(hist_a, hist_b)), 0.0, 1.0))


def depth_similarity(depth_a, depth_b, norm_mm):
    if depth_a <= 0 or depth_b <= 0:
        return 0.5
    norm = max(float(norm_mm), 1.0)
    return float(np.clip(1.0 - abs(depth_a - depth_b) / norm, 0.0, 1.0))


def blend_geometry_evidence(base_motion, base_static, geometry_detail, geometry_cfg):
    if not geometry_cfg.get("enabled", False):
        return float(base_motion), float(base_static)

    support_points = int(geometry_detail.get("support_points", 0))
    min_support = max(int(geometry_cfg.get("min_mask_points", 6)), 1)
    if support_points < min_support:
        return float(base_motion), float(base_static)

    dynamic_score = float(geometry_detail.get("dynamic_score", 0.0))
    static_score = float(geometry_detail.get("static_score", 0.0))
    motion_weight = float(np.clip(geometry_cfg.get("motion_weight", 0.32), 0.0, 1.0))
    static_weight = float(np.clip(geometry_cfg.get("static_weight", 0.24), 0.0, 1.0))
    support_norm = max(float(geometry_cfg.get("verification_min_support_points", min_support)), 1.0)
    support_scale = float(np.clip(support_points / support_norm, 0.0, 1.0))
    dominance_margin = float(geometry_cfg.get("dominance_margin", 0.08))

    motion_mix = float(
        np.clip(
            (1.0 - motion_weight * support_scale) * base_motion
            + motion_weight * support_scale * dynamic_score,
            0.0,
            1.0,
        )
    )
    static_mix = float(
        np.clip(
            (1.0 - static_weight * support_scale) * base_static
            + static_weight * support_scale * static_score,
            0.0,
            1.0,
        )
    )

    if dynamic_score >= static_score + dominance_margin:
        motion_mix = max(motion_mix, dynamic_score)
    elif static_score >= dynamic_score + dominance_margin:
        static_mix = max(static_mix, static_score)

    return float(np.clip(motion_mix, 0.0, 1.0)), float(np.clip(static_mix, 0.0, 1.0))


def class_dynamic_prior(class_name, dynamic_cfg):
    priors = dynamic_cfg.get("semantic_priors", {})
    default_prior = float(dynamic_cfg.get("default_semantic_prior", 0.10))
    return float(priors.get(class_name, default_prior))


def compute_dynamic_memory_observation(
    detection,
    task_score,
    foundation_score,
    motion_score,
    tube_motion_score,
    static_score,
    track_confirmation,
    temporal_score,
    dynamic_cfg,
):
    weights = dynamic_cfg.get("weights", {})
    semantic_prior = class_dynamic_prior(detection.canonical_name, dynamic_cfg)
    motion_evidence = float(np.clip(max(motion_score, tube_motion_score, 1.0 - static_score), 0.0, 1.0))
    semantic_support = float(
        np.clip(
            semantic_prior
            * np.clip(
                0.30 * temporal_score
                + 0.20 * foundation_score
                + 0.50 * motion_evidence,
                0.0,
                1.0,
            ),
            0.0,
            1.0,
        )
    )
    observation = float(
        np.clip(
            float(weights.get("motion", 0.30)) * motion_score
            + float(weights.get("tube_motion", 0.18)) * tube_motion_score
            + float(weights.get("track_confirmation", 0.20)) * track_confirmation
            + float(weights.get("task", 0.12)) * task_score
            + float(weights.get("foundation", 0.10)) * foundation_score
            + float(weights.get("anti_static", 0.10)) * (1.0 - static_score),
            0.0,
            1.0,
        )
    )
    high_conf_threshold = float(dynamic_cfg.get("high_confidence_detection_score", 0.75))
    high_conf_boost = float(dynamic_cfg.get("high_confidence_boost", 0.10))
    high_conf_motion_floor = float(dynamic_cfg.get("high_confidence_motion_floor", 0.12))
    if float(detection.score) >= high_conf_threshold and motion_evidence >= high_conf_motion_floor:
        observation = max(observation, min(1.0, semantic_support + high_conf_boost * float(detection.score)))
    return float(np.clip(max(observation, semantic_support), 0.0, 1.0))


def is_stable_track_match(match_score, match_components, confirmed_dynamic_cfg):
    if not match_components:
        return False
    if float(match_components.get("id_match", 0.0)) >= 0.5:
        return True

    stable_match_score = float(
        confirmed_dynamic_cfg.get(
            "stable_match_score",
            confirmed_dynamic_cfg.get("min_match_score", 0.42),
        )
    )
    if float(match_score) < stable_match_score:
        return False

    bbox_iou = float(match_components.get("bbox_iou", 0.0))
    mask_iou = float(match_components.get("mask_iou", 0.0))
    appearance = float(match_components.get("appearance", 0.0))
    depth = float(match_components.get("depth", 0.0))
    overlap_ok = (
        bbox_iou >= float(confirmed_dynamic_cfg.get("min_bbox_iou", 0.18))
        or mask_iou >= float(confirmed_dynamic_cfg.get("min_mask_iou", 0.10))
    )
    if not overlap_ok:
        return False

    votes = 0
    if bbox_iou >= float(confirmed_dynamic_cfg.get("min_bbox_iou", 0.18)):
        votes += 1
    if mask_iou >= float(confirmed_dynamic_cfg.get("min_mask_iou", 0.10)):
        votes += 1
    if appearance >= float(confirmed_dynamic_cfg.get("min_appearance", 0.18)):
        votes += 1
    if depth >= float(confirmed_dynamic_cfg.get("min_depth_similarity", 0.12)):
        votes += 1
    return votes >= max(int(confirmed_dynamic_cfg.get("min_component_votes", 2)), 1)


def temporal_geometry_dynamic_ready(track, geometry_detail, confirmed_dynamic_cfg, geometry_margin):
    min_support = max(int(confirmed_dynamic_cfg.get("min_geometry_support_points", 8)), 1)
    min_dynamic_score = float(confirmed_dynamic_cfg.get("min_geometry_dynamic_score", 0.32))
    return bool(
        (
            bool(geometry_detail.get("verified_dynamic", False))
            or track.geometry_dynamic_score >= track.geometry_static_score + geometry_margin
        )
        and int(track.geometry_support) >= min_support
        and float(track.geometry_dynamic_score) >= min_dynamic_score
    )


def temporal_geometry_static_ready(track, geometry_detail, confirmed_dynamic_cfg, geometry_margin):
    min_support = max(int(confirmed_dynamic_cfg.get("min_geometry_support_points", 8)), 1)
    return bool(
        (
            bool(geometry_detail.get("verified_static", False))
            or track.geometry_static_score >= track.geometry_dynamic_score + geometry_margin
        )
        and int(track.geometry_support) >= min_support
    )


def temporal_dynamic_evidence_score(track, match_score, stable_match_ready, geometry_detail, confirmed_dynamic_cfg):
    stable_floor = max(
        float(
            confirmed_dynamic_cfg.get(
                "stable_match_score",
                confirmed_dynamic_cfg.get("min_match_score", 0.42),
            )
        ),
        1e-6,
    )
    stable_component = 1.0 if stable_match_ready else float(np.clip(float(match_score) / stable_floor, 0.0, 1.0)) * 0.5
    geometry_margin = float(confirmed_dynamic_cfg.get("geometry_margin", 0.06))
    gap_norm = float(
        np.clip(
            (float(track.geometry_dynamic_score) - float(track.geometry_static_score) - geometry_margin)
            / max(1.0 - geometry_margin, 1e-6),
            0.0,
            1.0,
        )
    )
    geometry_component = float(
        np.clip(
            0.60 * float(track.geometry_dynamic_score)
            + 0.25 * gap_norm
            + 0.15 * (1.0 if geometry_detail.get("verified_dynamic", False) else 0.0),
            0.0,
            1.0,
        )
    )
    motion_component = float(
        np.clip(
            max(
                float(track.motion_score),
                float(track.tube_motion_score),
                float(track.geometry_dynamic_score),
            ),
            0.0,
            1.0,
        )
    )
    return float(np.clip(0.40 * stable_component + 0.40 * geometry_component + 0.20 * motion_component, 0.0, 1.0))


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scale_fill=False, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = (r, r)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = (new_shape[1] / shape[1], new_shape[0] / shape[0])

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


def postprocess_onnx(output, ratio, pad, image_shape, conf_thres=0.25, iou_thres=0.45):
    outputs = np.transpose(np.squeeze(output[0]))
    rows = outputs.shape[0]
    boxes = []
    scores = []
    class_ids = []
    dw, dh = pad

    x_factor = 1.0 / ratio[0]
    y_factor = 1.0 / ratio[1]
    img_h, img_w = image_shape[:2]

    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = float(np.amax(classes_scores))
        if max_score < conf_thres:
            continue

        class_id = int(np.argmax(classes_scores))
        x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
        if w <= 0 or h <= 0:
            continue

        left = int((x - w / 2 - dw) * x_factor)
        top = int((y - h / 2 - dh) * y_factor)
        width = int(w * x_factor)
        height = int(h * y_factor)
        x1 = max(0, left)
        y1 = max(0, top)
        x2 = min(img_w - 1, left + width)
        y2 = min(img_h - 1, top + height)
        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(max_score)
        class_ids.append(class_id)

    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
    if len(indices) == 0:
        return []

    detections = []
    for idx in np.array(indices).reshape(-1):
        x1, y1, w, h = boxes[int(idx)]
        detections.append((x1, y1, x1 + w, y1 + h, float(scores[int(idx)]), int(class_ids[int(idx)])))
    return detections


class BaseDetector:
    def __init__(self, config):
        self.config = dict(config)
        self.config["device"] = resolve_device(self.config.get("device", "auto"))
        self.source_name = self.config.get("source_name", "detector")

    def predict(self, image_bgr):
        raise NotImplementedError


# 专用 ONNX 检测器接口。
# 这部分主要是为早期无人机/专用目标检测分支保留的兼容入口。
class DedicatedONNXDetector(BaseDetector):
    def __init__(self, config):
        super().__init__(config)
        self.session = None
        self.input_name = None
        self.class_names = list(self.config.get("class_names", ["person", "drone"]))
        self.target_classes = {name.lower() for name in self.config.get("target_classes", ["drone"])}
        self.input_shape = int(self.config.get("imgsz", 640))
        self._load_model()

    def _load_model(self):
        try:
            import onnxruntime as ort  # pylint: disable=import-error
        except Exception as exc:
            raise RuntimeError(
                "Dedicated ONNX detector requires onnxruntime or onnxruntime-gpu in the runtime environment."
            ) from exc

        providers = ["CPUExecutionProvider"]
        if self.config.get("device", "cpu").startswith("cuda"):
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(self.config["model"], providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image_bgr):
        image, ratio, pad = letterbox(image_bgr, (self.input_shape, self.input_shape), auto=False)
        tensor = image.transpose((2, 0, 1))[::-1]
        tensor = np.ascontiguousarray(tensor).astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, 0)

        outputs = self.session.run(None, {self.input_name: tensor})
        raw_detections = postprocess_onnx(
            outputs,
            ratio,
            pad,
            image_bgr.shape,
            conf_thres=float(self.config.get("conf", 0.25)),
            iou_thres=float(self.config.get("iou", 0.45)),
        )

        detections = []
        for x1, y1, x2, y2, score, class_id in raw_detections:
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)
            if class_name.lower() not in self.target_classes:
                continue
            detections.append(
                Detection(
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                    score=float(score),
                    class_name=class_name,
                    class_id=int(class_id),
                    source_name=self.source_name,
                )
            )
        return detections


# 当前主线最重要的开放词汇检测器。
# 它负责根据收缩后的提示词生成动态候选框，例如 person / pedestrian / worker。
class YOLOWorldDetector(BaseDetector):
    def __init__(self, config):
        super().__init__(config)
        self.model = None
        self.names = {}
        self.class_names = list(self.config.get("classes", ["person", "pedestrian"]))
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLOWorld  # pylint: disable=import-error
        except Exception as exc:
            raise RuntimeError(
                "YOLO-World backend is unavailable. Install ultralytics with YOLO-World support."
            ) from exc

        model_name = self.config.get("model", "yolov8s-world.pt")
        self.model = YOLOWorld(model_name)
        if hasattr(self.model, "set_classes"):
            self.model.set_classes(self.class_names)
        self.names = {idx: name for idx, name in enumerate(self.class_names)}

    def predict(self, image_bgr):
        results = self.model.predict(
            source=image_bgr,
            imgsz=self.config.get("imgsz", 960),
            conf=self.config.get("conf", 0.20),
            iou=self.config.get("iou", 0.45),
            device=self.config.get("device", "cpu"),
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        names = getattr(result, "names", self.names)

        detections = []
        for idx, box in enumerate(xyxy):
            x1, y1, x2, y2 = box.astype(int).tolist()
            class_id = int(cls[idx])
            class_name = names.get(class_id, self.names.get(class_id, str(class_id)))
            detections.append(
                Detection(
                    x1=max(0, x1),
                    y1=max(0, y1),
                    x2=max(0, x2),
                    y2=max(0, y2),
                    score=float(conf[idx]),
                    class_name=str(class_name),
                    class_id=class_id,
                    source_name=self.source_name,
                )
            )
        return detections


class YOLOEDetector(BaseDetector):
    def __init__(self, config):
        super().__init__(config)
        self.model = None
        self.names = {}
        self.class_names = list(self.config.get("classes", ["person", "pedestrian"]))
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLOE  # pylint: disable=import-error
        except Exception as exc:
            raise RuntimeError(
                "YOLOE backend is unavailable. Install ultralytics with YOLOE support."
            ) from exc

        model_name = self.config.get("model", "yoloe-11s-seg.pt")
        self.model = YOLOE(model_name)
        if hasattr(self.model, "set_classes"):
            self.model.set_classes(self.class_names)
        self.names = {idx: name for idx, name in enumerate(self.class_names)}

    def predict(self, image_bgr):
        results = self.model.predict(
            source=image_bgr,
            imgsz=self.config.get("imgsz", 960),
            conf=self.config.get("conf", 0.20),
            iou=self.config.get("iou", 0.45),
            device=self.config.get("device", "cpu"),
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        names = getattr(result, "names", self.names)

        detections = []
        for idx, box in enumerate(xyxy):
            x1, y1, x2, y2 = box.astype(int).tolist()
            class_id = int(cls[idx])
            class_name = names.get(class_id, self.names.get(class_id, str(class_id)))
            detections.append(
                Detection(
                    x1=max(0, x1),
                    y1=max(0, y1),
                    x2=max(0, x2),
                    y2=max(0, y2),
                    score=float(conf[idx]),
                    class_name=str(class_name),
                    class_id=class_id,
                    source_name=self.source_name,
                )
            )
        return detections


# 闭集 YOLO 检测器接口，主要用于开放词汇 vs 闭集的对照实验。
class UltralyticsYOLODetector(BaseDetector):
    def __init__(self, config):
        super().__init__(config)
        self.model = None
        self.names = {}
        self.target_classes = {str(name).strip().lower() for name in self.config.get("target_classes", [])}
        self.target_class_ids = None
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO  # pylint: disable=import-error
        except Exception as exc:
            raise RuntimeError(
                "Ultralytics YOLO backend is unavailable. Install ultralytics in the runtime environment."
            ) from exc

        model_name = self.config.get("model", "yolov8n.pt")
        self.model = YOLO(model_name)
        self.names = getattr(self.model, "names", {}) or {}
        if self.target_classes:
            class_ids = []
            for class_id, class_name in self.names.items():
                lower = str(class_name).strip().lower()
                canonical = canonical_class_name(lower)
                if lower in self.target_classes or canonical in self.target_classes:
                    class_ids.append(int(class_id))
            self.target_class_ids = class_ids or None

    def predict(self, image_bgr):
        results = self.model.predict(
            source=image_bgr,
            imgsz=self.config.get("imgsz", 640),
            conf=self.config.get("conf", 0.25),
            iou=self.config.get("iou", 0.45),
            device=self.config.get("device", "cpu"),
            classes=self.target_class_ids,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        names = getattr(result, "names", self.names)

        detections = []
        for idx, box in enumerate(xyxy):
            x1, y1, x2, y2 = box.astype(int).tolist()
            class_id = int(cls[idx])
            class_name = str(names.get(class_id, self.names.get(class_id, str(class_id))))
            lower = class_name.strip().lower()
            canonical = canonical_class_name(lower)
            if self.target_classes and lower not in self.target_classes and canonical not in self.target_classes:
                continue
            detections.append(
                Detection(
                    x1=max(0, x1),
                    y1=max(0, y1),
                    x2=max(0, x2),
                    y2=max(0, y2),
                    score=float(conf[idx]),
                    class_name=class_name,
                    class_id=class_id,
                    source_name=self.source_name,
                )
            )
        return detections


def build_detector(detector_config):
    detector_type = str(detector_config.get("type", "")).strip().lower()
    if detector_type == "onnx_yolo":
        return DedicatedONNXDetector(detector_config)
    if detector_type == "yolo_world":
        return YOLOWorldDetector(detector_config)
    if detector_type == "yoloe":
        return YOLOEDetector(detector_config)
    if detector_type == "ultralytics_yolo":
        return UltralyticsYOLODetector(detector_config)
    raise RuntimeError(f"Unsupported detector type: {detector_type}")


# 检测器集成器：按配置实例化多个检测器，并在输出后做融合。
# 当前论文主线通常只有一个开放词汇检测器开启，但代码层面保留了多检测器并行入口。
class DetectorEnsemble:
    def __init__(self, config):
        self.config = config
        self.detectors = []
        self.auxiliary_mask_sources = {
            str(item).strip()
            for item in self.config.get("auxiliary_mask_sources", [])
            if str(item).strip()
        }
        detector_cfg = self.config.get("detectors", {})

        for _, single_detector_cfg in detector_cfg.items():
            if not isinstance(single_detector_cfg, dict):
                continue
            if not single_detector_cfg.get("enabled", False):
                continue
            self.detectors.append(build_detector(single_detector_cfg))

        if not self.detectors:
            raise RuntimeError("At least one detector must be enabled in the pipeline config.")

    def predict(self, image_bgr):
        per_source = {}
        all_detections = []
        for detector in self.detectors:
            detections = detector.predict(image_bgr)
            per_source[detector.source_name] = len(detections)
            all_detections.extend(detections)
        if not self.auxiliary_mask_sources:
            fused = fuse_detections(all_detections, self.config.get("fusion", {}))
            return fused, per_source

        primary_detections = []
        auxiliary_detections = []
        for detection in all_detections:
            if detection.source_name in self.auxiliary_mask_sources:
                auxiliary_detections.append(detection)
            else:
                primary_detections.append(detection)

        fused_primary = fuse_detections(primary_detections, self.config.get("fusion", {}))
        fused_auxiliary = fuse_detections(auxiliary_detections, self.config.get("fusion", {}))
        fused = fused_primary + fused_auxiliary
        return fused, per_source


def fuse_detections(detections, fusion_cfg):
    if not detections:
        return []
    if not fusion_cfg.get("enabled", True):
        return detections

    class_agnostic = bool(fusion_cfg.get("class_agnostic", False))
    boxes = np.array([[det.x1, det.y1, det.x2, det.y2] for det in detections], dtype=np.float32)
    scores = np.array([score_with_priority(det.score, det.source_name, fusion_cfg) for det in detections], dtype=np.float32)
    labels = [det.canonical_name for det in detections]
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break

        rest = order[1:]
        rest_boxes = boxes[rest]
        ious = compute_iou(boxes[current], rest_boxes)
        threshold = float(fusion_cfg.get("iou", 0.50))

        if class_agnostic:
            mask = ious <= threshold
        else:
            same_label = np.array([labels[idx] == labels[current] for idx in rest], dtype=bool)
            mask = (ious <= threshold) | (~same_label)
        order = rest[mask]

    return [detections[idx] for idx in keep]


# SAM 分割模块：把检测框细化为实例掩膜。
# 若 SAM 不可用且允许回退，则退化为 box mask，保证整条流水线不断。
class SamSegmenter:
    def __init__(self, config):
        self.config = dict(config)
        self.config["device"] = resolve_device(self.config.get("device", "auto"))
        self.backend = str(self.config.get("type", "sam")).strip().lower()
        self.mode = None
        self.predictor = None
        self.temporal_tracker = None
        self._load_predictor()

    def _load_predictor(self):
        checkpoint = self.config.get("checkpoint", "")
        model_type = self.config.get("model_type", "vit_b")
        model_cfg = self.config.get("model_cfg", "")
        device = self.config.get("device", "cpu")
        temporal_cfg = dict(self.config.get("temporal_tracking", {}))

        if self.backend in {"sam", "sam1", "auto"} and checkpoint:
            try:
                from segment_anything import SamPredictor, sam_model_registry  # pylint: disable=import-error

                sam = sam_model_registry[model_type](checkpoint=checkpoint)
                sam.to(device=device)
                self.predictor = SamPredictor(sam)
                self.mode = "sam1"
                return
            except Exception:
                pass

        if self.backend in {"sam3", "sam3.1", "sam31"}:
            try:
                from ultralytics import SAM  # pylint: disable=import-error

                checkpoint_path = self.config.get("checkpoint_path", checkpoint) or None
                if checkpoint_path is None:
                    raise RuntimeError("SAM3/SAM3.1 requires a local checkpoint_path.")
                self.predictor = SAM(checkpoint_path)
                self.mode = "sam3"
                return
            except Exception as exc:
                self.load_error = exc
                if not self.config.get("allow_box_fallback", True):
                    raise RuntimeError(f"SAM3/SAM3.1 backend is unavailable: {exc}") from exc
                print(
                    f"[SamSegmenter] SAM3/SAM3.1 backend unavailable, falling back to box masks: {exc}",
                    file=sys.stderr,
                )

        if self.backend in {"sam", "sam2", "auto"} and checkpoint and model_cfg:
            try:
                from sam2.build_sam import build_sam2, build_sam2_video_predictor  # pylint: disable=import-error
                from sam2.sam2_image_predictor import SAM2ImagePredictor  # pylint: disable=import-error

                self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=device))
                self.mode = "sam2"
                if temporal_cfg.get("enabled", False):
                    try:
                        video_predictor = build_sam2_video_predictor(
                            model_cfg,
                            checkpoint,
                            device=device,
                            vos_optimized=bool(temporal_cfg.get("vos_optimized", False)),
                        )
                        self.temporal_tracker = Sam2TemporalMaskTracker(video_predictor, temporal_cfg)
                    except Exception:
                        self.temporal_tracker = None
                return
            except Exception:
                pass

        if not self.config.get("allow_box_fallback", True):
            raise RuntimeError("SAM backend is unavailable and box fallback is disabled.")
        self.mode = "box"

    def segment(self, image_bgr, detections):
        if not detections:
            return []

        if self.mode == "box":
            return [
                {"mask": self._box_mask(image_bgr.shape[:2], det), "sam_score": float(np.clip(det.score * 0.7, 0.0, 1.0)), "segmenter_mode": "box"}
                for det in detections
            ]

        if self.mode == "sam3":
            records = []
            device = self.config.get("device", "cpu")
            predict_device = 0 if str(device).startswith("cuda") else "cpu"
            conf = float(self.config.get("sam3_conf", 0.001))
            iou = float(self.config.get("sam3_iou", 0.90))
            imgsz = int(self.config.get("sam3_imgsz", 1036))
            for det in detections:
                box = [[float(det.x1), float(det.y1), float(det.x2), float(det.y2)]]
                result = self.predictor.predict(
                    image_bgr,
                    bboxes=box,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    device=predict_device,
                    verbose=False,
                )[0]
                if result.masks is None or len(result.masks.data) == 0:
                    records.append({
                        "mask": self._box_mask(image_bgr.shape[:2], det),
                        "sam_score": float(np.clip(det.score * 0.25, 0.0, 1.0)),
                        "segmenter_mode": "sam3_box_fallback",
                    })
                    continue
                masks = result.masks.data.detach().cpu().numpy().astype(np.uint8)
                scores = result.boxes.conf.detach().cpu().numpy() if result.boxes is not None else np.ones((len(masks),), dtype=np.float32)
                best_index = int(np.argmax(scores))
                records.append({
                    "mask": masks[best_index].astype(np.uint8),
                    "sam_score": float(np.clip(scores[best_index], 0.0, 1.0)),
                    "segmenter_mode": self.mode,
                })
            return records

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        records = []
        inference_ctx = nullcontext()
        autocast_ctx = nullcontext()
        if self.mode == "sam2":
            try:
                import torch  # pylint: disable=import-error

                inference_ctx = torch.inference_mode()
                if str(self.config.get("device", "")).startswith("cuda"):
                    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            except Exception:
                inference_ctx = nullcontext()
                autocast_ctx = nullcontext()

        with inference_ctx:
            with autocast_ctx:
                self.predictor.set_image(image_rgb)
                for det in detections:
                    box = np.array([det.x1, det.y1, det.x2, det.y2], dtype=np.float32)
                    multimask_output = bool(self.config.get("box_multimask_output", self.config.get("multimask", True)))
                    sam_masks, scores, _ = self.predictor.predict(
                        box=box,
                        multimask_output=multimask_output,
                    )
                    best_index = int(np.argmax(scores))
                    records.append({
                        "mask": sam_masks[best_index].astype(np.uint8),
                        "sam_score": float(np.clip(scores[best_index], 0.0, 1.0)),
                        "segmenter_mode": self.mode,
                    })
        return records

    def set_temporal_anchor(self, image_bgr, detections, mask_records):
        if self.temporal_tracker is None:
            return
        self.temporal_tracker.set_anchor(image_bgr, detections, mask_records)

    def begin_temporal_refresh(self, image_bgr):
        if self.temporal_tracker is None:
            return None
        return self.temporal_tracker.begin_refresh_frame(image_bgr)

    def commit_temporal_refresh(self, image_bgr, detections, mask_records, refresh_state=None):
        if self.temporal_tracker is None:
            return
        self.temporal_tracker.commit_refresh_frame(
            image_bgr,
            detections,
            mask_records,
            refresh_state=refresh_state,
        )

    def propagate_temporal(self, image_bgr):
        if self.temporal_tracker is None:
            return [], []
        return self.temporal_tracker.propagate(image_bgr)

    def get_temporal_stats(self):
        if self.temporal_tracker is None:
            return {}
        return self.temporal_tracker.get_stats()

    def temporal_refresh_requested(self):
        if self.temporal_tracker is None:
            return False
        return self.temporal_tracker.wants_refresh()

    @staticmethod
    def _box_mask(shape_hw, detection):
        mask = np.zeros(shape_hw, dtype=np.uint8)
        mask[detection.y1:detection.y2, detection.x1:detection.x2] = 1
        return mask


class Sam2TemporalMaskTracker:
    def __init__(self, predictor, config):
        self.predictor = predictor
        self.config = dict(config)
        self.prompt_records = []
        self.frame_window = []
        self.object_metadata = {}
        self.local_object_counter = 1
        self.last_stats = {}
        self.inference_state = None
        self.sequence_dir = None
        self.last_frame_idx = -1
        self.rebase_count = 0
        self._torch = None
        self._img_mean = None
        self._img_std = None
        self.force_refresh_requested = False
        self.last_propagation_runtime_ms = 0.0
        self.max_propagation_runtime_ms = 0.0
        self.slow_propagation_events = 0
        self.slow_reset_count = 0
        self.last_removed_objects = 0
        self.last_anchor_candidates = 0
        self.last_anchor_inserted = 0
        self.last_anchor_skipped = 0
        self.last_anchor_mean_quality = 0.0
        self.last_inserted_anchor_mean_quality = 0.0
        self._update_stats()

    def reset(self):
        self.prompt_records = []
        self.frame_window = []
        self.object_metadata = {}
        self.local_object_counter = 1
        self.rebase_count = 0
        self.force_refresh_requested = False
        self.last_propagation_runtime_ms = 0.0
        self.max_propagation_runtime_ms = 0.0
        self.slow_propagation_events = 0
        self.slow_reset_count = 0
        self.last_removed_objects = 0
        self.last_anchor_candidates = 0
        self.last_anchor_inserted = 0
        self.last_anchor_skipped = 0
        self.last_anchor_mean_quality = 0.0
        self.last_inserted_anchor_mean_quality = 0.0
        self._dispose_inference_state()
        self._update_stats()

    def _history_max_frames(self):
        return max(int(self.config.get("history_max_frames", self.config.get("max_frames", 4))), 2)

    def _prompt_state_history_max_frames(self):
        default_prompt_history = max(self._history_max_frames() * 2, 24)
        return max(
            int(self.config.get("prompt_state_history_max_frames", default_prompt_history)),
            self._history_max_frames(),
        )

    def _key_prompt_history_max_frames(self):
        default_key_history = max(self._prompt_state_history_max_frames() * 2, 48)
        return max(
            int(self.config.get("key_prompt_history_max_frames", default_key_history)),
            self._prompt_state_history_max_frames(),
        )

    def _max_prompt_records_per_object(self):
        return max(int(self.config.get("max_prompt_records_per_object", 4)), 1)

    def _max_key_prompt_records_per_object(self):
        return max(
            min(
                int(self.config.get("max_key_prompt_records_per_object", 2)),
                self._max_prompt_records_per_object(),
            ),
            1,
        )

    def _rebase_max_frames(self):
        default_rebase = max(self._history_max_frames() * 2, int(self.config.get("max_frames", 4)) * 6, 24)
        return max(int(self.config.get("rebase_max_frames", default_rebase)), self._history_max_frames())

    def _object_idle_max_frames(self):
        default_idle = max(self._history_max_frames(), int(self.config.get("max_frames", 4)) * 2, 6)
        return max(int(self.config.get("object_max_idle_frames", default_idle)), 2)

    def _slow_propagation_limit_ms(self):
        return max(float(self.config.get("slow_propagation_sec", 8.0)), 0.0) * 1000.0

    def _slow_propagation_streak(self):
        return max(int(self.config.get("slow_propagation_streak", 1)), 1)

    def _slow_propagation_min_frame_idx(self):
        default_min_frame = max(self._history_max_frames(), int(self.config.get("max_frames", 4)) * 3)
        return max(int(self.config.get("slow_propagation_min_frame_idx", default_min_frame)), 0)

    def _predictor_device(self):
        device = getattr(self.predictor, "device", None)
        if device is not None:
            return str(device)
        return "cpu"

    def wants_refresh(self):
        return bool(self.force_refresh_requested)

    def _get_torch(self):
        if self._torch is None:
            import torch  # pylint: disable=import-error

            self._torch = torch
        return self._torch

    def _inference_contexts(self):
        inference_ctx = nullcontext()
        autocast_ctx = nullcontext()
        try:
            torch = self._get_torch()

            inference_ctx = torch.inference_mode()
            if self._predictor_device().startswith("cuda"):
                autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        except Exception:
            inference_ctx = nullcontext()
            autocast_ctx = nullcontext()
        return inference_ctx, autocast_ctx

    def _dispose_inference_state(self):
        self.last_frame_idx = -1
        if self.predictor is not None and self.inference_state is not None:
            try:
                self.predictor.reset_state(self.inference_state)
            except Exception:
                pass
        self.inference_state = None
        if self.sequence_dir is not None:
            self.sequence_dir.cleanup()
            self.sequence_dir = None

    def _init_kwargs(self):
        return {
            "offload_video_to_cpu": bool(self.config.get("offload_video_to_cpu", True)),
            "offload_state_to_cpu": bool(self.config.get("offload_state_to_cpu", False)),
            "async_loading_frames": bool(self.config.get("async_loading_frames", False)),
        }

    def _jpeg_quality(self):
        return int(np.clip(self.config.get("jpeg_quality", 95), 40, 100))

    def _prepare_frame_tensor(self, image_bgr):
        from PIL import Image  # pylint: disable=import-error

        torch = self._get_torch()
        image_size = int(getattr(self.predictor, "image_size", 1024))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = Image.fromarray(image_rgb).resize((image_size, image_size))
        img_np = np.asarray(resized, dtype=np.float32) / 255.0
        frame_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        if self.inference_state is None:
            return frame_tensor

        if self._img_mean is None:
            self._img_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
            self._img_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]

        offload_video_to_cpu = bool(self.inference_state.get("offload_video_to_cpu", True))
        target_device = self.inference_state.get("device")
        if not offload_video_to_cpu and target_device is not None:
            frame_tensor = frame_tensor.to(target_device, non_blocking=True)

        img_mean = self._img_mean.to(frame_tensor.device, non_blocking=True)
        img_std = self._img_std.to(frame_tensor.device, non_blocking=True)
        frame_tensor = frame_tensor.float()
        frame_tensor -= img_mean
        frame_tensor /= img_std
        return frame_tensor

    def _ensure_state_initialized(self, image_bgr):
        if self.inference_state is not None:
            return

        self.sequence_dir = tempfile.TemporaryDirectory(prefix="sam2_temporal_seq_")
        frame_path = os.path.join(self.sequence_dir.name, "00000.jpg")
        ok = cv2.imwrite(frame_path, image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality()])
        if not ok:
            raise RuntimeError(f"Failed to initialize SAM2 temporal state at {frame_path}")

        inference_ctx, autocast_ctx = self._inference_contexts()
        with inference_ctx:
            with autocast_ctx:
                self.inference_state = self.predictor.init_state(self.sequence_dir.name, **self._init_kwargs())
        images = self.inference_state.get("images")
        try:
            torch = self._get_torch()
            if isinstance(images, torch.Tensor):
                self.inference_state["images"] = [frame for frame in images]
        except Exception:
            pass
        self.last_frame_idx = 0

    def _trim_history(self, current_frame_idx=None):
        if current_frame_idx is None:
            current_frame_idx = self.last_frame_idx
        frame_cutoff = int(current_frame_idx) - self._history_max_frames() + 1
        self.frame_window = [idx for idx in self.frame_window if int(idx) >= frame_cutoff]
        regular_prompt_cutoff = int(current_frame_idx) - self._prompt_state_history_max_frames() + 1
        key_prompt_cutoff = int(current_frame_idx) - self._key_prompt_history_max_frames() + 1
        trimmed_prompt_records = []
        for item in self.prompt_records:
            frame_idx = int(item.get("frame_idx", -1))
            if frame_idx < 0:
                continue
            is_keyframe = bool(item.get("is_keyframe", False))
            keep_until_frame = int(
                item.get(
                    "keep_until_frame",
                    frame_idx
                    + (
                        self._key_prompt_history_max_frames()
                        if is_keyframe
                        else self._prompt_state_history_max_frames()
                    )
                    - 1,
                )
            )
            cutoff = key_prompt_cutoff if is_keyframe else regular_prompt_cutoff
            if frame_idx >= cutoff and keep_until_frame >= int(current_frame_idx):
                item["keep_until_frame"] = keep_until_frame
                trimmed_prompt_records.append(item)
        self.prompt_records = trimmed_prompt_records
        self._limit_prompt_records_per_object()

    def _limit_prompt_records_per_object(self):
        if not self.prompt_records:
            return

        max_total = self._max_prompt_records_per_object()
        max_key = self._max_key_prompt_records_per_object()
        grouped_records = {}
        for item in self.prompt_records:
            obj_id = int(item.get("obj_id", -1))
            if obj_id <= 0:
                continue
            grouped_records.setdefault(obj_id, []).append(item)

        limited_records = []
        for obj_id in sorted(grouped_records):
            records = grouped_records[obj_id]
            key_records = [item for item in records if bool(item.get("is_keyframe", False))]
            normal_records = [item for item in records if not bool(item.get("is_keyframe", False))]

            key_records_sorted = sorted(
                key_records,
                key=lambda item: (
                    int(item.get("frame_idx", -1)),
                    float(item.get("quality", 0.0)),
                ),
                reverse=True,
            )
            kept_key_records = []
            if key_records_sorted:
                kept_key_records.append(key_records_sorted[0])
                if len(key_records_sorted) > 1 and max_key > 1:
                    remaining_key_records = sorted(
                        key_records_sorted[1:],
                        key=lambda item: (
                            float(item.get("quality", 0.0)),
                            int(item.get("frame_idx", -1)),
                        ),
                        reverse=True,
                    )
                    kept_key_records.extend(remaining_key_records[: max_key - 1])

            remaining_slots = max(max_total - len(kept_key_records), 0)
            kept_normal_records = sorted(
                normal_records,
                key=lambda item: (
                    int(item.get("frame_idx", -1)),
                    float(item.get("quality", 0.0)),
                ),
                reverse=True,
            )[:remaining_slots]

            limited_records.extend(
                sorted(
                    kept_key_records + kept_normal_records,
                    key=lambda item: (int(item.get("frame_idx", -1)), int(item.get("obj_id", -1))),
                )
            )

        self.prompt_records = limited_records

    def _prompt_frames_by_object(self):
        prompt_frames = {}
        for item in self.prompt_records:
            obj_id = int(item.get("obj_id", -1))
            frame_idx = int(item.get("frame_idx", -1))
            if obj_id <= 0 or frame_idx < 0:
                continue
            prompt_frames.setdefault(obj_id, set()).add(frame_idx)
        return prompt_frames

    @staticmethod
    def _prune_frame_dict(frame_dict, keep_frames=None, cutoff=None, keep_latest=False):
        if not isinstance(frame_dict, dict) or not frame_dict:
            return frame_dict

        keep_frame_set = None if keep_frames is None else {int(frame_idx) for frame_idx in keep_frames}
        latest_frame = max(int(frame_idx) for frame_idx in frame_dict)
        kept = {}
        for frame_idx, value in frame_dict.items():
            frame_idx = int(frame_idx)
            keep = False
            if keep_frame_set is not None:
                keep = frame_idx in keep_frame_set
            elif cutoff is not None:
                keep = frame_idx >= int(cutoff)
            if keep_latest and frame_idx == latest_frame:
                keep = True
            if keep:
                kept[frame_idx] = value
        frame_dict.clear()
        frame_dict.update(kept)
        return frame_dict

    def _prune_prompt_state(self):
        if self.inference_state is None:
            return

        prompt_frames_by_object = self._prompt_frames_by_object()
        obj_id_to_idx = self.inference_state.get("obj_id_to_idx", {})
        point_inputs_per_obj = self.inference_state.get("point_inputs_per_obj", {})
        mask_inputs_per_obj = self.inference_state.get("mask_inputs_per_obj", {})
        output_dict_per_obj = self.inference_state.get("output_dict_per_obj", {})
        temp_output_dict_per_obj = self.inference_state.get("temp_output_dict_per_obj", {})

        for obj_id, obj_idx in list(obj_id_to_idx.items()):
            keep_frames = prompt_frames_by_object.get(int(obj_id), set())
            self._prune_frame_dict(point_inputs_per_obj.get(obj_idx, {}), keep_frames=keep_frames)
            self._prune_frame_dict(mask_inputs_per_obj.get(obj_idx, {}), keep_frames=keep_frames)

            obj_output = output_dict_per_obj.get(obj_idx, {})
            self._prune_frame_dict(
                obj_output.get("cond_frame_outputs", {}),
                keep_frames=keep_frames,
            )

            temp_output = temp_output_dict_per_obj.get(obj_idx, {})
            self._prune_frame_dict(
                temp_output.get("cond_frame_outputs", {}),
                keep_frames=keep_frames,
            )

    def _remove_object_from_state(self, obj_id):
        obj_id = int(obj_id)
        self.object_metadata.pop(obj_id, None)
        if self.predictor is None or self.inference_state is None:
            return
        if not hasattr(self.predictor, "remove_object"):
            return
        try:
            self.predictor.remove_object(self.inference_state, obj_id, strict=False, need_output=False)
        except Exception:
            pass

    def _prune_stale_objects(self, current_frame_idx):
        self.last_removed_objects = 0
        if not self.object_metadata:
            return

        cutoff = int(current_frame_idx) - self._object_idle_max_frames() + 1
        active_prompt_ids = {
            int(item.get("obj_id", -1))
            for item in self.prompt_records
            if int(item.get("frame_idx", -1)) >= cutoff
        }
        stale_obj_ids = [
            int(obj_id)
            for obj_id in list(self.object_metadata.keys())
            if int(obj_id) not in active_prompt_ids
        ]
        for obj_id in stale_obj_ids:
            self._remove_object_from_state(obj_id)
        self.last_removed_objects = int(len(stale_obj_ids))

    def _prune_objects_without_inputs(self):
        if self.inference_state is None or not self.object_metadata:
            return

        obj_id_to_idx = self.inference_state.get("obj_id_to_idx", {})
        point_inputs_per_obj = self.inference_state.get("point_inputs_per_obj", {})
        mask_inputs_per_obj = self.inference_state.get("mask_inputs_per_obj", {})
        output_dict_per_obj = self.inference_state.get("output_dict_per_obj", {})
        temp_output_dict_per_obj = self.inference_state.get("temp_output_dict_per_obj", {})

        removed = 0
        for obj_id in list(self.object_metadata.keys()):
            obj_idx = obj_id_to_idx.get(int(obj_id))
            if obj_idx is None:
                self._remove_object_from_state(obj_id)
                removed += 1
                continue

            has_points = bool(point_inputs_per_obj.get(obj_idx, {}))
            has_masks = bool(mask_inputs_per_obj.get(obj_idx, {}))
            has_outputs = bool(output_dict_per_obj.get(obj_idx, {}).get("cond_frame_outputs", {}))
            has_temp_outputs = bool(temp_output_dict_per_obj.get(obj_idx, {}).get("cond_frame_outputs", {}))
            if not (has_points or has_masks or has_outputs or has_temp_outputs):
                self._remove_object_from_state(obj_id)
                removed += 1

        if removed > 0:
            self.last_removed_objects += int(removed)

    def _handle_slow_propagation(self, last_frame_idx, runtime_ms):
        self.last_propagation_runtime_ms = float(runtime_ms)
        self.max_propagation_runtime_ms = max(float(self.max_propagation_runtime_ms), float(runtime_ms))

        runtime_limit_ms = self._slow_propagation_limit_ms()
        if runtime_limit_ms <= 0.0:
            self.slow_propagation_events = 0
            return

        is_slow = (
            float(runtime_ms) >= runtime_limit_ms
            and int(last_frame_idx) >= self._slow_propagation_min_frame_idx()
        )
        if is_slow:
            self.slow_propagation_events += 1
        else:
            self.slow_propagation_events = 0

        if self.slow_propagation_events < self._slow_propagation_streak():
            return

        if bool(self.config.get("force_refresh_on_slow_propagation", True)):
            self.force_refresh_requested = True

        if bool(self.config.get("reset_on_slow_propagation", True)):
            self.slow_reset_count += 1
            self.prompt_records = []
            self.frame_window = []
            self.object_metadata = {}
            self._dispose_inference_state()
            self.rebase_count = 0

    def _prune_inference_state(self, current_frame_idx):
        if self.inference_state is None:
            return

        cutoff = int(current_frame_idx) - self._history_max_frames() + 1

        cached_features = self.inference_state.get("cached_features", {})
        if isinstance(cached_features, dict):
            self.inference_state["cached_features"] = {
                int(frame_idx): value
                for frame_idx, value in cached_features.items()
                if int(frame_idx) >= cutoff
            }

        self._prune_prompt_state()
        self._prune_objects_without_inputs()

        for obj_output in self.inference_state.get("output_dict_per_obj", {}).values():
            self._prune_frame_dict(
                obj_output.get("non_cond_frame_outputs", {}),
                cutoff=cutoff,
                keep_latest=False,
            )

        for obj_output in self.inference_state.get("temp_output_dict_per_obj", {}).values():
            self._prune_frame_dict(
                obj_output.get("non_cond_frame_outputs", {}),
                cutoff=cutoff,
                keep_latest=False,
            )

        for frame_meta in self.inference_state.get("frames_tracked_per_obj", {}).values():
            self._prune_frame_dict(frame_meta, cutoff=cutoff, keep_latest=True)

    def _append_to_inference_state(self, image_bgr):
        if self.inference_state is None:
            self._ensure_state_initialized(image_bgr)
            return int(self.last_frame_idx)

        frame_tensor = self._prepare_frame_tensor(image_bgr)
        images = self.inference_state["images"]
        if hasattr(images, "images") and isinstance(images.images, list):
            images.images.append(frame_tensor)
        elif isinstance(images, list):
            images.append(frame_tensor)
        else:
            raise RuntimeError(f"Unsupported SAM2 image container type: {type(images)!r}")

        self.inference_state["num_frames"] = int(self.inference_state["num_frames"]) + 1
        self.last_frame_idx = int(self.inference_state["num_frames"]) - 1
        return self.last_frame_idx

    def _append_frame(self, image_bgr):
        frame_idx = self._append_to_inference_state(image_bgr)
        self.frame_window.append(int(frame_idx))
        self._trim_history()
        return frame_idx

    @staticmethod
    def _extract_missing_input_object_id(error_message):
        marker = "object id "
        if marker not in error_message:
            return -1
        suffix = error_message.split(marker, 1)[1]
        digits = []
        for char in suffix:
            if char.isdigit():
                digits.append(char)
            else:
                break
        return int("".join(digits)) if digits else -1

    def _update_stats(self, last_frame_idx=None):
        if last_frame_idx is None:
            last_frame_idx = self.last_frame_idx
        visible_prompt_cutoff = int(last_frame_idx) - self._history_max_frames() + 1
        prompt_frames = sorted(
            {
                int(item.get("frame_idx", 0))
                for item in self.prompt_records
                if int(item.get("frame_idx", -1)) >= visible_prompt_cutoff
            }
        )
        key_prompt_frames = sorted(
            {
                int(item.get("frame_idx", 0))
                for item in self.prompt_records
                if bool(item.get("is_keyframe", False)) and int(item.get("frame_idx", -1)) >= visible_prompt_cutoff
            }
        )
        if prompt_frames and last_frame_idx >= 0:
            oldest_prompt_age = max(int(last_frame_idx - prompt_frames[0]), 0)
            latest_prompt_age = max(int(last_frame_idx - prompt_frames[-1]), 0)
            propagation_span = int(oldest_prompt_age + 1)
        else:
            oldest_prompt_age = 0
            latest_prompt_age = 0
            propagation_span = 0
        self.last_stats = {
            "window_length": int(self.inference_state["num_frames"]) if self.inference_state is not None else 0,
            "prompt_records": int(len(self.prompt_records)),
            "prompt_frames": int(len(prompt_frames)),
            "key_prompt_records": int(sum(1 for item in self.prompt_records if bool(item.get("is_keyframe", False)))),
            "key_prompt_frames": int(len(key_prompt_frames)),
            "oldest_prompt_age": int(oldest_prompt_age),
            "latest_prompt_age": int(latest_prompt_age),
            "propagation_span_frames": int(propagation_span),
            "history_max_frames": int(self._history_max_frames()),
            "prompt_state_history_max_frames": int(self._prompt_state_history_max_frames()),
            "key_prompt_history_max_frames": int(self._key_prompt_history_max_frames()),
            "rebase_max_frames": int(self._rebase_max_frames()),
            "rebase_count": int(self.rebase_count),
            "active_objects": int(len(self.object_metadata)),
            "removed_objects": int(self.last_removed_objects),
            "last_anchor_candidates": int(self.last_anchor_candidates),
            "last_anchor_inserted": int(self.last_anchor_inserted),
            "last_anchor_skipped": int(self.last_anchor_skipped),
            "last_anchor_mean_quality": float(self.last_anchor_mean_quality),
            "last_inserted_anchor_mean_quality": float(self.last_inserted_anchor_mean_quality),
            "last_propagation_runtime_ms": float(self.last_propagation_runtime_ms),
            "max_propagation_runtime_ms": float(self.max_propagation_runtime_ms),
            "slow_propagation_events": int(self.slow_propagation_events),
            "slow_reset_count": int(self.slow_reset_count),
            "force_refresh_requested": bool(self.force_refresh_requested),
        }

    def get_stats(self):
        return dict(self.last_stats)

    def _should_rebase(self):
        if self.inference_state is None:
            return False
        return int(self.inference_state.get("num_frames", 0)) >= self._rebase_max_frames()

    def _reinitialize_from_anchor(self, image_bgr, records):
        self.prompt_records = []
        self.frame_window = []
        self.object_metadata = {}
        self._dispose_inference_state()

        current_frame_idx = self._append_frame(image_bgr)
        inference_ctx, autocast_ctx = self._inference_contexts()
        with inference_ctx:
            with autocast_ctx:
                for item in records:
                    self.object_metadata[int(item["obj_id"])] = {
                        "score": float(item["score"]),
                        "class_name": item["class_name"],
                        "class_id": int(item["class_id"]),
                        "source_name": item["source_name"],
                    }
                    self.predictor.add_new_mask(
                        self.inference_state,
                        frame_idx=int(current_frame_idx),
                        obj_id=int(item["obj_id"]),
                        mask=item["mask"].astype(bool),
                    )
                    self.prompt_records.append(
                        {
                            "obj_id": int(item["obj_id"]),
                            "frame_idx": int(current_frame_idx),
                            "quality": float(item.get("anchor_quality", 0.0)),
                            "is_keyframe": bool(item.get("anchor_is_keyframe", False)),
                            "keep_until_frame": int(
                                current_frame_idx + max(int(item.get("anchor_keep_frames", 1)), 1) - 1
                            ),
                        }
                    )

        self.rebase_count += 1
        self.force_refresh_requested = False
        self._trim_history(current_frame_idx)
        self._prune_stale_objects(current_frame_idx)
        self._update_stats(last_frame_idx=current_frame_idx)

    def _prepare_anchor_records(self, detections, mask_records):
        self.force_refresh_requested = False
        self.last_anchor_candidates = 0
        self.last_anchor_inserted = 0
        self.last_anchor_skipped = 0
        self.last_anchor_mean_quality = 0.0
        self.last_inserted_anchor_mean_quality = 0.0
        records = []
        candidate_qualities = []
        for detection, mask_record in zip(detections, mask_records):
            raw_mask = mask_record.get("mask") if isinstance(mask_record, dict) else mask_record
            if raw_mask is None:
                continue
            mask = np.asarray(raw_mask, dtype=np.uint8)
            if np.count_nonzero(mask) == 0:
                continue
            obj_id = int(getattr(detection, "track_id", -1))
            if obj_id <= 0:
                obj_id = self.local_object_counter
                self.local_object_counter += 1
            anchor_quality = float(mask_record.get("anchor_quality", 0.0))
            candidate_qualities.append(anchor_quality)
            records.append(
                {
                    "obj_id": int(obj_id),
                    "mask": (mask > 0).astype(np.uint8),
                    "score": float(getattr(detection, "score", 0.0)),
                    "class_name": str(getattr(detection, "class_name", "unknown")),
                    "class_id": int(getattr(detection, "class_id", -1)),
                    "source_name": str(getattr(detection, "source_name", "unknown")),
                    "anchor_quality": anchor_quality,
                    "anchor_is_keyframe": bool(mask_record.get("anchor_is_keyframe", False)),
                    "anchor_keep_frames": int(mask_record.get("anchor_keep_frames", self._prompt_state_history_max_frames())),
                    "anchor_memory_eligible": bool(mask_record.get("anchor_memory_eligible", True)),
                    "anchor_memory_bootstrap_eligible": bool(mask_record.get("anchor_memory_bootstrap_eligible", True)),
                }
            )

        self.last_anchor_candidates = int(len(records))
        if candidate_qualities:
            self.last_anchor_mean_quality = float(np.mean(candidate_qualities))

        accepted_records = []
        for item in records:
            obj_id = int(item["obj_id"])
            existing_object = obj_id in self.object_metadata
            if existing_object:
                if not bool(item.get("anchor_memory_eligible", True)):
                    self.last_anchor_skipped += 1
                    continue
            elif not bool(item.get("anchor_memory_bootstrap_eligible", item.get("anchor_memory_eligible", True))):
                self.last_anchor_skipped += 1
                continue
            accepted_records.append(item)

        self.last_anchor_inserted = int(len(accepted_records))
        if accepted_records:
            self.last_inserted_anchor_mean_quality = float(
                np.mean([float(item.get("anchor_quality", 0.0)) for item in accepted_records])
            )
        return accepted_records

    def _commit_anchor_records_to_frame(self, current_frame_idx, accepted_records):
        if not accepted_records:
            return

        inference_ctx, autocast_ctx = self._inference_contexts()
        self.prompt_records = [
            item for item in self.prompt_records if int(item.get("frame_idx", -1)) != int(current_frame_idx)
        ]
        with inference_ctx:
            with autocast_ctx:
                for item in accepted_records:
                    self.object_metadata[int(item["obj_id"])] = {
                        "score": float(item["score"]),
                        "class_name": item["class_name"],
                        "class_id": int(item["class_id"]),
                        "source_name": item["source_name"],
                    }
                    self.predictor.add_new_mask(
                        self.inference_state,
                        frame_idx=int(current_frame_idx),
                        obj_id=int(item["obj_id"]),
                        mask=item["mask"].astype(bool),
                    )
                    self.prompt_records.append(
                        {
                            "obj_id": int(item["obj_id"]),
                            "frame_idx": int(current_frame_idx),
                            "quality": float(item.get("anchor_quality", 0.0)),
                            "is_keyframe": bool(item.get("anchor_is_keyframe", False)),
                            "keep_until_frame": int(
                                current_frame_idx + max(int(item.get("anchor_keep_frames", 1)), 1) - 1
                            ),
                        }
                    )

    def set_anchor(self, image_bgr, detections, mask_records):
        self.commit_refresh_frame(image_bgr, detections, mask_records, refresh_state=None)

    def commit_refresh_frame(self, image_bgr, detections, mask_records, refresh_state=None):
        accepted_records = self._prepare_anchor_records(detections, mask_records)

        if accepted_records and self._should_rebase():
            self._reinitialize_from_anchor(image_bgr, accepted_records)
            return

        current_frame_idx = None
        if isinstance(refresh_state, dict):
            preview_frame_idx = int(refresh_state.get("frame_idx", -1))
            if self.inference_state is not None and preview_frame_idx == int(self.last_frame_idx):
                current_frame_idx = preview_frame_idx

        if current_frame_idx is None:
            current_frame_idx = self._append_frame(image_bgr)

        self._commit_anchor_records_to_frame(current_frame_idx, accepted_records)
        self._trim_history(current_frame_idx)
        self._prune_stale_objects(current_frame_idx)
        self._prune_inference_state(current_frame_idx)
        self._update_stats(last_frame_idx=current_frame_idx)

    def _predict_frame_masks(self, last_frame_idx):
        mask_threshold = float(self.config.get("mask_threshold", 0.0))
        min_mask_pixels = max(int(self.config.get("min_mask_pixels", 96)), 1)
        detections = []
        mask_records = []
        video_res_masks = None
        obj_ids = []
        propagation_start = time.time()
        inference_ctx, autocast_ctx = self._inference_contexts()
        retries = 0
        while retries < 2:
            try:
                with inference_ctx:
                    with autocast_ctx:
                        for frame_idx, current_obj_ids, current_masks in self.predictor.propagate_in_video(
                            self.inference_state,
                            start_frame_idx=int(last_frame_idx),
                            max_frame_num_to_track=1,
                        ):
                            if frame_idx == last_frame_idx:
                                obj_ids = [int(item) for item in current_obj_ids]
                                video_res_masks = current_masks.detach().cpu().numpy()
                                break
                break
            except RuntimeError as exc:
                message = str(exc)
                missing_obj_id = self._extract_missing_input_object_id(message)
                if missing_obj_id <= 0 or "No input points or masks are provided" not in message:
                    raise
                self._remove_object_from_state(missing_obj_id)
                retries += 1
                if self.inference_state is None or not self.object_metadata:
                    break
        propagation_runtime_ms = float((time.time() - propagation_start) * 1000.0)

        if video_res_masks is None:
            return [], [], propagation_runtime_ms

        for idx, obj_id in enumerate(obj_ids):
            anchor = self.object_metadata.get(int(obj_id))
            if anchor is None:
                continue
            logits = np.asarray(video_res_masks[idx, 0], dtype=np.float32)
            mask = (logits > mask_threshold).astype(np.uint8)
            mask_pixels = int(np.count_nonzero(mask))
            if mask_pixels < min_mask_pixels:
                continue
            ys, xs = np.where(mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue
            positive_logits = logits[mask > 0]
            if positive_logits.size == 0:
                sam_score = 0.0
            else:
                clipped_logits = np.clip(positive_logits, -20.0, 20.0)
                sam_score = float(np.mean(1.0 / (1.0 + np.exp(-clipped_logits))))
            detections.append(
                Detection(
                    x1=int(xs.min()),
                    y1=int(ys.min()),
                    x2=int(xs.max()) + 1,
                    y2=int(ys.max()) + 1,
                    score=float(anchor["score"]),
                    class_name=anchor["class_name"],
                    class_id=int(anchor["class_id"]),
                    source_name=anchor["source_name"],
                    track_id=int(obj_id),
                )
            )
            mask_records.append(
                {
                    "mask": mask,
                    "sam_score": float(np.clip(sam_score, 0.0, 1.0)),
                    "segmenter_mode": "sam2_video",
                    "temporal_track_id": int(obj_id),
                }
            )
        return detections, mask_records, propagation_runtime_ms

    def begin_refresh_frame(self, image_bgr):
        if self.predictor is None or self.inference_state is None or not self.object_metadata:
            return None

        current_frame_idx = self._append_frame(image_bgr)
        if current_frame_idx <= 0:
            return {
                "frame_idx": int(current_frame_idx),
                "detections": [],
                "mask_records": [],
                "propagation_runtime_ms": 0.0,
            }

        self._prune_stale_objects(current_frame_idx)
        self._prune_objects_without_inputs()
        if not self.object_metadata or self.inference_state is None:
            return {
                "frame_idx": int(current_frame_idx),
                "detections": [],
                "mask_records": [],
                "propagation_runtime_ms": 0.0,
            }

        detections, mask_records, propagation_runtime_ms = self._predict_frame_masks(current_frame_idx)
        self._handle_slow_propagation(current_frame_idx, propagation_runtime_ms)
        return {
            "frame_idx": int(current_frame_idx),
            "detections": detections,
            "mask_records": mask_records,
            "propagation_runtime_ms": float(propagation_runtime_ms),
        }

    def propagate(self, image_bgr):
        if self.predictor is None or not self.object_metadata:
            return [], []

        last_frame_idx = self._append_frame(image_bgr)
        if last_frame_idx <= 0:
            self._update_stats(last_frame_idx=last_frame_idx)
            return [], []

        self._prune_stale_objects(last_frame_idx)
        self._prune_objects_without_inputs()
        if not self.object_metadata or self.inference_state is None:
            self._update_stats(last_frame_idx=last_frame_idx)
            return [], []

        detections, mask_records, propagation_runtime_ms = self._predict_frame_masks(last_frame_idx)
        self._handle_slow_propagation(last_frame_idx, propagation_runtime_ms)
        self._prune_inference_state(last_frame_idx)
        self._update_stats(last_frame_idx=last_frame_idx)
        return detections, mask_records


# 主流程类：当前仓库的动态目标过滤前端主入口。
# 外部脚本通常只需要构造这个类，然后对每一帧调用 process()。
class WorldSamFilterPipeline:
    def __init__(self, config_path=None):
        # 初始化配置、检测器、分割器，以及跨帧状态缓存。
        self.config = load_pipeline_config(config_path)
        self.detector_ensemble = DetectorEnsemble(self.config)
        self.segmenter = SamSegmenter(self.config["segmenter"])
        self.frame_index = 0
        self.prev_gray = None
        self.prev_mask_records = []
        self.prev_detections = []
        self.last_source_counts = {}
        self.tracks = {}
        self.next_track_id = 1
        self.scene_dynamic_context = 0.0

    def _stage_mask_schedule(self):
        return dict(self.config.get("mask_schedule", {}))

    def _current_mask_stage(self):
        cfg = self._stage_mask_schedule()
        if not cfg.get("enabled", False):
            return "default"
        sam_frames = max(int(cfg.get("sam_frames", 0)), 0)
        if self.frame_index <= sam_frames:
            return "sam_init"
        return str(cfg.get("post_init_mode", "box"))

    def _apply_mask_schedule(self, detections, mask_records, image_shape):
        cfg = self._stage_mask_schedule()
        stage = self._current_mask_stage()
        if not cfg.get("enabled", False) or stage in {"default", "sam_init"}:
            return mask_records, stage

        scheduled_records = []
        dilate_pixels = max(int(cfg.get("post_init_box_dilate_pixels", 0)), 0)
        for detection, mask_record in zip(detections, mask_records):
            record = dict(mask_record or {})
            box_mask = self.segmenter._box_mask(image_shape[:2], detection)
            if stage in {"dilated_box", "dilated-box"} and dilate_pixels > 0:
                box_mask = dilate_mask(box_mask, dilate_pixels)
            record["mask"] = box_mask.astype(np.uint8)
            record["segmenter_mode"] = stage
            record["mask_schedule_stage"] = stage
            scheduled_records.append(record)
        return scheduled_records, stage

    def _split_auxiliary_mask_instances(self, detections, mask_records):
        auxiliary_sources = {
            str(item).strip()
            for item in self.config.get("auxiliary_mask_sources", [])
            if str(item).strip()
        }
        if not auxiliary_sources:
            return detections, mask_records, [], []

        primary_detections = []
        primary_mask_records = []
        auxiliary_detections = []
        auxiliary_mask_records = []
        for detection, mask_record in zip(detections, mask_records):
            if detection.source_name in auxiliary_sources:
                auxiliary_detections.append(detection)
                auxiliary_mask_records.append(mask_record)
            else:
                primary_detections.append(detection)
                primary_mask_records.append(mask_record)
        return primary_detections, primary_mask_records, auxiliary_detections, auxiliary_mask_records

    @staticmethod
    def _point_in_box(x, y, detection, margin=0.0):
        return (
            x >= float(detection.x1) - margin
            and x <= float(detection.x2) + margin
            and y >= float(detection.y1) - margin
            and y <= float(detection.y2) + margin
        )

    def _augment_primary_masks_with_auxiliary(self, detections, mask_records, auxiliary_detections, auxiliary_mask_records):
        if not detections or not auxiliary_detections:
            return mask_records

        cfg = self.config.get("auxiliary_mask_merge", {})
        if not cfg.get("enabled", True):
            return mask_records

        min_iou = float(cfg.get("min_iou", 0.02))
        center_margin = float(cfg.get("center_margin_pixels", 12.0))
        dilate_pixels = int(cfg.get("aux_dilate_pixels", 3))
        max_area_ratio = float(cfg.get("max_aux_area_ratio", 0.35))
        support_classes = {
            canonical_class_name(item)
            for item in cfg.get("support_classes", ["person", "pedestrian", "worker"])
        }

        merged_records = [dict(item) for item in mask_records]
        for index, (detection, mask_record) in enumerate(zip(detections, merged_records)):
            if detection.canonical_name not in support_classes:
                continue

            base_mask = mask_record.get("mask", mask_record).astype(np.uint8)
            if np.count_nonzero(base_mask) == 0:
                continue

            base_area = float(np.count_nonzero(base_mask))
            union_mask = base_mask.copy()
            merged_aux_count = 0
            for aux_detection, aux_mask_record in zip(auxiliary_detections, auxiliary_mask_records):
                aux_mask = aux_mask_record.get("mask", aux_mask_record).astype(np.uint8)
                if np.count_nonzero(aux_mask) == 0:
                    continue
                bbox_iou = compute_box_iou(
                    (detection.x1, detection.y1, detection.x2, detection.y2),
                    (aux_detection.x1, aux_detection.y1, aux_detection.x2, aux_detection.y2),
                )
                aux_center = mask_center(aux_mask, aux_detection)
                center_inside = self._point_in_box(aux_center[0], aux_center[1], detection, margin=center_margin)
                if bbox_iou < min_iou and not center_inside:
                    continue

                candidate_mask = aux_mask
                if dilate_pixels > 1:
                    candidate_mask = dilate_mask(candidate_mask, dilate_pixels)

                candidate_area = float(np.count_nonzero(candidate_mask))
                if candidate_area > max(base_area * max_area_ratio, 1.0):
                    continue

                union_mask = np.maximum(union_mask, candidate_mask).astype(np.uint8)
                merged_aux_count += 1

            if merged_aux_count > 0:
                merged_record = dict(mask_record)
                merged_record["mask"] = union_mask
                merged_record["auxiliary_mask_merges"] = int(merged_aux_count)
                merged_record["segmenter_mode"] = f"{mask_record.get('segmenter_mode', 'unknown')}+aux"
                merged_records[index] = merged_record
        return merged_records

    def process(self, image_bgr, depth_mm=None):
        # 单帧主入口：检测/传播 -> 分割 -> 打分 -> 时序记忆 -> 门控 -> 输出过滤结果。
        self.frame_index += 1
        start_time = time.time()
        current_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # 为了兼顾实时性，并不是每一帧都重新检测；中间帧可以走传播分支。
        should_refresh = (
            not self.prev_mask_records
            or self.frame_index == 1
            or self.frame_index % int(self.config["runtime"]["detector_interval"]) == 0
            or self.segmenter.temporal_refresh_requested()
        )

        refresh_state = None
        refresh_fusion_summary = {
            "preview_candidates": 0,
            "matched": 0,
            "strong_matched": 0,
            "mean_match_score": 0.0,
            "mean_id_consistency": 0.0,
        }
        if should_refresh:
            # 刷新帧：重新做检测和 SAM 分割，得到新的候选实例。
            refresh_state = self.segmenter.begin_temporal_refresh(image_bgr)
            detections, source_counts = self.detector_ensemble.predict(image_bgr)
            mask_records = self.segmenter.segment(image_bgr, detections)
            detections, mask_records, refresh_fusion_summary = self._fuse_refresh_with_temporal_preview(
                detections,
                mask_records,
                refresh_state,
            )
            self.last_source_counts = source_counts
        else:
            # 传播帧：优先利用 SAM2 video predictor 传播实例 mask / ID，失败时再回退到光流。
            detections, mask_records = self.segmenter.propagate_temporal(image_bgr)
            if not mask_records:
                detections, mask_records = self._propagate_masks(image_bgr)
            source_counts = self.last_source_counts
            mask_records = [self._ensure_temporal_fusion_metadata(item) for item in mask_records]

        detections, mask_records, auxiliary_detections, auxiliary_mask_records = self._split_auxiliary_mask_instances(
            detections,
            mask_records,
        )
        mask_records = self._augment_primary_masks_with_auxiliary(
            detections,
            mask_records,
            auxiliary_detections,
            auxiliary_mask_records,
        )
        mask_records, mask_schedule_stage = self._apply_mask_schedule(
            detections,
            mask_records,
            image_bgr.shape,
        )

        all_masks = []
        filtered_masks = []
        foundation_details = []
        task_details = []
        segment_records = []
        instance_records = []

        # 对每个候选实例做单帧细化与打分。
        for detection, mask_record in zip(detections, mask_records):
            raw_mask = mask_record.get("mask", mask_record).astype(np.uint8)
            sam_score = float(mask_record.get("sam_score", detection.score))
            refined_mask = self._refine_mask(raw_mask, detection, image_bgr, depth_mm)
            foundation = self._score_foundation_reliability(
                detection,
                refined_mask,
                image_bgr.shape[:2],
                sam_score,
                image_bgr=image_bgr,
                depth_mm=depth_mm,
            )
            task = self._score_task_relevance(detection, refined_mask, image_bgr.shape[:2], depth_mm, sam_score=sam_score, foundation_score=foundation["score"])
            all_masks.append(refined_mask)
            foundation_details.append(foundation)
            task_details.append(task)
            segment_record = dict(mask_record)
            segment_record.update(
                {
                    "mask": refined_mask,
                    "sam_score": sam_score,
                    "segmenter_mode": mask_record.get("segmenter_mode", "unknown"),
                    "mask_schedule_stage": mask_record.get("mask_schedule_stage", mask_schedule_stage),
                }
            )
            segment_records.append(segment_record)

        geometry_details = self._compute_geometry_consistency(current_gray, detections, all_masks)

        # 跨帧更新实例轨迹，累计运动、静态和动态记忆证据。
        track_infos = self._update_panoptic_memory(
            detections,
            all_masks,
            task_details,
            foundation_details,
            geometry_details,
            segment_records,
            depth_mm,
            image_bgr.shape[:2],
            image_bgr,
        )
        relevance_details = []
        # 最终门控：决定每个实例是保留还是过滤。
        for detection, mask, foundation, task, track_info, segment_record in zip(
            detections,
            all_masks,
            foundation_details,
            task_details,
            track_infos,
            segment_records,
        ):
            decision = self._apply_panoptic_gate(detection, foundation, task, track_info)
            relevance_details.append(decision)
            if decision["filter_out"]:
                filtered_masks.append(mask)
            detection.track_id = int(track_info.get("track_id", -1))
            temporal_fusion_score = float(
                np.clip(
                    max(
                        decision.get("temporal_fusion_score", 0.0),
                        segment_record.get("temporal_fusion_confidence", 0.0),
                    ),
                    0.0,
                    1.0,
                )
            )
            temporal_id_consistency = float(
                np.clip(
                    max(
                        decision.get("temporal_id_consistency", 0.0),
                        segment_record.get("anchor_temporal_id_consistency", 0.0),
                    ),
                    0.0,
                    1.0,
                )
            )
            temporal_mask_agreement = float(
                np.clip(
                    max(
                        decision.get("temporal_mask_agreement", 0.0),
                        segment_record.get("temporal_mask_agreement", 0.0),
                    ),
                    0.0,
                    1.0,
                )
            )
            backend_dynamic_score = float(
                np.clip(
                    max(
                        decision.get("dynamic_memory_score", 0.0),
                        decision.get("motion_score", 0.0),
                        decision.get("tube_motion_score", 0.0),
                        decision.get("geometry_dynamic_score", 0.0),
                        (
                            0.55 * temporal_fusion_score + 0.45 * temporal_id_consistency
                            if decision.get("filter_out", False)
                            else 0.0
                        ),
                    ),
                    0.0,
                    1.0,
                )
            )
            backend_temporal_consistency = float(
                np.clip(
                    max(
                        decision.get("track_confirmation", 0.0),
                        decision.get("dynamic_memory_score", 0.0),
                        0.48 * temporal_fusion_score + 0.32 * temporal_id_consistency + 0.20 * temporal_mask_agreement,
                    ),
                    0.0,
                    1.0,
                )
            )
            instance_records.append(
                {
                    "track_id": int(decision.get("track_id", -1)),
                    "class_name": str(decision.get("class_name", detection.class_name)),
                    "canonical_name": str(decision.get("canonical_name", detection.canonical_name)),
                    "source_name": str(decision.get("source_name", detection.source_name)),
                    "x1": int(detection.x1),
                    "y1": int(detection.y1),
                    "x2": int(detection.x2),
                    "y2": int(detection.y2),
                    "score": float(decision.get("score", task.get("score", 0.0))),
                    "foundation_score": float(decision.get("foundation_score", foundation.get("score", 0.0))),
                    "track_confirmation": float(decision.get("track_confirmation", 0.0)),
                    "dynamic_memory_score": float(decision.get("dynamic_memory_score", 0.0)),
                    "motion_score": float(decision.get("motion_score", 0.0)),
                    "tube_motion_score": float(decision.get("tube_motion_score", 0.0)),
                    "geometry_dynamic_score": float(decision.get("geometry_dynamic_score", 0.0)),
                    "geometry_static_score": float(decision.get("geometry_static_score", 0.0)),
                    "scene_dynamic_context": float(decision.get("scene_dynamic_context", 0.0)),
                    "temporal_fusion_score": temporal_fusion_score,
                    "temporal_id_consistency": temporal_id_consistency,
                    "temporal_mask_agreement": temporal_mask_agreement,
                    "backend_dynamic_score": backend_dynamic_score,
                    "backend_temporal_consistency": backend_temporal_consistency,
                    "filter_out": bool(decision.get("filter_out", False)),
                    "decision_reason": str(decision.get("decision_reason", "unknown")),
                    "gate_stage": str(decision.get("gate_stage", "other")),
                }
            )

        for segment_record, foundation, task, decision in zip(
            segment_records,
            foundation_details,
            task_details,
            relevance_details,
        ):
            segment_record.update(
                self._build_temporal_anchor_metadata(
                    foundation,
                    task,
                    decision,
                    segment_record,
                )
            )

        gate_stage_summary = self._summarize_gate_stages(relevance_details)

        # 将被判定为动态的实例掩膜合并，并同步作用到 RGB / Depth。
        merged_mask = self._merge_masks(filtered_masks, image_bgr.shape[:2])
        filtered_rgb = self._filter_rgb(image_bgr, merged_mask)
        filtered_depth = self._filter_depth(depth_mm, merged_mask)
        overlay = self._build_overlay(image_bgr, detections, all_masks, relevance_details)

        self.prev_gray = current_gray
        self.prev_mask_records = segment_records
        self.prev_detections = detections
        if should_refresh:
            self.segmenter.commit_temporal_refresh(
                image_bgr,
                detections,
                segment_records,
                refresh_state=refresh_state,
            )
        temporal_stats = self.segmenter.get_temporal_stats()

        runtime_ms = (time.time() - start_time) * 1000.0
        filtered_count = sum(1 for item in relevance_details if item["filter_out"])
        return {
            "filtered_rgb": filtered_rgb,
            "filtered_depth": filtered_depth,
            "overlay": overlay,
            "mask": merged_mask,
            "detections": detections,
            "stats": {
                "frame_index": self.frame_index,
                "detections": len(detections),
                "filtered_detections": filtered_count,
                "detections_by_source": source_counts,
                "mask_ratio": float(np.count_nonzero(merged_mask)) / float(merged_mask.size),
                "mean_relevance": float(np.mean([item["score"] for item in relevance_details])) if relevance_details else 0.0,
                "mean_foundation": float(np.mean([item.get("foundation_score", 0.0) for item in relevance_details])) if relevance_details else 0.0,
                "mean_motion": float(np.mean([item.get("motion_score", 0.0) for item in relevance_details])) if relevance_details else 0.0,
                "mean_geometry_dynamic": float(np.mean([item.get("geometry_dynamic_score", 0.0) for item in relevance_details])) if relevance_details else 0.0,
                "mean_tube_motion": float(np.mean([item.get("tube_motion_score", 0.0) for item in relevance_details])) if relevance_details else 0.0,
                "mean_track_confirmation": float(np.mean([item.get("track_confirmation", 0.0) for item in relevance_details])) if relevance_details else 0.0,
                "mean_dynamic_memory": float(np.mean([item.get("dynamic_memory_score", 0.0) for item in relevance_details])) if relevance_details else 0.0,
                "mean_temporal_fusion_score": float(np.mean([item.get("temporal_fusion_score", 0.0) for item in instance_records])) if instance_records else 0.0,
                "mean_temporal_id_consistency": float(np.mean([item.get("temporal_id_consistency", 0.0) for item in instance_records])) if instance_records else 0.0,
                "scene_dynamic_context": float(self.scene_dynamic_context),
                "mask_schedule_stage": str(mask_schedule_stage),
                "temporal_refresh_preview_candidates": int(refresh_fusion_summary.get("preview_candidates", 0)),
                "temporal_refresh_matches": int(refresh_fusion_summary.get("matched", 0)),
                "temporal_refresh_strong_matches": int(refresh_fusion_summary.get("strong_matched", 0)),
                "temporal_refresh_mean_match_score": float(refresh_fusion_summary.get("mean_match_score", 0.0)),
                "temporal_refresh_mean_id_consistency": float(refresh_fusion_summary.get("mean_id_consistency", 0.0)),
                "temporal_window_length": int(temporal_stats.get("window_length", 0)),
                "temporal_prompt_records": int(temporal_stats.get("prompt_records", 0)),
                "temporal_prompt_frames": int(temporal_stats.get("prompt_frames", 0)),
                "temporal_key_prompt_records": int(temporal_stats.get("key_prompt_records", 0)),
                "temporal_key_prompt_frames": int(temporal_stats.get("key_prompt_frames", 0)),
                "temporal_oldest_prompt_age": int(temporal_stats.get("oldest_prompt_age", 0)),
                "temporal_latest_prompt_age": int(temporal_stats.get("latest_prompt_age", 0)),
                "temporal_propagation_span_frames": int(temporal_stats.get("propagation_span_frames", 0)),
                "temporal_history_max_frames": int(temporal_stats.get("history_max_frames", 0)),
                "temporal_prompt_state_history_max_frames": int(temporal_stats.get("prompt_state_history_max_frames", 0)),
                "temporal_key_prompt_history_max_frames": int(temporal_stats.get("key_prompt_history_max_frames", 0)),
                "temporal_rebase_max_frames": int(temporal_stats.get("rebase_max_frames", 0)),
                "temporal_rebase_count": int(temporal_stats.get("rebase_count", 0)),
                "temporal_active_objects": int(temporal_stats.get("active_objects", 0)),
                "temporal_removed_objects": int(temporal_stats.get("removed_objects", 0)),
                "temporal_last_anchor_candidates": int(temporal_stats.get("last_anchor_candidates", 0)),
                "temporal_last_anchor_inserted": int(temporal_stats.get("last_anchor_inserted", 0)),
                "temporal_last_anchor_skipped": int(temporal_stats.get("last_anchor_skipped", 0)),
                "temporal_last_anchor_mean_quality": float(temporal_stats.get("last_anchor_mean_quality", 0.0)),
                "temporal_last_inserted_anchor_mean_quality": float(temporal_stats.get("last_inserted_anchor_mean_quality", 0.0)),
                "temporal_last_runtime_ms": float(temporal_stats.get("last_propagation_runtime_ms", 0.0)),
                "temporal_max_runtime_ms": float(temporal_stats.get("max_propagation_runtime_ms", 0.0)),
                "temporal_slow_events": int(temporal_stats.get("slow_propagation_events", 0)),
                "temporal_slow_reset_count": int(temporal_stats.get("slow_reset_count", 0)),
                "temporal_force_refresh_requested": bool(temporal_stats.get("force_refresh_requested", False)),
                "task_relevance_threshold": float(self.config.get("task_relevance", {}).get("min_score", 0.0)),
                "gate_stage_counts_total": gate_stage_summary["total_counts"],
                "gate_stage_counts_filtered": gate_stage_summary["filtered_counts"],
                "gate_stage_counts_kept": gate_stage_summary["kept_counts"],
                "instance_records": instance_records,
                "relevance_details": relevance_details,
                "runtime_ms": runtime_ms,
                "mode": "refresh" if should_refresh else "propagate",
            },
        }

    @staticmethod
    def _decision_reason_to_gate_stage(decision_reason):
        mapping = {
            "low_task_score": "stage1_task_gate",
            "low_foundation_score": "stage2_foundation_gate",
            "panoptic_static_protection": "stage3_static_protection",
            "weak_dynamic_guard": "stage4_weak_dynamic_guard",
            "dynamic_memory_propagation": "stage5_dynamic_memory_gate",
            "motion_supported": "stage6_motion_gate",
            "tube_motion_supported": "stage7_tube_motion_gate",
            "high_confidence_override": "stage8_high_confidence_override",
            "confirmed_track_support": "stage9_confirmed_track_gate",
            "rigid_static_prior": "keep_rigid_static_prior",
            "insufficient_temporal_support": "keep_insufficient_temporal_support",
        }
        return mapping.get(str(decision_reason), "other")

    @classmethod
    def _summarize_gate_stages(cls, relevance_details):
        total_counts = {}
        filtered_counts = {}
        kept_counts = {}
        for item in relevance_details:
            stage = cls._decision_reason_to_gate_stage(item.get("decision_reason", "other"))
            total_counts[stage] = int(total_counts.get(stage, 0)) + 1
            if item.get("filter_out", False):
                filtered_counts[stage] = int(filtered_counts.get(stage, 0)) + 1
            else:
                kept_counts[stage] = int(kept_counts.get(stage, 0)) + 1
        return {
            "total_counts": total_counts,
            "filtered_counts": filtered_counts,
            "kept_counts": kept_counts,
        }

    @staticmethod
    def _mask_bbox(mask, fallback_detection=None):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            if fallback_detection is None:
                return 0, 0, 0, 0
            return (
                int(fallback_detection.x1),
                int(fallback_detection.y1),
                int(fallback_detection.x2),
                int(fallback_detection.y2),
            )
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    def _ensure_temporal_fusion_metadata(self, mask_record):
        record = dict(mask_record or {})
        temporal_track_id = int(record.get("temporal_track_id", -1))
        record.setdefault("temporal_track_id", temporal_track_id)
        record.setdefault("temporal_mask_agreement", 0.0)
        record.setdefault("temporal_box_agreement", 0.0)
        record.setdefault("temporal_score_agreement", 0.0)
        record.setdefault("temporal_fusion_confidence", 0.0)
        record.setdefault("temporal_preview_sam_score", 0.0)
        record.setdefault("temporal_fusion_mode", "none")
        record.setdefault("temporal_source_used", False)
        record.setdefault("temporal_match_found", False)
        record.setdefault("temporal_id_hint", 1.0 if temporal_track_id > 0 else 0.0)
        return record

    def _temporal_refresh_fusion_config(self):
        temporal_cfg = self.config.get("segmenter", {}).get("temporal_tracking", {})
        fusion_cfg = temporal_cfg.get("refresh_fusion", {})
        weights = dict(fusion_cfg.get("weights", {}))
        return {
            "enabled": bool(fusion_cfg.get("enabled", False)),
            "class_agnostic": bool(fusion_cfg.get("class_agnostic", False)),
            "min_mask_iou": float(fusion_cfg.get("min_mask_iou", 0.12)),
            "min_box_iou": float(fusion_cfg.get("min_box_iou", 0.08)),
            "match_score_threshold": float(fusion_cfg.get("match_score_threshold", 0.34)),
            "strong_match_score": float(fusion_cfg.get("strong_match_score", 0.62)),
            "max_area_ratio": float(fusion_cfg.get("max_area_ratio", 2.4)),
            "mask_expand_pixels": max(int(fusion_cfg.get("mask_expand_pixels", 5)), 0),
            "weights": {
                "mask_iou": float(weights.get("mask_iou", 0.46)),
                "box_iou": float(weights.get("box_iou", 0.20)),
                "score_agreement": float(weights.get("score_agreement", 0.18)),
                "class_match": float(weights.get("class_match", 0.16)),
            },
        }

    def _fuse_refresh_with_temporal_preview(self, detections, mask_records, refresh_state):
        mask_records = [self._ensure_temporal_fusion_metadata(item) for item in mask_records]
        summary = {
            "preview_candidates": 0,
            "matched": 0,
            "strong_matched": 0,
            "mean_match_score": 0.0,
            "mean_id_consistency": 0.0,
        }
        if not detections or not mask_records:
            return detections, mask_records, summary

        cfg = self._temporal_refresh_fusion_config()
        if not cfg["enabled"] or not isinstance(refresh_state, dict):
            return detections, mask_records, summary

        temporal_detections = list(refresh_state.get("detections", []))
        temporal_records = [self._ensure_temporal_fusion_metadata(item) for item in refresh_state.get("mask_records", [])]
        summary["preview_candidates"] = int(min(len(temporal_detections), len(temporal_records)))
        if not temporal_detections or not temporal_records:
            return detections, mask_records, summary

        weights = cfg["weights"]
        total_weight = max(sum(float(value) for value in weights.values()), 1e-6)
        used_temporal = set()
        match_scores = []
        id_scores = []

        for detection, mask_record in zip(detections, mask_records):
            current_mask = np.asarray(mask_record.get("mask"), dtype=np.uint8)
            if current_mask is None or np.count_nonzero(current_mask) == 0:
                continue

            current_box = (detection.x1, detection.y1, detection.x2, detection.y2)
            current_score = float(mask_record.get("sam_score", detection.score))
            current_area = float(max(np.count_nonzero(current_mask), 1))
            best_idx = -1
            best_score = -1.0
            best_payload = None

            for idx, (temporal_detection, temporal_record) in enumerate(zip(temporal_detections, temporal_records)):
                if idx in used_temporal:
                    continue
                if (
                    not cfg["class_agnostic"]
                    and detection.canonical_name != temporal_detection.canonical_name
                ):
                    continue

                temporal_mask = np.asarray(temporal_record.get("mask"), dtype=np.uint8)
                if temporal_mask is None or np.count_nonzero(temporal_mask) == 0:
                    continue

                temporal_box = (
                    int(temporal_detection.x1),
                    int(temporal_detection.y1),
                    int(temporal_detection.x2),
                    int(temporal_detection.y2),
                )
                mask_iou = compute_mask_iou(current_mask, temporal_mask)
                box_iou = compute_box_iou(current_box, temporal_box)
                if mask_iou < cfg["min_mask_iou"] and box_iou < cfg["min_box_iou"]:
                    continue

                temporal_area = float(max(np.count_nonzero(temporal_mask), 1))
                area_ratio = max(current_area, temporal_area) / max(min(current_area, temporal_area), 1.0)
                if area_ratio > cfg["max_area_ratio"]:
                    continue

                temporal_score = float(temporal_record.get("sam_score", temporal_detection.score))
                score_agreement = float(np.clip(1.0 - abs(current_score - temporal_score), 0.0, 1.0))
                class_match = 1.0 if detection.canonical_name == temporal_detection.canonical_name else 0.0
                match_score = float(
                    np.clip(
                        (
                            weights["mask_iou"] * mask_iou
                            + weights["box_iou"] * box_iou
                            + weights["score_agreement"] * score_agreement
                            + weights["class_match"] * class_match
                        )
                        / total_weight,
                        0.0,
                        1.0,
                    )
                )
                if match_score > best_score:
                    best_idx = idx
                    best_score = match_score
                    best_payload = {
                        "temporal_detection": temporal_detection,
                        "temporal_record": temporal_record,
                        "temporal_mask": temporal_mask,
                        "mask_iou": float(mask_iou),
                        "box_iou": float(box_iou),
                        "score_agreement": score_agreement,
                        "temporal_score": temporal_score,
                    }

            if best_idx < 0 or best_payload is None or best_score < cfg["match_score_threshold"]:
                continue

            used_temporal.add(best_idx)
            match_scores.append(float(best_score))

            temporal_detection = best_payload["temporal_detection"]
            temporal_mask = best_payload["temporal_mask"]
            temporal_track_id = int(getattr(temporal_detection, "track_id", -1))
            id_consistency = 1.0 if temporal_track_id > 0 else 0.0
            id_scores.append(id_consistency)
            strong_match = bool(best_score >= cfg["strong_match_score"])

            fused_mask = current_mask.astype(np.uint8)
            fusion_mode = "soft_current_only"
            if strong_match:
                guided_temporal = temporal_mask.astype(np.uint8)
                if cfg["mask_expand_pixels"] > 0:
                    current_guard = dilate_mask(current_mask, cfg["mask_expand_pixels"])
                    guided_temporal = np.logical_and(guided_temporal > 0, current_guard > 0).astype(np.uint8)
                if np.count_nonzero(guided_temporal) > 0:
                    fused_mask = np.maximum(fused_mask, guided_temporal).astype(np.uint8)
                    fusion_mode = "strong_guided_union"
                else:
                    fusion_mode = "strong_current_only"
                summary["strong_matched"] += 1

            if temporal_track_id > 0:
                detection.track_id = temporal_track_id

            detection.x1, detection.y1, detection.x2, detection.y2 = self._mask_bbox(
                fused_mask,
                fallback_detection=detection,
            )
            blended_sam_score = float(
                np.clip(
                    0.78 * current_score + 0.22 * best_payload["temporal_score"],
                    0.0,
                    1.0,
                )
            )
            mask_record.update(
                {
                    "mask": fused_mask,
                    "sam_score": blended_sam_score,
                    "temporal_track_id": temporal_track_id,
                    "temporal_mask_agreement": float(best_payload["mask_iou"]),
                    "temporal_box_agreement": float(best_payload["box_iou"]),
                    "temporal_score_agreement": float(best_payload["score_agreement"]),
                    "temporal_fusion_confidence": float(best_score),
                    "temporal_preview_sam_score": float(best_payload["temporal_score"]),
                    "temporal_fusion_mode": fusion_mode,
                    "temporal_source_used": True,
                    "temporal_match_found": True,
                    "temporal_id_hint": id_consistency,
                }
            )

        summary["matched"] = int(len(match_scores))
        if match_scores:
            summary["mean_match_score"] = float(np.mean(match_scores))
        if id_scores:
            summary["mean_id_consistency"] = float(np.mean(id_scores))
        return detections, mask_records, summary

    def _temporal_memory_policy_config(self):
        temporal_cfg = self.config.get("segmenter", {}).get("temporal_tracking", {})
        regular_history = max(int(temporal_cfg.get("prompt_state_history_max_frames", temporal_cfg.get("history_max_frames", 12) * 2)), 1)
        key_history = max(int(temporal_cfg.get("key_prompt_history_max_frames", regular_history * 2)), regular_history)
        return {
            "regular_history_frames": regular_history,
            "key_history_frames": key_history,
            "key_min_quality": float(temporal_cfg.get("key_prompt_min_quality", 0.70)),
            "key_min_track_confirmation": float(temporal_cfg.get("key_prompt_min_track_confirmation", 0.60)),
            "key_min_dynamic_score": float(temporal_cfg.get("key_prompt_min_dynamic_score", 0.55)),
            "key_min_foundation_score": float(temporal_cfg.get("key_prompt_min_foundation_score", 0.50)),
            "memory_min_quality": float(temporal_cfg.get("memory_min_quality", 0.64)),
            "memory_min_foundation_score": float(temporal_cfg.get("memory_min_foundation_score", 0.50)),
            "memory_min_track_confirmation": float(temporal_cfg.get("memory_min_track_confirmation", 0.52)),
            "memory_bootstrap_min_quality": float(temporal_cfg.get("memory_bootstrap_min_quality", 0.56)),
            "memory_bootstrap_min_foundation_score": float(temporal_cfg.get("memory_bootstrap_min_foundation_score", 0.46)),
            "memory_require_filter_out": bool(temporal_cfg.get("memory_require_filter_out", True)),
            "key_min_temporal_support": float(temporal_cfg.get("key_prompt_min_temporal_support", 0.28)),
            "key_require_consistent_track_id": bool(temporal_cfg.get("key_require_consistent_track_id", False)),
            "memory_min_temporal_support": float(temporal_cfg.get("memory_min_temporal_support", 0.18)),
            "memory_require_consistent_track_id": bool(temporal_cfg.get("memory_require_consistent_track_id", True)),
        }

    def _build_temporal_anchor_metadata(self, foundation, task, decision, segment_record):
        cfg = self._temporal_memory_policy_config()
        foundation_score = float(decision.get("foundation_score", foundation.get("score", 0.0)))
        track_confirmation = float(decision.get("track_confirmation", 0.0))
        dynamic_score = float(
            np.clip(
                max(
                    float(decision.get("dynamic_memory_score", 0.0)),
                    float(decision.get("motion_score", 0.0)),
                    float(decision.get("tube_motion_score", 0.0)),
                    float(decision.get("geometry_dynamic_score", 0.0)),
                ),
                0.0,
                1.0,
            )
        )
        geometry_dynamic_score = float(decision.get("geometry_dynamic_score", 0.0))
        dynamic_memory_score = float(decision.get("dynamic_memory_score", 0.0))
        sam_score = float(segment_record.get("sam_score", 0.0))
        task_score = float(task.get("score", 0.0))
        filter_out = bool(decision.get("filter_out", False))
        decision_reason = str(decision.get("decision_reason", "unknown"))
        temporal_fusion_confidence = float(
            np.clip(
                max(
                    float(segment_record.get("temporal_fusion_confidence", 0.0)),
                    float(decision.get("temporal_fusion_score", 0.0)),
                ),
                0.0,
                1.0,
            )
        )
        temporal_mask_agreement = float(
            np.clip(
                max(
                    float(segment_record.get("temporal_mask_agreement", 0.0)),
                    float(decision.get("temporal_mask_agreement", 0.0)),
                ),
                0.0,
                1.0,
            )
        )
        temporal_box_agreement = float(
            np.clip(float(segment_record.get("temporal_box_agreement", 0.0)), 0.0, 1.0)
        )
        temporal_track_id = int(segment_record.get("temporal_track_id", -1))
        resolved_track_id = int(decision.get("track_id", -1))
        temporal_id_consistency = float(np.clip(decision.get("temporal_id_consistency", 0.0), 0.0, 1.0))
        if temporal_track_id > 0 and resolved_track_id > 0:
            temporal_id_consistency = 1.0 if temporal_track_id == resolved_track_id else 0.0
        elif temporal_track_id > 0:
            temporal_id_consistency = max(temporal_id_consistency, 0.55)
        temporal_id_ok = temporal_track_id <= 0 or resolved_track_id <= 0 or temporal_track_id == resolved_track_id
        temporal_support = float(
            np.clip(
                0.44 * temporal_fusion_confidence
                + 0.24 * temporal_mask_agreement
                + 0.16 * temporal_box_agreement
                + 0.16 * temporal_id_consistency,
                0.0,
                1.0,
            )
        )

        anchor_quality = float(
            np.clip(
                0.24 * foundation_score
                + 0.20 * track_confirmation
                + 0.16 * dynamic_score
                + 0.08 * geometry_dynamic_score
                + 0.08 * dynamic_memory_score
                + 0.06 * sam_score
                + 0.04 * task_score
                + 0.14 * temporal_support,
                0.0,
                1.0,
            )
        )
        eligible_dynamic_reasons = {
            "dynamic_memory_propagation",
            "motion_supported",
            "tube_motion_supported",
            "high_confidence_override",
            "confirmed_track_support",
        }
        is_key_anchor = (
            filter_out
            and decision_reason in eligible_dynamic_reasons
            and foundation_score >= cfg["key_min_foundation_score"]
            and track_confirmation >= cfg["key_min_track_confirmation"]
            and dynamic_score >= cfg["key_min_dynamic_score"]
            and temporal_support >= cfg["key_min_temporal_support"]
            and (temporal_id_ok or (not cfg["key_require_consistent_track_id"]))
            and anchor_quality >= cfg["key_min_quality"]
        )
        memory_filter_ok = filter_out or (not cfg["memory_require_filter_out"])
        memory_eligible = (
            memory_filter_ok
            and foundation_score >= cfg["memory_min_foundation_score"]
            and track_confirmation >= cfg["memory_min_track_confirmation"]
            and temporal_support >= cfg["memory_min_temporal_support"]
            and (temporal_id_ok or (not cfg["memory_require_consistent_track_id"]))
            and anchor_quality >= cfg["memory_min_quality"]
        )
        memory_bootstrap_eligible = (
            memory_filter_ok
            and foundation_score >= cfg["memory_bootstrap_min_foundation_score"]
            and (temporal_id_ok or temporal_track_id <= 0)
            and anchor_quality >= cfg["memory_bootstrap_min_quality"]
        )
        anchor_keep_frames = cfg["key_history_frames"] if is_key_anchor else cfg["regular_history_frames"]
        return {
            "anchor_quality": anchor_quality,
            "anchor_is_keyframe": bool(is_key_anchor),
            "anchor_keep_frames": int(anchor_keep_frames),
            "anchor_memory_eligible": bool(memory_eligible),
            "anchor_memory_bootstrap_eligible": bool(memory_bootstrap_eligible),
            "anchor_temporal_support": temporal_support,
            "anchor_temporal_id_consistency": temporal_id_consistency,
            "anchor_temporal_track_id_ok": bool(temporal_id_ok),
        }

    def _propagate_masks(self, image_bgr):
        # 使用稠密光流传播上一帧掩膜，减少每帧重新检测的成本。
        if self.prev_gray is None or not self.prev_mask_records:
            detections, source_counts = self.detector_ensemble.predict(image_bgr)
            self.last_source_counts = source_counts
            return detections, self.segmenter.segment(image_bgr, detections)

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )

        h, w = gray.shape
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x - flow[..., 0]).astype(np.float32)
        map_y = (grid_y - flow[..., 1]).astype(np.float32)

        warped_records = []
        warped_detections = []
        for detection, record in zip(self.prev_detections, self.prev_mask_records):
            warped = cv2.remap(
                record["mask"].astype(np.float32),
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            warped = (warped > 0.35).astype(np.uint8)
            if np.count_nonzero(warped) == 0:
                continue
            ys, xs = np.where(warped > 0)
            warped_detections.append(
                Detection(
                    x1=int(xs.min()),
                    y1=int(ys.min()),
                    x2=int(xs.max()) + 1,
                    y2=int(ys.max()) + 1,
                    score=detection.score,
                    class_name=detection.class_name,
                    class_id=detection.class_id,
                    source_name=detection.source_name,
                    track_id=detection.track_id,
                )
            )
            warped_records.append(
                {
                    "mask": warped,
                    "sam_score": float(record.get("sam_score", detection.score)),
                    "segmenter_mode": "propagate",
                }
            )

        if not warped_records:
            detections, source_counts = self.detector_ensemble.predict(image_bgr)
            self.last_source_counts = source_counts
            return detections, self.segmenter.segment(image_bgr, detections)
        return warped_detections, warped_records

    def _refine_mask(self, mask, detection, image_bgr, depth_mm):
        # 先做形态学膨胀，再用深度一致性把疑似背景泄漏区域收回来。
        refined = mask.astype(np.uint8)
        runtime_cfg = self.config["runtime"]
        dilate_pixels = int(runtime_cfg.get("dilate_pixels", 5))
        if dilate_pixels > 0:
            kernel = np.ones((dilate_pixels, dilate_pixels), dtype=np.uint8)
            refined = cv2.dilate(refined, kernel, iterations=1)

        depth_cfg = self.config["depth_filter"]
        if not depth_cfg.get("enabled", True) or depth_mm is None:
            return refined

        depth_mask = self._depth_gate(refined, depth_mm, depth_cfg)
        if np.count_nonzero(depth_mask) == 0:
            depth_mask = refined
        return self._boundary_trim(depth_mask, image_bgr, depth_mm)

    def _boundary_trim(self, mask, image_bgr, depth_mm):
        cfg = self.config.get("mask_boundary_refine", {})
        if not cfg.get("enabled", False) or np.count_nonzero(mask) == 0:
            return mask

        edge_width = max(int(cfg.get("edge_width", 3)), 1)
        boundary = mask_boundary(mask, edge_width)
        if np.count_nonzero(boundary) < int(cfg.get("min_boundary_pixels", 24)):
            return mask

        core = np.clip(mask.astype(np.uint8) - boundary, 0, 1)
        keep_boundary = np.zeros_like(mask, dtype=np.uint8)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        keep_boundary[(boundary > 0) & (grad_mag >= float(cfg.get("image_grad_threshold", 18.0)))] = 1

        if depth_mm is not None:
            ring = mask_outer_ring(mask, edge_width)
            ring_values = depth_mm[(ring > 0) & (depth_mm > 0)]
            if len(ring_values) >= int(self.config.get("depth_filter", {}).get("min_valid_pixels", 40)):
                bg_depth = float(np.median(ring_values))
                depth_margin = float(cfg.get("depth_margin_mm", 120.0))
                keep_boundary[(boundary > 0) & (depth_mm > 0) & (np.abs(depth_mm.astype(np.float32) - bg_depth) > depth_margin)] = 1

        refined = np.maximum(core, keep_boundary).astype(np.uint8)
        keep_ratio = float(np.count_nonzero(refined)) / max(float(np.count_nonzero(mask)), 1.0)
        if keep_ratio < float(cfg.get("min_keep_ratio", 0.58)):
            return mask
        return refined

    def _depth_gate(self, mask, depth_mm, cfg):
        kernel_size = int(cfg.get("dilate_pixels", 12))
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        outer = cv2.dilate(mask, kernel, iterations=1)
        ring = np.clip(outer - mask, 0, 1)

        bg_values = depth_mm[(ring > 0) & (depth_mm > 0)]
        fg_values = depth_mm[(mask > 0) & (depth_mm > 0)]
        if len(bg_values) < int(cfg.get("min_valid_pixels", 40)) or len(fg_values) < int(cfg.get("min_valid_pixels", 40)):
            return mask

        bg_depth = float(np.median(bg_values))
        margin = float(cfg.get("depth_margin_mm", 250.0))
        keep = np.zeros_like(mask, dtype=np.uint8)
        keep[(mask > 0) & (np.abs(depth_mm.astype(np.float32) - bg_depth) > margin)] = 1

        keep_ratio = float(np.count_nonzero(keep)) / max(1.0, float(np.count_nonzero(mask)))
        if keep_ratio < float(cfg.get("min_keep_ratio", 0.18)):
            return mask
        return keep

    def _score_mask_boundary_quality(self, image_bgr, depth_mm, mask):
        cfg = self.config.get("foundation_reliability", {}).get("boundary_quality", {})
        if not cfg.get("enabled", False) or image_bgr is None or np.count_nonzero(mask) == 0:
            return 1.0

        edge_width = max(int(cfg.get("edge_width", 3)), 1)
        boundary = mask_boundary(mask, edge_width)
        if np.count_nonzero(boundary) < int(cfg.get("min_boundary_pixels", 24)):
            return 0.65

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        image_score = float(
            np.clip(
                np.median(grad_mag[boundary > 0]) / max(float(cfg.get("image_grad_norm", 36.0)), 1.0),
                0.0,
                1.0,
            )
        )

        depth_score = 0.5
        if depth_mm is not None:
            ring = mask_outer_ring(mask, edge_width)
            inner_values = depth_mm[(boundary > 0) & (depth_mm > 0)]
            outer_values = depth_mm[(ring > 0) & (depth_mm > 0)]
            if len(inner_values) > 20 and len(outer_values) > 20:
                depth_delta = abs(float(np.median(inner_values)) - float(np.median(outer_values)))
                depth_score = float(np.clip(depth_delta / max(float(cfg.get("depth_norm_mm", 180.0)), 1.0), 0.0, 1.0))

        return float(np.clip(0.55 * image_score + 0.45 * depth_score, 0.0, 1.0))

    def _score_foundation_reliability(self, detection, mask, image_shape, sam_score, image_bgr=None, depth_mm=None):
        # 基础模型可靠性评分：回答“这个检测+分割结果能不能信”。
        cfg = self.config.get("foundation_reliability", {})
        if not cfg.get("enabled", True):
            return {
                "score": 1.0,
                "components": {
                    "detector_confidence": 1.0,
                    "segment_quality": 1.0,
                    "mask_area": 1.0,
                    "compactness": 1.0,
                    "border": 1.0,
                },
            }

        mask_pixels = float(np.count_nonzero(mask))
        mask_ratio = mask_pixels / max(float(mask.size), 1.0)
        min_pixels = max(float(cfg.get("min_mask_pixels", 96)), 1.0)
        min_ratio = max(float(cfg.get("min_mask_ratio", 0.00018)), 1e-8)
        mask_area_score = float(np.clip(0.5 * (mask_pixels / min_pixels) + 0.5 * (mask_ratio / min_ratio), 0.0, 1.0))
        compactness_score = mask_compactness(mask, detection)
        border_score = box_border_score(detection, image_shape, int(cfg.get("border_margin_pixels", 14)))
        boundary_score = self._score_mask_boundary_quality(image_bgr, depth_mm, mask)
        components = {
            "detector_confidence": float(np.clip(detection.score, 0.0, 1.0)),
            "segment_quality": float(np.clip(sam_score, 0.0, 1.0)),
            "mask_area": mask_area_score,
            "compactness": compactness_score,
            "border": border_score,
            "boundary": boundary_score,
        }
        weight_cfg = cfg.get("weights", {})
        total_weight = sum(float(weight_cfg.get(key, 0.0)) for key in components.keys())
        if total_weight <= 0:
            total_weight = 1.0
        score = 0.0
        for key, value in components.items():
            score += float(weight_cfg.get(key, 0.0)) * float(value)
        score = float(np.clip(score / total_weight, 0.0, 1.0))
        return {"score": score, "components": components}

    def _compute_geometry_consistency(self, current_gray, detections, masks):
        cfg = self.config.get("geometry_consistency", {})
        empty_template = {
            "dynamic_score": 0.0,
            "static_score": 0.0,
            "support_points": 0,
            "ring_points": 0,
            "median_residual_px": 0.0,
            "residual_contrast": 0.0,
            "verified_dynamic": False,
            "verified_static": False,
        }
        empty = [dict(empty_template) for _ in detections]
        if not detections or not cfg.get("enabled", False) or self.prev_gray is None:
            return empty

        prev_points = cv2.goodFeaturesToTrack(
            self.prev_gray,
            maxCorners=int(cfg.get("max_corners", 400)),
            qualityLevel=float(cfg.get("quality_level", 0.01)),
            minDistance=float(cfg.get("min_distance", 7)),
            blockSize=int(cfg.get("block_size", 7)),
        )
        if prev_points is None:
            return empty

        curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            current_gray,
            prev_points,
            None,
            winSize=tuple(int(v) for v in cfg.get("lk_win_size", [21, 21])),
            maxLevel=int(cfg.get("lk_max_level", 3)),
        )
        if curr_points is None or status is None:
            return empty

        valid = status.reshape(-1) > 0
        prev_pts = prev_points.reshape(-1, 2)[valid]
        curr_pts = curr_points.reshape(-1, 2)[valid]
        min_global_points = max(int(cfg.get("min_global_points", 24)), 4)
        if len(prev_pts) < min_global_points or len(curr_pts) < min_global_points:
            return empty

        model, _ = cv2.estimateAffinePartial2D(
            prev_pts,
            curr_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(cfg.get("ransac_reproj_threshold", 3.0)),
        )
        if model is None:
            return empty

        predicted = cv2.transform(prev_pts[None, :, :], model)[0]
        residuals = np.linalg.norm(curr_pts - predicted, axis=1)
        xs = np.clip(np.round(curr_pts[:, 0]).astype(np.int32), 0, current_gray.shape[1] - 1)
        ys = np.clip(np.round(curr_pts[:, 1]).astype(np.int32), 0, current_gray.shape[0] - 1)
        residual_norm = max(float(cfg.get("residual_norm_px", 4.0)), 1e-6)
        inlier_residual = max(float(cfg.get("inlier_residual_px", 2.0)), 1e-6)
        min_mask_points = max(int(cfg.get("min_mask_points", 6)), 1)
        verify_support = max(int(cfg.get("verification_min_support_points", 8)), min_mask_points)
        ring_width = max(int(cfg.get("ring_dilate_pixels", 9)), 1)
        dynamic_margin = float(cfg.get("dynamic_margin", 0.10))

        details = []
        for mask in masks:
            inside = mask[ys, xs] > 0
            support = int(np.count_nonzero(inside))
            if support < min_mask_points:
                details.append(dict(empty_template, support_points=support))
                continue

            local_residuals = residuals[inside]
            ring_mask = mask_outer_ring(mask, ring_width)
            ring_points = ring_mask[ys, xs] > 0
            ring_support = int(np.count_nonzero(ring_points))
            ring_median = float(np.median(residuals[ring_points])) if ring_support >= min_mask_points else float(np.median(local_residuals))
            local_median = float(np.median(local_residuals))
            contrast = float(np.clip((local_median - ring_median) / residual_norm, -1.0, 1.0))
            median_norm = float(np.clip(local_median / residual_norm, 0.0, 1.0))
            p90_norm = float(np.clip(np.percentile(local_residuals, 90) / (1.5 * residual_norm), 0.0, 1.0))
            outlier_ratio = float(np.mean(local_residuals >= residual_norm))
            inlier_ratio = float(np.mean(local_residuals <= inlier_residual))
            support_scale = float(np.clip(support / float(verify_support), 0.0, 1.0))
            dynamic_score = float(
                np.clip(
                    (0.40 * median_norm + 0.25 * p90_norm + 0.20 * outlier_ratio + 0.15 * max(contrast, 0.0))
                    * support_scale,
                    0.0,
                    1.0,
                )
            )
            static_score = float(
                np.clip(
                    (0.50 * inlier_ratio + 0.25 * (1.0 - median_norm) + 0.25 * max(-contrast, 0.0))
                    * support_scale,
                    0.0,
                    1.0,
                )
            )
            details.append(
                {
                    "dynamic_score": dynamic_score,
                    "static_score": static_score,
                    "support_points": support,
                    "ring_points": ring_support,
                    "median_residual_px": local_median,
                    "residual_contrast": contrast,
                    "verified_dynamic": bool(
                        support >= verify_support
                        and dynamic_score >= float(cfg.get("dynamic_confirm_threshold", 0.58))
                        and dynamic_score >= static_score + dynamic_margin
                    ),
                    "verified_static": bool(
                        support >= verify_support
                        and static_score >= float(cfg.get("static_veto_threshold", 0.70))
                        and static_score >= dynamic_score + dynamic_margin
                    ),
                }
            )
        return details

    def _score_task_relevance(self, detection, mask, image_shape, depth_mm, sam_score=1.0, foundation_score=1.0):
        # 任务相关性评分：回答“这个实例值不值得在 SLAM 前端重点处理”。
        cfg = self.config.get("task_relevance", {})
        if not cfg.get("enabled", True):
            return {
                "class_name": detection.class_name,
                "canonical_name": detection.canonical_name,
                "source_name": detection.source_name,
                "score": 1.0,
                "decision": "filter",
                "filter_out": True,
                "components": {
                    "semantic": 1.0,
                    "area": 1.0,
                    "center": 1.0,
                    "depth": 1.0,
                    "confidence": 1.0,
                    "sam": 1.0,
                    "foundation": 1.0,
                },
            }

        image_h, image_w = image_shape
        area_ratio = float(np.count_nonzero(mask)) / max(1.0, float(mask.size))
        area_norm = max(float(cfg.get("area_norm_ratio", 0.08)), 1e-6)
        area_score = float(np.clip(area_ratio / area_norm, 0.0, 1.0))

        cx = 0.5 * float(detection.x1 + detection.x2)
        cy = 0.5 * float(detection.y1 + detection.y2)
        nx = (cx - 0.5 * image_w) / max(0.5 * image_w, 1.0)
        ny = (cy - 0.5 * image_h) / max(0.5 * image_h, 1.0)
        center_dist = float(np.sqrt(nx * nx + ny * ny))
        center_score = float(np.clip(1.0 - center_dist / np.sqrt(2.0), 0.0, 1.0))

        semantic_weights = cfg.get("semantic_weights", {})
        class_key = detection.canonical_name
        semantic_score = float(semantic_weights.get(class_key, semantic_weights.get(detection.class_name, cfg.get("default_semantic_weight", 0.55))))

        depth_score = 0.5
        if depth_mm is not None and np.count_nonzero(mask) > 0:
            kernel = np.ones((9, 9), dtype=np.uint8)
            outer = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
            ring = np.clip(outer - mask.astype(np.uint8), 0, 1)
            fg_values = depth_mm[(mask > 0) & (depth_mm > 0)]
            bg_values = depth_mm[(ring > 0) & (depth_mm > 0)]
            if len(fg_values) > 20 and len(bg_values) > 20:
                fg_depth = float(np.median(fg_values))
                bg_depth = float(np.median(bg_values))
                depth_delta = abs(fg_depth - bg_depth)
                depth_norm = max(float(cfg.get("depth_norm_mm", 1500.0)), 1.0)
                depth_score = float(np.clip(depth_delta / depth_norm, 0.0, 1.0))

        confidence_score = float(np.clip(detection.score, 0.0, 1.0))
        components = {
            "semantic": semantic_score,
            "area": area_score,
            "center": center_score,
            "depth": depth_score,
            "confidence": confidence_score,
            "sam": float(np.clip(sam_score, 0.0, 1.0)),
            "foundation": float(np.clip(foundation_score, 0.0, 1.0)),
        }
        weight_cfg = cfg.get("weights", {})
        total_weight = sum(float(weight_cfg.get(key, 0.0)) for key in components.keys())
        if total_weight <= 0:
            total_weight = 1.0
        score = 0.0
        for key, value in components.items():
            score += float(weight_cfg.get(key, 0.0)) * float(value)
        score = float(np.clip(score / total_weight, 0.0, 1.0))
        threshold = float(cfg.get("min_score", 0.45))
        filter_out = score >= threshold
        return {
            "class_name": detection.class_name,
            "canonical_name": class_key,
            "source_name": detection.source_name,
            "score": score,
            "decision": "filter" if filter_out else "keep",
            "filter_out": filter_out,
            "components": components,
        }

    def _match_track(self, detection, mask, appearance_hist, current_depth, assigned_track_ids):
        cfg = self.config.get("panoptic_memory", {})
        threshold = float(cfg.get("match_iou", 0.35))
        weights = cfg.get("association_weights", {})
        best_track = None
        best_score = -1.0
        best_components = {}
        current_box = (detection.x1, detection.y1, detection.x2, detection.y2)
        current_label = detection.canonical_name
        preferred_track_id = int(getattr(detection, "track_id", -1))
        if preferred_track_id > 0:
            preferred_track = self.tracks.get(preferred_track_id)
            if (
                preferred_track is not None
                and preferred_track_id not in assigned_track_ids
                and preferred_track.class_name == current_label
            ):
                bbox_iou = compute_box_iou(current_box, preferred_track.bbox)
                mask_iou = compute_mask_iou(mask, preferred_track.mask) if preferred_track.mask is not None else 0.0
                app_sim = appearance_similarity(appearance_hist, preferred_track.appearance)
                dep_sim = depth_similarity(current_depth, preferred_track.last_depth, cfg.get("depth_motion_norm_mm", 600.0))
                return (
                    preferred_track,
                    1.0,
                    {
                        "bbox_iou": float(bbox_iou),
                        "mask_iou": float(mask_iou),
                        "appearance": float(app_sim),
                        "depth": float(dep_sim),
                        "id_match": 1.0,
                    },
                )
        for track_id, track in self.tracks.items():
            if track_id in assigned_track_ids:
                continue
            if track.class_name != current_label:
                continue
            bbox_iou = compute_box_iou(current_box, track.bbox)
            mask_iou = compute_mask_iou(mask, track.mask) if track.mask is not None else 0.0
            app_sim = appearance_similarity(appearance_hist, track.appearance)
            dep_sim = depth_similarity(current_depth, track.last_depth, cfg.get("depth_motion_norm_mm", 600.0))
            score = (
                float(weights.get("bbox_iou", 0.28)) * bbox_iou
                + float(weights.get("mask_iou", 0.22)) * mask_iou
                + float(weights.get("appearance", 0.30)) * app_sim
                + float(weights.get("depth", 0.20)) * dep_sim
            )
            if score > best_score and score >= threshold:
                best_score = score
                best_track = track
                best_components = {
                    "bbox_iou": float(bbox_iou),
                    "mask_iou": float(mask_iou),
                    "appearance": float(app_sim),
                    "depth": float(dep_sim),
                }
        return best_track, max(best_score, 0.0), best_components

    def _update_panoptic_memory(self, detections, masks, task_details, foundation_details, geometry_details, segment_records, depth_mm, image_shape, image_bgr):
        # 跨帧时序记忆：为实例建立轨迹，并更新运动、静态、tube motion 和 dynamic memory。
        cfg = self.config.get("panoptic_memory", {})
        if not detections:
            self._age_tracks()
            return []

        if not cfg.get("enabled", True):
            return [
                {
                    "track_id": -1,
                    "track_hits": 0,
                    "track_confirmation": 0.0,
                    "motion_score": 0.0,
                    "static_score": 0.0,
                    "tube_motion_score": 0.0,
                    "dynamic_memory_score": 0.0,
                    "geometry_dynamic_score": 0.0,
                    "geometry_static_score": 0.0,
                    "geometry_support_points": 0,
                    "temporal_fusion_score": 0.0,
                    "temporal_id_consistency": 0.0,
                    "temporal_mask_agreement": 0.0,
                    "temporal_box_agreement": 0.0,
                    "track_confirmed": False,
                    "match_score": 0.0,
                    "association_components": {},
                }
                for _ in detections
            ]

        assigned_track_ids = set()
        touched_track_ids = set()
        track_infos = []
        momentum = float(cfg.get("ema_momentum", 0.65))
        dynamic_cfg = self.config.get("dynamic_memory", {})
        dynamic_enabled = bool(dynamic_cfg.get("enabled", True))
        dynamic_momentum = float(dynamic_cfg.get("ema_momentum", 0.72))
        min_hits = max(int(cfg.get("min_hits", 2)), 1)
        confirm_threshold = float(cfg.get("confirm_threshold", 0.58))
        center_motion_norm = max(float(cfg.get("center_motion_norm", 0.04)), 1e-6)
        depth_motion_norm = max(float(cfg.get("depth_motion_norm_mm", 600.0)), 1.0)
        displacement_norm_ratio = max(float(cfg.get("tube_displacement_norm_ratio", 0.12)), 1e-6)
        appearance_bins = tuple(cfg.get("appearance_bins", [8, 8]))
        geometry_cfg = self.config.get("geometry_consistency", {})
        confirmed_dynamic_cfg = dict(cfg.get("confirmed_dynamic_track", {}))
        confirmed_dynamic_enabled = bool(confirmed_dynamic_cfg.get("enabled", False))
        confirmed_dynamic_classes = {
            canonical_class_name(item)
            for item in confirmed_dynamic_cfg.get("classes", ["person", "pedestrian", "worker"])
        }
        confirmed_dynamic_min_hits = max(int(confirmed_dynamic_cfg.get("min_hits", min_hits)), 1)
        confirmed_dynamic_min_streak = max(int(confirmed_dynamic_cfg.get("min_streak", 3)), 1)
        confirmed_dynamic_stable_match = float(
            confirmed_dynamic_cfg.get(
                "stable_match_score",
                confirmed_dynamic_cfg.get("min_match_score", max(float(cfg.get("match_iou", 0.35)), 0.42)),
            )
        )
        confirmed_dynamic_min_stable_streak = max(
            int(confirmed_dynamic_cfg.get("min_stable_match_streak", max(confirmed_dynamic_min_streak, 1))),
            1,
        )
        confirmed_dynamic_min_geometry_streak = max(
            int(confirmed_dynamic_cfg.get("min_geometry_streak", max(confirmed_dynamic_min_streak, 1))),
            1,
        )
        confirmed_dynamic_confirmation_floor = float(
            confirmed_dynamic_cfg.get("confirmation_floor", max(confirm_threshold * 0.95, 0.0))
        )
        confirmed_dynamic_geometry_margin = float(
            confirmed_dynamic_cfg.get("geometry_margin", geometry_cfg.get("dynamic_margin", 0.08))
        )
        confirmed_dynamic_temporal_momentum = float(
            confirmed_dynamic_cfg.get("temporal_score_momentum", 0.65)
        )
        confirmed_dynamic_activation_score = float(
            confirmed_dynamic_cfg.get("activation_score", confirmed_dynamic_confirmation_floor)
        )
        confirmed_dynamic_release_score = float(
            confirmed_dynamic_cfg.get("release_score", max(confirmed_dynamic_activation_score - 0.16, 0.0))
        )
        confirmed_dynamic_motion_floor = float(
            confirmed_dynamic_cfg.get("motion_floor", max(float(cfg.get("motion_filter_threshold", 0.18)), 0.10))
        )
        confirmed_dynamic_tube_floor = float(
            confirmed_dynamic_cfg.get(
                "tube_motion_floor",
                max(float(cfg.get("tube_motion_threshold", 0.28)) * 0.75, 0.16),
            )
        )
        confirmed_dynamic_static_reset = float(confirmed_dynamic_cfg.get("static_reset_floor", 0.74))
        confirmed_dynamic_uncertain_decay = max(int(confirmed_dynamic_cfg.get("uncertain_decay", 1)), 0)
        confirmed_dynamic_unconfirmed_cap = float(
            confirmed_dynamic_cfg.get(
                "unconfirmed_memory_cap",
                max(float(dynamic_cfg.get("min_score", 0.60)) - 0.04, 0.0),
            )
        )
        image_h, image_w = image_shape
        diag = max(float(np.sqrt(image_h * image_h + image_w * image_w)), 1.0)

        for detection, mask, task, foundation, geometry, segment_record in zip(
            detections,
            masks,
            task_details,
            foundation_details,
            geometry_details,
            segment_records,
        ):
            current_center = mask_center(mask, detection)
            current_depth = median_mask_depth(depth_mm, mask)
            current_appearance = extract_instance_appearance(image_bgr, mask, bins=appearance_bins)
            track, match_score, match_components = self._match_track(detection, mask, current_appearance, current_depth, assigned_track_ids)
            temporal_score = 1.0 / float(min_hits)
            motion_score = 0.0
            static_score = 0.0
            dynamic_memory_score = 0.0
            geometry_dynamic_score = float(geometry.get("dynamic_score", 0.0))
            geometry_static_score = float(geometry.get("static_score", 0.0))
            geometry_support = int(geometry.get("support_points", 0))
            temporal_fusion_observation = float(
                np.clip(float(segment_record.get("temporal_fusion_confidence", 0.0)), 0.0, 1.0)
            )
            temporal_mask_observation = float(
                np.clip(float(segment_record.get("temporal_mask_agreement", 0.0)), 0.0, 1.0)
            )
            temporal_box_observation = float(
                np.clip(float(segment_record.get("temporal_box_agreement", 0.0)), 0.0, 1.0)
            )
            temporal_track_hint = int(segment_record.get("temporal_track_id", -1))
            temporal_id_observation = 1.0 if temporal_track_hint > 0 else 0.0
            temporal_support_observation = float(
                np.clip(
                    0.50 * temporal_fusion_observation
                    + 0.30 * temporal_mask_observation
                    + 0.20 * temporal_box_observation,
                    0.0,
                    1.0,
                )
            )

            if track is None:
                motion_score, static_score = blend_geometry_evidence(0.0, 0.0, geometry, geometry_cfg)
                track = TrackState(
                    track_id=self.next_track_id,
                    class_name=detection.canonical_name,
                    bbox=(detection.x1, detection.y1, detection.x2, detection.y2),
                    mask=mask.copy(),
                    center=current_center,
                    hits=1,
                    misses=0,
                    age=1,
                    ema_confidence=float(detection.score),
                    ema_relevance=float(task["score"]),
                    ema_reliability=float(foundation["score"]),
                    confirmation_score=float(
                        np.clip(
                            0.30 * foundation["score"]
                            + 0.28 * task["score"]
                            + 0.10 * (1.0 - static_score)
                            + 0.12 * geometry_dynamic_score
                            + 0.08 * temporal_support_observation
                            + 0.10,
                            0.0,
                            1.0,
                        )
                    ),
                    motion_score=motion_score,
                    static_score=static_score,
                    last_depth=current_depth,
                    appearance=current_appearance,
                    start_center=current_center,
                    cumulative_displacement=0.0,
                    max_center_distance=0.0,
                    tube_motion_score=0.0,
                    max_motion_score=0.0,
                    dynamic_confirmation_streak=0,
                    stable_match_streak=0,
                    geometry_confirmation_streak=0,
                    temporal_dynamic_score=0.0,
                    geometry_dynamic_score=geometry_dynamic_score,
                    geometry_static_score=geometry_static_score,
                    geometry_support=geometry_support,
                    temporal_fusion_score=temporal_support_observation,
                    temporal_id_consistency=temporal_id_observation,
                    temporal_mask_agreement=temporal_mask_observation,
                    temporal_box_agreement=temporal_box_observation,
                )
                if dynamic_enabled:
                    initial_dynamic_memory = compute_dynamic_memory_observation(
                        detection,
                        float(task["score"]),
                        float(foundation["score"]),
                        motion_score,
                        track.tube_motion_score,
                        static_score,
                        float(track.confirmation_score),
                        temporal_score,
                        dynamic_cfg,
                    )
                    if confirmed_dynamic_enabled and detection.canonical_name in confirmed_dynamic_classes:
                        initial_dynamic_memory = min(initial_dynamic_memory, confirmed_dynamic_unconfirmed_cap)
                    track.dynamic_memory_score = float(np.clip(initial_dynamic_memory, 0.0, 1.0))
                else:
                    track.dynamic_memory_score = 0.0
                dynamic_memory_score = track.dynamic_memory_score
                self.tracks[track.track_id] = track
                self.next_track_id += 1
            else:
                prev_center = track.center
                center_delta = float(np.sqrt((current_center[0] - prev_center[0]) ** 2 + (current_center[1] - prev_center[1]) ** 2))
                center_motion = float(np.clip(center_delta / (center_motion_norm * diag), 0.0, 1.0))
                mask_iou = compute_mask_iou(track.mask, mask)
                depth_motion = 0.0
                if current_depth > 0 and track.last_depth > 0:
                    depth_motion = float(np.clip(abs(current_depth - track.last_depth) / depth_motion_norm, 0.0, 1.0))
                base_motion = float(np.clip(0.45 * center_motion + 0.35 * (1.0 - mask_iou) + 0.20 * depth_motion, 0.0, 1.0))
                base_static = float(np.clip(0.55 * mask_iou + 0.45 * (1.0 - center_motion), 0.0, 1.0))
                motion_observation, static_observation = blend_geometry_evidence(base_motion, base_static, geometry, geometry_cfg)
                track.cumulative_displacement += center_delta
                if track.start_center is None:
                    track.start_center = prev_center
                net_distance = float(np.sqrt((current_center[0] - track.start_center[0]) ** 2 + (current_center[1] - track.start_center[1]) ** 2))
                cumulative_score = float(np.clip(track.cumulative_displacement / (diag * displacement_norm_ratio), 0.0, 1.0))
                net_score = float(np.clip(net_distance / (diag * displacement_norm_ratio), 0.0, 1.0))
                tube_motion = float(np.clip(0.45 * cumulative_score + 0.35 * net_score + 0.20 * motion_observation, 0.0, 1.0))

                track.hits += 1
                track.misses = 0
                track.age += 1
                track.ema_confidence = momentum * track.ema_confidence + (1.0 - momentum) * float(detection.score)
                track.ema_relevance = momentum * track.ema_relevance + (1.0 - momentum) * float(task["score"])
                track.ema_reliability = momentum * track.ema_reliability + (1.0 - momentum) * float(foundation["score"])
                track.motion_score = momentum * track.motion_score + (1.0 - momentum) * motion_observation
                track.static_score = momentum * track.static_score + (1.0 - momentum) * static_observation
                track.max_motion_score = max(float(track.max_motion_score), motion_observation)
                track.max_center_distance = max(float(track.max_center_distance), net_distance)
                track.tube_motion_score = momentum * track.tube_motion_score + (1.0 - momentum) * tube_motion
                track.geometry_dynamic_score = momentum * track.geometry_dynamic_score + (1.0 - momentum) * geometry_dynamic_score
                track.geometry_static_score = momentum * track.geometry_static_score + (1.0 - momentum) * geometry_static_score
                track.geometry_support = geometry_support
                track.temporal_fusion_score = (
                    momentum * track.temporal_fusion_score + (1.0 - momentum) * temporal_support_observation
                )
                if temporal_track_hint > 0:
                    temporal_id_observation = 1.0 if track.track_id == temporal_track_hint else 0.0
                    track.temporal_id_consistency = (
                        momentum * track.temporal_id_consistency + (1.0 - momentum) * temporal_id_observation
                    )
                else:
                    track.temporal_id_consistency = momentum * track.temporal_id_consistency
                track.temporal_mask_agreement = (
                    momentum * track.temporal_mask_agreement + (1.0 - momentum) * temporal_mask_observation
                )
                track.temporal_box_agreement = (
                    momentum * track.temporal_box_agreement + (1.0 - momentum) * temporal_box_observation
                )
                temporal_score = float(np.clip(track.hits / float(min_hits), 0.0, 1.0))
                track.confirmation_score = float(
                    np.clip(
                        0.28 * temporal_score
                        + 0.22 * track.ema_reliability
                        + 0.14 * track.ema_relevance
                        + 0.10 * track.ema_confidence
                        + 0.12 * track.tube_motion_score
                        + 0.10 * (1.0 - track.static_score)
                        + 0.02 * track.temporal_id_consistency
                        + 0.04 * track.temporal_fusion_score
                        + 0.04 * max(track.geometry_dynamic_score, track.geometry_static_score),
                        0.0,
                        1.0,
                    )
                )
                stable_match_ready = (
                    confirmed_dynamic_enabled
                    and detection.canonical_name in confirmed_dynamic_classes
                    and is_stable_track_match(match_score, match_components, confirmed_dynamic_cfg)
                )
                if stable_match_ready:
                    track.stable_match_streak = min(
                        track.stable_match_streak + 1,
                        confirmed_dynamic_min_stable_streak + 8,
                    )
                else:
                    track.stable_match_streak = 0

                geometry_dynamic_ready = (
                    confirmed_dynamic_enabled
                    and detection.canonical_name in confirmed_dynamic_classes
                    and temporal_geometry_dynamic_ready(
                        track,
                        geometry,
                        confirmed_dynamic_cfg,
                        confirmed_dynamic_geometry_margin,
                    )
                )
                geometry_static_ready = (
                    confirmed_dynamic_enabled
                    and detection.canonical_name in confirmed_dynamic_classes
                    and temporal_geometry_static_ready(
                        track,
                        geometry,
                        confirmed_dynamic_cfg,
                        confirmed_dynamic_geometry_margin,
                    )
                )
                if geometry_dynamic_ready:
                    track.geometry_confirmation_streak = min(
                        track.geometry_confirmation_streak + 1,
                        confirmed_dynamic_min_geometry_streak + 8,
                    )
                else:
                    track.geometry_confirmation_streak = 0

                if confirmed_dynamic_enabled and detection.canonical_name in confirmed_dynamic_classes:
                    temporal_evidence = temporal_dynamic_evidence_score(
                        track,
                        match_score,
                        stable_match_ready,
                        geometry,
                        confirmed_dynamic_cfg,
                    )
                    track.temporal_dynamic_score = float(
                        np.clip(
                            confirmed_dynamic_temporal_momentum * track.temporal_dynamic_score
                            + (1.0 - confirmed_dynamic_temporal_momentum) * temporal_evidence,
                            0.0,
                            1.0,
                        )
                    )
                else:
                    track.temporal_dynamic_score = 0.0

                dynamic_track_candidate = (
                    confirmed_dynamic_enabled
                    and detection.canonical_name in confirmed_dynamic_classes
                    and track.hits >= confirmed_dynamic_min_hits
                    and track.confirmation_score >= confirmed_dynamic_confirmation_floor
                    and track.temporal_dynamic_score >= confirmed_dynamic_activation_score
                    and track.stable_match_streak >= confirmed_dynamic_min_stable_streak
                    and track.geometry_confirmation_streak >= confirmed_dynamic_min_geometry_streak
                    and (
                        track.motion_score >= confirmed_dynamic_motion_floor
                        or track.tube_motion_score >= confirmed_dynamic_tube_floor
                        or bool(geometry.get("verified_dynamic", False))
                    )
                )
                if dynamic_track_candidate:
                    track.dynamic_confirmation_streak = min(
                        track.dynamic_confirmation_streak + 1,
                        confirmed_dynamic_min_streak + 6,
                    )
                elif geometry_static_ready and track.static_score >= confirmed_dynamic_static_reset:
                    track.dynamic_confirmation_streak = 0
                    track.stable_match_streak = 0
                    track.geometry_confirmation_streak = 0
                    track.temporal_dynamic_score = min(
                        track.temporal_dynamic_score,
                        confirmed_dynamic_release_score,
                    )
                elif confirmed_dynamic_uncertain_decay > 0:
                    track.dynamic_confirmation_streak = max(
                        track.dynamic_confirmation_streak - confirmed_dynamic_uncertain_decay,
                        0,
                    )
                    track.temporal_dynamic_score = float(
                        np.clip(
                            track.temporal_dynamic_score
                            - 0.5 * confirmed_dynamic_uncertain_decay / max(confirmed_dynamic_min_streak, 1),
                            0.0,
                            1.0,
                        )
                    )
                dynamic_observation = compute_dynamic_memory_observation(
                    detection,
                    float(task["score"]),
                    float(foundation["score"]),
                    float(track.motion_score),
                    float(track.tube_motion_score),
                    float(track.static_score),
                    float(track.confirmation_score),
                    temporal_score,
                    dynamic_cfg,
                )
                if dynamic_enabled:
                    updated_dynamic_memory = float(
                        np.clip(
                            dynamic_momentum * track.dynamic_memory_score
                            + (1.0 - dynamic_momentum) * dynamic_observation,
                            0.0,
                            1.0,
                        )
                    )
                    confirmed_dynamic_track = (
                        confirmed_dynamic_enabled
                        and track.class_name in confirmed_dynamic_classes
                        and track.dynamic_confirmation_streak >= confirmed_dynamic_min_streak
                        and track.temporal_dynamic_score >= confirmed_dynamic_release_score
                    )
                    if (
                        confirmed_dynamic_enabled
                        and detection.canonical_name in confirmed_dynamic_classes
                        and not confirmed_dynamic_track
                    ):
                        updated_dynamic_memory = min(
                            updated_dynamic_memory,
                            confirmed_dynamic_unconfirmed_cap,
                        )
                    track.dynamic_memory_score = float(updated_dynamic_memory)
                else:
                    track.dynamic_memory_score = 0.0
                track.bbox = (detection.x1, detection.y1, detection.x2, detection.y2)
                track.mask = mask.copy()
                track.center = current_center
                track.last_depth = current_depth
                track.appearance = current_appearance
                static_score = track.static_score
                motion_score = track.motion_score
                dynamic_memory_score = track.dynamic_memory_score
                geometry_dynamic_score = track.geometry_dynamic_score
                geometry_static_score = track.geometry_static_score
                geometry_support = track.geometry_support

            assigned_track_ids.add(track.track_id)
            touched_track_ids.add(track.track_id)
            detection.track_id = track.track_id
            temporal_score = float(np.clip(track.hits / float(min_hits), 0.0, 1.0))
            track_confirmed = track.hits >= min_hits and track.confirmation_score >= confirm_threshold
            confirmed_dynamic_track = (
                confirmed_dynamic_enabled
                and track.class_name in confirmed_dynamic_classes
                and track.dynamic_confirmation_streak >= confirmed_dynamic_min_streak
            )
            track_infos.append(
                {
                    "track_id": track.track_id,
                    "track_hits": track.hits,
                    "track_confirmation": float(track.confirmation_score),
                    "motion_score": float(track.motion_score),
                    "static_score": float(track.static_score),
                    "tube_motion_score": float(track.tube_motion_score),
                    "dynamic_memory_score": float(track.dynamic_memory_score),
                    "dynamic_confirmation_streak": int(track.dynamic_confirmation_streak),
                    "confirmed_dynamic_track": bool(confirmed_dynamic_track),
                    "geometry_dynamic_score": float(geometry_dynamic_score),
                    "geometry_static_score": float(geometry_static_score),
                    "geometry_support_points": int(geometry_support),
                    "temporal_fusion_score": float(track.temporal_fusion_score),
                    "temporal_id_consistency": float(track.temporal_id_consistency),
                    "temporal_mask_agreement": float(track.temporal_mask_agreement),
                    "temporal_box_agreement": float(track.temporal_box_agreement),
                    "geometry_verified_dynamic": bool(geometry.get("verified_dynamic", False)),
                    "geometry_verified_static": bool(geometry.get("verified_static", False)),
                    "max_motion_score": float(track.max_motion_score),
                    "track_confirmed": bool(track_confirmed),
                    "match_score": float(match_score),
                    "temporal_score": temporal_score,
                    "association_components": match_components,
                    "cumulative_displacement": float(track.cumulative_displacement),
                    "max_center_distance": float(track.max_center_distance),
                }
            )

        scene_context = self._estimate_scene_dynamic_context(track_infos)
        for item in track_infos:
            item["scene_dynamic_context"] = float(scene_context["current"])
            item["scene_dynamic_context_ema"] = float(scene_context["ema"])

        stale_track_ids = []
        max_age = max(int(cfg.get("max_age", 4)), 1)
        for track_id, track in self.tracks.items():
            if track_id in touched_track_ids:
                continue
            track.misses += 1
            if track.misses > max_age:
                stale_track_ids.append(track_id)
        for track_id in stale_track_ids:
            self.tracks.pop(track_id, None)
        return track_infos

    def _age_tracks(self):
        cfg = self.config.get("panoptic_memory", {})
        max_age = max(int(cfg.get("max_age", 4)), 1)
        dynamic_cfg = self.config.get("dynamic_memory", {})
        propagation_decay = float(dynamic_cfg.get("propagation_decay", 0.94))
        stale_track_ids = []
        for track_id, track in self.tracks.items():
            track.misses += 1
            track.dynamic_memory_score = float(np.clip(track.dynamic_memory_score * propagation_decay, 0.0, 1.0))
            track.dynamic_confirmation_streak = max(int(track.dynamic_confirmation_streak) - 1, 0)
            if track.misses > max_age:
                stale_track_ids.append(track_id)
        for track_id in stale_track_ids:
            self.tracks.pop(track_id, None)

    def _estimate_scene_dynamic_context(self, track_infos):
        if not track_infos:
            return {"current": 0.0, "ema": float(self.scene_dynamic_context)}
        mean_motion = float(np.mean([item.get("motion_score", 0.0) for item in track_infos]))
        mean_anti_static = float(np.mean([1.0 - item.get("static_score", 0.0) for item in track_infos]))
        mean_geometry = float(np.mean([item.get("geometry_dynamic_score", 0.0) for item in track_infos]))
        context = float(np.clip(0.45 * mean_motion + 0.25 * mean_anti_static + 0.30 * mean_geometry, 0.0, 1.0))
        cfg = self.config.get("dynamic_memory", {})
        momentum = float(cfg.get("scene_context_momentum", 0.70))
        self.scene_dynamic_context = float(np.clip(momentum * self.scene_dynamic_context + (1.0 - momentum) * context, 0.0, 1.0))
        return {"current": context, "ema": float(self.scene_dynamic_context)}

    def _apply_panoptic_gate(self, detection, foundation, task, track_info):
        # 多级安全门控：不是“检测到就删”，而是结合单帧与时序证据做最终裁决。
        cfg = self.config.get("panoptic_memory", {})
        threshold = float(self.config.get("task_relevance", {}).get("min_score", 0.45))
        foundation_min_score = float(self.config.get("foundation_reliability", {}).get("min_score", 0.32))
        canonical_name = detection.canonical_name
        class_motion_thresholds = cfg.get("class_motion_thresholds", {})
        class_high_confidence = cfg.get("class_high_confidence_overrides", {})
        class_static_thresholds = cfg.get("class_static_protection_thresholds", {})
        class_motion_foundation_ceiling = cfg.get("class_motion_foundation_ceiling", {})
        confirmed_support_classes = set(cfg.get("confirmed_support_classes", ["person"]))
        class_confirmed_hit_requirements = cfg.get("class_confirmed_hit_requirements", {})
        high_conf_override = float(class_high_confidence.get(canonical_name, cfg.get("high_confidence_override", 0.90)))
        motion_threshold = float(class_motion_thresholds.get(canonical_name, cfg.get("motion_filter_threshold", 0.18)))
        static_protection_threshold = float(class_static_thresholds.get(canonical_name, cfg.get("static_protection_threshold", 0.72)))
        confirmed_hits_required = int(class_confirmed_hit_requirements.get(canonical_name, cfg.get("min_hits", 2)))

        foundation_score = float(foundation["score"])
        base_score = float(task["score"])
        motion_score = float(track_info.get("motion_score", 0.0))
        static_score = float(track_info.get("static_score", 0.0))
        tube_motion_score = float(track_info.get("tube_motion_score", 0.0))
        track_confirmation = float(track_info.get("track_confirmation", 0.0))
        geometry_dynamic_score = float(track_info.get("geometry_dynamic_score", 0.0))
        geometry_static_score = float(track_info.get("geometry_static_score", 0.0))
        geometry_support_points = int(track_info.get("geometry_support_points", 0))
        geometry_verified_dynamic = bool(track_info.get("geometry_verified_dynamic", False))
        geometry_verified_static = bool(track_info.get("geometry_verified_static", False))
        dynamic_cfg = self.config.get("dynamic_memory", {})
        dynamic_enabled = bool(dynamic_cfg.get("enabled", True))
        dynamic_memory_score = float(track_info.get("dynamic_memory_score", 0.0))
        scene_dynamic_context = float(track_info.get("scene_dynamic_context", 0.0))
        track_confirmed = bool(track_info.get("track_confirmed", False))
        temporal_fusion_score = float(track_info.get("temporal_fusion_score", 0.0))
        temporal_id_consistency = float(track_info.get("temporal_id_consistency", 0.0))
        temporal_mask_agreement = float(track_info.get("temporal_mask_agreement", 0.0))
        temporal_box_agreement = float(track_info.get("temporal_box_agreement", 0.0))
        allow_confirmed_support = canonical_name in confirmed_support_classes
        geometry_cfg = self.config.get("geometry_consistency", {})

        adaptive_profiles_cfg = cfg.get("adaptive_profiles", {})
        selected_profile = {}
        if adaptive_profiles_cfg.get("enabled", False):
            split = float(adaptive_profiles_cfg.get("scene_context_split", 0.18))
            profile_key = "low_dynamic" if scene_dynamic_context < split else "high_dynamic"
            selected_profile = adaptive_profiles_cfg.get(profile_key, {}) or {}

        profile_panoptic_cfg = selected_profile.get("panoptic_memory", {})
        profile_dynamic_cfg = selected_profile.get("dynamic_memory", {})

        profile_class_motion_thresholds = profile_panoptic_cfg.get("class_motion_thresholds", {})
        profile_class_high_conf = profile_panoptic_cfg.get("class_high_confidence_overrides", {})
        profile_class_static_thresholds = profile_panoptic_cfg.get("class_static_protection_thresholds", {})
        profile_class_motion_foundation = profile_panoptic_cfg.get("class_motion_foundation_ceiling", {})

        motion_threshold = float(
            profile_class_motion_thresholds.get(
                canonical_name,
                profile_panoptic_cfg.get(
                    "motion_filter_threshold",
                    motion_threshold,
                ),
            )
        )
        high_conf_override = float(
            profile_class_high_conf.get(
                canonical_name,
                profile_panoptic_cfg.get(
                    "high_confidence_override",
                    high_conf_override,
                ),
            )
        )
        static_protection_threshold = float(
            profile_class_static_thresholds.get(
                canonical_name,
                profile_panoptic_cfg.get(
                    "static_protection_threshold",
                    static_protection_threshold,
                ),
            )
        )
        static_protection_scene_context_ceiling = float(
            profile_panoptic_cfg.get(
                "static_protection_scene_context_ceiling",
                cfg.get("static_protection_scene_context_ceiling", 1.01),
            )
        )

        dynamic_threshold = float(profile_dynamic_cfg.get("min_score", dynamic_cfg.get("min_score", 0.60)))
        dynamic_min_hits = max(int(profile_dynamic_cfg.get("min_hits", dynamic_cfg.get("min_hits", cfg.get("min_hits", 2)))), 1)
        dynamic_static_ceiling = float(profile_dynamic_cfg.get("static_ceiling", dynamic_cfg.get("static_ceiling", 0.92)))
        dynamic_foundation_floor = float(profile_dynamic_cfg.get("foundation_floor", dynamic_cfg.get("foundation_floor", max(foundation_min_score, 0.36))))
        dynamic_motion_floor = float(profile_dynamic_cfg.get("min_motion_score_for_activation", dynamic_cfg.get("min_motion_score_for_activation", motion_threshold)))
        dynamic_tube_motion_floor = float(profile_dynamic_cfg.get("min_tube_motion_score_for_activation", dynamic_cfg.get("min_tube_motion_score_for_activation", cfg.get("tube_motion_threshold", 0.28))))
        dynamic_evidence_floor = float(profile_dynamic_cfg.get("min_dynamic_evidence", dynamic_cfg.get("min_dynamic_evidence", min(dynamic_motion_floor, dynamic_tube_motion_floor))))
        adaptive_scene_gate_enabled = bool(profile_dynamic_cfg.get("adaptive_scene_gate_enabled", dynamic_cfg.get("adaptive_scene_gate_enabled", False)))
        scene_context_threshold = float(profile_dynamic_cfg.get("scene_context_threshold", dynamic_cfg.get("scene_context_threshold", 0.185)))

        weak_dynamic_guard_cfg = dict(cfg.get("weak_dynamic_guard", {}))
        weak_dynamic_guard_cfg.update(profile_panoptic_cfg.get("weak_dynamic_guard", {}))
        weak_dynamic_guard_enabled = bool(weak_dynamic_guard_cfg.get("enabled", False))
        weak_dynamic_guard_classes = {
            canonical_class_name(item) for item in weak_dynamic_guard_cfg.get("classes", ["person", "pedestrian", "worker"])
        }
        confirmed_dynamic_cfg = dict(cfg.get("confirmed_dynamic_track", {}))
        confirmed_dynamic_cfg.update(profile_panoptic_cfg.get("confirmed_dynamic_track", {}))
        confirmed_dynamic_enabled = bool(confirmed_dynamic_cfg.get("enabled", False))
        confirmed_dynamic_classes = {
            canonical_class_name(item)
            for item in confirmed_dynamic_cfg.get("classes", ["person", "pedestrian", "worker"])
        }
        require_confirmed_dynamic_for_memory = bool(
            confirmed_dynamic_cfg.get("required_for_dynamic_memory", False)
        )
        weak_dynamic_guard_context = float(weak_dynamic_guard_cfg.get("scene_context_threshold", 0.14))
        weak_dynamic_guard_motion = float(weak_dynamic_guard_cfg.get("motion_threshold", max(motion_threshold, 0.22)))
        weak_dynamic_guard_memory = float(weak_dynamic_guard_cfg.get("dynamic_memory_threshold", max(dynamic_threshold, 0.78)))
        weak_dynamic_guard_static = float(weak_dynamic_guard_cfg.get("static_score_floor", 0.62))
        static_veto_hits = max(int(geometry_cfg.get("static_veto_min_hits", 4)), max(int(cfg.get("min_hits", 2)), 1))
        static_veto_confirm = float(geometry_cfg.get("static_veto_min_confirmation", 0.55))
        if track_info.get("track_hits", 0) < static_veto_hits or track_confirmation < static_veto_confirm:
            geometry_verified_static = False
        motion_margin = float(geometry_cfg.get("motion_margin", 0.06))
        effective_motion_score = max(
            motion_score,
            geometry_dynamic_score
            if geometry_verified_dynamic
            and (
                scene_dynamic_context >= float(geometry_cfg.get("weak_scene_context_threshold", 0.22))
                or motion_score >= max(motion_threshold - motion_margin, 0.0)
            )
            else 0.0,
        )
        effective_static_score = max(static_score, geometry_static_score if geometry_verified_static else 0.0)
        dynamic_evidence = float(max(effective_motion_score, tube_motion_score, 1.0 - effective_static_score))
        confirmed_dynamic_track = bool(track_info.get("confirmed_dynamic_track", False))
        dynamic_memory_ready = dynamic_memory_score >= dynamic_threshold
        if (
            dynamic_memory_ready
            and require_confirmed_dynamic_for_memory
            and confirmed_dynamic_enabled
            and canonical_name in confirmed_dynamic_classes
            and not confirmed_dynamic_track
        ):
            dynamic_memory_ready = False

        decision_reason = "low_task_score"
        filter_out = False
        if base_score < threshold:
            decision_reason = "low_task_score"
        elif foundation_score < foundation_min_score:
            decision_reason = "low_foundation_score"
        elif (
            track_info.get("track_hits", 0) >= max(int(cfg.get("min_hits", 2)), 1)
            and effective_static_score >= static_protection_threshold
            and effective_motion_score < motion_threshold
            and scene_dynamic_context <= static_protection_scene_context_ceiling
        ):
            decision_reason = "panoptic_static_protection"
        elif weak_dynamic_guard_enabled and canonical_name in weak_dynamic_guard_classes and scene_dynamic_context < weak_dynamic_guard_context and effective_static_score >= weak_dynamic_guard_static and effective_motion_score < weak_dynamic_guard_motion and dynamic_memory_score < weak_dynamic_guard_memory:
            decision_reason = "weak_dynamic_guard"
        elif dynamic_enabled and track_info.get("track_hits", 0) >= dynamic_min_hits and dynamic_memory_ready and effective_static_score < dynamic_static_ceiling and foundation_score >= dynamic_foundation_floor and dynamic_evidence >= dynamic_evidence_floor and (effective_motion_score >= dynamic_motion_floor or tube_motion_score >= dynamic_tube_motion_floor) and ((not adaptive_scene_gate_enabled) or scene_dynamic_context >= scene_context_threshold):
            filter_out = True
            decision_reason = "dynamic_memory_propagation"
        elif effective_motion_score >= motion_threshold:
            motion_foundation_limit = float(
                profile_class_motion_foundation.get(
                    canonical_name,
                    profile_panoptic_cfg.get(
                        "motion_foundation_ceiling",
                        class_motion_foundation_ceiling.get(canonical_name, 1.01),
                    ),
                )
            )
            if foundation_score <= motion_foundation_limit:
                filter_out = True
                decision_reason = "motion_supported"
            else:
                decision_reason = "rigid_static_prior"
        elif canonical_name in set(cfg.get("tube_support_classes", ["box", "balloon"])) and track_info.get("track_hits", 0) >= int(cfg.get("tube_min_hits", 3)) and tube_motion_score >= float(cfg.get("class_tube_motion_thresholds", {}).get(canonical_name, cfg.get("tube_motion_threshold", 0.28))):
            motion_foundation_limit = float(class_motion_foundation_ceiling.get(canonical_name, 1.01))
            if foundation_score <= motion_foundation_limit:
                filter_out = True
                decision_reason = "tube_motion_supported"
            else:
                decision_reason = "rigid_static_prior"
        elif base_score >= high_conf_override and foundation_score >= max(foundation_min_score, 0.55):
            filter_out = True
            decision_reason = "high_confidence_override"
        elif allow_confirmed_support and track_confirmed and track_info.get("track_hits", 0) >= confirmed_hits_required and track_confirmation >= float(cfg.get("confirm_threshold", 0.58)) and effective_static_score < static_protection_threshold * 0.85:
            filter_out = True
            decision_reason = "confirmed_track_support"
        else:
            decision_reason = "insufficient_temporal_support"

        components = dict(task["components"])
        components.update({
            "foundation_score": foundation_score,
            "track_confirmation": track_confirmation,
            "motion": effective_motion_score,
            "tube_motion": tube_motion_score,
            "dynamic_memory": dynamic_memory_score,
            "confirmed_dynamic_track": 1.0 if confirmed_dynamic_track else 0.0,
            "scene_dynamic_context": scene_dynamic_context,
            "static": effective_static_score,
            "geometry_dynamic": geometry_dynamic_score,
            "geometry_static": geometry_static_score,
            "geometry_support_points": geometry_support_points,
            "temporal_fusion": temporal_fusion_score,
            "temporal_id_consistency": temporal_id_consistency,
            "temporal_mask_agreement": temporal_mask_agreement,
            "temporal_box_agreement": temporal_box_agreement,
        })
        return {
            "class_name": detection.class_name,
            "canonical_name": detection.canonical_name,
            "source_name": detection.source_name,
            "track_id": int(track_info.get("track_id", -1)),
            "track_hits": int(track_info.get("track_hits", 0)),
            "track_confirmation": track_confirmation,
            "dynamic_memory_score": dynamic_memory_score,
            "dynamic_confirmation_streak": int(track_info.get("dynamic_confirmation_streak", 0)),
            "confirmed_dynamic_track": bool(confirmed_dynamic_track),
            "scene_dynamic_context": scene_dynamic_context,
            "motion_score": effective_motion_score,
            "tube_motion_score": tube_motion_score,
            "static_score": effective_static_score,
            "geometry_dynamic_score": geometry_dynamic_score,
            "geometry_static_score": geometry_static_score,
            "geometry_support_points": geometry_support_points,
            "temporal_fusion_score": temporal_fusion_score,
            "temporal_id_consistency": temporal_id_consistency,
            "temporal_mask_agreement": temporal_mask_agreement,
            "temporal_box_agreement": temporal_box_agreement,
            "foundation_score": foundation_score,
            "base_score": base_score,
            "score": base_score,
            "decision": "filter" if filter_out else "keep",
            "filter_out": filter_out,
            "decision_reason": decision_reason,
            "gate_stage": self._decision_reason_to_gate_stage(decision_reason),
            "components": components,
        }

    @staticmethod
    def _merge_masks(masks, shape_hw):
        merged = np.zeros(shape_hw, dtype=np.uint8)
        for mask in masks:
            merged = np.maximum(merged, mask.astype(np.uint8))
        return merged

    def _filter_rgb(self, image_bgr, merged_mask):
        # 将最终动态掩膜作用到 RGB 图像；默认采用模糊，也支持小区域修补。
        if np.count_nonzero(merged_mask) == 0:
            return image_bgr.copy()

        area_ratio = float(np.count_nonzero(merged_mask)) / float(merged_mask.size)
        fill_mode = self.config["runtime"].get("rgb_fill_mode", "blur")
        if fill_mode == "inpaint" and area_ratio <= float(self.config["runtime"].get("max_inpaint_area_ratio", 0.12)):
            inpaint_mask = (merged_mask * 255).astype(np.uint8)
            return cv2.inpaint(image_bgr, inpaint_mask, 3, cv2.INPAINT_TELEA)

        kernel_size = int(self.config["runtime"].get("mask_blur_kernel", 21))
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.GaussianBlur(image_bgr, (kernel_size, kernel_size), 0)
        alpha = cv2.GaussianBlur(merged_mask.astype(np.float32), (kernel_size, kernel_size), 0)
        alpha = np.clip(alpha[..., None], 0.0, 1.0)
        return (blurred * alpha + image_bgr * (1.0 - alpha)).astype(np.uint8)

    @staticmethod
    def _filter_depth(depth_mm, merged_mask):
        # 对深度图做一致性过滤，阻断动态深度继续进入后端建图。
        if depth_mm is None:
            return None
        filtered = depth_mm.copy()
        filtered[merged_mask > 0] = 0
        return filtered

    @staticmethod
    def _build_overlay(image_bgr, detections, masks, relevance_details):
        overlay = image_bgr.copy()
        source_colors = {
            "drone_detector": (0, 0, 255),
            "open_vocab_detector": (0, 200, 255),
        }
        for idx, detection in enumerate(detections):
            color = source_colors.get(detection.source_name, (0, 255, 0))
            cv2.rectangle(overlay, (detection.x1, detection.y1), (detection.x2, detection.y2), color, 2)
            if idx < len(relevance_details):
                relevance = relevance_details[idx]
                label = (
                    f"T{relevance.get('track_id', -1)}:"
                    f"{detection.class_name}:"
                    f"{detection.score:.2f}:"
                    f"R{relevance['score']:.2f}:"
                    f"F{relevance.get('foundation_score', 0.0):.2f}:"
                    f"M{relevance.get('motion_score', 0.0):.2f}:"
                    f"{relevance['decision']}"
                )
            else:
                label = f"{detection.source_name}:{detection.class_name}:{detection.score:.2f}"
            cv2.putText(
                overlay,
                label,
                (detection.x1, max(20, detection.y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                color,
                1,
                cv2.LINE_AA,
            )
            if idx < len(masks):
                mask = masks[idx].astype(np.uint8)
                colored = np.zeros_like(overlay)
                is_filtered = idx < len(relevance_details) and relevance_details[idx].get("filter_out", True)
                if is_filtered:
                    colored[:, :, 1] = mask * 180
                    overlay = cv2.addWeighted(overlay, 1.0, colored, 0.25, 0)
                else:
                    colored[:, :, 2] = mask * 160
                    overlay = cv2.addWeighted(overlay, 1.0, colored, 0.18, 0)
        return overlay


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent)
