"""Adapt backend packets into DynoSAM-ready frame records or file bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator
import json
import os
import shutil
import sys

import cv2
import numpy as np


OPTICAL_FLOW_METHODS = {
    "none",
    "farneback",
    "raft",
    "raft_small",
    "raft_large",
    "gmflow",
    "unimatch_gmflow",
    "unimatch_gmflow_scale2_mixdata",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def read_tum_depth_metric(path: Path, depth_scale: float) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Failed to read depth image: {path}")
    return depth.astype(np.float64) / float(depth_scale)


def build_instance_mask(
    *,
    dynamic_mask_path: Path,
    objects: list[dict[str, Any]],
    height: int,
    width: int,
) -> np.ndarray:
    """Create a dense instance-id mask from binary dynamic mask and object boxes."""

    dynamic_mask = cv2.imread(str(dynamic_mask_path), cv2.IMREAD_UNCHANGED)
    if dynamic_mask is None:
        raise RuntimeError(f"Failed to read dynamic mask: {dynamic_mask_path}")
    if dynamic_mask.ndim == 3:
        dynamic_mask = dynamic_mask[:, :, 0]
    if dynamic_mask.shape[:2] != (height, width):
        raise RuntimeError(
            f"Mask shape mismatch for {dynamic_mask_path}: got {dynamic_mask.shape[:2]}, expected {(height, width)}"
        )

    fg = dynamic_mask > 0
    assigned = np.zeros((height, width), dtype=bool)
    instance_mask = np.zeros((height, width), dtype=np.int32)
    ordered = sorted(
        objects,
        key=lambda obj: (
            float(obj.get("confidence", 0.0)),
            float(obj.get("dynamic_score", 0.0)),
            float(obj.get("temporal_consistency", 0.0)),
        ),
        reverse=True,
    )
    for obj in ordered:
        object_id = int(obj["object_id"])
        x1, y1, x2, y2 = [int(v) for v in obj["bbox_2d"]]
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))
        if x2 < x1 or y2 < y1:
            continue
        roi = fg[y1 : y2 + 1, x1 : x2 + 1] & ~assigned[y1 : y2 + 1, x1 : x2 + 1]
        instance_mask[y1 : y2 + 1, x1 : x2 + 1][roi] = object_id
        assigned[y1 : y2 + 1, x1 : x2 + 1] |= roi
    return instance_mask


def load_dense_instance_mask(path: Path, *, height: int, width: int) -> np.ndarray:
    instance_mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if instance_mask is None:
        raise RuntimeError(f"Failed to read instance mask: {path}")
    if instance_mask.ndim == 3:
        instance_mask = instance_mask[:, :, 0]
    if instance_mask.shape[:2] != (height, width):
        raise RuntimeError(
            f"Instance mask shape mismatch for {path}: got {instance_mask.shape[:2]}, expected {(height, width)}"
        )
    return instance_mask.astype(np.int32)


def _frame_stats_by_index(frame_stats_path: Path) -> dict[int, dict[str, Any]]:
    if not frame_stats_path.is_file():
        return {}
    rows = load_json(frame_stats_path)
    return {int(item.get("frame_index", i + 1)) - 1: item for i, item in enumerate(rows)}


def _jump_index(packet_analysis_path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    if not packet_analysis_path.is_file():
        return {}
    data = load_json(packet_analysis_path)
    jumps: dict[tuple[int, int], dict[str, Any]] = {}
    for track in data.get("tracks_detail", []):
        object_id = int(track["object_id"])
        for jump in track.get("large_jumps", []):
            jumps[(object_id, int(jump["from_frame"]))] = {
                "flag": "large_world_centroid_jump_to_next",
                "step_m": float(jump["step_m"]),
                "paired_frame": int(jump["to_frame"]),
            }
            jumps[(object_id, int(jump["to_frame"]))] = {
                "flag": "large_world_centroid_jump_from_previous",
                "step_m": float(jump["step_m"]),
                "paired_frame": int(jump["from_frame"]),
            }
    return jumps


def _depth_scale_from_packet(frames: list[dict[str, Any]], object_observations: list[dict[str, Any]]) -> float:
    for obj in object_observations:
        if obj.get("num_depth_pixels", 0) > 0:
            return 5000.0
    return 5000.0


def _make_torchvision_raft_estimator(method: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Create a pretrained torchvision RAFT estimator returning HxWx2 flow."""

    try:
        import torch
        import torch.nn.functional as torch_f
        from torchvision.models.optical_flow import (
            Raft_Large_Weights,
            Raft_Small_Weights,
            raft_large,
            raft_small,
        )
    except ImportError as exc:
        raise RuntimeError(
            "RAFT optical flow requires torch and torchvision in the active Python environment."
        ) from exc

    if method in {"raft", "raft_large"}:
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights, progress=True)
    elif method == "raft_small":
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights, progress=True)
    else:
        raise ValueError(f"Unsupported RAFT method: {method}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transforms = weights.transforms()
    model = model.to(device).eval()

    def to_tensor_bchw(image_bgr: np.ndarray) -> "torch.Tensor":
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1)
        return tensor.unsqueeze(0)

    def estimate(current_bgr: np.ndarray, next_bgr: np.ndarray) -> np.ndarray:
        if current_bgr.shape[:2] != next_bgr.shape[:2]:
            raise RuntimeError(
                f"RAFT input shape mismatch: current={current_bgr.shape[:2]}, next={next_bgr.shape[:2]}"
            )
        height, width = current_bgr.shape[:2]
        image1, image2 = transforms(to_tensor_bchw(current_bgr), to_tensor_bchw(next_bgr))
        image1 = image1.to(device, non_blocking=True)
        image2 = image2.to(device, non_blocking=True)

        pad_h = (8 - height % 8) % 8
        pad_w = (8 - width % 8) % 8
        if pad_h or pad_w:
            image1 = torch_f.pad(image1, (0, pad_w, 0, pad_h), mode="replicate")
            image2 = torch_f.pad(image2, (0, pad_w, 0, pad_h), mode="replicate")

        with torch.inference_mode():
            flow_predictions = model(image1, image2)
        flow = flow_predictions[-1][0, :, :height, :width].permute(1, 2, 0)
        return flow.detach().cpu().numpy().astype(np.float32, copy=False)

    return estimate


def _make_unimatch_gmflow_estimator(method: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Create a UniMatch GMFlow estimator returning HxWx2 flow."""

    repo_root = Path(__file__).resolve().parents[3] / "third_party" / "flow_models" / "unimatch"
    if not repo_root.is_dir():
        raise RuntimeError(f"UniMatch repository not found: {repo_root}")

    checkpoint = repo_root / "pretrained" / "gmflow-scale2-mixdata-train320x576-9ff1c094.pth"
    if not checkpoint.is_file():
        raise RuntimeError(
            "UniMatch GMFlow checkpoint is missing. Expected: "
            f"{checkpoint}. Download it from the official UniMatch model zoo."
        )

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        import torch
        import torch.nn.functional as torch_f
        from unimatch.unimatch import UniMatch
    except ImportError as exc:
        raise RuntimeError(
            "UniMatch GMFlow requires the official UniMatch repository and torch."
        ) from exc

    if method not in {"gmflow", "unimatch_gmflow", "unimatch_gmflow_scale2_mixdata"}:
        raise ValueError(f"Unsupported UniMatch GMFlow method: {method}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniMatch(
        feature_channels=128,
        num_scales=2,
        upsample_factor=4,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        reg_refine=False,
        task="flow",
    ).to(device)

    checkpoint_data = torch.load(str(checkpoint), map_location=device)
    state_dict = checkpoint_data["model"] if isinstance(checkpoint_data, dict) and "model" in checkpoint_data else checkpoint_data
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    def to_tensor_bchw(image_bgr: np.ndarray) -> "torch.Tensor":
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1).float()
        return tensor.unsqueeze(0).to(device, non_blocking=True)

    def estimate(current_bgr: np.ndarray, next_bgr: np.ndarray) -> np.ndarray:
        if current_bgr.shape[:2] != next_bgr.shape[:2]:
            raise RuntimeError(
                f"UniMatch GMFlow input shape mismatch: current={current_bgr.shape[:2]}, next={next_bgr.shape[:2]}"
            )
        height, width = current_bgr.shape[:2]
        image1 = to_tensor_bchw(current_bgr)
        image2 = to_tensor_bchw(next_bgr)

        inference_size = [
            int(np.ceil(height / 32.0)) * 32,
            int(np.ceil(width / 32.0)) * 32,
        ]
        if inference_size != [height, width]:
            image1 = torch_f.interpolate(image1, size=inference_size, mode="bilinear", align_corners=True)
            image2 = torch_f.interpolate(image2, size=inference_size, mode="bilinear", align_corners=True)

        with torch.inference_mode():
            results = model(
                image1,
                image2,
                attn_type="swin",
                attn_splits_list=[2, 8],
                corr_radius_list=[-1, 4],
                prop_radius_list=[-1, 1],
                num_reg_refine=1,
                task="flow",
            )
        flow = results["flow_preds"][-1]
        if inference_size != [height, width]:
            flow = torch_f.interpolate(flow, size=(height, width), mode="bilinear", align_corners=True)
            flow[:, 0] = flow[:, 0] * width / inference_size[1]
            flow[:, 1] = flow[:, 1] * height / inference_size[0]
        flow = flow[0].permute(1, 2, 0)
        return flow.detach().cpu().numpy().astype(np.float32, copy=False)

    return estimate


@dataclass(frozen=True)
class DynosamDirectFramePacket:
    frame_id: int
    timestamp: float
    raw_rgb: np.ndarray
    static_filtered_rgb: np.ndarray
    static_filtered_depth_metric: np.ndarray
    raw_depth_metric: np.ndarray
    instance_mask: np.ndarray
    optical_flow: np.ndarray | None
    camera_pose_twc: dict[str, Any] | None
    observations: list[dict[str, Any]]


@dataclass(frozen=True)
class DynosamFrameAdapterRecord:
    frame_id: int
    timestamp: float
    raw_rgb_path: Path
    static_filtered_rgb_path: Path
    static_filtered_depth_path: Path
    raw_depth_path: Path
    dynamic_mask_path: Path
    instance_mask_path: Path | None
    optical_flow_path: Path | None
    camera_pose_twc: dict[str, Any] | None
    observations: list[dict[str, Any]]
    depth_scale: float = 5000.0

    def load_raw_rgb(self) -> np.ndarray:
        rgb = cv2.imread(str(self.raw_rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError(f"Failed to read raw RGB: {self.raw_rgb_path}")
        return rgb

    def load_static_filtered_rgb(self) -> np.ndarray:
        rgb = cv2.imread(str(self.static_filtered_rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError(f"Failed to read filtered RGB: {self.static_filtered_rgb_path}")
        return rgb

    def load_raw_depth_metric(self) -> np.ndarray:
        return read_tum_depth_metric(self.raw_depth_path, self.depth_scale)

    def load_static_filtered_depth_metric(self) -> np.ndarray:
        return read_tum_depth_metric(self.static_filtered_depth_path, self.depth_scale)

    def load_instance_mask(self) -> np.ndarray:
        rgb = self.load_static_filtered_rgb()
        height, width = rgb.shape[:2]
        if self.instance_mask_path is not None and self.instance_mask_path.is_file():
            return load_dense_instance_mask(self.instance_mask_path, height=height, width=width)
        return build_instance_mask(
            dynamic_mask_path=self.dynamic_mask_path,
            objects=self.observations,
            height=height,
            width=width,
        )

    def load_optical_flow(self) -> np.ndarray | None:
        if self.optical_flow_path is None or not self.optical_flow_path.is_file():
            return None
        flow = np.load(self.optical_flow_path)
        if flow.ndim != 3 or flow.shape[2] != 2:
            raise RuntimeError(f"Optical flow must have shape HxWx2: {self.optical_flow_path}")
        return flow.astype(np.float32, copy=False)

    def to_direct_frame_packet(self) -> DynosamDirectFramePacket:
        return DynosamDirectFramePacket(
            frame_id=self.frame_id,
            timestamp=self.timestamp,
            raw_rgb=self.load_raw_rgb(),
            static_filtered_rgb=self.load_static_filtered_rgb(),
            static_filtered_depth_metric=self.load_static_filtered_depth_metric(),
            raw_depth_metric=self.load_raw_depth_metric(),
            instance_mask=self.load_instance_mask(),
            optical_flow=self.load_optical_flow(),
            camera_pose_twc=self.camera_pose_twc,
            observations=list(self.observations),
        )


@dataclass(frozen=True)
class DynosamAdapterBundle:
    source_packet_root: Path
    camera_info: dict[str, Any]
    frames: list[DynosamFrameAdapterRecord]
    object_observations: list[dict[str, Any]]
    tracks: list[dict[str, Any]]
    validation: dict[str, Any]
    manifest: dict[str, Any]

    def iter_direct_frame_packets(self) -> Iterator[DynosamDirectFramePacket]:
        for frame in self.frames:
            yield frame.to_direct_frame_packet()


def build_dynosam_adapter_bundle(
    *,
    packet_root: Path,
    benchmark_summary_path: Path | None = None,
    frame_stats_path: Path | None = None,
    packet_analysis_path: Path | None = None,
    low_quality_weight: float = 0.35,
) -> DynosamAdapterBundle:
    packet_root = packet_root.resolve()

    manifest = load_json(packet_root / "manifest.json")
    frames = load_json(packet_root / "frames.json")
    observations = load_json(packet_root / "object_observations.json")
    tracks = load_json(packet_root / "tracks.json")
    sequence_root = Path(manifest["sequence_root"]).resolve()
    source_sequence_root = None
    if benchmark_summary_path and benchmark_summary_path.is_file():
        source_sequence_root = Path(load_json(benchmark_summary_path)["sequence_root"]).resolve()
    stats = _frame_stats_by_index(frame_stats_path) if frame_stats_path else {}
    jumps = _jump_index(packet_analysis_path) if packet_analysis_path else {}
    depth_scale = _depth_scale_from_packet(frames, observations)

    frame_objects: dict[int, list[dict[str, Any]]] = {}
    for obj in observations:
        frame_objects.setdefault(int(obj["frame_id"]), []).append(obj)

    adapter_frames: list[DynosamFrameAdapterRecord] = []
    output_observations: list[dict[str, Any]] = []
    mask_nonzero_counts: list[int] = []

    image_width = 640
    image_height = 480

    for frame in frames:
        frame_id = int(frame["frame_id"])
        timestamp = float(frame["timestamp"])
        objects = frame_objects.get(frame_id, [])

        filtered_rgb_src = sequence_root / frame["rgb"]
        filtered_depth_src = sequence_root / frame["filtered_depth"]
        raw_depth_src = sequence_root / frame["raw_depth"]
        raw_rgb_src = filtered_rgb_src
        source_stats = stats.get(frame_id)
        if source_sequence_root is not None and source_stats and source_stats.get("rgb_path"):
            raw_rgb_src = source_sequence_root / str(source_stats["rgb_path"])

        rgb_img = cv2.imread(str(filtered_rgb_src), cv2.IMREAD_COLOR)
        if rgb_img is None:
            raise RuntimeError(f"Failed to read filtered RGB for shape: {filtered_rgb_src}")
        image_height, image_width = rgb_img.shape[:2]

        dense_instance_mask_path = sequence_root / frame["instance_mask"] if frame.get("instance_mask") else None
        if dense_instance_mask_path is not None and dense_instance_mask_path.is_file():
            instance_mask = load_dense_instance_mask(dense_instance_mask_path, height=image_height, width=image_width)
        else:
            instance_mask = build_instance_mask(
                dynamic_mask_path=sequence_root / frame["mask"],
                objects=objects,
                height=image_height,
                width=image_width,
            )
        mask_nonzero_counts.append(int(np.count_nonzero(instance_mask)))

        adapter_frames.append(
            DynosamFrameAdapterRecord(
                frame_id=frame_id,
                timestamp=timestamp,
                raw_rgb_path=raw_rgb_src,
                static_filtered_rgb_path=filtered_rgb_src,
                static_filtered_depth_path=filtered_depth_src,
                raw_depth_path=raw_depth_src,
                dynamic_mask_path=sequence_root / frame["mask"],
                instance_mask_path=dense_instance_mask_path,
                optical_flow_path=None,
                camera_pose_twc=frame.get("camera_pose_twc"),
                observations=list(objects),
                depth_scale=depth_scale,
            )
        )

        for obj in objects:
            object_id = int(obj["object_id"])
            jump = jumps.get((object_id, frame_id))
            quality_flags = []
            quality_weight = 1.0
            if jump is not None:
                quality_flags.append(jump["flag"])
                quality_weight = float(low_quality_weight)
            output_observations.append(
                {
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "object_id": object_id,
                    "semantic_id": int(obj.get("semantic_id", 0)),
                    "semantic_label": str(obj.get("semantic_label", "")),
                    "bbox_2d": obj.get("bbox_2d"),
                    "centroid_camera": obj.get("centroid_camera"),
                    "centroid_world": obj.get("centroid_world"),
                    "bbox_3d_camera_min": obj.get("bbox_3d_camera_min"),
                    "bbox_3d_camera_max": obj.get("bbox_3d_camera_max"),
                    "bbox_3d_world_min": obj.get("bbox_3d_world_min"),
                    "bbox_3d_world_max": obj.get("bbox_3d_world_max"),
                    "point_cloud_path": obj.get("point_cloud_path"),
                    "num_depth_pixels": int(obj.get("num_depth_pixels", 0)),
                    "confidence": float(obj.get("confidence", 0.0)),
                    "match_score": float(obj.get("match_score", 0.0)),
                    "association_bbox_iou": float(obj.get("association_bbox_iou", 0.0)),
                    "association_mask_iou": float(obj.get("association_mask_iou", 0.0)),
                    "association_appearance": float(obj.get("association_appearance", 0.0)),
                    "association_depth": float(obj.get("association_depth", 0.0)),
                    "association_id_match": float(obj.get("association_id_match", 0.0)),
                    "temporal_fusion_score": float(obj.get("temporal_fusion_score", 0.0)),
                    "temporal_id_consistency": float(obj.get("temporal_id_consistency", 0.0)),
                    "temporal_mask_agreement": float(obj.get("temporal_mask_agreement", 0.0)),
                    "temporal_box_agreement": float(obj.get("temporal_box_agreement", 0.0)),
                    "quality_weight": quality_weight,
                    "quality_flags": quality_flags,
                    "quality_detail": jump,
                }
            )

    camera_info = {
        "model": "pinhole",
        "width": image_width,
        "height": image_height,
        "fx": 535.4,
        "fy": 539.2,
        "cx": 320.1,
        "cy": 247.6,
        "depth_units": "meters",
        "image_source": "raw_rgb",
        "depth_source": "raw_depth_metric_npy",
        "dynamic_label_source": "instance_mask_png_uint16",
        "optical_flow_source": "optional_forward_flow_npy_float32",
        "pose_convention": manifest.get("pose_convention"),
    }
    validation = {
        "frames": len(adapter_frames),
        "objects": len(output_observations),
        "tracks": len(tracks),
        "frames_with_nonzero_instance_mask": int(sum(1 for count in mask_nonzero_counts if count > 0)),
        "low_quality_observations": int(sum(1 for item in output_observations if item["quality_weight"] < 1.0)),
    }
    output_manifest = {
        "schema": "dynamic_slam_dynosam_adapter_v0",
        "source_packet": str(packet_root),
        "note": "DynoSAM adapter view: direct per-frame RGB/Depth/Mask access plus optional file-bundle materialization.",
        "validation": validation,
    }

    return DynosamAdapterBundle(
        source_packet_root=packet_root,
        camera_info=camera_info,
        frames=adapter_frames,
        object_observations=output_observations,
        tracks=tracks,
        validation=validation,
        manifest=output_manifest,
    )


def materialize_dynosam_bundle(
    *,
    adapter_bundle: DynosamAdapterBundle,
    output_root: Path,
    materialize_mode: str = "symlink",
    optical_flow_method: str = "none",
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    if materialize_mode not in {"symlink", "copy"}:
        raise ValueError("materialize_mode must be symlink or copy")
    if optical_flow_method not in OPTICAL_FLOW_METHODS:
        allowed = ", ".join(sorted(OPTICAL_FLOW_METHODS))
        raise ValueError(f"optical_flow_method must be one of: {allowed}")

    out_rgb = output_root / "rgb_raw"
    out_depth = output_root / "depth_metric"
    out_mask = output_root / "instance_masks"
    out_static = output_root / "static_slam"
    out_flow = output_root / "optical_flow"

    output_frames: list[dict[str, Any]] = []
    raw_rgb_cache: list[np.ndarray | None] = []
    if optical_flow_method != "none":
        for frame in adapter_bundle.frames:
            raw_rgb_cache.append(frame.load_raw_rgb())

    raft_estimator: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
    if optical_flow_method.startswith("raft"):
        raft_estimator = _make_torchvision_raft_estimator(optical_flow_method)

    gmflow_estimator: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
    if optical_flow_method in {"gmflow", "unimatch_gmflow", "unimatch_gmflow_scale2_mixdata"}:
        gmflow_estimator = _make_unimatch_gmflow_estimator(optical_flow_method)

    def compute_forward_flow(index: int) -> np.ndarray | None:
        if optical_flow_method == "none":
            return None
        current = raw_rgb_cache[index]
        if current is None:
            return None
        if index + 1 >= len(raw_rgb_cache):
            return np.zeros((current.shape[0], current.shape[1], 2), dtype=np.float32)
        nxt = raw_rgb_cache[index + 1]
        if nxt is None:
            return None
        if raft_estimator is not None:
            return raft_estimator(current, nxt)
        if gmflow_estimator is not None:
            return gmflow_estimator(current, nxt)
        current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            current_gray,
            next_gray,
            None,
            0.5,
            5,
            21,
            5,
            7,
            1.5,
            0,
        )
        return flow.astype(np.float32, copy=False)

    for index, frame in enumerate(adapter_bundle.frames):
        rgb_dst = out_rgb / frame.raw_rgb_path.name
        static_rgb_dst = out_static / "rgb_filtered" / frame.static_filtered_rgb_path.name
        static_depth_dst = out_static / "depth_filtered" / frame.static_filtered_depth_path.name
        link_or_copy(frame.raw_rgb_path, rgb_dst, materialize_mode)
        link_or_copy(frame.static_filtered_rgb_path, static_rgb_dst, materialize_mode)
        link_or_copy(frame.static_filtered_depth_path, static_depth_dst, materialize_mode)

        depth_metric = frame.load_raw_depth_metric()
        depth_npy = out_depth / f"{frame.frame_id:06d}_{frame.timestamp:.6f}.npy"
        depth_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(depth_npy, depth_metric)

        instance_mask = frame.load_instance_mask()
        mask_npy = out_mask / f"{frame.frame_id:06d}_{frame.timestamp:.6f}_instances.npy"
        mask_png = out_mask / f"{frame.frame_id:06d}_{frame.timestamp:.6f}_instances.png"
        mask_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(mask_npy, instance_mask.astype(np.int32))
        cv2.imwrite(str(mask_png), instance_mask.astype(np.uint16))

        output_frame = {
            "frame_id": frame.frame_id,
            "timestamp": frame.timestamp,
            "raw_rgb": str(rgb_dst.relative_to(output_root)),
            "raw_depth_metric_npy": str(depth_npy.relative_to(output_root)),
            "instance_mask_npy": str(mask_npy.relative_to(output_root)),
            "instance_mask_png_uint16": str(mask_png.relative_to(output_root)),
            "static_filtered_rgb": str(static_rgb_dst.relative_to(output_root)),
            "static_filtered_depth": str(static_depth_dst.relative_to(output_root)),
            "camera_pose_twc": frame.camera_pose_twc,
            "num_objects": len(frame.observations),
        }
        optical_flow = compute_forward_flow(index)
        if optical_flow is not None:
            flow_npy = out_flow / f"{frame.frame_id:06d}_{frame.timestamp:.6f}_flow.npy"
            flow_npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(flow_npy, optical_flow)
            output_frame["optical_flow_npy"] = str(flow_npy.relative_to(output_root))
        output_frames.append(output_frame)

    validation = {
        "frames": len(output_frames),
        "objects": len(adapter_bundle.object_observations),
        "tracks": len(adapter_bundle.tracks),
        "raw_rgb_missing": int(sum(1 for item in output_frames if not (output_root / item["raw_rgb"]).exists())),
        "raw_depth_metric_missing": int(sum(1 for item in output_frames if not (output_root / item["raw_depth_metric_npy"]).exists())),
        "instance_mask_missing": int(sum(1 for item in output_frames if not (output_root / item["instance_mask_npy"]).exists())),
        "optical_flow_method": optical_flow_method,
        "optical_flow_missing": int(sum(1 for item in output_frames if optical_flow_method != "none" and not (output_root / item.get("optical_flow_npy", "")).exists())),
        "frames_with_nonzero_instance_mask": adapter_bundle.validation["frames_with_nonzero_instance_mask"],
        "low_quality_observations": adapter_bundle.validation["low_quality_observations"],
    }
    output_manifest = {
        "schema": "dynamic_slam_dynosam_adapter_v0",
        "source_packet": str(adapter_bundle.source_packet_root),
        "note": "DynoSAM-style bundle: raw RGB-D + dense instance-id masks for dynamic backend, filtered RGB-D retained separately for static SLAM.",
        "data_contract": {
            "dynosam_frontend_rgb": "raw_rgb",
            "dynosam_frontend_depth": "raw_depth_metric_npy",
            "dynosam_dynamic_labels": "instance_mask_png_uint16",
            "dynosam_frontend_optical_flow": "optical_flow_npy_forward_float32_optional",
            "static_slam_baseline_rgb": "static_filtered_rgb",
            "static_slam_baseline_depth": "static_filtered_depth",
        },
        "files": {
            "camera_info": "camera_info.json",
            "frames": "frames.json",
            "object_observations": "object_observations.json",
            "tracks": "tracks.json",
            "validation": "validation.json",
        },
        "validation": validation,
    }

    write_json(output_root / "camera_info.json", adapter_bundle.camera_info)
    write_json(output_root / "frames.json", output_frames)
    write_json(output_root / "object_observations.json", adapter_bundle.object_observations)
    write_json(output_root / "tracks.json", adapter_bundle.tracks)
    write_json(output_root / "validation.json", validation)
    write_json(output_root / "manifest.json", output_manifest)
    return output_manifest


def export_dynosam_bundle(
    *,
    packet_root: Path,
    output_root: Path,
    benchmark_summary_path: Path | None = None,
    frame_stats_path: Path | None = None,
    packet_analysis_path: Path | None = None,
    materialize_mode: str = "symlink",
    low_quality_weight: float = 0.35,
    optical_flow_method: str = "none",
) -> dict[str, Any]:
    adapter_bundle = build_dynosam_adapter_bundle(
        packet_root=packet_root,
        benchmark_summary_path=benchmark_summary_path,
        frame_stats_path=frame_stats_path,
        packet_analysis_path=packet_analysis_path,
        low_quality_weight=low_quality_weight,
    )
    return materialize_dynosam_bundle(
        adapter_bundle=adapter_bundle,
        output_root=output_root,
        materialize_mode=materialize_mode,
        optical_flow_method=optical_flow_method,
    )
