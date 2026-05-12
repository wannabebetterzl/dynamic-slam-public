#!/usr/bin/env python3
"""Build DynoSAM KITTI motion masks with SAM prompted by KITTI track boxes."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch


SAM_CHECKPOINT = Path("/home/lj/multi-UAV-world-SLAM-work/weights/sam_vit_b_01ec64.pth")
SAM3_CHECKPOINT = Path("/home/lj/d-drive/CODEX/sam3.pt")
ALL_DYNAMIC_CLASSES = ["Car", "Van", "Truck", "Tram", "Pedestrian", "Person_sitting", "Cyclist"]
NONRIGID_CLASSES = ["Pedestrian", "Person_sitting", "Cyclist"]
RIGID_CLASSES = ["Car", "Van", "Truck", "Tram"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-root", default="/home/lj/dynamic_SLAM/datasets/kitti_tracking/0004")
    parser.add_argument("--src-name", default="data_gmflow_gmstereo")
    parser.add_argument("--dst-name", default="data_samgt_gmflow_gmstereo")
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--segmenter", choices=["sam1", "sam3", "box"], default="sam3")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--model-type", default="vit_b")
    parser.add_argument("--class-policy", choices=["all", "nonrigid", "rigid", "custom"], default="all")
    parser.add_argument(
        "--preserve-source-classes",
        nargs="*",
        default=[],
        help="Classes copied from the source KITTI label boxes before SAM masks are applied.",
    )
    parser.add_argument(
        "--base-source-motion",
        action="store_true",
        help="Initialize each output mask from src/motion instead of redrawing preserved classes from labels.",
    )
    parser.add_argument(
        "--clear-selected-source-labels",
        action="store_true",
        help="When --base-source-motion is used, clear selected KITTI object ids before writing SAM masks.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=[],
    )
    parser.add_argument("--min-box-area", type=float, default=80.0)
    parser.add_argument(
        "--dilate-pixels",
        type=int,
        default=0,
        help="Dilate each SAM instance mask by this many pixels before writing DynoSAM labels.",
    )
    parser.add_argument(
        "--render-filtered-image0",
        action="store_true",
        help="Apply the generated motion mask to dst/image_0 so downstream ORB actually sees filtered left images.",
    )
    parser.add_argument(
        "--filtered-fill",
        choices=["black", "gray"],
        default="black",
        help="Fill color used when rendering filtered image_0.",
    )
    return parser.parse_args()


def source_classes(args: argparse.Namespace) -> list[str]:
    return list(args.preserve_source_classes or [])


def select_classes(args: argparse.Namespace) -> list[str]:
    if args.class_policy == "custom":
        if not args.classes:
            raise RuntimeError("--class-policy custom requires --classes")
        return args.classes
    if args.class_policy == "nonrigid":
        return NONRIGID_CLASSES
    if args.class_policy == "rigid":
        return RIGID_CLASSES
    return ALL_DYNAMIC_CLASSES


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def materialize(src: Path, dst: Path, overwrite: bool) -> None:
    if overwrite and dst.exists():
        shutil.rmtree(dst)
    ensure_dir(dst)
    for name in ("image_0", "depth", "flow"):
        ensure_dir(dst / name)
        for item in sorted((src / name).iterdir()):
            if item.is_file():
                link_or_copy(item, dst / name / item.name)
    for name in ("times.txt", "pose_gt.txt", "object_pose.txt"):
        link_or_copy(src / name, dst / name)
    ensure_dir(dst / "motion")


def parse_labels(path: Path, keep_classes: set[str], n: int) -> dict[int, list[dict]]:
    by_frame: dict[int, list[dict]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 17:
            continue
        frame = int(parts[0])
        if frame >= n:
            continue
        object_id = int(parts[1])
        cls = parts[2]
        if object_id < 0 or cls not in keep_classes:
            continue
        x1, y1, x2, y2 = [float(v) for v in parts[6:10]]
        if (x2 - x1) * (y2 - y1) <= 0:
            continue
        by_frame.setdefault(frame, []).append(
            {
                "object_id": object_id + 1,
                "class_name": cls,
                "box": np.array([x1, y1, x2, y2], dtype=np.float32),
                "area": float((x2 - x1) * (y2 - y1)),
            }
        )
    return by_frame


def draw_source_labels(
    labels: dict[int, list[dict]],
    frame: int,
    shape_hw: tuple[int, int],
) -> np.ndarray:
    h, w = shape_hw
    instance = np.zeros((h, w), dtype=np.int32)
    items = sorted(labels.get(frame, []), key=lambda item: item["area"], reverse=True)
    for item in items:
        mask = box_mask(shape_hw, item["box"])
        object_id = int(item["object_id"])
        instance[(instance == 0) & (mask > 0)] = object_id
    return instance


def load_source_motion(src: Path, frame: int, shape_hw: tuple[int, int]) -> np.ndarray:
    path = src / "motion" / f"{frame:06d}.txt"
    if not path.exists():
        return np.zeros(shape_hw, dtype=np.int32)
    instance = np.loadtxt(path, dtype=np.int32)
    if instance.shape != shape_hw:
        raise RuntimeError(f"Source motion mask has shape {instance.shape}, expected {shape_hw}: {path}")
    return instance


def clear_label_ids(instance: np.ndarray, items: list[dict]) -> None:
    for item in items:
        instance[instance == int(item["object_id"])] = 0


def box_mask(shape: tuple[int, int], box: np.ndarray) -> np.ndarray:
    h, w = shape
    x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)
    mask = np.zeros((h, w), dtype=np.uint8)
    if x2 > x1 and y2 > y1:
        mask[y1 : y2 + 1, x1 : x2 + 1] = 1
    return mask


def dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask.astype(np.uint8)
    radius = int(pixels)
    kernel_size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)


def render_filtered_image(image_bgr: np.ndarray, instance: np.ndarray, fill_mode: str) -> np.ndarray:
    filtered = image_bgr.copy()
    dynamic = instance > 0
    if not np.any(dynamic):
        return filtered
    if fill_mode == "gray":
        filtered[dynamic] = 127
    else:
        filtered[dynamic] = 0
    return filtered


def overwrite_image(path: Path, image_bgr: np.ndarray) -> None:
    if path.exists() or path.is_symlink():
        path.unlink()
    ok = cv2.imwrite(str(path), image_bgr)
    if not ok:
        raise RuntimeError(f"Failed to write filtered image: {path}")


class BoxPromptSegmenter:
    def __init__(self, args: argparse.Namespace, device: str):
        self.mode = args.segmenter
        self.device = device
        if self.mode == "box":
            self.predictor = None
            return
        if self.mode == "sam1":
            from segment_anything import SamPredictor, sam_model_registry

            checkpoint = args.checkpoint or str(SAM_CHECKPOINT)
            sam = sam_model_registry[args.model_type](checkpoint=checkpoint)
            sam.to(device=device)
            self.predictor = SamPredictor(sam)
            return
        from ultralytics import SAM

        checkpoint = args.checkpoint or str(SAM3_CHECKPOINT)
        self.predictor = SAM(checkpoint)

    def set_image(self, image_bgr: np.ndarray) -> None:
        if self.mode == "sam1":
            self.predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    def predict(self, image_bgr: np.ndarray, box: np.ndarray, shape_hw: tuple[int, int]) -> tuple[np.ndarray, float]:
        if self.mode == "box":
            return box_mask(shape_hw, box), 0.0
        if self.mode == "sam1":
            masks, scores, _ = self.predictor.predict(box=box, multimask_output=True)
            idx = int(np.argmax(scores))
            return masks[idx].astype(np.uint8), float(scores[idx])

        result = self.predictor.predict(
            image_bgr,
            bboxes=[box.astype(float).tolist()],
            conf=0.001,
            iou=0.90,
            imgsz=1036,
            device=0 if self.device.startswith("cuda") else "cpu",
            verbose=False,
        )[0]
        if result.masks is None or len(result.masks.data) == 0:
            return box_mask(shape_hw, box), 0.0
        masks = result.masks.data.detach().cpu().numpy().astype(np.uint8)
        scores = (
            result.boxes.conf.detach().cpu().numpy()
            if result.boxes is not None
            else np.ones((len(masks),), dtype=np.float32)
        )
        idx = int(np.argmax(scores))
        return masks[idx].astype(np.uint8), float(scores[idx])


def main() -> None:
    args = parse_args()
    seq_root = Path(args.seq_root)
    src = seq_root / args.src_name
    dst = seq_root / args.dst_name
    materialize(src, dst, args.overwrite)

    frames = sorted((dst / "image_0").glob("*.png"))
    n = len(frames) if args.max_frames < 0 else min(args.max_frames, len(frames))
    label_path = seq_root / "raw" / "training" / "label_02" / f"{seq_root.name}.txt"
    selected_classes = select_classes(args)
    preserved_classes = source_classes(args)
    labels = parse_labels(label_path, set(selected_classes), n)
    preserved_labels = parse_labels(label_path, set(preserved_classes), n) if preserved_classes else {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmenter = BoxPromptSegmenter(args, device)
    print(
        f"[sam-mask] frames={n}, device={device}, segmenter={args.segmenter}, "
        f"class_policy={args.class_policy}, classes={selected_classes}, "
        f"preserve_source_classes={preserved_classes}, "
        f"src={src.name}, dst={dst.name}, dilate={args.dilate_pixels}px"
    )

    stats = []
    for frame in range(n):
        image_bgr = cv2.imread(str(dst / "image_0" / f"{frame:06d}.png"), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Could not read image {frame:06d}")
        h, w = image_bgr.shape[:2]
        if args.base_source_motion:
            instance = load_source_motion(src, frame, (h, w))
        else:
            instance = draw_source_labels(preserved_labels, frame, (h, w)) if preserved_labels else np.zeros((h, w), dtype=np.int32)
        items = sorted(labels.get(frame, []), key=lambda item: item["area"], reverse=True)
        if args.base_source_motion and args.clear_selected_source_labels:
            clear_label_ids(instance, items)
        if items:
            segmenter.set_image(image_bgr)
        kept = 0
        fallback_count = 0
        for item in items:
            if item["area"] < args.min_box_area:
                continue
            try:
                mask, score = segmenter.predict(image_bgr, item["box"], (h, w))
            except Exception:
                mask = box_mask((h, w), item["box"])
                score = 0.0
            if np.count_nonzero(mask) == 0:
                mask = box_mask((h, w), item["box"])
                score = 0.0
            if score <= 0.0:
                fallback_count += 1
            mask = dilate_mask(mask, args.dilate_pixels)
            object_id = int(item["object_id"])
            free = (instance == 0) & (mask > 0)
            instance[free] = object_id
            kept += 1
        np.savetxt(dst / "motion" / f"{frame:06d}.txt", instance, fmt="%d")
        if args.render_filtered_image0:
            filtered_bgr = render_filtered_image(image_bgr, instance, args.filtered_fill)
            overwrite_image(dst / "image_0" / f"{frame:06d}.png", filtered_bgr)
        stats.append({
            "frame": frame,
            "objects": kept,
            "preserved_objects": len(preserved_labels.get(frame, [])),
            "fallback_objects": fallback_count,
            "mask_pixels": int(np.count_nonzero(instance)),
        })
        if (frame + 1) % 25 == 0 or frame + 1 == n:
            print(f"[sam-mask] {frame + 1}/{n}")
    (dst / "sam_motion_stats.json").write_text(
        __import__("json").dumps(
            {
                "frames": n,
                "segmenter": args.segmenter,
                "class_policy": args.class_policy,
                "classes": selected_classes,
                "preserve_source_classes": preserved_classes,
                "dilate_pixels": int(args.dilate_pixels),
                "render_filtered_image0": bool(args.render_filtered_image0),
                "filtered_fill": args.filtered_fill,
                "mean_objects": float(np.mean([s["objects"] for s in stats])) if stats else 0.0,
                "mean_preserved_objects": float(np.mean([s["preserved_objects"] for s in stats])) if stats else 0.0,
                "mean_fallback_objects": float(np.mean([s["fallback_objects"] for s in stats])) if stats else 0.0,
                "mean_mask_ratio": float(np.mean([s["mask_pixels"] / (h * w) for s in stats])) if stats else 0.0,
                "records": stats,
            },
            indent=2,
        )
    )
    print(f"[done] {dst}")


if __name__ == "__main__":
    main()
