#!/usr/bin/env python3
# coding=utf-8

import argparse
import json
import os

import cv2

from rflysim_slam_nav.world_sam_pipeline import WorldSamFilterPipeline


def main():
    parser = argparse.ArgumentParser(description="Run a local smoke test for the YOLO-World + SAM SLAM filter.")
    parser.add_argument("--config", required=True, help="Pipeline config JSON.")
    parser.add_argument("--image", required=True, help="Input RGB image.")
    parser.add_argument("--output-dir", required=True, help="Directory to save debug outputs.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Failed to read image: {args.image}")

    pipeline = WorldSamFilterPipeline(args.config)
    result = pipeline.process(image, depth_mm=None)

    cv2.imwrite(os.path.join(args.output_dir, "filtered_rgb.jpg"), result["filtered_rgb"])
    cv2.imwrite(os.path.join(args.output_dir, "overlay.jpg"), result["overlay"])
    cv2.imwrite(os.path.join(args.output_dir, "mask.png"), result["mask"] * 255)
    with open(os.path.join(args.output_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(result["stats"], f, indent=2, ensure_ascii=False)

    print(json.dumps(result["stats"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
