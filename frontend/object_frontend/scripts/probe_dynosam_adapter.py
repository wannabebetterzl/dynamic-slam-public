#!/usr/bin/env python3
"""Probe the in-memory DynoSAM adapter interface without materializing a bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dynamic_object_frontend import build_dynosam_adapter_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect direct DynoSAM adapter frame packets.")
    parser.add_argument("--packet", required=True, help="Backend packet directory.")
    parser.add_argument("--benchmark-summary", default="", help="Benchmark summary JSON used to recover raw RGB paths.")
    parser.add_argument("--frame-stats", default="", help="Frame stats JSON used to recover source RGB paths.")
    parser.add_argument("--packet-analysis", default="", help="Packet analysis JSON used for observation quality flags.")
    parser.add_argument("--max-frames", type=int, default=3, help="Maximum number of frames to inspect.")
    args = parser.parse_args()

    adapter_bundle = build_dynosam_adapter_bundle(
        packet_root=Path(args.packet),
        benchmark_summary_path=Path(args.benchmark_summary) if args.benchmark_summary else None,
        frame_stats_path=Path(args.frame_stats) if args.frame_stats else None,
        packet_analysis_path=Path(args.packet_analysis) if args.packet_analysis else None,
    )
    print(json.dumps(adapter_bundle.validation, indent=2, ensure_ascii=False))

    for index, packet in enumerate(adapter_bundle.iter_direct_frame_packets()):
        if index >= max(args.max_frames, 0):
            break
        summary = {
            "frame_id": packet.frame_id,
            "timestamp": packet.timestamp,
            "raw_rgb_shape": list(packet.raw_rgb.shape),
            "filtered_rgb_shape": list(packet.static_filtered_rgb.shape),
            "filtered_depth_shape": list(packet.static_filtered_depth_metric.shape),
            "raw_depth_shape": list(packet.raw_depth_metric.shape),
            "instance_mask_shape": list(packet.instance_mask.shape),
            "num_objects": len(packet.observations),
            "instance_pixels": int((packet.instance_mask > 0).sum()),
        }
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
