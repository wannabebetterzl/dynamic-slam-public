#!/usr/bin/env python3
"""Export a DynoSAM-style input bundle from a backend packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dynamic_object_frontend.dynosam_adapter import export_dynosam_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a backend packet into a DynoSAM-style bundle.")
    parser.add_argument("--packet", required=True, help="Backend packet directory.")
    parser.add_argument("--out", required=True, help="Output DynoSAM-style bundle directory.")
    parser.add_argument("--benchmark-summary", default="", help="Benchmark summary JSON used to recover raw RGB paths.")
    parser.add_argument("--frame-stats", default="", help="Frame stats JSON used to recover source RGB paths.")
    parser.add_argument("--packet-analysis", default="", help="Packet analysis JSON used for observation quality flags.")
    parser.add_argument("--materialize-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--low-quality-weight", type=float, default=0.35)
    parser.add_argument(
        "--optical-flow-method",
        choices=[
            "none",
            "farneback",
            "raft",
            "raft_small",
            "raft_large",
            "gmflow",
            "unimatch_gmflow",
            "unimatch_gmflow_scale2_mixdata",
        ],
        default="none",
        help="Optionally materialize forward dense optical flow for DynoSAM. 'raft' aliases raft_large; 'gmflow' aliases UniMatch GMFlow scale2 mixdata.",
    )
    args = parser.parse_args()

    manifest = export_dynosam_bundle(
        packet_root=Path(args.packet),
        output_root=Path(args.out),
        benchmark_summary_path=Path(args.benchmark_summary) if args.benchmark_summary else None,
        frame_stats_path=Path(args.frame_stats) if args.frame_stats else None,
        packet_analysis_path=Path(args.packet_analysis) if args.packet_analysis else None,
        materialize_mode=args.materialize_mode,
        low_quality_weight=args.low_quality_weight,
        optical_flow_method=args.optical_flow_method,
    )
    print(json.dumps(manifest["validation"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
