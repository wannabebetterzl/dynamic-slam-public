#!/usr/bin/env python3
"""Analyze object tracks from exported object observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dynamic_object_frontend import analyze_object_tracks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--static-translation-m", type=float, default=0.03)
    parser.add_argument("--moving-translation-m", type=float, default=0.10)
    parser.add_argument("--static-speed-mps", type=float, default=0.20)
    parser.add_argument("--moving-speed-mps", type=float, default=0.75)
    parser.add_argument("--min-depth-observations", type=int, default=2)
    parser.add_argument("--min-motion-steps", type=int, default=2)
    args = parser.parse_args()

    summaries = analyze_object_tracks(
        args.observations,
        static_translation_m=args.static_translation_m,
        moving_translation_m=args.moving_translation_m,
        static_speed_mps=args.static_speed_mps,
        moving_speed_mps=args.moving_speed_mps,
        min_depth_observations=args.min_depth_observations,
        min_motion_steps=args.min_motion_steps,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "observations": str(args.observations),
        "tracks": [summary.to_json_dict() for summary in summaries],
        "counts": {
            "total": len(summaries),
            "static": sum(1 for item in summaries if item.motion_state == "static"),
            "moving": sum(1 for item in summaries if item.motion_state == "moving"),
            "uncertain": sum(1 for item in summaries if item.motion_state == "uncertain"),
        },
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["counts"], indent=2))
    for item in summaries:
        print(
            f"object={item.object_id} semantic={item.semantic_label} "
            f"frames={item.frames} depth_obs={item.observations_with_depth} "
            f"state={item.motion_state} conf={item.confidence:.2f} "
            f"mean_step={item.mean_step_translation_m} mean_speed={item.mean_speed_mps}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

