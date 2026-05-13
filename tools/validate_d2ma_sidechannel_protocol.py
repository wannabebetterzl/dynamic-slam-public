#!/usr/bin/env python3
"""Validate that a backend run used the D2MA side-channel isolation protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


BASE_REQUIRED = {
    "ORB_SLAM3_MASK_MODE": "off",
    "STSLAM_PANOPTIC_SIDE_CHANNEL_ONLY": "1",
    "STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES": "0",
    "STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURE_STAGES": "none",
    "STSLAM_RGBD_DYNAMIC_FRONTEND_SPLIT": "0",
    "STSLAM_SEMANTIC_GEOMETRIC_VERIFICATION": "0",
    "STSLAM_SEMANTIC_CANDIDATE_GEOMETRY_GATE": "0",
    "STSLAM_SEMANTIC_CANDIDATE_SPARSE_FLOW_GATE": "0",
    "STSLAM_SEMANTIC_CONSERVATIVE_DYNAMIC_DELETE": "0",
    "STSLAM_SEMANTIC_STRICT_STATIC_KEEP": "0",
    "STSLAM_GEOMETRIC_DYNAMIC_REJECTION": "0",
    "STSLAM_DYNAMIC_DEPTH_INVALIDATION": "0",
    "STSLAM_DYNAMIC_MAP_ADMISSION_VETO": "0",
    "STSLAM_DYNAMIC_MAP_ADMISSION_VETO_STEREO_INITIALIZATION": "0",
    "STSLAM_DYNAMIC_MAP_ADMISSION_VETO_NEED_NEW_KEYFRAME": "0",
}


METHOD_REQUIRED = {
    "raw": {
        "STSLAM_DYNAMIC_MAP_ADMISSION_VETO_CREATE_NEW_KEYFRAME": "0",
        "STSLAM_DYNAMIC_MAP_ADMISSION_VETO_LOCAL_MAPPING_CREATE_NEW_MAPPOINTS": "0",
        "STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_VETO": "0",
        "STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_VETO_CREATE_NEW_KEYFRAME": "0",
        "STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_VETO_LOCAL_MAPPING_CREATE_NEW_MAPPOINTS": "0",
        "STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_SAME_COUNT_CONTROL": "0",
    },
    "d2ma_min": {
        "STSLAM_DYNAMIC_MAP_ADMISSION_VETO_CREATE_NEW_KEYFRAME": "1",
        "STSLAM_DYNAMIC_MAP_ADMISSION_VETO_LOCAL_MAPPING_CREATE_NEW_MAPPOINTS": "1",
        "STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_VETO": "0",
        "STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_SAME_COUNT_CONTROL": "0",
    },
    "d2ma_b_r5": {
        "STSLAM_DYNAMIC_MAP_ADMISSION_VETO_CREATE_NEW_KEYFRAME": "1",
        "STSLAM_DYNAMIC_MAP_ADMISSION_VETO_LOCAL_MAPPING_CREATE_NEW_MAPPOINTS": "1",
        "STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_VETO": "1",
        "STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_VETO_CREATE_NEW_KEYFRAME": "1",
        "STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_VETO_LOCAL_MAPPING_CREATE_NEW_MAPPOINTS": "1",
        "STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_RADIUS_PX": "5",
        "STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_SAME_COUNT_CONTROL": "0",
    },
    "samecount_nonboundary_r5": {
        "STSLAM_DYNAMIC_MAP_ADMISSION_VETO_CREATE_NEW_KEYFRAME": "1",
        "STSLAM_DYNAMIC_MAP_ADMISSION_VETO_LOCAL_MAPPING_CREATE_NEW_MAPPOINTS": "1",
        "STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_VETO": "0",
        "STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_SAME_COUNT_CONTROL": "1",
        "STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_SAME_COUNT_CONTROL_CREATE_NEW_KEYFRAME": "1",
        "STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_SAME_COUNT_CONTROL_LOCAL_MAPPING_CREATE_NEW_MAPPOINTS": "1",
        "STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_RADIUS_PX": "5",
    },
}


def parse_key_values(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def validate(manifest: Dict[str, str], method: str) -> Dict[str, object]:
    if method not in METHOD_REQUIRED:
        raise SystemExit(f"Unknown method: {method}")

    required = dict(BASE_REQUIRED)
    required.update(METHOD_REQUIRED[method])

    errors: List[Dict[str, str]] = []
    for key, expected in required.items():
        actual = manifest.get(key)
        if actual != expected:
            errors.append({"key": key, "expected": expected, "actual": "" if actual is None else actual})

    return {
        "valid": not errors,
        "method": method,
        "errors": errors,
        "checked_keys": sorted(required),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--method", choices=sorted(METHOD_REQUIRED), required=True)
    parser.add_argument("--write-json", type=Path)
    args = parser.parse_args()

    manifest_path = args.run_dir / "run_manifest.txt"
    if not manifest_path.exists():
        raise SystemExit(f"Missing run manifest: {manifest_path}")

    payload = validate(parse_key_values(manifest_path), args.method)
    payload["run_dir"] = str(args.run_dir)
    payload["manifest"] = str(manifest_path)

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.write_json:
        args.write_json.write_text(text + "\n", encoding="utf-8")

    if not payload["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
