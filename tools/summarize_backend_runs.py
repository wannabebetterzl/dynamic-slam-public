#!/usr/bin/env python3
"""Summarize ORB-SLAM backend run directories into a CSV table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def parse_manifest(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def file_sha16(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_eval(run_dir: Path) -> Optional[Dict[str, object]]:
    path = run_dir / "eval_unified_all.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    by_alignment = {item["alignment"]: item for item in data.get("results", [])}
    se3 = by_alignment.get("se3")
    sim3 = by_alignment.get("sim3")
    if not se3 or not sim3:
        return None
    return {
        "matched_poses": se3.get("matched_poses"),
        "coverage": se3.get("trajectory_coverage"),
        "ate_se3_rmse_m": se3.get("ate_rmse_m"),
        "ate_sim3_rmse_m": sim3.get("ate_rmse_m"),
        "sim3_scale": sim3.get("alignment_scale"),
        "rpet_rmse_m": se3.get("rpet_rmse_m"),
        "rper_rmse_deg": se3.get("rper_rmse_deg"),
        "estimated_poses": se3.get("estimated_poses"),
    }


def summarize_run(run_dir: Path) -> Dict[str, object]:
    manifest = parse_manifest(run_dir / "run_manifest.txt")
    protocol = parse_manifest(run_dir / "d2ma_protocol_manifest.txt")
    row: Dict[str, object] = {
        "case": run_dir.name,
        "run_dir": str(run_dir),
        "dataset_id": manifest.get("dataset_id", ""),
        "profile": manifest.get("profile", ""),
        "method": protocol.get("d2ma_method", ""),
        "protocol_valid": "",
        "camera_sha16": file_sha16(run_dir / "CameraTrajectory.txt"),
        "keyframe_timeline_sha16": file_sha16(run_dir / "KeyFrameTimeline.csv"),
    }
    validation_path = run_dir / "d2ma_protocol_validation.json"
    if validation_path.exists():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        row["protocol_valid"] = int(bool(validation.get("valid")))
    metrics = load_eval(run_dir)
    if metrics:
        row.update(metrics)
    return row


def iter_run_dirs(root: Path) -> Iterable[Path]:
    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / "run_manifest.txt").exists():
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--csv-out", type=Path, required=True)
    args = parser.parse_args()

    rows: List[Dict[str, object]] = [summarize_run(path) for path in iter_run_dirs(args.run_root)]
    fieldnames = [
        "case",
        "dataset_id",
        "method",
        "profile",
        "protocol_valid",
        "matched_poses",
        "coverage",
        "ate_se3_rmse_m",
        "ate_sim3_rmse_m",
        "sim3_scale",
        "rpet_rmse_m",
        "rper_rmse_deg",
        "estimated_poses",
        "camera_sha16",
        "keyframe_timeline_sha16",
        "run_dir",
    ]
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"wrote {args.csv_out}")
    print("| case | method | matched | ATE-SE3 | ATE-Sim3 | scale | protocol |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row.get('case','')} | {row.get('method','')} | "
            f"{row.get('matched_poses','')} | {row.get('ate_se3_rmse_m','')} | "
            f"{row.get('ate_sim3_rmse_m','')} | {row.get('sim3_scale','')} | "
            f"{row.get('protocol_valid','')} |"
        )


if __name__ == "__main__":
    main()
