#!/usr/bin/env python3
"""Summarize nested side-channel-isolated backend repeat runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from parse_map_admission_events import parse_stdout_events


def parse_key_values(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    values: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def sha16(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def read_eval(run_dir: Path) -> Dict[str, object]:
    path = run_dir / "eval_unified_all.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    by_alignment = {item.get("alignment"): item for item in data.get("results", [])}
    se3 = by_alignment.get("se3", {})
    sim3 = by_alignment.get("sim3", {})
    return {
        "matched_poses": se3.get("matched_poses", ""),
        "coverage": se3.get("trajectory_coverage", ""),
        "ate_se3_rmse_m": se3.get("ate_rmse_m", ""),
        "ate_sim3_rmse_m": sim3.get("ate_rmse_m", ""),
        "sim3_scale": sim3.get("alignment_scale", ""),
        "rpet_rmse_m": se3.get("rpet_rmse_m", ""),
        "rper_rmse_deg": se3.get("rper_rmse_deg", ""),
        "estimated_poses": se3.get("estimated_poses", ""),
    }


def read_last_observability(run_dir: Path) -> Dict[str, object]:
    path = run_dir / "observability_frame_stats.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    last = rows[-1]
    return {
        "final_keyframes": last.get("num_keyframes", ""),
        "final_mappoints": last.get("num_mappoints", ""),
        "estimated_accum_path_m": last.get("estimated_accum_path_m", ""),
        "keyframe_created_count": sum(1 for row in rows if row.get("is_keyframe_created") == "1"),
    }


def protocol_valid(run_dir: Path) -> object:
    path = run_dir / "d2ma_protocol_validation.json"
    if not path.exists():
        return ""
    data = json.loads(path.read_text(encoding="utf-8"))
    return int(bool(data.get("valid")))


def discover_run_dirs(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("run_manifest.txt")):
        yield path.parent


def case_repeat_for(run_root: Path, run_dir: Path) -> Dict[str, str]:
    rel = run_dir.relative_to(run_root)
    parts = rel.parts
    if len(parts) >= 2 and parts[-1].startswith("r"):
        return {"case": parts[-2], "repeat": parts[-1]}
    return {"case": parts[-1], "repeat": ""}


def collect_run(run_root: Path, run_dir: Path) -> Dict[str, object]:
    manifest = parse_key_values(run_dir / "run_manifest.txt")
    protocol = parse_key_values(run_dir / "d2ma_protocol_manifest.txt")
    events, _ = parse_stdout_events(run_dir)
    row: Dict[str, object] = {
        **case_repeat_for(run_root, run_dir),
        "dataset_id": manifest.get("dataset_id", ""),
        "method": protocol.get("d2ma_method", ""),
        "profile": manifest.get("profile", ""),
        "protocol_valid": protocol_valid(run_dir),
        "camera_sha16": sha16(run_dir / "CameraTrajectory.txt"),
        "keyframe_timeline_sha16": sha16(run_dir / "KeyFrameTimeline.csv"),
        "run_dir": str(run_dir),
    }
    row.update(read_eval(run_dir))
    row.update(read_last_observability(run_dir))
    for key in [
        "ckf_direct_vetoed_candidates",
        "ckf_accepted_depth_candidates",
        "ckf_boundary_skipped_new_candidates",
        "ckf_boundary_existing_supported_candidates",
        "lm_skipped_instance_pairs",
        "lm_skipped_boundary_pairs",
    ]:
        row[key] = events.get(key, 0)
    return row


def numeric(values: Sequence[object]) -> List[float]:
    out: List[float] = []
    for value in values:
        if value in ("", None):
            continue
        out.append(float(value))
    return out


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    metrics = [
        "matched_poses",
        "coverage",
        "ate_se3_rmse_m",
        "ate_sim3_rmse_m",
        "sim3_scale",
        "rpet_rmse_m",
        "rper_rmse_deg",
        "estimated_poses",
        "final_keyframes",
        "final_mappoints",
        "estimated_accum_path_m",
        "ckf_direct_vetoed_candidates",
        "ckf_boundary_skipped_new_candidates",
        "lm_skipped_instance_pairs",
        "lm_skipped_boundary_pairs",
    ]
    cases = sorted({str(row["case"]) for row in rows})
    summary_rows: List[Dict[str, object]] = []
    for case in cases:
        case_rows = [row for row in rows if row["case"] == case]
        summary: Dict[str, object] = {
            "case": case,
            "method": case_rows[0].get("method", ""),
            "dataset_id": case_rows[0].get("dataset_id", ""),
            "n": len(case_rows),
            "protocol_valid_all": int(all(row.get("protocol_valid") == 1 for row in case_rows)),
        }
        for metric in metrics:
            vals = numeric([row.get(metric, "") for row in case_rows])
            if not vals:
                continue
            summary[f"{metric}_mean"] = statistics.mean(vals)
            summary[f"{metric}_std"] = statistics.pstdev(vals)
            summary[f"{metric}_min"] = min(vals)
            summary[f"{metric}_max"] = max(vals)
        hashes = sorted({str(row.get("camera_sha16", "")) for row in case_rows})
        summary["camera_sha16_unique"] = ";".join(hashes)
        summary_rows.append(summary)
    return summary_rows


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--raw-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    rows = [collect_run(args.run_root, run_dir) for run_dir in discover_run_dirs(args.run_root)]
    summary_rows = summarize(rows)
    write_csv(args.raw_out, rows)
    write_csv(args.summary_out, summary_rows)

    print(f"raw={args.raw_out}")
    print(f"summary={args.summary_out}")
    print("| case | n | valid | ATE-SE3 mean±std | ATE-Sim3 mean±std | scale mean±std | final MPs mean±std |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        def pair(metric: str) -> str:
            return f"{row[f'{metric}_mean']:.6f}±{row[f'{metric}_std']:.6f}"

        print(
            f"| {row['case']} | {row['n']} | {row['protocol_valid_all']} | "
            f"{pair('ate_se3_rmse_m')} | {pair('ate_sim3_rmse_m')} | "
            f"{pair('sim3_scale')} | {pair('final_mappoints')} |"
        )


if __name__ == "__main__":
    main()
