#!/usr/bin/env python3
"""Parse D2MA map-admission stdout events and observability logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

EVENT_RE = re.compile(r"(\w+)=([^\s]+)")


def parse_case(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("case must be NAME=/path/to/run_dir")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("case name cannot be empty")
    return name, Path(path)


def read_eval(run_dir: Path) -> Dict[str, object]:
    path = run_dir / "eval_unified_all.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    by_alignment = {item.get("alignment"): item for item in data.get("results", [])}
    se3 = by_alignment.get("se3", {})
    sim3 = by_alignment.get("sim3", {})
    return {
        "matched": se3.get("matched_poses", ""),
        "coverage": se3.get("trajectory_coverage", ""),
        "ate_se3": se3.get("ate_rmse_m", ""),
        "ate_sim3": sim3.get("ate_rmse_m", ""),
        "scale": sim3.get("alignment_scale", ""),
        "rpet": se3.get("rpet_rmse_m", ""),
        "rper": se3.get("rper_rmse_deg", ""),
    }


def read_observability(run_dir: Path) -> Tuple[List[Dict[str, str]], Dict[int, Dict[str, str]]]:
    path = run_dir / "observability_frame_stats.csv"
    if not path.exists():
        return [], {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_frame: Dict[int, Dict[str, str]] = {}
    for row in rows:
        frame_id = row.get("frame_id")
        if frame_id in (None, ""):
            continue
        by_frame[int(float(frame_id))] = row
    return rows, by_frame


def add_event(
    totals: Dict[str, int],
    per_frame: Dict[int, Dict[str, int]],
    values: Dict[str, str],
    key: str,
    source_key: str,
) -> None:
    if source_key not in values:
        return
    try:
        amount = int(float(values[source_key]))
    except ValueError:
        return
    totals[key] += amount
    frame = values.get("frame")
    if frame is not None:
        per_frame[int(float(frame))][key] += amount


def parse_stdout_events(run_dir: Path) -> Tuple[Dict[str, int], Dict[int, Dict[str, int]]]:
    totals: Dict[str, int] = defaultdict(int)
    per_frame: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    path = run_dir / "stdout.log"
    if not path.exists():
        return totals, per_frame

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("[STSLAM_"):
            continue
        tag = line.split("]", 1)[0].strip("[")
        values = dict(EVENT_RE.findall(line))
        stage = values.get("stage", "unknown")

        if tag == "STSLAM_DYNAMIC_MAP_ADMISSION_VETO" and stage == "create_new_keyframe":
            add_event(totals, per_frame, values, "ckf_direct_vetoed_candidates", "vetoed_candidates")
            add_event(totals, per_frame, values, "ckf_accepted_depth_candidates", "accepted_depth_candidates")
        elif tag == "STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_VETO" and stage == "create_new_keyframe":
            add_event(totals, per_frame, values, "ckf_boundary_skipped_new_candidates", "skipped_new_candidates")
            add_event(totals, per_frame, values, "ckf_boundary_existing_supported_candidates", "existing_supported_candidates")
        elif tag == "STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_CONTROL" and stage == "create_new_keyframe":
            add_event(totals, per_frame, values, "ckf_boundary_budget", "boundary_budget")
            add_event(
                totals,
                per_frame,
                values,
                "ckf_control_skipped_nonboundary_new_candidates",
                "skipped_nonboundary_new_candidates",
            )
        elif tag == "STSLAM_DYNAMIC_MAP_ADMISSION_VETO" and stage == "create_new_map_points":
            add_event(totals, per_frame, values, "lm_skipped_instance_pairs", "skipped_instance_pairs")
            add_event(totals, per_frame, values, "lm_kept_static_pairs_after_instance", "kept_static_pairs")
        elif tag == "STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_VETO" and stage == "create_new_map_points":
            add_event(totals, per_frame, values, "lm_skipped_boundary_pairs", "skipped_boundary_pairs")
            add_event(totals, per_frame, values, "lm_kept_pairs_after_boundary", "kept_pairs")
        elif tag == "STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_CONTROL" and stage == "create_new_map_points":
            add_event(totals, per_frame, values, "lm_boundary_budget", "boundary_budget")
            add_event(totals, per_frame, values, "lm_control_skipped_nonboundary_pairs", "skipped_nonboundary_pairs")
            add_event(totals, per_frame, values, "lm_kept_pairs_after_control", "kept_pairs")

    return totals, per_frame


def collect_case(name: str, run_dir: Path) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    totals, per_frame_events = parse_stdout_events(run_dir)
    obs_rows, obs_by_frame = read_observability(run_dir)
    eval_metrics = read_eval(run_dir)
    last_obs = obs_rows[-1] if obs_rows else {}
    keyframe_count = sum(1 for row in obs_rows if row.get("is_keyframe_created") == "1")

    summary: Dict[str, object] = {
        "case": name,
        "matched": eval_metrics.get("matched", ""),
        "coverage": eval_metrics.get("coverage", ""),
        "ate_se3": eval_metrics.get("ate_se3", ""),
        "ate_sim3": eval_metrics.get("ate_sim3", ""),
        "scale": eval_metrics.get("scale", ""),
        "rpet": eval_metrics.get("rpet", ""),
        "rper": eval_metrics.get("rper", ""),
        "final_keyframes": last_obs.get("num_keyframes", ""),
        "final_mappoints": last_obs.get("num_mappoints", ""),
        "estimated_accum_path_m": last_obs.get("estimated_accum_path_m", ""),
        "keyframe_created_count": keyframe_count,
        "run_dir": str(run_dir),
    }
    event_keys = [
        "ckf_direct_vetoed_candidates",
        "ckf_accepted_depth_candidates",
        "ckf_boundary_skipped_new_candidates",
        "ckf_boundary_existing_supported_candidates",
        "ckf_boundary_budget",
        "ckf_control_skipped_nonboundary_new_candidates",
        "lm_skipped_instance_pairs",
        "lm_skipped_boundary_pairs",
        "lm_boundary_budget",
        "lm_control_skipped_nonboundary_pairs",
    ]
    for key in event_keys:
        summary[key] = totals.get(key, 0)

    frame_rows: List[Dict[str, object]] = []
    for frame_id, values in sorted(per_frame_events.items()):
        obs = obs_by_frame.get(frame_id, {})
        row: Dict[str, object] = {
            "case": name,
            "frame_id": frame_id,
            "timestamp": obs.get("timestamp", ""),
            "is_keyframe_created": obs.get("is_keyframe_created", ""),
            "num_keyframes": obs.get("num_keyframes", ""),
            "num_mappoints": obs.get("num_mappoints", ""),
            "local_map_matches_before_pose": obs.get("local_map_matches_before_pose", ""),
            "inlier_map_matches_after_pose": obs.get("inlier_map_matches_after_pose", ""),
            "estimated_accum_path_m": obs.get("estimated_accum_path_m", ""),
            "estimated_frame_step_m": obs.get("estimated_frame_step_m", ""),
            "mask_ratio": obs.get("mask_ratio", ""),
        }
        row.update(values)
        frame_rows.append(row)

    return summary, frame_rows


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=parse_case, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--per-frame-out", type=Path, required=True)
    args = parser.parse_args()

    summaries: List[Dict[str, object]] = []
    frame_rows: List[Dict[str, object]] = []
    for name, run_dir in args.case:
        summary, rows = collect_case(name, run_dir)
        summaries.append(summary)
        frame_rows.extend(rows)

    summary_fields = list(summaries[0].keys()) if summaries else []
    frame_prefix = [
        "case",
        "frame_id",
        "timestamp",
        "is_keyframe_created",
        "num_keyframes",
        "num_mappoints",
        "local_map_matches_before_pose",
        "inlier_map_matches_after_pose",
        "estimated_accum_path_m",
        "estimated_frame_step_m",
        "mask_ratio",
    ]
    frame_event_fields = sorted({key for row in frame_rows for key in row.keys()} - set(frame_prefix))

    write_csv(args.summary_out, summaries, summary_fields)
    write_csv(args.per_frame_out, frame_rows, frame_prefix + frame_event_fields)

    print(f"summary={args.summary_out}")
    print(f"per_frame={args.per_frame_out}")
    print("| case | ATE-SE3 | scale | final KFs | final MPs | CKF direct | CKF boundary/control | LM boundary/control |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summaries:
        def fmt(value: object) -> str:
            try:
                return f"{float(value):.6f}"
            except (TypeError, ValueError):
                return str(value)
        print(
            f"| {row['case']} | {fmt(row.get('ate_se3'))} | {fmt(row.get('scale'))} | "
            f"{row.get('final_keyframes')} | {row.get('final_mappoints')} | "
            f"{row.get('ckf_direct_vetoed_candidates')} | "
            f"{row.get('ckf_boundary_skipped_new_candidates')}/{row.get('ckf_control_skipped_nonboundary_new_candidates')} | "
            f"{row.get('lm_skipped_boundary_pairs')}/{row.get('lm_control_skipped_nonboundary_pairs')} |"
        )


if __name__ == "__main__":
    main()
