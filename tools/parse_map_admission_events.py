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
        elif tag == "STSLAM_DYNAMIC_MAP_ADMISSION_DELAYED_BOUNDARY" and stage == "create_new_keyframe":
            add_event(totals, per_frame, values, "ckf_delayed_rejected_new_candidates", "delayed_rejected_new_candidates")
            add_event(totals, per_frame, values, "ckf_support_promoted_new_candidates", "support_promoted_new_candidates")
            add_event(totals, per_frame, values, "ckf_delayed_existing_supported_candidates", "existing_supported_candidates")
            add_event(totals, per_frame, values, "ckf_delayed_support_sum", "support_sum")
            add_event(totals, per_frame, values, "ckf_quality_rejected_new_candidates", "quality_rejected_new_candidates")
            add_event(totals, per_frame, values, "ckf_quality_raw_support_sum", "quality_raw_support_sum")
            add_event(totals, per_frame, values, "ckf_quality_found_support_sum", "quality_found_support_sum")
            add_event(totals, per_frame, values, "ckf_quality_frame_support_sum", "quality_frame_support_sum")
            add_event(totals, per_frame, values, "ckf_quality_raw_depth_support_sum", "quality_raw_depth_support_sum")
            add_event(totals, per_frame, values, "ckf_quality_reliable_support_sum", "quality_reliable_support_sum")
            add_event(totals, per_frame, values, "ckf_quality_residual_support_sum", "quality_residual_support_sum")
            add_event(totals, per_frame, values, "ckf_quality_depth_support_sum", "quality_depth_support_sum")
        elif tag == "STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_CONTROL" and stage == "create_new_keyframe":
            add_event(totals, per_frame, values, "ckf_boundary_budget", "boundary_budget")
            add_event(
                totals,
                per_frame,
                values,
                "ckf_control_skipped_nonboundary_new_candidates",
                "skipped_nonboundary_new_candidates",
            )
        elif tag == "STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_MATCHED_CONTROL" and stage == "create_new_keyframe":
            add_event(totals, per_frame, values, "ckf_matched_boundary_budget", "boundary_budget")
            add_event(
                totals,
                per_frame,
                values,
                "ckf_matched_skipped_nonboundary_new_candidates",
                "skipped_matched_nonboundary_new_candidates",
            )
            add_event(
                totals,
                per_frame,
                values,
                "ckf_matched_exact_skipped_new_candidates",
                "exact_skipped_new_candidates",
            )
            add_event(
                totals,
                per_frame,
                values,
                "ckf_matched_fallback_skipped_new_candidates",
                "fallback_skipped_new_candidates",
            )
        elif tag == "STSLAM_DYNAMIC_MAP_ADMISSION_VETO" and stage == "create_new_map_points":
            add_event(totals, per_frame, values, "lm_skipped_instance_pairs", "skipped_instance_pairs")
            add_event(totals, per_frame, values, "lm_kept_static_pairs_after_instance", "kept_static_pairs")
        elif tag == "STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_VETO" and stage == "create_new_map_points":
            add_event(totals, per_frame, values, "lm_skipped_boundary_pairs", "skipped_boundary_pairs")
            add_event(totals, per_frame, values, "lm_kept_pairs_after_boundary", "kept_pairs")
        elif tag == "STSLAM_DYNAMIC_MAP_ADMISSION_DELAYED_BOUNDARY" and stage == "create_new_map_points":
            add_event(totals, per_frame, values, "lm_delayed_rejected_boundary_pairs", "delayed_rejected_boundary_pairs")
            add_event(totals, per_frame, values, "lm_support_promoted_boundary_pairs", "support_promoted_boundary_pairs")
            add_event(totals, per_frame, values, "lm_delayed_support_sum", "support_sum")
            add_event(totals, per_frame, values, "lm_quality_rejected_boundary_pairs", "quality_rejected_boundary_pairs")
            add_event(totals, per_frame, values, "lm_quality_raw_support_sum", "quality_raw_support_sum")
            add_event(totals, per_frame, values, "lm_quality_found_support_sum", "quality_found_support_sum")
            add_event(totals, per_frame, values, "lm_quality_frame_support_sum", "quality_frame_support_sum")
            add_event(totals, per_frame, values, "lm_quality_raw_depth_support_sum", "quality_raw_depth_support_sum")
            add_event(totals, per_frame, values, "lm_quality_reliable_support_sum", "quality_reliable_support_sum")
            add_event(totals, per_frame, values, "lm_quality_residual_support_sum", "quality_residual_support_sum")
            add_event(totals, per_frame, values, "lm_quality_depth_support_sum", "quality_depth_support_sum")
            add_event(totals, per_frame, values, "lm_promoted_geom_enter", "promoted_geom_enter")
            add_event(totals, per_frame, values, "lm_promoted_geom_parallax", "promoted_geom_parallax")
            add_event(totals, per_frame, values, "lm_promoted_geom_triangulated", "promoted_geom_triangulated")
            add_event(totals, per_frame, values, "lm_promoted_geom_depth", "promoted_geom_depth")
            add_event(totals, per_frame, values, "lm_promoted_geom_reproj1", "promoted_geom_reproj1")
            add_event(totals, per_frame, values, "lm_promoted_geom_reproj2", "promoted_geom_reproj2")
            add_event(totals, per_frame, values, "lm_promoted_geom_scale", "promoted_geom_scale")
            add_event(totals, per_frame, values, "lm_promoted_geom_created", "promoted_geom_created")
            add_event(totals, per_frame, values, "lm_kept_pairs_after_delayed_boundary", "kept_pairs")
        elif tag == "STSLAM_DYNAMIC_MAP_ADMISSION_PROMOTED_GEOM" and stage == "create_new_map_points":
            add_event(totals, per_frame, values, "lm_promoted_geom_enter", "promoted_geom_enter")
            add_event(totals, per_frame, values, "lm_promoted_geom_parallax", "promoted_geom_parallax")
            add_event(totals, per_frame, values, "lm_promoted_geom_triangulated", "promoted_geom_triangulated")
            add_event(totals, per_frame, values, "lm_promoted_geom_depth", "promoted_geom_depth")
            add_event(totals, per_frame, values, "lm_promoted_geom_reproj1", "promoted_geom_reproj1")
            add_event(totals, per_frame, values, "lm_promoted_geom_reproj2", "promoted_geom_reproj2")
            add_event(totals, per_frame, values, "lm_promoted_geom_scale", "promoted_geom_scale")
            add_event(totals, per_frame, values, "lm_promoted_geom_created", "promoted_geom_created")
        elif tag == "STSLAM_DYNAMIC_MAP_ADMISSION_SCORE_BASED" and stage == "create_new_map_points":
            add_event(totals, per_frame, values, "lm_score_support_candidates", "support_candidates")
            add_event(totals, per_frame, values, "lm_score_support_accepted", "support_accepted")
            add_event(totals, per_frame, values, "lm_score_support_rejected", "support_rejected")
            add_event(totals, per_frame, values, "lm_score_geom_evaluated", "geom_evaluated")
            add_event(totals, per_frame, values, "lm_score_post_geom_rejected", "post_geom_rejected")
            add_event(totals, per_frame, values, "lm_score_created", "score_created")
        elif tag == "STSLAM_DYNAMIC_MAP_ADMISSION_STATE_AWARE" and stage == "create_new_map_points":
            add_event(totals, per_frame, values, "lm_state_candidates", "state_candidates")
            add_event(totals, per_frame, values, "lm_state_allowed", "state_allowed")
            add_event(totals, per_frame, values, "lm_state_rejected", "state_rejected")
            add_event(totals, per_frame, values, "lm_state_tracking_pressure", "tracking_pressure")
            add_event(totals, per_frame, values, "lm_state_keyframe_pressure", "keyframe_pressure")
            add_event(totals, per_frame, values, "lm_state_scale_pressure", "scale_pressure")
            add_event(totals, per_frame, values, "lm_state_lba_pressure", "lba_pressure")
        elif tag == "STSLAM_DYNAMIC_MAP_ADMISSION_V5_CANDIDATE":
            add_event(totals, per_frame, values, "lm_v5_support_candidates", "support_candidate")
            add_event(totals, per_frame, values, "lm_v5_support_accepted", "support_accepted")
            add_event(totals, per_frame, values, "lm_v5_reject_support", "reject_support")
            add_event(totals, per_frame, values, "lm_v5_geom_candidates", "geom_candidate")
            add_event(totals, per_frame, values, "lm_v5_reject_parallax", "reject_parallax")
            add_event(totals, per_frame, values, "lm_v5_reject_triangulate", "reject_triangulate")
            add_event(totals, per_frame, values, "lm_v5_reject_depth", "reject_depth")
            add_event(totals, per_frame, values, "lm_v5_reject_reproj1", "reject_reproj1")
            add_event(totals, per_frame, values, "lm_v5_reject_reproj2", "reject_reproj2")
            add_event(totals, per_frame, values, "lm_v5_reject_scale", "reject_scale")
            add_event(totals, per_frame, values, "lm_v5_reject_score", "reject_score")
            add_event(totals, per_frame, values, "lm_v5_created", "created")
        elif tag == "STSLAM_DYNAMIC_MAP_ADMISSION_V5_LIFECYCLE":
            add_event(totals, per_frame, values, "lm_v5_lifecycle_score_recent", "score_recent")
            add_event(totals, per_frame, values, "lm_v5_lifecycle_score_prebad", "score_prebad")
            add_event(totals, per_frame, values, "lm_v5_lifecycle_score_culled_found_ratio", "score_culled_found_ratio")
            add_event(totals, per_frame, values, "lm_v5_lifecycle_score_culled_low_obs", "score_culled_low_obs")
            add_event(totals, per_frame, values, "lm_v5_lifecycle_score_survived", "score_survived")
            add_event(totals, per_frame, values, "lm_v5_lifecycle_score_matured", "score_matured")
            add_event(totals, per_frame, values, "lm_v5_lifecycle_pose_use_edges", "score_pose_use_edges")
            add_event(totals, per_frame, values, "lm_v5_lifecycle_pose_use_inliers", "score_pose_use_inliers")
        elif tag == "STSLAM_DYNAMIC_MAP_ADMISSION_V5_SUMMARY":
            add_event(totals, per_frame, values, "lm_v5_support_candidates", "support_candidates")
            add_event(totals, per_frame, values, "lm_v5_support_accepted", "support_accepted")
            add_event(totals, per_frame, values, "lm_v5_reject_support", "reject_support")
            add_event(totals, per_frame, values, "lm_v5_geom_candidates", "geom_events")
            add_event(totals, per_frame, values, "lm_v5_reject_parallax", "reject_parallax")
            add_event(totals, per_frame, values, "lm_v5_reject_triangulate", "reject_triangulate")
            add_event(totals, per_frame, values, "lm_v5_reject_depth", "reject_depth")
            add_event(totals, per_frame, values, "lm_v5_reject_reproj1", "reject_reproj1")
            add_event(totals, per_frame, values, "lm_v5_reject_reproj2", "reject_reproj2")
            add_event(totals, per_frame, values, "lm_v5_reject_scale", "reject_scale")
            add_event(totals, per_frame, values, "lm_v5_reject_score", "reject_score")
            add_event(totals, per_frame, values, "lm_v5_created", "created")
            add_event(totals, per_frame, values, "lm_v5_lifecycle_score_recent", "lifecycle_score_recent_sum")
            add_event(totals, per_frame, values, "lm_v5_lifecycle_score_prebad", "lifecycle_score_prebad_sum")
            add_event(
                totals,
                per_frame,
                values,
                "lm_v5_lifecycle_score_culled_found_ratio",
                "lifecycle_score_culled_found_ratio_sum",
            )
            add_event(
                totals,
                per_frame,
                values,
                "lm_v5_lifecycle_score_culled_low_obs",
                "lifecycle_score_culled_low_obs_sum",
            )
            add_event(totals, per_frame, values, "lm_v5_lifecycle_score_survived", "lifecycle_score_survived_sum")
            add_event(totals, per_frame, values, "lm_v5_lifecycle_score_matured", "lifecycle_score_matured_sum")
            add_event(totals, per_frame, values, "lm_v5_lifecycle_pose_use_edges", "lifecycle_pose_use_edges_sum")
            add_event(totals, per_frame, values, "lm_v5_lifecycle_pose_use_inliers", "lifecycle_pose_use_inliers_sum")
        elif tag == "STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_CONTROL" and stage == "create_new_map_points":
            add_event(totals, per_frame, values, "lm_boundary_budget", "boundary_budget")
            add_event(totals, per_frame, values, "lm_control_skipped_nonboundary_pairs", "skipped_nonboundary_pairs")
            add_event(totals, per_frame, values, "lm_kept_pairs_after_control", "kept_pairs")
        elif tag == "STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_MATCHED_CONTROL" and stage == "create_new_map_points":
            add_event(totals, per_frame, values, "lm_matched_boundary_budget", "boundary_budget")
            add_event(
                totals,
                per_frame,
                values,
                "lm_matched_skipped_nonboundary_pairs",
                "skipped_matched_nonboundary_pairs",
            )
            add_event(totals, per_frame, values, "lm_matched_exact_skipped_pairs", "exact_skipped_pairs")
            add_event(totals, per_frame, values, "lm_matched_fallback_skipped_pairs", "fallback_skipped_pairs")
            add_event(totals, per_frame, values, "lm_kept_pairs_after_matched_control", "kept_pairs")

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
        "ckf_delayed_rejected_new_candidates",
        "ckf_support_promoted_new_candidates",
        "ckf_delayed_existing_supported_candidates",
        "ckf_delayed_support_sum",
        "ckf_quality_rejected_new_candidates",
        "ckf_quality_raw_support_sum",
        "ckf_quality_found_support_sum",
        "ckf_quality_frame_support_sum",
        "ckf_quality_raw_depth_support_sum",
        "ckf_quality_reliable_support_sum",
        "ckf_quality_residual_support_sum",
        "ckf_quality_depth_support_sum",
        "ckf_control_skipped_nonboundary_new_candidates",
        "ckf_matched_boundary_budget",
        "ckf_matched_skipped_nonboundary_new_candidates",
        "ckf_matched_exact_skipped_new_candidates",
        "ckf_matched_fallback_skipped_new_candidates",
        "lm_skipped_instance_pairs",
        "lm_skipped_boundary_pairs",
        "lm_boundary_budget",
        "lm_delayed_rejected_boundary_pairs",
        "lm_support_promoted_boundary_pairs",
        "lm_delayed_support_sum",
        "lm_quality_rejected_boundary_pairs",
        "lm_quality_raw_support_sum",
        "lm_quality_found_support_sum",
        "lm_quality_frame_support_sum",
        "lm_quality_raw_depth_support_sum",
        "lm_quality_reliable_support_sum",
        "lm_quality_residual_support_sum",
        "lm_quality_depth_support_sum",
        "lm_promoted_geom_enter",
        "lm_promoted_geom_parallax",
        "lm_promoted_geom_triangulated",
        "lm_promoted_geom_depth",
        "lm_promoted_geom_reproj1",
        "lm_promoted_geom_reproj2",
        "lm_promoted_geom_scale",
        "lm_promoted_geom_created",
        "lm_score_support_candidates",
        "lm_score_support_accepted",
        "lm_score_support_rejected",
        "lm_score_geom_evaluated",
        "lm_score_post_geom_rejected",
        "lm_score_created",
        "lm_state_candidates",
        "lm_state_allowed",
        "lm_state_rejected",
        "lm_state_tracking_pressure",
        "lm_state_keyframe_pressure",
        "lm_state_scale_pressure",
        "lm_state_lba_pressure",
        "lm_v5_support_candidates",
        "lm_v5_support_accepted",
        "lm_v5_reject_support",
        "lm_v5_geom_candidates",
        "lm_v5_reject_parallax",
        "lm_v5_reject_triangulate",
        "lm_v5_reject_depth",
        "lm_v5_reject_reproj1",
        "lm_v5_reject_reproj2",
        "lm_v5_reject_scale",
        "lm_v5_reject_score",
        "lm_v5_created",
        "lm_v5_lifecycle_score_recent",
        "lm_v5_lifecycle_score_prebad",
        "lm_v5_lifecycle_score_culled_found_ratio",
        "lm_v5_lifecycle_score_culled_low_obs",
        "lm_v5_lifecycle_score_survived",
        "lm_v5_lifecycle_score_matured",
        "lm_v5_lifecycle_pose_use_edges",
        "lm_v5_lifecycle_pose_use_inliers",
        "lm_kept_pairs_after_delayed_boundary",
        "lm_control_skipped_nonboundary_pairs",
        "lm_matched_boundary_budget",
        "lm_matched_skipped_nonboundary_pairs",
        "lm_matched_exact_skipped_pairs",
        "lm_matched_fallback_skipped_pairs",
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
