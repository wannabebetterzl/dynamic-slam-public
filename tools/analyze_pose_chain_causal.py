#!/usr/bin/env python3
"""Build causal timeline tables for pose-chain admission experiments."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from evaluate_trajectory_ate import load_tum_trajectory


EVENT_RE = re.compile(r"(\w+)=([^\s]+)")


@dataclass
class CaseData:
    name: str
    run_dir: Path
    timeline: Dict[int, Dict[str, float]]
    lba_rows: List[Dict[str, object]]
    frames: List[int]


def parse_case(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("case must be NAME=/path/to/run_dir")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("case name cannot be empty")
    return name, Path(path)


def parse_frame_ranges(value: str) -> List[Tuple[str, int, int]]:
    ranges: List[Tuple[str, int, int]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" not in item:
            raise argparse.ArgumentTypeError("ranges must be comma-separated START-END values")
        start_s, end_s = item.split("-", 1)
        start = int(start_s)
        end = int(end_s)
        if start < 0 or end < start:
            raise argparse.ArgumentTypeError("each range must satisfy 0 <= START <= END")
        ranges.append((f"{start}-{end}", start, end))
    if not ranges:
        raise argparse.ArgumentTypeError("at least one range is required")
    return ranges


def to_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def add_sum(row: Dict[str, float], key: str, value: object) -> None:
    row[key] = row.get(key, 0.0) + to_float(value)


def add_max(row: Dict[str, float], key: str, value: object) -> None:
    row[key] = max(row.get(key, 0.0), to_float(value))


def add_min(row: Dict[str, float], key: str, value: object) -> None:
    v = to_float(value)
    if key not in row:
        row[key] = v
    else:
        row[key] = min(row[key], v)


def read_association_times(path: Optional[Path]) -> List[float]:
    if path is None or not path.exists():
        return []
    times: List[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) >= 2:
            times.append(float(parts[0]))
    return times


def nearest_index(times: np.ndarray, value: float, max_diff: float) -> int:
    if times.size == 0:
        return -1
    idx = int(np.argmin(np.abs(times - value)))
    return idx if abs(float(times[idx]) - value) <= max_diff else -1


def nearest_frame_id(value: float, association_times: Sequence[float], max_diff: float) -> int:
    if not association_times:
        return -1
    arr = np.asarray(association_times, dtype=np.float64)
    idx = int(np.argmin(np.abs(arr - value)))
    return idx if abs(float(arr[idx]) - value) <= max_diff else -1


def read_keyframe_timeline(run_dir: Path, density_window: int) -> Dict[int, Dict[str, float]]:
    path = run_dir / "KeyFrameTimeline.csv"
    if not path.exists():
        return {}
    rows: List[Dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    result: Dict[int, Dict[str, float]] = {}
    prev_frame: Optional[int] = None
    prev_xyz: Optional[np.ndarray] = None
    frame_list: List[int] = []
    for row in rows:
        frame = to_int(row.get("frame_id"), -1)
        if frame < 0:
            continue
        xyz = np.asarray(
            [to_float(row.get("x")), to_float(row.get("y")), to_float(row.get("z"))],
            dtype=np.float64,
        )
        kf_step = float(np.linalg.norm(xyz - prev_xyz)) if prev_xyz is not None else 0.0
        kf_gap = float(frame - prev_frame) if prev_frame is not None else 0.0
        frame_list.append(frame)
        density = sum(1 for f in frame_list if frame - density_window + 1 <= f <= frame)
        result[frame] = {
            "is_keyframe": 1.0,
            "kf_id": float(to_int(row.get("keyframe_id"), -1)),
            "kf_step": kf_step,
            "kf_gap": kf_gap,
            "kf_density": float(density),
            "kf_x": float(xyz[0]),
            "kf_y": float(xyz[1]),
            "kf_z": float(xyz[2]),
        }
        prev_frame = frame
        prev_xyz = xyz
    return result


def add_trajectory_steps(
    timeline: Dict[int, Dict[str, float]],
    run_dir: Path,
    ground_truth: Optional[Path],
    associations: Optional[Path],
    max_diff: float,
) -> None:
    est_path = run_dir / "CameraTrajectory.txt"
    if not est_path.exists():
        return
    est = load_tum_trajectory(est_path)
    est_times = est["times"]
    est_xyz = est["xyz"]
    assoc_times = read_association_times(associations)

    gt_times: Optional[np.ndarray] = None
    gt_xyz: Optional[np.ndarray] = None
    if ground_truth is not None and ground_truth.exists():
        gt = load_tum_trajectory(ground_truth)
        gt_times = gt["times"]
        gt_xyz = gt["xyz"]

    prev_frame = -1
    prev_est: Optional[np.ndarray] = None
    prev_gt: Optional[np.ndarray] = None
    for idx, timestamp in enumerate(est_times):
        frame = nearest_frame_id(float(timestamp), assoc_times, max_diff)
        if frame < 0:
            frame = idx
        current_est = est_xyz[idx]
        current_gt: Optional[np.ndarray] = None
        if gt_times is not None and gt_xyz is not None:
            gt_idx = nearest_index(gt_times, float(timestamp), max_diff)
            if gt_idx >= 0:
                current_gt = gt_xyz[gt_idx]
        row = timeline[frame]
        row["traj_present"] = 1.0
        row["traj_timestamp"] = float(timestamp)
        if prev_est is not None and prev_frame >= 0:
            est_step = float(np.linalg.norm(current_est - prev_est))
            row["traj_est_step"] = est_step
            row["traj_frame_gap"] = float(frame - prev_frame)
            if current_gt is not None and prev_gt is not None:
                gt_step = float(np.linalg.norm(current_gt - prev_gt))
                row["traj_gt_step"] = gt_step
                row["traj_step_ratio"] = est_step / gt_step if gt_step > 1e-12 else 0.0
        prev_frame = frame
        prev_est = current_est
        prev_gt = current_gt


def parse_stdout(run_dir: Path) -> Tuple[Dict[int, Dict[str, float]], List[Dict[str, object]]]:
    timeline: Dict[int, Dict[str, float]] = defaultdict(dict)
    lba_rows: List[Dict[str, object]] = []
    path = run_dir / "stdout.log"
    if not path.exists():
        return timeline, lba_rows

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("[STSLAM_"):
            continue
        tag = line.split("]", 1)[0].strip("[")
        values = dict(EVENT_RE.findall(line))
        frame_value = values.get("frame", values.get("frame_index"))
        if frame_value is None:
            continue
        frame = to_int(frame_value, -1)
        if frame < 0:
            continue
        row = timeline[frame]
        stage = values.get("stage", "")

        if tag == "STSLAM_SEQUENTIAL_LOCAL_MAPPING":
            add_sum(row, "sequential_mapping_events", 1)
            add_sum(row, "sequential_processed_keyframes", values.get("processed_keyframes", 0))
            add_max(row, "sequential_queue_max", values.get("queue", 0))
        elif tag == "STSLAM_DYNAMIC_MAP_ADMISSION_LBA_DELAY_V9" and stage == "before_optimize":
            add_sum(row, "lba_count", 1)
            for key in ("local_kf", "fixed_kf", "local_mp", "delayed_mp", "optimizer_points", "optimizer_edges"):
                add_sum(row, f"lba_{key}_sum", values.get(key, 0))
                add_max(row, f"lba_{key}_max", values.get(key, 0))
            local_mp = to_float(values.get("local_mp"), 0.0)
            delayed_mp = to_float(values.get("delayed_mp"), 0.0)
            delayed_ratio = delayed_mp / local_mp if local_mp > 0 else 0.0
            add_max(row, "lba_delayed_ratio_max", delayed_ratio)
            lba_rows.append(
                {
                    "frame_id": frame,
                    "current_kf": to_int(values.get("current_kf"), -1),
                    "local_kf": to_int(values.get("local_kf"), 0),
                    "fixed_kf": to_int(values.get("fixed_kf"), 0),
                    "local_mp": to_int(values.get("local_mp"), 0),
                    "delayed_mp": to_int(values.get("delayed_mp"), 0),
                    "optimizer_points": to_int(values.get("optimizer_points"), 0),
                    "optimizer_edges": to_int(values.get("optimizer_edges"), 0),
                    "delayed_ratio": delayed_ratio,
                }
            )
        elif tag == "STSLAM_DYNAMIC_MAP_ADMISSION_RECOVERY_V11" and stage == "create_new_map_points":
            add_sum(row, "recovery_events", 1)
            for src, dst in [
                ("candidates", "recovery_candidates"),
                ("pregeom_allowed", "recovery_pregeom_allowed"),
                ("rejected", "recovery_rejected"),
                ("geom_rejected", "recovery_geom_rejected"),
                ("created", "recovery_created"),
            ]:
                add_sum(row, dst, values.get(src, 0))
            for key in ("tracking_pressure", "keyframe_pressure", "scale_pressure", "lba_pressure"):
                add_sum(row, f"recovery_{key}_sum", values.get(key, 0))
            add_max(row, "recovery_need_score_max", values.get("need_score", 0))
        elif tag == "STSLAM_DYNAMIC_MAP_ADMISSION_POSE_CHAIN_GUARD_V13":
            add_sum(row, "pose_chain_guard_lines", 1)
            for src, dst in [
                ("active", "pose_chain_guard_active_lines"),
                ("candidates", "pose_chain_guard_candidates"),
                ("marked", "pose_chain_guard_marked"),
                ("created", "pose_chain_guard_created"),
            ]:
                add_sum(row, dst, values.get(src, 0))
            for key in ("tracking_pressure", "keyframe_pressure", "scale_pressure", "lba_pressure"):
                add_sum(row, f"pose_chain_{key}_sum", values.get(key, 0))
            add_min(row, "pose_chain_tracking_inliers_min", values.get("tracking_inliers", 0))
            add_max(row, "pose_chain_need_score_max", values.get("need_score", 0))
            add_max(row, "pose_chain_keyframe_step_max", values.get("keyframe_step", 0))
            add_max(row, "pose_chain_keyframe_step_ratio_max", values.get("keyframe_step_ratio", 0))
        elif tag == "STSLAM_DYNAMIC_MAP_ADMISSION_V7_PROBATION" and stage == "map_point_culling":
            add_sum(row, "probation_events", 1)
            for key in [
                "score_recent",
                "score_survived",
                "score_matured",
                "score_prebad",
                "score_culled_found_ratio",
                "score_culled_low_obs",
                "v7_residual_rejected",
                "v7_low_use_rejected",
                "score_pose_use_edges",
                "score_pose_use_inliers",
                "score_lba_edges",
                "score_lba_inliers",
                "score_lba_window_points",
                "score_lba_edge_points",
            ]:
                add_sum(row, f"probation_{key}", values.get(key, 0))
            add_max(row, "probation_pose_use_chi2_mean_max", values.get("score_pose_use_chi2_mean", 0))
            add_max(row, "probation_lba_chi2_mean_max", values.get("score_lba_chi2_mean", 0))
    return timeline, lba_rows


def merge_timeline(
    keyframes: Dict[int, Dict[str, float]],
    events: Dict[int, Dict[str, float]],
) -> Dict[int, Dict[str, float]]:
    result: Dict[int, Dict[str, float]] = defaultdict(dict)
    for frame, values in keyframes.items():
        result[frame].update(values)
    for frame, values in events.items():
        result[frame].update(values)
    return result


def load_case(
    name: str,
    run_dir: Path,
    ground_truth: Optional[Path],
    associations: Optional[Path],
    max_diff: float,
    density_window: int,
) -> CaseData:
    keyframes = read_keyframe_timeline(run_dir, density_window)
    events, lba_rows = parse_stdout(run_dir)
    timeline = merge_timeline(keyframes, events)
    add_trajectory_steps(timeline, run_dir, ground_truth, associations, max_diff)
    frames = sorted(timeline)
    return CaseData(name=name, run_dir=run_dir, timeline=timeline, lba_rows=lba_rows, frames=frames)


def segment_label(frame: int, ranges: Sequence[Tuple[str, int, int]]) -> str:
    for label, start, end in ranges:
        if start <= frame <= end:
            return label
    return ""


def fmt(value: object) -> object:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.9f}"
    return value


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(value) for key, value in row.items()})


TIMELINE_METRICS = [
    "is_keyframe",
    "kf_id",
    "kf_step",
    "kf_gap",
    "kf_density",
    "traj_est_step",
    "traj_gt_step",
    "traj_step_ratio",
    "lba_count",
    "lba_local_kf_max",
    "lba_fixed_kf_max",
    "lba_local_mp_max",
    "lba_delayed_mp_sum",
    "lba_delayed_mp_max",
    "lba_delayed_ratio_max",
    "lba_optimizer_edges_max",
    "recovery_candidates",
    "recovery_pregeom_allowed",
    "recovery_created",
    "probation_score_recent",
    "probation_score_survived",
    "probation_score_matured",
    "probation_score_pose_use_edges",
    "probation_score_lba_edges",
    "pose_chain_guard_active_lines",
    "pose_chain_guard_candidates",
    "pose_chain_guard_created",
    "pose_chain_need_score_max",
    "pose_chain_keyframe_step_ratio_max",
]


def build_timeline_rows(cases: Sequence[CaseData], ranges: Sequence[Tuple[str, int, int]]) -> List[Dict[str, object]]:
    all_frames = sorted({frame for case in cases for frame in case.frames})
    rows: List[Dict[str, object]] = []
    base = cases[0] if cases else None
    comp = cases[1] if len(cases) > 1 else None
    for frame in all_frames:
        row: Dict[str, object] = {"frame_id": frame, "segment": segment_label(frame, ranges)}
        for case in cases:
            values = case.timeline.get(frame, {})
            for metric in TIMELINE_METRICS:
                row[f"{case.name}_{metric}"] = values.get(metric, 0.0)
        if base is not None and comp is not None:
            for metric in [
                "is_keyframe",
                "kf_density",
                "kf_step",
                "traj_est_step",
                "traj_step_ratio",
                "lba_delayed_mp_sum",
                "lba_delayed_mp_max",
                "lba_local_mp_max",
                "recovery_created",
                "pose_chain_guard_active_lines",
            ]:
                row[f"{comp.name}_minus_{base.name}_{metric}"] = (
                    comp.timeline.get(frame, {}).get(metric, 0.0)
                    - base.timeline.get(frame, {}).get(metric, 0.0)
                )
        rows.append(row)
    return rows


def build_lba_rows(cases: Sequence[CaseData], ranges: Sequence[Tuple[str, int, int]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for case in cases:
        for item in case.lba_rows:
            row = {"case": case.name, **item}
            row["segment"] = segment_label(to_int(item.get("frame_id"), -1), ranges)
            rows.append(row)
    rows.sort(key=lambda r: (to_int(r.get("frame_id")), str(r.get("case"))))
    return rows


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def sum_metric(timeline: Dict[int, Dict[str, float]], frames: Iterable[int], key: str) -> float:
    return float(sum(timeline.get(frame, {}).get(key, 0.0) for frame in frames))


def max_metric(timeline: Dict[int, Dict[str, float]], frames: Iterable[int], key: str) -> float:
    vals = [timeline.get(frame, {}).get(key, 0.0) for frame in frames]
    return float(max(vals)) if vals else 0.0


def segment_case_summary(
    case: CaseData,
    label: str,
    start: int,
    end: int,
) -> Dict[str, object]:
    frames = [frame for frame in range(start, end + 1) if frame in case.timeline]
    kf_steps = [case.timeline[f].get("kf_step", 0.0) for f in frames if case.timeline[f].get("is_keyframe", 0.0) > 0]
    est_steps = [case.timeline[f].get("traj_est_step", 0.0) for f in frames if case.timeline[f].get("traj_est_step", 0.0) > 0]
    gt_steps = [case.timeline[f].get("traj_gt_step", 0.0) for f in frames if case.timeline[f].get("traj_gt_step", 0.0) > 0]
    est_path = float(sum(est_steps))
    gt_path = float(sum(gt_steps))
    return {
        "case": case.name,
        "segment": label,
        "frame_start": start,
        "frame_end": end,
        "observed_frames": len(frames),
        "keyframes": int(sum_metric(case.timeline, frames, "is_keyframe")),
        "kf_step_sum": float(sum(kf_steps)),
        "kf_step_mean": mean(kf_steps),
        "kf_step_max": float(max(kf_steps)) if kf_steps else 0.0,
        "traj_est_path": est_path,
        "traj_gt_path": gt_path,
        "traj_path_ratio": est_path / gt_path if gt_path > 1e-12 else 0.0,
        "lba_count": int(sum_metric(case.timeline, frames, "lba_count")),
        "lba_delayed_mp_sum": int(sum_metric(case.timeline, frames, "lba_delayed_mp_sum")),
        "lba_delayed_mp_max": int(max_metric(case.timeline, frames, "lba_delayed_mp_max")),
        "lba_local_mp_max": int(max_metric(case.timeline, frames, "lba_local_mp_max")),
        "recovery_candidates": int(sum_metric(case.timeline, frames, "recovery_candidates")),
        "recovery_pregeom_allowed": int(sum_metric(case.timeline, frames, "recovery_pregeom_allowed")),
        "recovery_created": int(sum_metric(case.timeline, frames, "recovery_created")),
        "probation_score_recent": int(sum_metric(case.timeline, frames, "probation_score_recent")),
        "probation_score_survived": int(sum_metric(case.timeline, frames, "probation_score_survived")),
        "probation_score_matured": int(sum_metric(case.timeline, frames, "probation_score_matured")),
        "probation_pose_use_edges": int(sum_metric(case.timeline, frames, "probation_score_pose_use_edges")),
        "probation_lba_edges": int(sum_metric(case.timeline, frames, "probation_score_lba_edges")),
        "pose_chain_active_lines": int(sum_metric(case.timeline, frames, "pose_chain_guard_active_lines")),
        "pose_chain_candidates": int(sum_metric(case.timeline, frames, "pose_chain_guard_candidates")),
        "pose_chain_created": int(sum_metric(case.timeline, frames, "pose_chain_guard_created")),
        "pose_chain_step_ratio_max": max_metric(case.timeline, frames, "pose_chain_keyframe_step_ratio_max"),
    }


def build_segment_rows(cases: Sequence[CaseData], ranges: Sequence[Tuple[str, int, int]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    summaries: Dict[Tuple[str, str], Dict[str, object]] = {}
    for label, start, end in ranges:
        for case in cases:
            summary = segment_case_summary(case, label, start, end)
            rows.append(summary)
            summaries[(label, case.name)] = summary
        if len(cases) >= 2:
            base = summaries[(label, cases[0].name)]
            comp = summaries[(label, cases[1].name)]
            diff: Dict[str, object] = {
                "case": f"{cases[1].name}_minus_{cases[0].name}",
                "segment": label,
                "frame_start": start,
                "frame_end": end,
            }
            for key in [
                "keyframes",
                "kf_step_sum",
                "kf_step_mean",
                "kf_step_max",
                "traj_est_path",
                "traj_path_ratio",
                "lba_count",
                "lba_delayed_mp_sum",
                "lba_delayed_mp_max",
                "lba_local_mp_max",
                "recovery_candidates",
                "recovery_pregeom_allowed",
                "recovery_created",
                "probation_score_recent",
                "probation_score_survived",
                "probation_score_matured",
                "probation_pose_use_edges",
                "probation_lba_edges",
                "pose_chain_active_lines",
                "pose_chain_candidates",
                "pose_chain_created",
                "pose_chain_step_ratio_max",
            ]:
                diff[key] = to_float(comp.get(key), 0.0) - to_float(base.get(key), 0.0)
            rows.append(diff)
    return rows


def write_summary(path: Path, segment_rows: Sequence[Dict[str, object]], cases: Sequence[CaseData]) -> None:
    lines = ["Pose-chain causal summary", ""]
    case_names = ", ".join(case.name for case in cases)
    lines.append(f"cases: {case_names}")
    lines.append("")
    for row in segment_rows:
        case = row.get("case", "")
        segment = row.get("segment", "")
        if "minus" not in str(case):
            continue
        lines.append(
            "{case} segment {segment}: keyframes={keyframes}, delayed_mp_sum={delayed}, "
            "traj_path_ratio_delta={ratio}, recovery_created={recovery}, pose_chain_active={active}".format(
                case=case,
                segment=segment,
                keyframes=fmt(row.get("keyframes", 0.0)),
                delayed=fmt(row.get("lba_delayed_mp_sum", 0.0)),
                ratio=fmt(row.get("traj_path_ratio", 0.0)),
                recovery=fmt(row.get("recovery_created", 0.0)),
                active=fmt(row.get("pose_chain_active_lines", 0.0)),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=parse_case, action="append", required=True)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--associations", type=Path)
    parser.add_argument("--frame-ranges", type=parse_frame_ranges, required=True)
    parser.add_argument("--out-timeline-csv", type=Path, required=True)
    parser.add_argument("--out-lba-csv", type=Path, required=True)
    parser.add_argument("--out-segment-csv", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--max-diff", type=float, default=0.03)
    parser.add_argument("--keyframe-density-window", type=int, default=30)
    args = parser.parse_args()

    cases = [
        load_case(
            name,
            path,
            args.ground_truth,
            args.associations,
            args.max_diff,
            args.keyframe_density_window,
        )
        for name, path in args.case
    ]
    timeline_rows = build_timeline_rows(cases, args.frame_ranges)
    lba_rows = build_lba_rows(cases, args.frame_ranges)
    segment_rows = build_segment_rows(cases, args.frame_ranges)

    write_csv(args.out_timeline_csv, timeline_rows)
    write_csv(args.out_lba_csv, lba_rows)
    write_csv(args.out_segment_csv, segment_rows)
    write_summary(args.out_summary, segment_rows, cases)
    print(args.out_summary)


if __name__ == "__main__":
    main()
