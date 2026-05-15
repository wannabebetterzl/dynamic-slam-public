#!/usr/bin/env python3
"""Correlate trajectory residual bins with keyframe and map-admission events."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from evaluate_trajectory_ate import align_positions, load_tum_trajectory
from parse_map_admission_events import parse_stdout_events


EVENT_RE = re.compile(r"(\w+)=([^\s]+)")


def parse_case(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("case must be NAME=/path/to/run_dir")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("case name cannot be empty")
    return name, Path(path)


def parse_range(value: str) -> Tuple[int, int]:
    if "-" not in value:
        raise argparse.ArgumentTypeError("range must be START-END")
    start_s, end_s = value.split("-", 1)
    start = int(start_s)
    end = int(end_s)
    if start < 0 or end < start:
        raise argparse.ArgumentTypeError("range must satisfy 0 <= START <= END")
    return start, end


def rmse(values: Sequence[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(math.sqrt(np.mean(arr * arr)))


def path_ratio(est_xyz: np.ndarray, gt_xyz: np.ndarray) -> float:
    if len(est_xyz) < 2 or len(gt_xyz) < 2:
        return float("nan")
    est_len = float(np.sum(np.linalg.norm(np.diff(est_xyz, axis=0), axis=1)))
    gt_len = float(np.sum(np.linalg.norm(np.diff(gt_xyz, axis=0), axis=1)))
    if abs(gt_len) <= 1e-12:
        return float("nan")
    return est_len / gt_len


def associate_with_indices(gt: Dict[str, np.ndarray], est: Dict[str, np.ndarray], max_diff: float) -> Dict[str, np.ndarray]:
    gt_times = gt["times"]
    est_times = est["times"]
    gt_idx = 0
    est_idx = 0
    matched_gt_idx: List[int] = []
    matched_est_idx: List[int] = []
    while gt_idx < len(gt_times) and est_idx < len(est_times):
        diff = est_times[est_idx] - gt_times[gt_idx]
        if abs(diff) <= max_diff:
            matched_gt_idx.append(gt_idx)
            matched_est_idx.append(est_idx)
            gt_idx += 1
            est_idx += 1
        elif diff > 0:
            gt_idx += 1
        else:
            est_idx += 1
    if not matched_gt_idx:
        raise RuntimeError("No matched trajectory pairs were found.")
    gt_indices = np.asarray(matched_gt_idx, dtype=np.int64)
    est_indices = np.asarray(matched_est_idx, dtype=np.int64)
    return {
        "gt_indices": gt_indices,
        "est_indices": est_indices,
        "gt_times": gt_times[gt_indices],
        "est_times": est_times[est_indices],
        "gt_xyz": gt["xyz"][gt_indices],
        "est_xyz": est["xyz"][est_indices],
    }


def load_case(run_dir: Path, gt_path: Path, max_diff: float) -> Dict[str, np.ndarray | float]:
    gt = load_tum_trajectory(gt_path)
    est = load_tum_trajectory(run_dir / "CameraTrajectory.txt")
    matched = associate_with_indices(gt, est, max_diff)
    gt_xyz = matched["gt_xyz"]
    est_xyz = matched["est_xyz"]
    assert isinstance(gt_xyz, np.ndarray)
    assert isinstance(est_xyz, np.ndarray)
    se3_aligned, _, _, _, _ = align_positions(gt_xyz, est_xyz, "se3")
    sim3_aligned, _, _, sim3_scale, _ = align_positions(gt_xyz, est_xyz, "sim3")
    matched["se3_error"] = np.linalg.norm(gt_xyz - se3_aligned, axis=1)
    matched["sim3_error"] = np.linalg.norm(gt_xyz - sim3_aligned, axis=1)
    matched["sim3_scale"] = float(sim3_scale)
    return matched


def read_association_times(path: Path) -> List[float]:
    times: List[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) >= 2:
            times.append(float(parts[0]))
    return times


def nearest_frame_ids(times: np.ndarray, association_times: Sequence[float], max_diff: float) -> List[int]:
    result: List[int] = []
    assoc = np.asarray(association_times, dtype=np.float64)
    for value in times:
        idx = int(np.argmin(np.abs(assoc - value)))
        if abs(float(assoc[idx]) - float(value)) <= max_diff:
            result.append(idx)
        else:
            result.append(-1)
    return result


def read_keyframe_frames(run_dir: Path) -> List[int]:
    path = run_dir / "KeyFrameTimeline.csv"
    if not path.exists():
        return []
    frames: List[int] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frame_id = row.get("frame_id", "")
            if frame_id:
                frames.append(int(float(frame_id)))
    return frames


def parse_v9_lba_delay(run_dir: Path) -> Dict[int, Dict[str, float]]:
    per_frame: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    path = run_dir / "stdout.log"
    if not path.exists():
        return per_frame
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "[STSLAM_DYNAMIC_MAP_ADMISSION_LBA_DELAY_V9]" not in line:
            continue
        values = {key: value for key, value in EVENT_RE.findall(line)}
        if values.get("stage") != "before_optimize" or "frame" not in values:
            continue
        frame = int(float(values["frame"]))
        per_frame[frame]["v9_lba_delay_windows"] += 1.0
        for key in ("local_kf", "fixed_kf", "local_mp", "delayed_mp", "optimizer_points", "optimizer_edges"):
            if key in values:
                per_frame[frame][f"v9_{key}_sum"] += float(values[key])
                per_frame[frame][f"v9_{key}_max"] = max(
                    per_frame[frame].get(f"v9_{key}_max", 0.0),
                    float(values[key]),
                )
    return per_frame


def merge_events(run_dir: Path) -> Dict[int, Dict[str, float]]:
    _, parsed = parse_stdout_events(run_dir)
    merged: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for frame, values in parsed.items():
        for key, value in values.items():
            merged[int(frame)][key] += float(value)
    v9_delay = parse_v9_lba_delay(run_dir)
    for frame, values in v9_delay.items():
        for key, value in values.items():
            merged[int(frame)][key] += float(value)
    return merged


def sum_events(events: Dict[int, Dict[str, float]], frame_start: int, frame_end: int, keys: Iterable[str]) -> Dict[str, float]:
    totals = {key: 0.0 for key in keys}
    for frame, values in events.items():
        if frame_start <= frame <= frame_end:
            for key in keys:
                totals[key] += float(values.get(key, 0.0))
    return totals


def mean_event(events: Dict[int, Dict[str, float]], frame_start: int, frame_end: int, sum_key: str, count_key: str) -> float:
    total = 0.0
    count = 0.0
    for frame, values in events.items():
        if frame_start <= frame <= frame_end:
            total += float(values.get(sum_key, 0.0))
            count += float(values.get(count_key, 0.0))
    return total / count if count > 0 else float("nan")


def fmt(value: float) -> str:
    if isinstance(value, float) and math.isnan(value):
        return ""
    return f"{value:.9f}"


def build_rows(
    cases: Dict[str, Dict[str, np.ndarray | float]],
    keyframes: Dict[str, List[int]],
    events: Dict[str, Dict[int, Dict[str, float]]],
    frame_ids: List[int],
    ranges: Sequence[Tuple[int, int]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    event_keys = [
        "ckf_direct_vetoed_candidates",
        "ckf_accepted_depth_candidates",
        "ckf_boundary_skipped_new_candidates",
        "ckf_boundary_existing_supported_candidates",
        "lm_skipped_boundary_pairs",
        "lm_delayed_rejected_boundary_pairs",
        "lm_quality_rejected_boundary_pairs",
        "lm_support_promoted_boundary_pairs",
        "lm_score_support_candidates",
        "lm_score_support_accepted",
        "lm_score_support_rejected",
        "lm_score_geom_evaluated",
        "lm_score_post_geom_rejected",
        "lm_score_created",
        "v9_lba_delay_windows",
        "v9_delayed_mp_sum",
        "v9_delayed_mp_max",
    ]
    for start, requested_end in ranges:
        first_case = next(iter(cases.values()))
        n = len(first_case["se3_error"])  # type: ignore[arg-type]
        if start >= n:
            continue
        end = min(requested_end, n - 1)
        segment_frames = [frame for frame in frame_ids[start : end + 1] if frame >= 0]
        frame_start = min(segment_frames) if segment_frames else -1
        frame_end = max(segment_frames) if segment_frames else -1
        row: Dict[str, object] = {
            "match_start": start,
            "match_end": end,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "count": end - start + 1,
        }
        for name, data in cases.items():
            se3 = data["se3_error"]
            sim3 = data["sim3_error"]
            est_xyz = data["est_xyz"]
            gt_xyz = data["gt_xyz"]
            assert isinstance(se3, np.ndarray)
            assert isinstance(sim3, np.ndarray)
            assert isinstance(est_xyz, np.ndarray)
            assert isinstance(gt_xyz, np.ndarray)
            row[f"{name}_se3_rmse_m"] = fmt(rmse(se3[start : end + 1]))
            row[f"{name}_sim3_rmse_m"] = fmt(rmse(sim3[start : end + 1]))
            row[f"{name}_path_ratio"] = fmt(path_ratio(est_xyz[start : end + 1], gt_xyz[start : end + 1]))
            row[f"{name}_keyframes"] = sum(1 for frame in keyframes.get(name, []) if frame_start <= frame <= frame_end)
        names = list(cases.keys())
        if "mainline" in cases and "v8" in cases and "v9" in cases:
            for lhs, rhs in [("v8", "mainline"), ("v9", "mainline"), ("v9", "v8")]:
                row[f"{lhs}_minus_{rhs}_se3_rmse_m"] = fmt(
                    float(row[f"{lhs}_se3_rmse_m"]) - float(row[f"{rhs}_se3_rmse_m"])
                )
                row[f"{lhs}_minus_{rhs}_sim3_rmse_m"] = fmt(
                    float(row[f"{lhs}_sim3_rmse_m"]) - float(row[f"{rhs}_sim3_rmse_m"])
                )
        for name in names:
            totals = sum_events(events.get(name, {}), frame_start, frame_end, event_keys)
            for key, value in totals.items():
                row[f"{name}_{key}"] = fmt(value)
            row[f"{name}_v9_delayed_mp_mean"] = fmt(
                mean_event(events.get(name, {}), frame_start, frame_end, "v9_delayed_mp_sum", "v9_lba_delay_windows")
            )
            row[f"{name}_v9_optimizer_edges_mean"] = fmt(
                mean_event(events.get(name, {}), frame_start, frame_end, "v9_optimizer_edges_sum", "v9_lba_delay_windows")
            )
        rows.append(row)
    return rows


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
        writer.writerows(rows)


def render_summary(rows: List[Dict[str, object]]) -> str:
    lines = ["Segment-aware mechanism summary", ""]
    for row in rows:
        label = f"matches {row['match_start']}-{row['match_end']} frames {row['frame_start']}-{row['frame_end']}"
        lines.append(label)
        if "v9_minus_mainline_se3_rmse_m" in row:
            lines.append(
                "  v9-mainline: SE3 {se3}, Sim3 {sim3}; v9-v8: SE3 {se3v8}, Sim3 {sim3v8}".format(
                    se3=row["v9_minus_mainline_se3_rmse_m"],
                    sim3=row["v9_minus_mainline_sim3_rmse_m"],
                    se3v8=row["v9_minus_v8_se3_rmse_m"],
                    sim3v8=row["v9_minus_v8_sim3_rmse_m"],
                )
            )
        lines.append(
            "  v9 events: score_created={created}, score_candidates={candidates}, delayed_mp_sum={delayed}, delayed_mp_mean={mean}, lba_windows={windows}".format(
                created=row.get("v9_lm_score_created", ""),
                candidates=row.get("v9_lm_score_support_candidates", ""),
                delayed=row.get("v9_v9_delayed_mp_sum", ""),
                mean=row.get("v9_v9_delayed_mp_mean", ""),
                windows=row.get("v9_v9_lba_delay_windows", ""),
            )
        )
        lines.append(
            "  keyframes: mainline={main}, v8={v8}, v9={v9}".format(
                main=row.get("mainline_keyframes", ""),
                v8=row.get("v8_keyframes", ""),
                v9=row.get("v9_keyframes", ""),
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--associations", type=Path, required=True)
    parser.add_argument("--case", type=parse_case, action="append", required=True)
    parser.add_argument("--range", type=parse_range, action="append", required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--max-diff", type=float, default=0.03)
    args = parser.parse_args()

    cases = {name: load_case(path, args.ground_truth, args.max_diff) for name, path in args.case}
    run_dirs = {name: path for name, path in args.case}
    association_times = read_association_times(args.associations)
    ref_name = "v9" if "v9" in cases else next(iter(cases))
    ref_times = cases[ref_name]["est_times"]
    assert isinstance(ref_times, np.ndarray)
    frame_ids = nearest_frame_ids(ref_times, association_times, args.max_diff)
    keyframes = {name: read_keyframe_frames(path) for name, path in run_dirs.items()}
    events = {name: merge_events(path) for name, path in run_dirs.items()}
    rows = build_rows(cases, keyframes, events, frame_ids, args.range)
    write_csv(args.out_csv, rows)
    summary = render_summary(rows)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(summary + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
