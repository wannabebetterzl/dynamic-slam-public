#!/usr/bin/env python3
"""Align WR-PY support-quality admissions with trajectory and observability.

The goal is to explain whether the small number of admitted LocalMapping
boundary pairs happen in the right temporal window and whether they correlate
with trajectory/path-scale changes relative to the hard-boundary reference.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from evaluate_trajectory_ate import align_positions, load_tum_trajectory


TrajectoryTriples = Tuple[np.ndarray, np.ndarray, np.ndarray]


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: object, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: object, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def associate_with_times(gt: Dict[str, np.ndarray], est: Dict[str, np.ndarray], max_diff: float) -> TrajectoryTriples:
    gt_times = gt["times"]
    est_times = est["times"]
    gt_xyz = gt["xyz"]
    est_xyz = est["xyz"]

    gt_idx = 0
    est_idx = 0
    times: List[float] = []
    matched_gt: List[np.ndarray] = []
    matched_est: List[np.ndarray] = []

    while gt_idx < len(gt_times) and est_idx < len(est_times):
        diff = est_times[est_idx] - gt_times[gt_idx]
        if abs(diff) <= max_diff:
            times.append(float(est_times[est_idx]))
            matched_gt.append(gt_xyz[gt_idx])
            matched_est.append(est_xyz[est_idx])
            gt_idx += 1
            est_idx += 1
        elif diff > 0:
            gt_idx += 1
        else:
            est_idx += 1

    if not times:
        raise RuntimeError("No matched trajectory pairs found.")

    return (
        np.array(times, dtype=np.float64),
        np.array(matched_gt, dtype=np.float64),
        np.array(matched_est, dtype=np.float64),
    )


def cumulative_path(xyz: np.ndarray) -> np.ndarray:
    if len(xyz) == 0:
        return np.array([], dtype=np.float64)
    result = np.zeros(len(xyz), dtype=np.float64)
    if len(xyz) > 1:
        result[1:] = np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))
    return result


def safe_ratio(numerator: float, denominator: float) -> str:
    if abs(denominator) <= 1e-12:
        return ""
    return f"{numerator / denominator:.6f}"


def find_nearest_index(times: np.ndarray, timestamp: float) -> int:
    return int(np.argmin(np.abs(times - timestamp)))


def row_for_frame(rows: Iterable[Dict[str, str]], frame_id: int) -> Dict[str, str]:
    for row in rows:
        if parse_int(row.get("frame_id")) == frame_id:
            return row
    return {}


def rows_by_frame(rows: Iterable[Dict[str, str]]) -> Dict[int, Dict[str, str]]:
    return {parse_int(row.get("frame_id")): row for row in rows if row.get("frame_id") not in (None, "")}


def mean_numeric(rows: List[Dict[str, str]], key: str) -> str:
    values = [parse_float(row.get(key), math.nan) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return ""
    return f"{float(np.mean(values)):.6f}"


def segment_path_ratio(xyz: np.ndarray, gt_xyz: np.ndarray, start: int, end: int) -> str:
    start = max(0, start)
    end = min(len(xyz) - 1, end)
    if end <= start:
        return ""
    est_len = float(np.sum(np.linalg.norm(np.diff(xyz[start : end + 1], axis=0), axis=1)))
    gt_len = float(np.sum(np.linalg.norm(np.diff(gt_xyz[start : end + 1], axis=0), axis=1)))
    return safe_ratio(est_len, gt_len)


def load_run_alignment(run_dir: Path, ground_truth: Path, max_diff: float) -> Dict[str, object]:
    gt = load_tum_trajectory(ground_truth)
    est = load_tum_trajectory(run_dir / "CameraTrajectory.txt")
    times, gt_xyz, est_xyz = associate_with_times(gt, est, max_diff)
    aligned_se3, _, _, _, _ = align_positions(gt_xyz, est_xyz, "se3")
    aligned_sim3, _, _, sim3_scale, _ = align_positions(gt_xyz, est_xyz, "sim3")
    return {
        "times": times,
        "gt_xyz": gt_xyz,
        "est_xyz": est_xyz,
        "se3_error": np.linalg.norm(gt_xyz - aligned_se3, axis=1),
        "sim3_error": np.linalg.norm(gt_xyz - aligned_sim3, axis=1),
        "gt_cum": cumulative_path(gt_xyz),
        "est_cum": cumulative_path(est_xyz),
        "sim3_scale": sim3_scale,
    }


def window_rows(obs_by_frame: Dict[int, Dict[str, str]], frame_id: int, before: int, after: int) -> List[Dict[str, str]]:
    return [
        obs_by_frame[frame]
        for frame in range(frame_id - before, frame_id + after + 1)
        if frame in obs_by_frame
    ]


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_promoted_context(
    support_events: List[Dict[str, str]],
    support_obs: Dict[int, Dict[str, str]],
    hard_obs: Dict[int, Dict[str, str]],
    support_align: Dict[str, object],
    hard_align: Dict[str, object],
) -> List[Dict[str, object]]:
    support_times = support_align["times"]
    hard_times = hard_align["times"]
    assert isinstance(support_times, np.ndarray)
    assert isinstance(hard_times, np.ndarray)

    rows: List[Dict[str, object]] = []
    for event in support_events:
        promoted = parse_int(event.get("lm_support_promoted_boundary_pairs"))
        if promoted <= 0:
            continue
        frame_id = parse_int(event.get("frame_id"))
        timestamp = parse_float(event.get("timestamp"))
        support_idx = find_nearest_index(support_times, timestamp)
        hard_idx = find_nearest_index(hard_times, timestamp)

        support_gt = support_align["gt_xyz"]
        support_est = support_align["est_xyz"]
        hard_gt = hard_align["gt_xyz"]
        hard_est = hard_align["est_xyz"]
        assert isinstance(support_gt, np.ndarray)
        assert isinstance(support_est, np.ndarray)
        assert isinstance(hard_gt, np.ndarray)
        assert isinstance(hard_est, np.ndarray)

        support_frame = support_obs.get(frame_id, {})
        hard_frame = hard_obs.get(frame_id, {})
        support_win30 = window_rows(support_obs, frame_id, 0, 30)
        hard_win30 = window_rows(hard_obs, frame_id, 0, 30)
        support_win60 = window_rows(support_obs, frame_id, 0, 60)
        hard_win60 = window_rows(hard_obs, frame_id, 0, 60)

        row: Dict[str, object] = {
            "frame_id": frame_id,
            "timestamp": f"{timestamp:.6f}",
            "promoted_pairs": promoted,
            "support_num_kfs": support_frame.get("num_keyframes", ""),
            "support_num_mps": support_frame.get("num_mappoints", ""),
            "hard_num_kfs": hard_frame.get("num_keyframes", ""),
            "hard_num_mps": hard_frame.get("num_mappoints", ""),
            "support_inliers": support_frame.get("inlier_map_matches_after_pose", ""),
            "hard_inliers": hard_frame.get("inlier_map_matches_after_pose", ""),
            "support_mask_ratio": support_frame.get("mask_ratio", ""),
            "hard_mask_ratio": hard_frame.get("mask_ratio", ""),
            "support_se3_error_m": f"{float(support_align['se3_error'][support_idx]):.6f}",
            "hard_se3_error_m": f"{float(hard_align['se3_error'][hard_idx]):.6f}",
            "support_sim3_error_m": f"{float(support_align['sim3_error'][support_idx]):.6f}",
            "hard_sim3_error_m": f"{float(hard_align['sim3_error'][hard_idx]):.6f}",
            "support_cum_est_gt_ratio": safe_ratio(
                float(support_align["est_cum"][support_idx]),
                float(support_align["gt_cum"][support_idx]),
            ),
            "hard_cum_est_gt_ratio": safe_ratio(
                float(hard_align["est_cum"][hard_idx]),
                float(hard_align["gt_cum"][hard_idx]),
            ),
            "support_next30_est_gt_ratio": segment_path_ratio(support_est, support_gt, support_idx, support_idx + 30),
            "hard_next30_est_gt_ratio": segment_path_ratio(hard_est, hard_gt, hard_idx, hard_idx + 30),
            "support_next60_est_gt_ratio": segment_path_ratio(support_est, support_gt, support_idx, support_idx + 60),
            "hard_next60_est_gt_ratio": segment_path_ratio(hard_est, hard_gt, hard_idx, hard_idx + 60),
            "support_mean_inliers_next30": mean_numeric(support_win30, "inlier_map_matches_after_pose"),
            "hard_mean_inliers_next30": mean_numeric(hard_win30, "inlier_map_matches_after_pose"),
            "support_mean_mask_next30": mean_numeric(support_win30, "mask_ratio"),
            "hard_mean_mask_next30": mean_numeric(hard_win30, "mask_ratio"),
            "lm_quality_raw_support_sum": event.get("lm_quality_raw_support_sum", ""),
            "lm_quality_found_support_sum": event.get("lm_quality_found_support_sum", ""),
            "lm_quality_frame_support_sum": event.get("lm_quality_frame_support_sum", ""),
            "lm_quality_raw_depth_support_sum": event.get("lm_quality_raw_depth_support_sum", ""),
            "lm_quality_reliable_support_sum": event.get("lm_quality_reliable_support_sum", ""),
            "lm_quality_residual_support_sum": event.get("lm_quality_residual_support_sum", ""),
            "lm_quality_depth_support_sum": event.get("lm_quality_depth_support_sum", ""),
            "lm_promoted_geom_enter": event.get("lm_promoted_geom_enter", ""),
            "lm_promoted_geom_parallax": event.get("lm_promoted_geom_parallax", ""),
            "lm_promoted_geom_triangulated": event.get("lm_promoted_geom_triangulated", ""),
            "lm_promoted_geom_depth": event.get("lm_promoted_geom_depth", ""),
            "lm_promoted_geom_reproj1": event.get("lm_promoted_geom_reproj1", ""),
            "lm_promoted_geom_reproj2": event.get("lm_promoted_geom_reproj2", ""),
            "lm_promoted_geom_scale": event.get("lm_promoted_geom_scale", ""),
            "lm_promoted_geom_created": event.get("lm_promoted_geom_created", ""),
            "support_delta_mps_next30": parse_int(support_win30[-1].get("num_mappoints")) - parse_int(support_frame.get("num_mappoints")) if support_win30 else "",
            "hard_delta_mps_next30": parse_int(hard_win30[-1].get("num_mappoints")) - parse_int(hard_frame.get("num_mappoints")) if hard_win30 else "",
            "support_delta_kfs_next30": parse_int(support_win30[-1].get("num_keyframes")) - parse_int(support_frame.get("num_keyframes")) if support_win30 else "",
            "hard_delta_kfs_next30": parse_int(hard_win30[-1].get("num_keyframes")) - parse_int(hard_frame.get("num_keyframes")) if hard_win30 else "",
            "support_delta_mps_next60": parse_int(support_win60[-1].get("num_mappoints")) - parse_int(support_frame.get("num_mappoints")) if support_win60 else "",
            "hard_delta_mps_next60": parse_int(hard_win60[-1].get("num_mappoints")) - parse_int(hard_frame.get("num_mappoints")) if hard_win60 else "",
        }
        rows.append(row)
    return rows


def build_temporal_bins(
    support_events: List[Dict[str, str]],
    support_obs: Dict[int, Dict[str, str]],
    hard_obs: Dict[int, Dict[str, str]],
    support_align: Dict[str, object],
    hard_align: Dict[str, object],
    bin_size: int,
) -> List[Dict[str, object]]:
    support_times = support_align["times"]
    hard_times = hard_align["times"]
    assert isinstance(support_times, np.ndarray)
    assert isinstance(hard_times, np.ndarray)
    support_event_by_frame = rows_by_frame(support_events)
    max_frame = max(max(support_obs), max(hard_obs))

    rows: List[Dict[str, object]] = []
    for start in range(0, max_frame + 1, bin_size):
        end = min(max_frame, start + bin_size - 1)
        support_frames = [support_obs[f] for f in range(start, end + 1) if f in support_obs]
        hard_frames = [hard_obs[f] for f in range(start, end + 1) if f in hard_obs]
        if not support_frames and not hard_frames:
            continue
        support_promoted = sum(
            parse_int(support_event_by_frame.get(f, {}).get("lm_support_promoted_boundary_pairs"))
            for f in range(start, end + 1)
        )
        start_ts = parse_float((support_frames or hard_frames)[0].get("timestamp"))
        end_ts = parse_float((support_frames or hard_frames)[-1].get("timestamp"))
        s0 = find_nearest_index(support_times, start_ts)
        s1 = find_nearest_index(support_times, end_ts)
        h0 = find_nearest_index(hard_times, start_ts)
        h1 = find_nearest_index(hard_times, end_ts)
        if s1 < s0:
            s0, s1 = s1, s0
        if h1 < h0:
            h0, h1 = h1, h0

        support_se3 = support_align["se3_error"]
        hard_se3 = hard_align["se3_error"]
        support_gt = support_align["gt_xyz"]
        support_est = support_align["est_xyz"]
        hard_gt = hard_align["gt_xyz"]
        hard_est = hard_align["est_xyz"]
        assert isinstance(support_se3, np.ndarray)
        assert isinstance(hard_se3, np.ndarray)
        assert isinstance(support_gt, np.ndarray)
        assert isinstance(support_est, np.ndarray)
        assert isinstance(hard_gt, np.ndarray)
        assert isinstance(hard_est, np.ndarray)

        rows.append(
            {
                "frame_start": start,
                "frame_end": end,
                "support_promoted_pairs": support_promoted,
                "support_mean_mask": mean_numeric(support_frames, "mask_ratio"),
                "hard_mean_mask": mean_numeric(hard_frames, "mask_ratio"),
                "support_mean_inliers": mean_numeric(support_frames, "inlier_map_matches_after_pose"),
                "hard_mean_inliers": mean_numeric(hard_frames, "inlier_map_matches_after_pose"),
                "support_mean_mps": mean_numeric(support_frames, "num_mappoints"),
                "hard_mean_mps": mean_numeric(hard_frames, "num_mappoints"),
                "support_se3_rmse_m": f"{float(math.sqrt(np.mean(support_se3[s0:s1+1] ** 2))):.6f}",
                "hard_se3_rmse_m": f"{float(math.sqrt(np.mean(hard_se3[h0:h1+1] ** 2))):.6f}",
                "support_segment_est_gt_ratio": segment_path_ratio(support_est, support_gt, s0, s1),
                "hard_segment_est_gt_ratio": segment_path_ratio(hard_est, hard_gt, h0, h1),
            }
        )
    return rows


def render_summary(
    context_rows: List[Dict[str, object]],
    bin_rows: List[Dict[str, object]],
    support_align: Dict[str, object],
    hard_align: Dict[str, object],
) -> str:
    lines = [
        "WR-PY support-quality LM-only admission effect audit",
        f"support_sim3_scale={float(support_align['sim3_scale']):.9f}",
        f"hard_sim3_scale={float(hard_align['sim3_scale']):.9f}",
        f"promoted_event_frames={','.join(str(r['frame_id']) for r in context_rows)}",
        f"promoted_pairs_total={sum(parse_int(r['promoted_pairs']) for r in context_rows)}",
        "",
        "Promoted frame context:",
    ]
    for row in context_rows:
        lines.append(
            "frame={frame_id} promoted={promoted_pairs} "
            "support_se3={support_se3_error_m} hard_se3={hard_se3_error_m} "
            "support_cum_ratio={support_cum_est_gt_ratio} hard_cum_ratio={hard_cum_est_gt_ratio} "
            "support_next30_ratio={support_next30_est_gt_ratio} hard_next30_ratio={hard_next30_est_gt_ratio} "
            "support_inliers={support_inliers} hard_inliers={hard_inliers}".format(**row)
        )
        if row.get("lm_promoted_geom_enter") not in ("", None):
            lines.append(
                "  geom funnel enter/parallax/triangulated/depth/reproj1/reproj2/scale/created="
                "{lm_promoted_geom_enter}/{lm_promoted_geom_parallax}/"
                "{lm_promoted_geom_triangulated}/{lm_promoted_geom_depth}/"
                "{lm_promoted_geom_reproj1}/{lm_promoted_geom_reproj2}/"
                "{lm_promoted_geom_scale}/{lm_promoted_geom_created}".format(**row)
            )
    lines.append("")
    lines.append("Temporal bins with promoted events:")
    for row in bin_rows:
        if parse_int(row["support_promoted_pairs"]) <= 0:
            continue
        lines.append(
            "frames={frame_start}-{frame_end} promoted={support_promoted_pairs} "
            "support_rmse={support_se3_rmse_m} hard_rmse={hard_se3_rmse_m} "
            "support_ratio={support_segment_est_gt_ratio} hard_ratio={hard_segment_est_gt_ratio}".format(**row)
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--support-run-dir", type=Path, required=True)
    parser.add_argument("--hard-run-dir", type=Path, required=True)
    parser.add_argument("--support-events", type=Path, required=True)
    parser.add_argument("--support-case", help="Filter support event CSV to one case name.")
    parser.add_argument("--max-diff", type=float, default=0.03)
    parser.add_argument("--bin-size", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    support_events = read_csv_rows(args.support_events)
    if args.support_case:
        support_events = [
            row for row in support_events if row.get("case") == args.support_case
        ]
        if not support_events:
            raise SystemExit(f"No support event rows matched case={args.support_case}")
    support_obs = rows_by_frame(read_csv_rows(args.support_run_dir / "observability_frame_stats.csv"))
    hard_obs = rows_by_frame(read_csv_rows(args.hard_run_dir / "observability_frame_stats.csv"))
    support_align = load_run_alignment(args.support_run_dir, args.ground_truth, args.max_diff)
    hard_align = load_run_alignment(args.hard_run_dir, args.ground_truth, args.max_diff)

    context_rows = build_promoted_context(
        support_events,
        support_obs,
        hard_obs,
        support_align,
        hard_align,
    )
    bin_rows = build_temporal_bins(
        support_events,
        support_obs,
        hard_obs,
        support_align,
        hard_align,
        args.bin_size,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    context_fields = list(context_rows[0].keys()) if context_rows else []
    bin_fields = list(bin_rows[0].keys()) if bin_rows else []
    write_csv(args.out_dir / "wrpy_supportq_promoted_context.csv", context_rows, context_fields)
    write_csv(args.out_dir / "wrpy_supportq_temporal_bins.csv", bin_rows, bin_fields)
    summary = render_summary(context_rows, bin_rows, support_align, hard_align)
    (args.out_dir / "wrpy_supportq_admission_effect_summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
