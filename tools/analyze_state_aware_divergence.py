#!/usr/bin/env python3
"""Compare a state-aware admission run against a same-build baseline.

The script aligns trajectory, observability, and parsed map-admission events
into frame bins.  It is intended for mechanism diagnosis: where does a stricter
state gate start to change local path scale, keyframe/mappoint growth, or
tracking quality?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from evaluate_trajectory_ate import align_positions, load_tum_trajectory


TrajectoryAlignment = Dict[str, object]


def parse_case(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("case must be NAME=/path/to/run_dir")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("case name cannot be empty")
    return name, Path(path)


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


def rows_by_frame(rows: Iterable[Dict[str, str]]) -> Dict[int, Dict[str, str]]:
    return {
        parse_int(row.get("frame_id")): row
        for row in rows
        if row.get("frame_id") not in (None, "")
    }


def mean_numeric(rows: List[Dict[str, str]], key: str) -> str:
    values = [parse_float(row.get(key), math.nan) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return ""
    return f"{float(np.mean(values)):.6f}"


def sum_numeric(rows: List[Dict[str, str]], key: str) -> int:
    return sum(parse_int(row.get(key)) for row in rows)


def safe_ratio(numerator: float, denominator: float) -> str:
    if abs(denominator) <= 1e-12:
        return ""
    return f"{numerator / denominator:.6f}"


def associate_with_times(gt: Dict[str, np.ndarray], est: Dict[str, np.ndarray], max_diff: float):
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
    result = np.zeros(len(xyz), dtype=np.float64)
    if len(xyz) > 1:
        result[1:] = np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))
    return result


def load_run_alignment(run_dir: Path, ground_truth: Path, max_diff: float) -> TrajectoryAlignment:
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
        "sim3_scale": float(sim3_scale),
    }


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
        "ate_se3": se3.get("ate_rmse_m", ""),
        "ate_sim3": sim3.get("ate_rmse_m", ""),
        "scale": sim3.get("alignment_scale", ""),
        "rpet": se3.get("rpet_rmse_m", ""),
        "rper": se3.get("rper_rmse_deg", ""),
    }


def nearest_index(times: np.ndarray, timestamp: float) -> int:
    return int(np.argmin(np.abs(times - timestamp)))


def frame_window(rows_by_id: Dict[int, Dict[str, str]], start: int, end: int) -> List[Dict[str, str]]:
    return [rows_by_id[f] for f in range(start, end + 1) if f in rows_by_id]


def time_index_window(align: TrajectoryAlignment, start_ts: float, end_ts: float) -> Tuple[int, int]:
    times = align["times"]
    assert isinstance(times, np.ndarray)
    i0 = nearest_index(times, start_ts)
    i1 = nearest_index(times, end_ts)
    return (i0, i1) if i0 <= i1 else (i1, i0)


def segment_path_ratio(align: TrajectoryAlignment, i0: int, i1: int) -> str:
    est = align["est_xyz"]
    gt = align["gt_xyz"]
    assert isinstance(est, np.ndarray)
    assert isinstance(gt, np.ndarray)
    if i1 <= i0:
        return ""
    est_len = float(np.sum(np.linalg.norm(np.diff(est[i0 : i1 + 1], axis=0), axis=1)))
    gt_len = float(np.sum(np.linalg.norm(np.diff(gt[i0 : i1 + 1], axis=0), axis=1)))
    return safe_ratio(est_len, gt_len)


def cumulative_ratio_at(align: TrajectoryAlignment, idx: int) -> str:
    est_cum = align["est_cum"]
    gt_cum = align["gt_cum"]
    assert isinstance(est_cum, np.ndarray)
    assert isinstance(gt_cum, np.ndarray)
    return safe_ratio(float(est_cum[idx]), float(gt_cum[idx]))


def rmse(values: np.ndarray, i0: int, i1: int) -> str:
    if i1 < i0:
        return ""
    return f"{float(math.sqrt(np.mean(values[i0 : i1 + 1] ** 2))):.6f}"


def float_or_nan(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def ratio_delta(probe_value: object, baseline_value: object) -> str:
    p = float_or_nan(probe_value)
    b = float_or_nan(baseline_value)
    if not math.isfinite(p) or not math.isfinite(b):
        return ""
    return f"{p - b:.6f}"


def build_bin_rows(
    baseline_events: Dict[int, Dict[str, str]],
    probe_events: Dict[int, Dict[str, str]],
    baseline_obs: Dict[int, Dict[str, str]],
    probe_obs: Dict[int, Dict[str, str]],
    baseline_align: TrajectoryAlignment,
    probe_align: TrajectoryAlignment,
    bin_size: int,
) -> List[Dict[str, object]]:
    max_frame = max(max(baseline_obs), max(probe_obs), max(probe_events or {0: {}}))
    rows: List[Dict[str, object]] = []
    for start in range(0, max_frame + 1, bin_size):
        end = min(max_frame, start + bin_size - 1)
        b_obs = frame_window(baseline_obs, start, end)
        p_obs = frame_window(probe_obs, start, end)
        if not b_obs and not p_obs:
            continue
        b_events = frame_window(baseline_events, start, end)
        p_events = frame_window(probe_events, start, end)
        first = (p_obs or b_obs)[0]
        last = (p_obs or b_obs)[-1]
        start_ts = parse_float(first.get("timestamp"))
        end_ts = parse_float(last.get("timestamp"))
        b0, b1 = time_index_window(baseline_align, start_ts, end_ts)
        p0, p1 = time_index_window(probe_align, start_ts, end_ts)

        b_se3 = baseline_align["se3_error"]
        p_se3 = probe_align["se3_error"]
        b_sim3 = baseline_align["sim3_error"]
        p_sim3 = probe_align["sim3_error"]
        assert isinstance(b_se3, np.ndarray)
        assert isinstance(p_se3, np.ndarray)
        assert isinstance(b_sim3, np.ndarray)
        assert isinstance(p_sim3, np.ndarray)

        b_seg_ratio = segment_path_ratio(baseline_align, b0, b1)
        p_seg_ratio = segment_path_ratio(probe_align, p0, p1)
        b_cum_ratio = cumulative_ratio_at(baseline_align, b1)
        p_cum_ratio = cumulative_ratio_at(probe_align, p1)

        row: Dict[str, object] = {
            "frame_start": start,
            "frame_end": end,
            "timestamp_start": f"{start_ts:.6f}",
            "timestamp_end": f"{end_ts:.6f}",
            "baseline_frames": len(b_obs),
            "probe_frames": len(p_obs),
            "probe_state_candidates": sum_numeric(p_events, "lm_state_candidates"),
            "probe_state_allowed": sum_numeric(p_events, "lm_state_allowed"),
            "probe_state_rejected": sum_numeric(p_events, "lm_state_rejected"),
            "probe_tracking_pressure": sum_numeric(p_events, "lm_state_tracking_pressure"),
            "probe_keyframe_pressure": sum_numeric(p_events, "lm_state_keyframe_pressure"),
            "probe_scale_pressure": sum_numeric(p_events, "lm_state_scale_pressure"),
            "probe_lba_pressure": sum_numeric(p_events, "lm_state_lba_pressure"),
            "baseline_score_created": sum_numeric(b_events, "lm_score_created"),
            "probe_score_created": sum_numeric(p_events, "lm_score_created"),
            "baseline_mean_inliers": mean_numeric(b_obs, "inlier_map_matches_after_pose"),
            "probe_mean_inliers": mean_numeric(p_obs, "inlier_map_matches_after_pose"),
            "baseline_mean_step_m": mean_numeric(b_obs, "estimated_frame_step_m"),
            "probe_mean_step_m": mean_numeric(p_obs, "estimated_frame_step_m"),
            "baseline_mean_mappoints": mean_numeric(b_obs, "num_mappoints"),
            "probe_mean_mappoints": mean_numeric(p_obs, "num_mappoints"),
            "baseline_mean_keyframes": mean_numeric(b_obs, "num_keyframes"),
            "probe_mean_keyframes": mean_numeric(p_obs, "num_keyframes"),
            "baseline_segment_est_gt_ratio": b_seg_ratio,
            "probe_segment_est_gt_ratio": p_seg_ratio,
            "segment_ratio_delta_probe_minus_baseline": ratio_delta(p_seg_ratio, b_seg_ratio),
            "baseline_cum_est_gt_ratio_end": b_cum_ratio,
            "probe_cum_est_gt_ratio_end": p_cum_ratio,
            "cum_ratio_delta_probe_minus_baseline": ratio_delta(p_cum_ratio, b_cum_ratio),
            "baseline_se3_rmse_m": rmse(b_se3, b0, b1),
            "probe_se3_rmse_m": rmse(p_se3, p0, p1),
            "se3_rmse_delta_probe_minus_baseline": ratio_delta(rmse(p_se3, p0, p1), rmse(b_se3, b0, b1)),
            "baseline_sim3_rmse_m": rmse(b_sim3, b0, b1),
            "probe_sim3_rmse_m": rmse(p_sim3, p0, p1),
            "sim3_rmse_delta_probe_minus_baseline": ratio_delta(rmse(p_sim3, p0, p1), rmse(b_sim3, b0, b1)),
        }
        rows.append(row)
    return rows


def build_state_event_rows(
    baseline_obs: Dict[int, Dict[str, str]],
    probe_events: Dict[int, Dict[str, str]],
    probe_obs: Dict[int, Dict[str, str]],
    baseline_align: TrajectoryAlignment,
    probe_align: TrajectoryAlignment,
    window: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for frame_id, event in sorted(probe_events.items()):
        candidates = parse_int(event.get("lm_state_candidates"))
        rejected = parse_int(event.get("lm_state_rejected"))
        if candidates <= 0 and rejected <= 0:
            continue
        obs = probe_obs.get(frame_id, {})
        baseline = baseline_obs.get(frame_id, {})
        timestamp = parse_float(obs.get("timestamp") or event.get("timestamp"))
        b0, b1 = time_index_window(baseline_align, timestamp, timestamp + 1e-9)
        p0, p1 = time_index_window(probe_align, timestamp, timestamp + 1e-9)
        b_next0, b_next1 = time_index_window(
            baseline_align,
            timestamp,
            parse_float(frame_window(probe_obs, frame_id, frame_id + window)[-1].get("timestamp"), timestamp)
            if frame_window(probe_obs, frame_id, frame_id + window)
            else timestamp,
        )
        p_next0, p_next1 = time_index_window(
            probe_align,
            timestamp,
            parse_float(frame_window(probe_obs, frame_id, frame_id + window)[-1].get("timestamp"), timestamp)
            if frame_window(probe_obs, frame_id, frame_id + window)
            else timestamp,
        )
        row: Dict[str, object] = {
            "frame_id": frame_id,
            "timestamp": f"{timestamp:.6f}",
            "state_candidates": candidates,
            "state_allowed": parse_int(event.get("lm_state_allowed")),
            "state_rejected": rejected,
            "tracking_pressure": parse_int(event.get("lm_state_tracking_pressure")),
            "keyframe_pressure": parse_int(event.get("lm_state_keyframe_pressure")),
            "scale_pressure": parse_int(event.get("lm_state_scale_pressure")),
            "lba_pressure": parse_int(event.get("lm_state_lba_pressure")),
            "score_created": parse_int(event.get("lm_score_created")),
            "probe_inliers": obs.get("inlier_map_matches_after_pose", ""),
            "baseline_inliers": baseline.get("inlier_map_matches_after_pose", ""),
            "probe_mappoints": obs.get("num_mappoints", ""),
            "baseline_mappoints": baseline.get("num_mappoints", ""),
            "probe_keyframes": obs.get("num_keyframes", ""),
            "baseline_keyframes": baseline.get("num_keyframes", ""),
            "probe_step_m": obs.get("estimated_frame_step_m", ""),
            "baseline_step_m": baseline.get("estimated_frame_step_m", ""),
            "probe_cum_ratio": cumulative_ratio_at(probe_align, p0),
            "baseline_cum_ratio": cumulative_ratio_at(baseline_align, b0),
            "cum_ratio_delta_probe_minus_baseline": ratio_delta(
                cumulative_ratio_at(probe_align, p0),
                cumulative_ratio_at(baseline_align, b0),
            ),
            f"probe_next{window}_ratio": segment_path_ratio(probe_align, p_next0, p_next1),
            f"baseline_next{window}_ratio": segment_path_ratio(baseline_align, b_next0, b_next1),
            f"next{window}_ratio_delta_probe_minus_baseline": ratio_delta(
                segment_path_ratio(probe_align, p_next0, p_next1),
                segment_path_ratio(baseline_align, b_next0, b_next1),
            ),
        }
        rows.append(row)
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def top_rows(rows: List[Dict[str, object]], key: str, limit: int, reverse: bool = True) -> List[Dict[str, object]]:
    def sort_value(row: Dict[str, object]) -> float:
        value = float_or_nan(row.get(key))
        return value if math.isfinite(value) else (-math.inf if reverse else math.inf)

    return sorted(rows, key=sort_value, reverse=reverse)[:limit]


def render_summary(
    baseline_name: str,
    probe_name: str,
    baseline_run: Path,
    probe_run: Path,
    bin_rows: List[Dict[str, object]],
    event_rows: List[Dict[str, object]],
) -> str:
    baseline_eval = read_eval(baseline_run)
    probe_eval = read_eval(probe_run)
    lines = [
        "State-aware admission divergence audit",
        f"baseline={baseline_name} run={baseline_run}",
        f"probe={probe_name} run={probe_run}",
        "",
        "Global metrics:",
        f"baseline ATE_SE3={baseline_eval.get('ate_se3')} ATE_Sim3={baseline_eval.get('ate_sim3')} scale={baseline_eval.get('scale')} RPEt={baseline_eval.get('rpet')} RPER={baseline_eval.get('rper')}",
        f"probe    ATE_SE3={probe_eval.get('ate_se3')} ATE_Sim3={probe_eval.get('ate_sim3')} scale={probe_eval.get('scale')} RPEt={probe_eval.get('rpet')} RPER={probe_eval.get('rper')}",
        "",
        "State event totals:",
        f"candidates={sum(parse_int(r.get('state_candidates')) for r in event_rows)} allowed={sum(parse_int(r.get('state_allowed')) for r in event_rows)} rejected={sum(parse_int(r.get('state_rejected')) for r in event_rows)}",
        f"tracking/keyframe/scale/lba pressures={sum(parse_int(r.get('tracking_pressure')) for r in event_rows)}/{sum(parse_int(r.get('keyframe_pressure')) for r in event_rows)}/{sum(parse_int(r.get('scale_pressure')) for r in event_rows)}/{sum(parse_int(r.get('lba_pressure')) for r in event_rows)}",
        "",
        "Worst bins by probe-baseline SE3 RMSE delta:",
    ]
    for row in top_rows(bin_rows, "se3_rmse_delta_probe_minus_baseline", 5):
        lines.append(
            "frames={frame_start}-{frame_end} se3_delta={se3_rmse_delta_probe_minus_baseline} "
            "cum_ratio_delta={cum_ratio_delta_probe_minus_baseline} "
            "seg_ratio_delta={segment_ratio_delta_probe_minus_baseline} "
            "state_rejected={probe_state_rejected} probe_created={probe_score_created} baseline_created={baseline_score_created}".format(**row)
        )
    lines.append("")
    lines.append("Bins with strongest path-ratio shrinkage in probe:")
    for row in top_rows(bin_rows, "segment_ratio_delta_probe_minus_baseline", 5, reverse=False):
        lines.append(
            "frames={frame_start}-{frame_end} seg_ratio_delta={segment_ratio_delta_probe_minus_baseline} "
            "baseline_ratio={baseline_segment_est_gt_ratio} probe_ratio={probe_segment_est_gt_ratio} "
            "state_rejected={probe_state_rejected} state_allowed={probe_state_allowed}".format(**row)
        )
    lines.append("")
    lines.append("State-event frames with largest next-window ratio loss:")
    if event_rows:
        next_keys = [key for key in event_rows[0] if key.startswith("next") and key.endswith("delta_probe_minus_baseline")]
        next_key = next_keys[0] if next_keys else ""
        for row in top_rows(event_rows, next_key, 8, reverse=False):
            lines.append(
                "frame={frame_id} rejected={state_rejected}/{state_candidates} "
                "pressures=t{tracking_pressure},k{keyframe_pressure},s{scale_pressure},ba{lba_pressure} "
                "{next_key}={next_delta} cum_delta={cum_ratio_delta_probe_minus_baseline} "
                "probe_inliers={probe_inliers} baseline_inliers={baseline_inliers}".format(
                    next_key=next_key,
                    next_delta=row.get(next_key, ""),
                    **row,
                )
            )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--baseline", type=parse_case, required=True)
    parser.add_argument("--probe", type=parse_case, required=True)
    parser.add_argument("--event-csv", type=Path, required=True)
    parser.add_argument("--bin-size", type=int, default=50)
    parser.add_argument("--next-window", type=int, default=50)
    parser.add_argument("--max-diff", type=float, default=0.03)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="state_aware_divergence")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_name, baseline_run = args.baseline
    probe_name, probe_run = args.probe
    event_rows = read_csv_rows(args.event_csv)
    baseline_events = rows_by_frame(row for row in event_rows if row.get("case") == baseline_name)
    probe_events = rows_by_frame(row for row in event_rows if row.get("case") == probe_name)
    if not baseline_events:
        raise SystemExit(f"No event rows found for baseline case: {baseline_name}")
    if not probe_events:
        raise SystemExit(f"No event rows found for probe case: {probe_name}")

    baseline_obs = rows_by_frame(read_csv_rows(baseline_run / "observability_frame_stats.csv"))
    probe_obs = rows_by_frame(read_csv_rows(probe_run / "observability_frame_stats.csv"))
    baseline_align = load_run_alignment(baseline_run, args.ground_truth, args.max_diff)
    probe_align = load_run_alignment(probe_run, args.ground_truth, args.max_diff)

    bin_rows = build_bin_rows(
        baseline_events,
        probe_events,
        baseline_obs,
        probe_obs,
        baseline_align,
        probe_align,
        args.bin_size,
    )
    state_rows = build_state_event_rows(
        baseline_obs,
        probe_events,
        probe_obs,
        baseline_align,
        probe_align,
        args.next_window,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = args.out_dir / f"{args.prefix}_bins.csv"
    state_path = args.out_dir / f"{args.prefix}_state_events.csv"
    summary_path = args.out_dir / f"{args.prefix}_summary.txt"
    write_csv(bin_path, bin_rows, list(bin_rows[0].keys()) if bin_rows else [])
    write_csv(state_path, state_rows, list(state_rows[0].keys()) if state_rows else [])
    summary = render_summary(
        baseline_name,
        probe_name,
        baseline_run,
        probe_run,
        bin_rows,
        state_rows,
    )
    summary_path.write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print(f"bins={bin_path}")
    print(f"state_events={state_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
