#!/usr/bin/env python3
"""Create WRPy failure-interval plots from observability and D2MA event logs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from parse_map_admission_events import parse_stdout_events, read_observability


DEFAULT_INTERVALS = (
    (48, 65, "early LM boundary"),
    (239, 290, "middle CKF/LM"),
    (574, 593, "mid-late CKF"),
    (812, 825, "late CKF/path"),
)


def parse_case(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("case must be NAME=/path/to/run_dir")
    name, path = value.split("=", 1)
    return name, Path(path)


def parse_interval(value: str) -> Tuple[int, int, str]:
    parts = value.split(":", 2)
    if len(parts) < 2:
        raise argparse.ArgumentTypeError("interval must be START:END[:LABEL]")
    start = int(parts[0])
    end = int(parts[1])
    label = parts[2] if len(parts) == 3 else f"{start}-{end}"
    return start, end, label


def to_float(value: object, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def build_timeseries(name: str, run_dir: Path) -> List[Dict[str, object]]:
    events_by_total, events_by_frame = parse_stdout_events(run_dir)
    del events_by_total
    obs_rows, _ = read_observability(run_dir)

    cumulative: Dict[str, int] = {}
    rows: List[Dict[str, object]] = []
    for obs in obs_rows:
        frame_id = int(float(obs.get("frame_id", 0)))
        events = events_by_frame.get(frame_id, {})
        for key, value in events.items():
            cumulative[key] = cumulative.get(key, 0) + int(value)

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
        for key in sorted(events):
            row[key] = events[key]
        for key in sorted(cumulative):
            row[f"cum_{key}"] = cumulative[key]
        rows.append(row)
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def summarize_intervals(
    rows: List[Dict[str, object]], intervals: Iterable[Tuple[int, int, str]]
) -> List[Dict[str, object]]:
    by_case: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case"]), []).append(row)

    summary: List[Dict[str, object]] = []
    event_keys = sorted(
        key
        for key in {k for row in rows for k in row.keys()}
        if key.startswith("ckf_") or key.startswith("lm_")
    )
    for case, case_rows in sorted(by_case.items()):
        case_rows = sorted(case_rows, key=lambda r: int(r["frame_id"]))
        for start, end, label in intervals:
            window = [r for r in case_rows if start <= int(r["frame_id"]) <= end]
            if not window:
                continue
            first = window[0]
            last = window[-1]
            row: Dict[str, object] = {
                "case": case,
                "interval": label,
                "start_frame": start,
                "end_frame": end,
                "frames": len(window),
                "path_delta_m": to_float(last.get("estimated_accum_path_m"))
                - to_float(first.get("estimated_accum_path_m")),
                "mappoint_delta": to_float(last.get("num_mappoints"))
                - to_float(first.get("num_mappoints")),
                "keyframe_delta": to_float(last.get("num_keyframes"))
                - to_float(first.get("num_keyframes")),
                "mean_inliers_after_pose": sum(
                    to_float(r.get("inlier_map_matches_after_pose"), -1.0) for r in window
                )
                / len(window),
            }
            for key in event_keys:
                row[key] = sum(int(to_float(r.get(key), 0.0)) for r in window)
            summary.append(row)
    return summary


def plot(rows: List[Dict[str, object]], intervals: Iterable[Tuple[int, int, str]], out_path: Path) -> None:
    by_case: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case"]), []).append(row)

    colors = {
        "wrpy_raw": "#6b7280",
        "wrpy_d2ma_min": "#2563eb",
        "wrpy_d2ma_b_r5": "#0f766e",
        "wrpy_samecount_nonboundary_r5": "#dc2626",
    }

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    for ax in axes:
        for start, end, label in intervals:
            ax.axvspan(start, end, color="#f59e0b", alpha=0.12)
            ax.text((start + end) / 2, 0.98, label, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8, color="#92400e")
        ax.grid(True, alpha=0.25)

    for case, case_rows in sorted(by_case.items()):
        case_rows = sorted(case_rows, key=lambda r: int(r["frame_id"]))
        x = [int(r["frame_id"]) for r in case_rows]
        color = colors.get(case)
        label = case.replace("wrpy_", "")
        axes[0].plot(x, [to_float(r.get("estimated_accum_path_m")) for r in case_rows], label=label, color=color)
        axes[1].plot(x, [to_float(r.get("num_mappoints")) for r in case_rows], label=label, color=color)
        axes[2].plot(x, [to_float(r.get("inlier_map_matches_after_pose"), -1) for r in case_rows], label=label, color=color)

    for case in ("wrpy_d2ma_b_r5", "wrpy_samecount_nonboundary_r5"):
        case_rows = sorted(by_case.get(case, []), key=lambda r: int(r["frame_id"]))
        x = [int(r["frame_id"]) for r in case_rows]
        color = colors.get(case)
        label = case.replace("wrpy_", "")
        axes[3].plot(
            x,
            [
                to_float(r.get("cum_ckf_boundary_skipped_new_candidates"))
                + to_float(r.get("cum_ckf_control_skipped_nonboundary_new_candidates"))
                for r in case_rows
            ],
            label=f"{label}: CKF boundary/control",
            color=color,
            linestyle="-",
        )
        axes[3].plot(
            x,
            [
                to_float(r.get("cum_lm_skipped_boundary_pairs"))
                + to_float(r.get("cum_lm_control_skipped_nonboundary_pairs"))
                for r in case_rows
            ],
            label=f"{label}: LM boundary/control",
            color=color,
            linestyle="--",
        )

    axes[0].set_ylabel("Estimated path (m)")
    axes[1].set_ylabel("MapPoints")
    axes[2].set_ylabel("Inliers after pose")
    axes[3].set_ylabel("Cumulative veto/control")
    axes[3].set_xlabel("Frame id")
    axes[0].set_title("walking_rpy failure-interval probe: path, map growth, inliers, and D2MA events")
    for ax in axes:
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=parse_case, required=True)
    parser.add_argument("--interval", action="append", type=parse_interval)
    parser.add_argument("--timeseries-out", type=Path, required=True)
    parser.add_argument("--interval-summary-out", type=Path, required=True)
    parser.add_argument("--plot-out", type=Path, required=True)
    args = parser.parse_args()

    intervals = tuple(args.interval) if args.interval else DEFAULT_INTERVALS
    rows: List[Dict[str, object]] = []
    for name, run_dir in args.case:
        rows.extend(build_timeseries(name, run_dir))

    fields = [
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
    ] + sorted({key for row in rows for key in row.keys()} - {
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
    })
    write_csv(args.timeseries_out, rows, fields)

    interval_rows = summarize_intervals(rows, intervals)
    interval_fields = list(interval_rows[0].keys()) if interval_rows else []
    write_csv(args.interval_summary_out, interval_rows, interval_fields)
    plot(rows, intervals, args.plot_out)

    print(f"timeseries={args.timeseries_out}")
    print(f"interval_summary={args.interval_summary_out}")
    print(f"plot={args.plot_out}")


if __name__ == "__main__":
    main()
