#!/usr/bin/env python3
# coding=utf-8

import argparse
import csv
import json
import os
import shutil
import subprocess
import time

import cv2
import numpy as np

from rflysim_slam_nav.world_sam_pipeline import WorldSamFilterPipeline


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def accumulate_dict_counts(dst, src):
    for key, value in (src or {}).items():
        dst[str(key)] = int(dst.get(str(key), 0)) + int(value)
    return dst


def aggregate_gate_stage_summary(frame_stats):
    total_counts = {}
    filtered_counts = {}
    kept_counts = {}
    for item in frame_stats:
        accumulate_dict_counts(total_counts, item.get("gate_stage_counts_total", {}))
        accumulate_dict_counts(filtered_counts, item.get("gate_stage_counts_filtered", {}))
        accumulate_dict_counts(kept_counts, item.get("gate_stage_counts_kept", {}))
    gate_order = sorted(set(total_counts) | set(filtered_counts) | set(kept_counts))
    stage_rows = []
    total_instances = int(sum(total_counts.values()))
    for stage in gate_order:
        stage_total = int(total_counts.get(stage, 0))
        stage_filtered = int(filtered_counts.get(stage, 0))
        stage_kept = int(kept_counts.get(stage, 0))
        stage_rows.append(
            {
                "stage": stage,
                "instances": stage_total,
                "filtered_instances": stage_filtered,
                "kept_instances": stage_kept,
                "instance_ratio": float(stage_total / total_instances) if total_instances > 0 else 0.0,
                "filtered_ratio_within_stage": float(stage_filtered / stage_total) if stage_total > 0 else 0.0,
                "filtered_ratio_overall": float(stage_filtered / total_instances) if total_instances > 0 else 0.0,
            }
        )
    return {
        "total_instances": total_instances,
        "stages": stage_rows,
        "total_counts": total_counts,
        "filtered_counts": filtered_counts,
        "kept_counts": kept_counts,
    }


def load_index(path):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            timestamp_str, rel_path = line.split(maxsplit=1)
            entries.append(
                {
                    "timestamp": float(timestamp_str),
                    "timestamp_str": timestamp_str,
                    "rel_path": rel_path,
                }
            )
    return entries


def associate_rgb_depth(rgb_entries, depth_entries, max_diff):
    depth_times = np.array([item["timestamp"] for item in depth_entries], dtype=np.float64)
    pairs = []
    for rgb_entry in rgb_entries:
        idx = int(np.argmin(np.abs(depth_times - rgb_entry["timestamp"])))
        depth_entry = depth_entries[idx]
        if abs(depth_entry["timestamp"] - rgb_entry["timestamp"]) <= max_diff:
            pairs.append(
                {
                    "rgb_time": rgb_entry["timestamp"],
                    "rgb_time_str": rgb_entry["timestamp_str"],
                    "rgb_rel": rgb_entry["rel_path"],
                    "depth_time": depth_entry["timestamp"],
                    "depth_time_str": depth_entry["timestamp_str"],
                    "depth_rel": depth_entry["rel_path"],
                }
            )
    return pairs


def write_index(path, title, entries):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n")
        f.write("# timestamp filename\n")
        for item in entries:
            f.write(f"{item['timestamp_str']} {item['rel_path']}\n")


def write_associations(path, pairs):
    with open(path, "w", encoding="utf-8") as f:
        for item in pairs:
            f.write(
                f"{item['rgb_time_str']} {item['rgb_rel']} {item['depth_time_str']} {item['depth_rel']}\n"
            )


def write_instance_metadata(path, instance_records):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("# track_id x1 y1 x2 y2 dynamic_score temporal_consistency geometry_dynamic_score filter_out\n")
        for item in instance_records or []:
            f.write(
                f"{int(item.get('track_id', -1))} "
                f"{int(item.get('x1', 0))} "
                f"{int(item.get('y1', 0))} "
                f"{int(item.get('x2', 0))} "
                f"{int(item.get('y2', 0))} "
                f"{float(item.get('backend_dynamic_score', 0.0)):.6f} "
                f"{float(item.get('backend_temporal_consistency', 0.0)):.6f} "
                f"{float(item.get('geometry_dynamic_score', 0.0)):.6f} "
                f"{1 if item.get('filter_out', False) else 0}\n"
            )


def copy_or_link(src, dst, mode):
    ensure_dir(os.path.dirname(dst))
    if os.path.lexists(dst):
        os.remove(dst)
    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def load_tum_xyz(path):
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    return data[:, 0], data[:, 1:4]


def associate_trajectories(gt_times, gt_xyz, est_times, est_xyz, max_diff):
    pairs = []
    gt_idx = 0
    est_idx = 0
    while gt_idx < len(gt_times) and est_idx < len(est_times):
        diff = est_times[est_idx] - gt_times[gt_idx]
        if abs(diff) <= max_diff:
            pairs.append((gt_xyz[gt_idx], est_xyz[est_idx]))
            gt_idx += 1
            est_idx += 1
        elif diff > 0:
            gt_idx += 1
        else:
            est_idx += 1
    return pairs


def align_points_rigid(est, gt):
    est_mean = np.mean(est, axis=0)
    gt_mean = np.mean(gt, axis=0)
    est_centered = est - est_mean
    gt_centered = gt - gt_mean

    covariance = est_centered.T @ gt_centered / max(len(est), 1)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = gt_mean - rotation @ est_mean
    aligned = (rotation @ est.T).T + translation
    return aligned, rotation, translation


def compute_trajectory_metrics(gt_path, est_path, max_diff):
    gt_times, gt_xyz = load_tum_xyz(gt_path)
    est_times, est_xyz = load_tum_xyz(est_path)
    pairs = associate_trajectories(gt_times, gt_xyz, est_times, est_xyz, max_diff)
    if not pairs:
        raise RuntimeError("No matched trajectory pairs were found. Check timestamps or max-diff.")

    gt = np.array([item[0] for item in pairs], dtype=np.float64)
    est = np.array([item[1] for item in pairs], dtype=np.float64)
    aligned_est, rotation, translation = align_points_rigid(est, gt)
    errors = np.linalg.norm(gt - aligned_est, axis=1)

    ate_rmse = float(np.sqrt(np.mean(errors ** 2)))
    ate_mean = float(np.mean(errors))
    if len(gt) < 2:
        rpe_rmse = 0.0
    else:
        gt_delta = gt[1:] - gt[:-1]
        est_delta = aligned_est[1:] - aligned_est[:-1]
        rpe = np.linalg.norm(gt_delta - est_delta, axis=1)
        rpe_rmse = float(np.sqrt(np.mean(rpe ** 2)))

    metrics = {
        "matched_poses": int(len(pairs)),
        "ground_truth_poses": int(len(gt_times)),
        "estimated_poses": int(len(est_times)),
        "trajectory_coverage": float(len(pairs) / max(len(gt_times), 1)),
        "alignment_method": "rigid_umeyama_se3",
        "ate_rmse_m": ate_rmse,
        "ate_mean_m": ate_mean,
        "rpe_rmse_m": rpe_rmse,
        "alignment_rotation": rotation.tolist(),
        "alignment_translation": translation.tolist(),
    }
    return metrics


def discover_trajectory(run_dir):
    candidates = [
        os.path.join(run_dir, "CameraTrajectory.txt"),
        os.path.join(run_dir, "KeyFrameTrajectory.txt"),
    ]
    for path in candidates:
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return path
    return ""


def load_source_frame_stats(sequence_root):
    parent_dir = os.path.dirname(os.path.abspath(sequence_root))
    stats_path = os.path.join(parent_dir, "frame_stats.json")
    if not os.path.isfile(stats_path):
        return []
    with open(stats_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def export_sequence(args):
    rgb_entries = load_index(os.path.join(args.sequence_root, "rgb.txt"))
    depth_entries = load_index(os.path.join(args.sequence_root, "depth.txt"))
    pairs = associate_rgb_depth(rgb_entries, depth_entries, args.max_diff)
    if args.max_frames > 0:
        pairs = pairs[: args.max_frames]
    if not pairs:
        raise RuntimeError("No RGB-D pairs were found for export.")

    export_root = os.path.join(args.output_dir, "sequence")
    rgb_dir = os.path.join(export_root, "rgb")
    depth_dir = os.path.join(export_root, "depth")
    mask_dir = os.path.join(export_root, "mask")
    meta_dir = os.path.join(export_root, "meta")
    ensure_dir(rgb_dir)
    ensure_dir(depth_dir)
    ensure_dir(mask_dir)
    ensure_dir(meta_dir)

    pipeline = None
    source_frame_stats = []
    if args.filter_mode == "filtered":
        pipeline = WorldSamFilterPipeline(args.config)
    else:
        source_frame_stats = load_source_frame_stats(args.sequence_root)

    exported_rgb_entries = []
    exported_depth_entries = []
    exported_pairs = []
    frame_stats = []
    export_start = time.time()

    for pair_index, pair in enumerate(pairs, 1):
        src_rgb = os.path.join(args.sequence_root, pair["rgb_rel"])
        src_depth = os.path.join(args.sequence_root, pair["depth_rel"])
        rgb_name = os.path.basename(pair["rgb_rel"])
        depth_name = os.path.basename(pair["depth_rel"])
        dst_rgb_rel = os.path.join("rgb", rgb_name)
        dst_depth_rel = os.path.join("depth", depth_name)
        dst_mask_rel = os.path.join("mask", rgb_name)
        dst_rgb = os.path.join(export_root, dst_rgb_rel)
        dst_depth = os.path.join(export_root, dst_depth_rel)
        dst_mask = os.path.join(export_root, dst_mask_rel)
        dst_meta = os.path.join(meta_dir, rgb_name + ".txt")

        if args.filter_mode == "raw":
            copy_or_link(src_rgb, dst_rgb, args.raw_export_mode)
            copy_or_link(src_depth, dst_depth, args.raw_export_mode)
            stats = dict(source_frame_stats[pair_index - 1]) if pair_index - 1 < len(source_frame_stats) else {
                "frame_index": pair_index,
                "rgb_timestamp": pair["rgb_time"],
                "depth_timestamp": pair["depth_time"],
                "detections": 0,
                "filtered_detections": 0,
                "mask_ratio": 0.0,
                "mean_relevance": 0.0,
                "mean_foundation": 0.0,
                "mean_motion": 0.0,
                "mean_tube_motion": 0.0,
                "mean_track_confirmation": 0.0,
                "mean_dynamic_memory": 0.0,
                "scene_dynamic_context": 0.0,
                "temporal_window_length": 0,
                "temporal_prompt_records": 0,
                "temporal_prompt_frames": 0,
                "temporal_oldest_prompt_age": 0,
                "temporal_latest_prompt_age": 0,
                "temporal_propagation_span_frames": 0,
                "temporal_history_max_frames": 0,
                "runtime_ms": 0.0,
                "mode": "raw_passthrough",
                "rgb_path": pair["rgb_rel"],
                "depth_path": pair["depth_rel"],
            }
            stats["frame_index"] = pair_index
            stats["rgb_timestamp"] = pair["rgb_time"]
            stats["depth_timestamp"] = pair["depth_time"]
            stats["rgb_path"] = pair["rgb_rel"]
            stats["depth_path"] = pair["depth_rel"]
            stats.setdefault("runtime_ms", 0.0)
            stats.setdefault("mode", "raw_passthrough_frontend_metadata")
            write_instance_metadata(dst_meta, stats.get("instance_records", []))
        else:
            image = cv2.imread(src_rgb, cv2.IMREAD_COLOR)
            depth = cv2.imread(src_depth, cv2.IMREAD_UNCHANGED)
            if image is None or depth is None:
                continue
            result = pipeline.process(image, depth_mm=depth)
            ok_rgb = cv2.imwrite(dst_rgb, result["filtered_rgb"])
            ok_depth = cv2.imwrite(dst_depth, result["filtered_depth"])
            ok_mask = cv2.imwrite(dst_mask, result["mask"])
            if not ok_rgb or not ok_depth or not ok_mask:
                raise RuntimeError(f"Failed to save filtered RGB-D frame: {dst_rgb}, {dst_depth}")
            stats = dict(result["stats"])
            stats["rgb_timestamp"] = pair["rgb_time"]
            stats["depth_timestamp"] = pair["depth_time"]
            stats["rgb_path"] = pair["rgb_rel"]
            stats["depth_path"] = pair["depth_rel"]
            write_instance_metadata(dst_meta, stats.get("instance_records", []))

        exported_rgb_entries.append({"timestamp_str": pair["rgb_time_str"], "rel_path": dst_rgb_rel})
        exported_depth_entries.append({"timestamp_str": pair["depth_time_str"], "rel_path": dst_depth_rel})
        exported_pairs.append(
            {
                "rgb_time_str": pair["rgb_time_str"],
                "rgb_rel": dst_rgb_rel,
                "depth_time_str": pair["depth_time_str"],
                "depth_rel": dst_depth_rel,
            }
        )
        frame_stats.append(stats)

    if not frame_stats:
        raise RuntimeError("No frames were exported successfully.")

    write_index(os.path.join(export_root, "rgb.txt"), "color images", exported_rgb_entries)
    write_index(os.path.join(export_root, "depth.txt"), "depth maps", exported_depth_entries)
    write_associations(os.path.join(export_root, "associations.txt"), exported_pairs)

    gt_src = os.path.join(args.sequence_root, "groundtruth.txt")
    gt_dst = os.path.join(export_root, "groundtruth.txt")
    if os.path.isfile(gt_src):
        try:
            shutil.copy2(gt_src, gt_dst)
        except PermissionError:
            shutil.copyfile(gt_src, gt_dst)

    summary = {
        "sequence_root": args.sequence_root,
        "export_root": export_root,
        "filter_mode": args.filter_mode,
        "config_path": args.config if args.filter_mode == "filtered" else "",
        "exported_frames": len(frame_stats),
        "export_runtime_sec": float(time.time() - export_start),
        "mean_runtime_ms": float(np.mean([item["runtime_ms"] for item in frame_stats])),
        "mean_mask_ratio": float(np.mean([item["mask_ratio"] for item in frame_stats])),
        "mean_filtered_detections": float(np.mean([item.get("filtered_detections", 0) for item in frame_stats])),
        "mean_relevance": float(np.mean([item.get("mean_relevance", 0.0) for item in frame_stats])),
        "mean_foundation": float(np.mean([item.get("mean_foundation", 0.0) for item in frame_stats])),
        "mean_motion": float(np.mean([item.get("mean_motion", 0.0) for item in frame_stats])),
        "mean_geometry_dynamic": float(np.mean([item.get("mean_geometry_dynamic", 0.0) for item in frame_stats])),
        "mean_tube_motion": float(np.mean([item.get("mean_tube_motion", 0.0) for item in frame_stats])),
        "mean_track_confirmation": float(np.mean([item.get("mean_track_confirmation", 0.0) for item in frame_stats])),
        "mean_dynamic_memory": float(np.mean([item.get("mean_dynamic_memory", 0.0) for item in frame_stats])),
        "mean_scene_dynamic_context": float(np.mean([item.get("scene_dynamic_context", 0.0) for item in frame_stats])),
        "mean_temporal_window_length": float(np.mean([item.get("temporal_window_length", 0.0) for item in frame_stats])),
        "max_temporal_window_length": int(np.max([item.get("temporal_window_length", 0) for item in frame_stats])),
        "mean_temporal_prompt_frames": float(np.mean([item.get("temporal_prompt_frames", 0.0) for item in frame_stats])),
        "max_temporal_prompt_frames": int(np.max([item.get("temporal_prompt_frames", 0) for item in frame_stats])),
        "mean_temporal_propagation_span_frames": float(np.mean([item.get("temporal_propagation_span_frames", 0.0) for item in frame_stats])),
        "max_temporal_propagation_span_frames": int(np.max([item.get("temporal_propagation_span_frames", 0) for item in frame_stats])),
        "gate_stage_summary": aggregate_gate_stage_summary(frame_stats),
    }

    with open(os.path.join(args.output_dir, "frame_stats.json"), "w", encoding="utf-8") as f:
        json.dump(frame_stats, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, "frame_stats.csv"), "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "frame_index",
            "rgb_timestamp",
            "depth_timestamp",
            "detections",
            "filtered_detections",
            "mask_ratio",
            "mean_relevance",
            "mean_foundation",
            "mean_motion",
            "mean_geometry_dynamic",
            "mean_tube_motion",
            "mean_track_confirmation",
            "mean_dynamic_memory",
            "scene_dynamic_context",
            "temporal_window_length",
            "temporal_prompt_records",
            "temporal_prompt_frames",
            "temporal_oldest_prompt_age",
            "temporal_latest_prompt_age",
            "temporal_propagation_span_frames",
            "temporal_history_max_frames",
            "mask_schedule_stage",
            "runtime_ms",
            "mode",
            "rgb_path",
            "depth_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in frame_stats:
            writer.writerow({key: item.get(key, "") for key in fieldnames})

    return summary


def run_orb_slam3(args, export_root):
    run_dir = os.path.join(args.output_dir, "orb_run")
    ensure_dir(run_dir)
    stats_csv = os.path.join(run_dir, "orb_frame_stats.csv")
    cmd = [args.orb_exec, args.orb_vocab, args.orb_config, export_root, os.path.join(export_root, "associations.txt"), stats_csv]
    start_time = time.time()
    proc = subprocess.run(
        cmd,
        cwd=run_dir,
        text=True,
        capture_output=True,
        timeout=args.orb_timeout,
        check=False,
    )
    elapsed = float(time.time() - start_time)
    trajectory_path = discover_trajectory(run_dir)
    result = {
        "command": cmd,
        "returncode": int(proc.returncode),
        "runtime_sec": elapsed,
        "stdout_tail": proc.stdout[-4000:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-4000:] if proc.stderr else "",
        "trajectory_path": trajectory_path,
        "stats_csv": stats_csv if os.path.isfile(stats_csv) else "",
    }
    with open(os.path.join(args.output_dir, "orb_run_stdout.log"), "w", encoding="utf-8") as f:
        f.write(proc.stdout or "")
    with open(os.path.join(args.output_dir, "orb_run_stderr.log"), "w", encoding="utf-8") as f:
        f.write(proc.stderr or "")
    return result


def main():
    parser = argparse.ArgumentParser(description="Export filtered RGB-D sequences and optionally run ORB-SLAM3 benchmarking.")
    parser.add_argument("--sequence-root", required=True, help="Dataset sequence root containing rgb.txt, depth.txt and groundtruth.txt.")
    parser.add_argument("--output-dir", required=True, help="Benchmark output directory.")
    parser.add_argument("--filter-mode", choices=["raw", "filtered"], default="filtered", help="Whether to export raw frames or filtered frames.")
    parser.add_argument("--config", default="", help="Pipeline config path. Required when --filter-mode=filtered.")
    parser.add_argument("--max-diff", type=float, default=0.04, help="Maximum timestamp difference when pairing RGB and depth.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame cap for smoke tests; 0 means full sequence.")
    parser.add_argument("--raw-export-mode", choices=["symlink", "hardlink", "copy"], default="symlink", help="How to materialize raw baseline frames.")
    parser.add_argument("--orb-exec", default="", help="ORB-SLAM3 RGB-D executable path, for example rgbd_tum.")
    parser.add_argument("--orb-vocab", default="", help="ORB-SLAM3 ORBvoc.txt path.")
    parser.add_argument("--orb-config", default="", help="ORB-SLAM3 RGB-D config yaml path.")
    parser.add_argument("--orb-timeout", type=int, default=1800, help="Timeout in seconds for the ORB-SLAM3 process.")
    parser.add_argument("--eval-max-diff", type=float, default=0.03, help="Maximum timestamp difference used in ATE/RPE evaluation.")
    args = parser.parse_args()

    if args.filter_mode == "filtered" and not args.config:
        raise RuntimeError("--config is required when --filter-mode=filtered.")

    args.sequence_root = os.path.abspath(args.sequence_root)
    args.output_dir = os.path.abspath(args.output_dir)
    if args.config:
        args.config = os.path.abspath(args.config)
    if args.orb_exec:
        args.orb_exec = os.path.abspath(args.orb_exec)
    if args.orb_vocab:
        args.orb_vocab = os.path.abspath(args.orb_vocab)
    if args.orb_config:
        args.orb_config = os.path.abspath(args.orb_config)

    ensure_dir(args.output_dir)
    benchmark_summary = export_sequence(args)

    export_root = benchmark_summary["export_root"]
    gt_path = os.path.join(export_root, "groundtruth.txt")
    benchmark_summary["orb_slam3"] = {
        "attempted": False,
        "status": "not_requested",
        "trajectory_path": "",
    }

    ready_for_orb = all([args.orb_exec, args.orb_vocab, args.orb_config])
    if ready_for_orb:
        benchmark_summary["orb_slam3"]["attempted"] = True
        orb_result = run_orb_slam3(args, export_root)
        benchmark_summary["orb_slam3"].update(orb_result)
        if orb_result["trajectory_path"] and os.path.isfile(gt_path):
            try:
                metrics = compute_trajectory_metrics(gt_path, orb_result["trajectory_path"], args.eval_max_diff)
                benchmark_summary["orb_slam3"]["status"] = "evaluated"
                benchmark_summary["trajectory_metrics"] = metrics
            except Exception as exc:
                benchmark_summary["orb_slam3"]["status"] = "trajectory_found_but_evaluation_failed"
                benchmark_summary["orb_slam3"]["evaluation_error"] = str(exc)
        else:
            benchmark_summary["orb_slam3"]["status"] = "run_finished_without_trajectory"
    else:
        benchmark_summary["orb_slam3"]["status"] = "export_only"

    with open(os.path.join(args.output_dir, "benchmark_summary.json"), "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(benchmark_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
