#!/usr/bin/env python3
# coding=utf-8
"""
Weight sensitivity analysis for the foundation-model filtering front end.

For each scoring formula (task relevance and foundation reliability), this script
perturbs the dominant weight by +/- 30% while renormalizing the remaining weights,
runs the full export-only pipeline on freiburg3_walking_xyz, and collects per-frame
gate stage statistics.  When ORB-SLAM3 paths are provided it also runs end-to-end
trajectory evaluation to produce ATE RMSE numbers for Table 5 in the paper.

Additionally, the script collects gate stage distribution statistics from the
default-weight run for Table 6 in the paper.

Usage (export-only, no ORB-SLAM3):
    python scripts/run_weight_sensitivity_study.py

Usage (with ORB-SLAM3 end-to-end evaluation):
    python scripts/run_weight_sensitivity_study.py \
        --orb-exec /path/to/rgbd_tum \
        --orb-vocab /path/to/ORBvoc.txt \
        --orb-config /path/to/TUM3.yaml
"""

import argparse
import copy
import json
import os
import sys
import time

import cv2
import numpy as np

# ensure project modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from rflysim_slam_nav.world_sam_pipeline import WorldSamFilterPipeline, load_pipeline_config

# reuse helpers from the main benchmark script
from run_rgbd_slam_benchmark import (
    aggregate_gate_stage_summary,
    associate_rgb_depth,
    compute_trajectory_metrics,
    ensure_dir,
    load_index,
    run_orb_slam3,
    write_associations,
    write_index,
)


def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Define perturbation experiments
# ---------------------------------------------------------------------------

PERTURBATIONS = [
    # (experiment_name, config_section, weight_key, factor)
    ("default", None, None, 1.0),
    ("alpha1_semantic_plus30", "task_relevance", "semantic", 1.30),
    ("alpha1_semantic_minus30", "task_relevance", "semantic", 0.70),
    ("alpha3_center_plus30", "task_relevance", "center", 1.30),
    ("alpha3_center_minus30", "task_relevance", "center", 0.70),
    ("beta1_det_conf_plus30", "foundation_reliability", "detector_confidence", 1.30),
    ("beta1_det_conf_minus30", "foundation_reliability", "detector_confidence", 0.70),
    ("beta2_seg_quality_plus30", "foundation_reliability", "segment_quality", 1.30),
    ("beta2_seg_quality_minus30", "foundation_reliability", "segment_quality", 0.70),
]


def apply_perturbation(base_config, section, weight_key, factor):
    """Perturb a single weight by `factor` and renormalize the rest."""
    config = copy.deepcopy(base_config)
    if section is None:
        return config  # default, no perturbation

    weights = config[section]["weights"]
    original_value = float(weights[weight_key])
    new_value = original_value * factor

    # compute remaining budget
    remaining_budget = 1.0 - new_value
    old_remaining = sum(float(v) for k, v in weights.items() if k != weight_key)
    if old_remaining <= 0:
        old_remaining = 1.0

    scale = remaining_budget / old_remaining
    for k in weights:
        if k == weight_key:
            weights[k] = round(new_value, 6)
        else:
            weights[k] = round(float(weights[k]) * scale, 6)

    return config


# ---------------------------------------------------------------------------
# Run a single experiment
# ---------------------------------------------------------------------------

def run_single_experiment(name, config, sequence_root, output_base, orb_args=None):
    """Run one perturbation experiment: export filtered sequence + optional SLAM."""
    output_dir = os.path.join(output_base, name)
    ensure_dir(output_dir)

    # load RGB-D pairs
    rgb_entries = load_index(os.path.join(sequence_root, "rgb.txt"))
    depth_entries = load_index(os.path.join(sequence_root, "depth.txt"))
    pairs = associate_rgb_depth(rgb_entries, depth_entries, max_diff=0.04)
    if not pairs:
        raise RuntimeError(f"No RGB-D pairs found in {sequence_root}")

    # write perturbed config to a temp file so the pipeline constructor can load it
    import tempfile
    config_tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", dir=output_dir, delete=False, encoding="utf-8"
    )
    json.dump(config, config_tmp, indent=2, ensure_ascii=False)
    config_tmp.close()
    pipeline = WorldSamFilterPipeline(config_path=config_tmp.name)

    export_root = os.path.join(output_dir, "sequence")
    rgb_dir = os.path.join(export_root, "rgb")
    depth_dir = os.path.join(export_root, "depth")
    ensure_dir(rgb_dir)
    ensure_dir(depth_dir)

    exported_rgb = []
    exported_depth = []
    exported_pairs = []
    frame_stats = []
    t0 = time.time()

    for idx, pair in enumerate(pairs, 1):
        src_rgb = os.path.join(sequence_root, pair["rgb_rel"])
        src_depth = os.path.join(sequence_root, pair["depth_rel"])
        rgb_name = os.path.basename(pair["rgb_rel"])
        depth_name = os.path.basename(pair["depth_rel"])
        dst_rgb_rel = os.path.join("rgb", rgb_name)
        dst_depth_rel = os.path.join("depth", depth_name)
        dst_rgb = os.path.join(export_root, dst_rgb_rel)
        dst_depth = os.path.join(export_root, dst_depth_rel)

        image = cv2.imread(src_rgb, cv2.IMREAD_COLOR)
        depth = cv2.imread(src_depth, cv2.IMREAD_UNCHANGED)
        if image is None or depth is None:
            continue

        result = pipeline.process(image, depth_mm=depth)
        cv2.imwrite(dst_rgb, result["filtered_rgb"])
        cv2.imwrite(dst_depth, result["filtered_depth"])

        stats = dict(result["stats"])
        stats["rgb_timestamp"] = pair["rgb_time"]
        stats["depth_timestamp"] = pair["depth_time"]
        stats["rgb_path"] = pair["rgb_rel"]
        stats["depth_path"] = pair["depth_rel"]
        frame_stats.append(stats)

        exported_rgb.append({"timestamp_str": pair["rgb_time_str"], "rel_path": dst_rgb_rel})
        exported_depth.append({"timestamp_str": pair["depth_time_str"], "rel_path": dst_depth_rel})
        exported_pairs.append({
            "rgb_time_str": pair["rgb_time_str"],
            "rgb_rel": dst_rgb_rel,
            "depth_time_str": pair["depth_time_str"],
            "depth_rel": dst_depth_rel,
        })

        if idx % 50 == 0:
            print(f"  [{name}] {idx}/{len(pairs)} frames processed")

    export_elapsed = time.time() - t0
    print(f"  [{name}] export done in {export_elapsed:.1f}s ({len(frame_stats)} frames)")

    # write index files
    write_index(os.path.join(export_root, "rgb.txt"), "color images", exported_rgb)
    write_index(os.path.join(export_root, "depth.txt"), "depth maps", exported_depth)
    write_associations(os.path.join(export_root, "associations.txt"), exported_pairs)

    # copy ground truth
    gt_src = os.path.join(sequence_root, "groundtruth.txt")
    gt_dst = os.path.join(export_root, "groundtruth.txt")
    if os.path.isfile(gt_src):
        import shutil
        try:
            shutil.copy2(gt_src, gt_dst)
        except PermissionError:
            shutil.copyfile(gt_src, gt_dst)

    # gate stage analysis
    gate_summary = aggregate_gate_stage_summary(frame_stats)

    # save per-frame stats
    with open(os.path.join(output_dir, "frame_stats.json"), "w", encoding="utf-8") as f:
        json.dump(frame_stats, f, indent=2, ensure_ascii=False)

    summary = {
        "name": name,
        "exported_frames": len(frame_stats),
        "export_runtime_sec": export_elapsed,
        "mean_runtime_ms": float(np.mean([s["runtime_ms"] for s in frame_stats])) if frame_stats else 0.0,
        "mean_mask_ratio": float(np.mean([s["mask_ratio"] for s in frame_stats])) if frame_stats else 0.0,
        "gate_stage_summary": gate_summary,
        "perturbed_config_weights": {
            "task_relevance": config.get("task_relevance", {}).get("weights", {}),
            "foundation_reliability": config.get("foundation_reliability", {}).get("weights", {}),
        },
    }

    # optional ORB-SLAM3 run
    if orb_args and orb_args.get("orb_exec"):
        import types
        fake_args = types.SimpleNamespace(
            output_dir=output_dir,
            orb_exec=orb_args["orb_exec"],
            orb_vocab=orb_args["orb_vocab"],
            orb_config=orb_args["orb_config"],
            orb_timeout=orb_args.get("orb_timeout", 1800),
            eval_max_diff=orb_args.get("eval_max_diff", 0.03),
        )
        orb_result = run_orb_slam3(fake_args, export_root)
        summary["orb_slam3"] = orb_result
        if orb_result.get("trajectory_path") and os.path.isfile(gt_dst):
            try:
                metrics = compute_trajectory_metrics(
                    gt_dst, orb_result["trajectory_path"], fake_args.eval_max_diff
                )
                summary["trajectory_metrics"] = metrics
                summary["ate_rmse"] = metrics["ate_rmse_m"]
                print(f"  [{name}] ATE RMSE = {metrics['ate_rmse_m']:.6f} m")
            except Exception as exc:
                summary["evaluation_error"] = str(exc)
                print(f"  [{name}] evaluation failed: {exc}")
    else:
        summary["orb_slam3"] = {"status": "not_requested"}

    with open(os.path.join(output_dir, "experiment_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


# ---------------------------------------------------------------------------
# Multi-sequence gate stage collection (for Table 6)
# ---------------------------------------------------------------------------

GATE_SEQUENCES = {
    "w_xyz": "tum_rgbd/freiburg3_walking_xyz/rgbd_dataset_freiburg3_walking_xyz",
    "w_static": "tum_rgbd/freiburg3_walking_static/rgbd_dataset_freiburg3_walking_static",
    "p_track": "bonn_rgbd_dynamic/person_tracking/rgbd_bonn_person_tracking",
    "crowd": "bonn_rgbd_dynamic/crowd/rgbd_bonn_crowd",
}


def collect_gate_stages_all_sequences(base_config, datasets_root, output_base, orb_args=None):
    """Run default config on all four main sequences, collecting gate stage stats."""
    gate_results = {}
    for short_name, rel_path in GATE_SEQUENCES.items():
        seq_root = os.path.join(datasets_root, rel_path)
        if not os.path.isdir(seq_root):
            print(f"  [gate_stages] Skipping {short_name}: {seq_root} not found")
            continue
        print(f"\n=== Gate stage collection: {short_name} ===")
        summary = run_single_experiment(
            f"gate_stages_{short_name}",
            base_config,
            seq_root,
            output_base,
            orb_args=orb_args,
        )
        gate_results[short_name] = summary.get("gate_stage_summary", {})
    return gate_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Weight sensitivity analysis study.")
    parser.add_argument(
        "--sequence-root",
        default="",
        help="Override sequence root. Defaults to datasets/tum_rgbd/freiburg3_walking_xyz",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Defaults to experiments/weight_sensitivity_YYYYMMDD",
    )
    parser.add_argument(
        "--config",
        default="",
        help="Base pipeline config. Defaults to the person-v2 mainline config.",
    )
    parser.add_argument("--skip-gate-stages", action="store_true",
                        help="Skip multi-sequence gate stage collection.")
    # ORB-SLAM3 optional args
    parser.add_argument("--orb-exec", default="", help="ORB-SLAM3 executable path.")
    parser.add_argument("--orb-vocab", default="", help="ORB-SLAM3 vocabulary path.")
    parser.add_argument("--orb-config", default="", help="ORB-SLAM3 config yaml path.")
    parser.add_argument("--orb-timeout", type=int, default=1800)
    parser.add_argument("--eval-max-diff", type=float, default=0.03)
    args = parser.parse_args()

    root = project_root()
    datasets_root = os.path.join(root, "datasets")

    # defaults
    if not args.sequence_root:
        args.sequence_root = os.path.join(
            datasets_root, "tum_rgbd", "freiburg3_walking_xyz",
            "rgbd_dataset_freiburg3_walking_xyz"
        )
    if not args.output_dir:
        datestamp = time.strftime("%Y%m%d")
        args.output_dir = os.path.join(root, "experiments", f"weight_sensitivity_{datestamp}")
    if not args.config:
        args.config = os.path.join(
            root, "config", "world_sam_pipeline_foundation_panoptic_person_v2_local.json"
        )

    ensure_dir(args.output_dir)
    base_config = load_pipeline_config(args.config)

    orb_args = None
    if args.orb_exec:
        orb_args = {
            "orb_exec": args.orb_exec,
            "orb_vocab": args.orb_vocab,
            "orb_config": args.orb_config,
            "orb_timeout": args.orb_timeout,
            "eval_max_diff": args.eval_max_diff,
        }

    # -----------------------------------------------------------------------
    # Part 1: Weight sensitivity on walking_xyz
    # -----------------------------------------------------------------------
    all_results = []
    for exp_name, section, weight_key, factor in PERTURBATIONS:
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name}")
        print(f"  section={section}, weight={weight_key}, factor={factor}")
        print(f"{'='*60}")

        perturbed_config = apply_perturbation(base_config, section, weight_key, factor)
        summary = run_single_experiment(
            exp_name,
            perturbed_config,
            args.sequence_root,
            args.output_dir,
            orb_args=orb_args,
        )
        all_results.append(summary)

    # summary table
    print("\n" + "=" * 80)
    print("WEIGHT SENSITIVITY SUMMARY")
    print("=" * 80)
    print(f"{'Experiment':<35} {'ATE RMSE (m)':>14} {'Mean Mask %':>12} {'Runtime ms':>12}")
    print("-" * 80)
    for r in all_results:
        ate = r.get("ate_rmse", "N/A")
        ate_str = f"{ate:.6f}" if isinstance(ate, float) else str(ate)
        mask = r.get("mean_mask_ratio", 0.0)
        rt = r.get("mean_runtime_ms", 0.0)
        print(f"{r['name']:<35} {ate_str:>14} {mask*100:>11.2f}% {rt:>11.1f}")

    # -----------------------------------------------------------------------
    # Part 2: Gate stage distribution on all sequences (Table 6)
    # -----------------------------------------------------------------------
    gate_results = {}
    if not args.skip_gate_stages:
        print("\n" + "=" * 80)
        print("GATE STAGE DISTRIBUTION COLLECTION")
        print("=" * 80)
        gate_results = collect_gate_stages_all_sequences(
            base_config, datasets_root, args.output_dir, orb_args=orb_args
        )

    # -----------------------------------------------------------------------
    # Save master results
    # -----------------------------------------------------------------------
    master = {
        "sensitivity_results": all_results,
        "gate_stage_results": gate_results,
        "base_config_path": args.config,
        "sequence_root": args.sequence_root,
    }
    master_path = os.path.join(args.output_dir, "master_results.json")
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2, ensure_ascii=False)
    print(f"\nMaster results saved to {master_path}")

    # Print gate stage summary for paper
    if gate_results:
        print("\n" + "=" * 80)
        print("GATE STAGE DISTRIBUTION (for Table 6)")
        print("=" * 80)
        for seq_name, gs in gate_results.items():
            total = gs.get("total_instances", 0)
            print(f"\n{seq_name} (total instances: {total}):")
            for stage in gs.get("stages", []):
                pct = stage["instance_ratio"] * 100
                filt_pct = stage["filtered_ratio_within_stage"] * 100
                print(f"  {stage['stage']:<35} {stage['instances']:>5} ({pct:5.1f}%)  filtered: {stage['filtered_instances']:>5} ({filt_pct:5.1f}%)")


if __name__ == "__main__":
    main()
