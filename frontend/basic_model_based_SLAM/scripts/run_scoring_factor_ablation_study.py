#!/usr/bin/env python3
# coding=utf-8

import argparse
import copy
import csv
import json
import os
import subprocess
import sys


DEFAULT_ORB_EXEC = "/home/lj/tools/slam_deps/ORB_SLAM3/Examples/RGB-D/rgbd_tum"
DEFAULT_ORB_VOCAB = "/home/lj/tools/slam_deps/ORB_SLAM3/Vocabulary/ORBvoc.txt"
DEFAULT_ORB_CONFIG = "/home/lj/tools/slam_deps/ORB_SLAM3/Examples/RGB-D/TUM3.yaml"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_command(cmd):
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def read_metrics(summary_path):
    data = load_json(summary_path)
    metrics = data.get("trajectory_metrics", {})
    return {
        "mean_runtime_ms": float(data.get("mean_runtime_ms", 0.0)),
        "mean_mask_ratio": float(data.get("mean_mask_ratio", 0.0)),
        "mean_filtered_detections": float(data.get("mean_filtered_detections", 0.0)),
        "mean_relevance": float(data.get("mean_relevance", 0.0)),
        "mean_foundation": float(data.get("mean_foundation", 0.0)),
        "ate_rmse_m": float(metrics.get("ate_rmse_m", 0.0)),
        "rpe_rmse_m": float(metrics.get("rpe_rmse_m", 0.0)),
        "trajectory_coverage": float(metrics.get("trajectory_coverage", 0.0)),
    }


def build_drop_config(base_config, group_name, factor_name):
    cfg = copy.deepcopy(base_config)
    weights = cfg[group_name]["weights"]
    if factor_name not in weights:
        raise KeyError(f"Unknown factor {factor_name} in {group_name}")
    weights[factor_name] = 0.0
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Run leave-one-factor-out ablation for task-relevance and foundation-reliability scoring.")
    parser.add_argument("--sequence-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--project-root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    parser.add_argument(
        "--base-config",
        default="",
        help="Base config path. Defaults to config/world_sam_pipeline_foundation_panoptic_person_v2_local.json.",
    )
    parser.add_argument("--orb-exec", default=DEFAULT_ORB_EXEC)
    parser.add_argument("--orb-vocab", default=DEFAULT_ORB_VOCAB)
    parser.add_argument("--orb-config", default=DEFAULT_ORB_CONFIG)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--eval-max-diff", type=float, default=0.03)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--include-raw", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.output_root)
    generated_config_dir = os.path.join(args.output_root, "generated_configs")
    ensure_dir(generated_config_dir)
    benchmark_script = os.path.join(args.project_root, "scripts", "run_rgbd_slam_benchmark.py")
    base_config_path = args.base_config or os.path.join(
        args.project_root,
        "config",
        "world_sam_pipeline_foundation_panoptic_person_v2_local.json",
    )
    base_config = load_json(base_config_path)

    studies = []
    if args.include_raw:
        studies.append(
            {
                "label": "raw",
                "mode": "raw",
                "group": "raw",
                "dropped_factor": "",
                "config": "",
                "output_dir": os.path.join(args.output_root, "raw"),
            }
        )

    baseline_config_path = os.path.join(generated_config_dir, "baseline_full.json")
    dump_json(baseline_config_path, base_config)
    studies.append(
        {
            "label": "baseline_full",
            "mode": "filtered",
            "group": "baseline",
            "dropped_factor": "",
            "config": baseline_config_path,
            "output_dir": os.path.join(args.output_root, "baseline_full"),
        }
    )

    for factor_name in base_config.get("task_relevance", {}).get("weights", {}).keys():
        cfg = build_drop_config(base_config, "task_relevance", factor_name)
        cfg_path = os.path.join(generated_config_dir, f"task_drop_{factor_name}.json")
        dump_json(cfg_path, cfg)
        studies.append(
            {
                "label": f"task_drop_{factor_name}",
                "mode": "filtered",
                "group": "task_relevance",
                "dropped_factor": factor_name,
                "config": cfg_path,
                "output_dir": os.path.join(args.output_root, f"task_drop_{factor_name}"),
            }
        )

    for factor_name in base_config.get("foundation_reliability", {}).get("weights", {}).keys():
        cfg = build_drop_config(base_config, "foundation_reliability", factor_name)
        cfg_path = os.path.join(generated_config_dir, f"foundation_drop_{factor_name}.json")
        dump_json(cfg_path, cfg)
        studies.append(
            {
                "label": f"foundation_drop_{factor_name}",
                "mode": "filtered",
                "group": "foundation_reliability",
                "dropped_factor": factor_name,
                "config": cfg_path,
                "output_dir": os.path.join(args.output_root, f"foundation_drop_{factor_name}"),
            }
        )

    rows = []
    baseline_ate = None
    baseline_rpe = None
    for study in studies:
        summary_path = os.path.join(study["output_dir"], "benchmark_summary.json")
        if not (args.reuse_existing and os.path.isfile(summary_path)):
            cmd = [
                sys.executable,
                benchmark_script,
                "--sequence-root",
                args.sequence_root,
                "--output-dir",
                study["output_dir"],
                "--filter-mode",
                study["mode"],
                "--orb-exec",
                args.orb_exec,
                "--orb-vocab",
                args.orb_vocab,
                "--orb-config",
                args.orb_config,
                "--eval-max-diff",
                str(args.eval_max_diff),
            ]
            if args.max_frames > 0:
                cmd.extend(["--max-frames", str(args.max_frames)])
            if study["mode"] == "filtered":
                cmd.extend(["--config", study["config"]])
            else:
                cmd.extend(["--raw-export-mode", "symlink"])
            run_command(cmd)

        metrics = read_metrics(summary_path)
        if study["label"] == "baseline_full":
            baseline_ate = metrics["ate_rmse_m"]
            baseline_rpe = metrics["rpe_rmse_m"]
        rows.append(
            {
                "label": study["label"],
                "group": study["group"],
                "dropped_factor": study["dropped_factor"],
                "config": study["config"],
                "output_dir": study["output_dir"],
                **metrics,
            }
        )

    for row in rows:
        row["ate_delta_vs_baseline_m"] = float(row["ate_rmse_m"] - baseline_ate) if baseline_ate is not None else 0.0
        row["rpe_delta_vs_baseline_m"] = float(row["rpe_rmse_m"] - baseline_rpe) if baseline_rpe is not None else 0.0
        row["ate_change_vs_baseline_pct"] = (
            float((row["ate_rmse_m"] - baseline_ate) / baseline_ate * 100.0) if baseline_ate not in (None, 0.0) else 0.0
        )

    csv_path = os.path.join(args.output_root, "scoring_factor_ablation_summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "group",
                "dropped_factor",
                "config",
                "output_dir",
                "mean_runtime_ms",
                "mean_mask_ratio",
                "mean_filtered_detections",
                "mean_relevance",
                "mean_foundation",
                "ate_rmse_m",
                "rpe_rmse_m",
                "trajectory_coverage",
                "ate_delta_vs_baseline_m",
                "rpe_delta_vs_baseline_m",
                "ate_change_vs_baseline_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(args.output_root, "scoring_factor_ablation_summary.json")
    dump_json(json_path, rows)
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
