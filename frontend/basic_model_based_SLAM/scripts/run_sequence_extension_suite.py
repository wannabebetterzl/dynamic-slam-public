#!/usr/bin/env python3
# coding=utf-8

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime


DEFAULT_ORB_EXEC = "/home/lj/tools/slam_deps/ORB_SLAM3/Examples/RGB-D/rgbd_tum"
DEFAULT_ORB_VOCAB = "/home/lj/tools/slam_deps/ORB_SLAM3/Vocabulary/ORBvoc.txt"
DEFAULT_ORB_CONFIG = "/home/lj/tools/slam_deps/ORB_SLAM3/Examples/RGB-D/TUM3.yaml"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_command(cmd):
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def read_metrics(summary_path):
    data = load_json(summary_path)
    metrics = data.get("trajectory_metrics", {})
    return {
        "exported_frames": int(data.get("exported_frames", 0)),
        "mean_runtime_ms": float(data.get("mean_runtime_ms", 0.0)),
        "mean_mask_ratio": float(data.get("mean_mask_ratio", 0.0)),
        "mean_filtered_detections": float(data.get("mean_filtered_detections", 0.0)),
        "mean_dynamic_memory": float(data.get("mean_dynamic_memory", 0.0)),
        "matched_poses": int(metrics.get("matched_poses", 0)),
        "trajectory_coverage": float(metrics.get("trajectory_coverage", 0.0)),
        "ate_rmse_m": float(metrics.get("ate_rmse_m", 0.0)),
        "rpe_rmse_m": float(metrics.get("rpe_rmse_m", 0.0)),
    }


def normalize_name(name):
    return name.replace("-", "_").replace(" ", "_")


def main():
    parser = argparse.ArgumentParser(
        description="Run the static-baseline and TUM-sitting extension benchmarks for foundation-enhanced RGB-D SLAM."
    )
    parser.add_argument("--project-root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    parser.add_argument("--output-root", default="", help="Root folder for benchmark outputs. Defaults to experiments/E12_sequence_extension_<date>.")
    parser.add_argument("--orb-exec", default=DEFAULT_ORB_EXEC)
    parser.add_argument("--orb-vocab", default=DEFAULT_ORB_VOCAB)
    parser.add_argument("--orb-config", default=DEFAULT_ORB_CONFIG)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--eval-max-diff", type=float, default=0.03)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument(
        "--only-sequence",
        action="append",
        default=[],
        help="Optional sequence tag filter. Repeatable. Available: bonn_static, tum_sitting_xyz, tum_sitting_static.",
    )
    parser.add_argument(
        "--only-method",
        action="append",
        default=[],
        help="Optional method filter. Repeatable. Available: raw, semantic_all_delete, person_v2_no_dynamic_memory, person_v2_dynamic_memory.",
    )
    args = parser.parse_args()

    project_root = os.path.abspath(args.project_root)
    today = datetime.now().strftime("%Y%m%d")
    output_root = args.output_root or os.path.join(project_root, "experiments", f"E12_sequence_extension_{today}")
    ensure_dir(output_root)

    benchmark_script = os.path.join(project_root, "scripts", "run_rgbd_slam_benchmark.py")
    config_root = os.path.join(project_root, "config")
    method_specs = {
        "raw": {
            "mode": "raw",
            "config": "",
        },
        "semantic_all_delete": {
            "mode": "filtered",
            "config": os.path.join(config_root, "world_sam_pipeline_semantic_all_delete_local.json"),
        },
        "person_v2_no_dynamic_memory": {
            "mode": "filtered",
            "config": os.path.join(config_root, "world_sam_pipeline_foundation_panoptic_person_v2_no_dynamic_memory_local.json"),
        },
        "person_v2_dynamic_memory": {
            "mode": "filtered",
            "config": os.path.join(config_root, "world_sam_pipeline_foundation_panoptic_person_v2_local.json"),
        },
    }
    sequence_specs = [
        {
            "tag": "bonn_static",
            "scene_type": "static_control",
            "dataset_id": "bonn_rgbd_static",
            "sequence_name": "rgbd_bonn_static",
            "sequence_root": os.path.join(project_root, "datasets", "bonn_rgbd_dynamic", "static", "rgbd_bonn_static"),
            "methods": ["raw", "person_v2_no_dynamic_memory", "person_v2_dynamic_memory"],
        },
        {
            "tag": "tum_sitting_xyz",
            "scene_type": "mild_dynamic_xyz",
            "dataset_id": "tum_rgbd_freiburg3_sitting_xyz",
            "sequence_name": "rgbd_dataset_freiburg3_sitting_xyz",
            "sequence_root": os.path.join(project_root, "datasets", "tum_rgbd", "freiburg3_sitting_xyz", "rgbd_dataset_freiburg3_sitting_xyz"),
            "methods": ["raw", "semantic_all_delete", "person_v2_no_dynamic_memory", "person_v2_dynamic_memory"],
        },
        {
            "tag": "tum_sitting_static",
            "scene_type": "mild_dynamic_static_cam",
            "dataset_id": "tum_rgbd_freiburg3_sitting_static",
            "sequence_name": "rgbd_dataset_freiburg3_sitting_static",
            "sequence_root": os.path.join(project_root, "datasets", "tum_rgbd", "freiburg3_sitting_static", "rgbd_dataset_freiburg3_sitting_static"),
            "methods": ["raw", "semantic_all_delete", "person_v2_no_dynamic_memory", "person_v2_dynamic_memory"],
        },
    ]

    only_sequences = {normalize_name(item) for item in args.only_sequence}
    only_methods = {normalize_name(item) for item in args.only_method}
    if only_sequences:
        sequence_specs = [item for item in sequence_specs if normalize_name(item["tag"]) in only_sequences]
        if not sequence_specs:
            raise RuntimeError("No matching sequence specs remain after --only-sequence filtering.")

    if only_methods:
        for sequence in sequence_specs:
            sequence["methods"] = [
                method_name for method_name in sequence["methods"] if normalize_name(method_name) in only_methods
            ]
        sequence_specs = [item for item in sequence_specs if item["methods"]]
        if not sequence_specs:
            raise RuntimeError("No matching method specs remain after --only-method filtering.")

    missing_roots = [item["sequence_root"] for item in sequence_specs if not os.path.isdir(item["sequence_root"])]
    if missing_roots:
        raise FileNotFoundError(
            "The following sequence roots are missing. Download the datasets first:\n" + "\n".join(missing_roots)
        )

    rows = []
    for sequence in sequence_specs:
        sequence_output_root = os.path.join(output_root, sequence["tag"])
        ensure_dir(sequence_output_root)
        raw_ate = None

        for method_name in sequence["methods"]:
            method = method_specs[method_name]
            study_output_dir = os.path.join(sequence_output_root, method_name)
            summary_path = os.path.join(study_output_dir, "benchmark_summary.json")
            if not (args.reuse_existing and os.path.isfile(summary_path)):
                cmd = [
                    sys.executable,
                    benchmark_script,
                    "--sequence-root",
                    sequence["sequence_root"],
                    "--output-dir",
                    study_output_dir,
                    "--filter-mode",
                    method["mode"],
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
                if method["mode"] == "filtered":
                    cmd.extend(["--config", method["config"]])
                else:
                    cmd.extend(["--raw-export-mode", "symlink"])
                run_command(cmd)

            metrics = read_metrics(summary_path)
            if method_name == "raw":
                raw_ate = metrics["ate_rmse_m"]

            rows.append(
                {
                    "suite": "sequence_extension",
                    "sequence_tag": sequence["tag"],
                    "scene_type": sequence["scene_type"],
                    "dataset_id": sequence["dataset_id"],
                    "sequence_name": sequence["sequence_name"],
                    "method": method_name,
                    "config": method["config"],
                    "output_dir": study_output_dir,
                    **metrics,
                    "ate_delta_vs_raw_m": (
                        metrics["ate_rmse_m"] - raw_ate if raw_ate is not None else 0.0
                    ),
                    "ate_improvement_vs_raw_pct": (
                        ((raw_ate - metrics["ate_rmse_m"]) / raw_ate) * 100.0 if raw_ate and raw_ate > 0 else 0.0
                    ),
                }
            )

    csv_path = os.path.join(output_root, f"sequence_extension_summary_{today}.csv")
    json_path = os.path.join(output_root, f"sequence_extension_summary_{today}.json")
    fieldnames = [
        "suite",
        "sequence_tag",
        "scene_type",
        "dataset_id",
        "sequence_name",
        "method",
        "config",
        "output_dir",
        "exported_frames",
        "mean_runtime_ms",
        "mean_mask_ratio",
        "mean_filtered_detections",
        "mean_dynamic_memory",
        "matched_poses",
        "trajectory_coverage",
        "ate_rmse_m",
        "rpe_rmse_m",
        "ate_delta_vs_raw_m",
        "ate_improvement_vs_raw_pct",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
