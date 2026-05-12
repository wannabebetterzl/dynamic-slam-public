#!/usr/bin/env python3
# coding=utf-8

import argparse
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


def run_command(cmd):
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def read_metrics(summary_path):
    data = load_json(summary_path)
    metrics = data.get("trajectory_metrics", {})
    return {
        "mean_runtime_ms": float(data.get("mean_runtime_ms", 0.0)),
        "mean_mask_ratio": float(data.get("mean_mask_ratio", 0.0)),
        "ate_rmse_m": float(metrics.get("ate_rmse_m", 0.0)),
        "rpe_rmse_m": float(metrics.get("rpe_rmse_m", 0.0)),
        "trajectory_coverage": float(metrics.get("trajectory_coverage", 0.0)),
    }


def main():
    parser = argparse.ArgumentParser(description="Run open-vocab vs closed-set detector ablation on RGB-D SLAM.")
    parser.add_argument("--sequence-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--project-root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    parser.add_argument("--orb-exec", default=DEFAULT_ORB_EXEC)
    parser.add_argument("--orb-vocab", default=DEFAULT_ORB_VOCAB)
    parser.add_argument("--orb-config", default=DEFAULT_ORB_CONFIG)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--eval-max-diff", type=float, default=0.03)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.output_root)
    benchmark_script = os.path.join(args.project_root, "scripts", "run_rgbd_slam_benchmark.py")
    studies = [
        {
            "label": "raw",
            "mode": "raw",
            "config": "",
            "output_dir": os.path.join(args.output_root, "raw"),
        },
        {
            "label": "open_vocab_world",
            "mode": "filtered",
            "config": os.path.join(args.project_root, "config", "world_sam_pipeline_foundation_panoptic_person_v2_local.json"),
            "output_dir": os.path.join(args.output_root, "open_vocab_world"),
        },
        {
            "label": "closed_set_yolov8n",
            "mode": "filtered",
            "config": os.path.join(args.project_root, "config", "world_sam_pipeline_closed_set_person_yolov8n_local.json"),
            "output_dir": os.path.join(args.output_root, "closed_set_yolov8n"),
        },
    ]

    rows = []
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
        rows.append(
            {
                "label": study["label"],
                "config": study["config"],
                "output_dir": study["output_dir"],
                **metrics,
            }
        )

    csv_path = os.path.join(args.output_root, "open_vocab_vs_closed_set_summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "config",
                "output_dir",
                "mean_runtime_ms",
                "mean_mask_ratio",
                "ate_rmse_m",
                "rpe_rmse_m",
                "trajectory_coverage",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(args.output_root, "open_vocab_vs_closed_set_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
