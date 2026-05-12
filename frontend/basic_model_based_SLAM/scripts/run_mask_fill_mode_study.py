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
        "ate_rmse_m": float(metrics.get("ate_rmse_m", 0.0)),
        "rpe_rmse_m": float(metrics.get("rpe_rmse_m", 0.0)),
        "trajectory_coverage": float(metrics.get("trajectory_coverage", 0.0)),
    }


def build_variant(base_config, fill_mode, max_inpaint_area_ratio):
    cfg = copy.deepcopy(base_config)
    cfg.setdefault("runtime", {})
    cfg["runtime"]["rgb_fill_mode"] = str(fill_mode)
    cfg["runtime"]["max_inpaint_area_ratio"] = float(max_inpaint_area_ratio)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Compare RGB fill modes used after dynamic mask filtering.")
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
    for label, fill_mode, max_ratio in [
        ("blur", "blur", base_config.get("runtime", {}).get("max_inpaint_area_ratio", 0.12)),
        ("inpaint_auto", "inpaint", base_config.get("runtime", {}).get("max_inpaint_area_ratio", 0.12)),
        ("inpaint_forced", "inpaint", 1.0),
    ]:
        cfg = build_variant(base_config, fill_mode, max_ratio)
        cfg_path = os.path.join(generated_config_dir, f"{label}.json")
        dump_json(cfg_path, cfg)
        studies.append(
            {
                "label": label,
                "config": cfg_path,
                "output_dir": os.path.join(args.output_root, label),
            }
        )

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
                "filtered",
                "--config",
                study["config"],
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

    csv_path = os.path.join(args.output_root, "mask_fill_mode_summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "config",
                "output_dir",
                "mean_runtime_ms",
                "mean_mask_ratio",
                "mean_filtered_detections",
                "ate_rmse_m",
                "rpe_rmse_m",
                "trajectory_coverage",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(args.output_root, "mask_fill_mode_summary.json")
    dump_json(json_path, rows)
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
