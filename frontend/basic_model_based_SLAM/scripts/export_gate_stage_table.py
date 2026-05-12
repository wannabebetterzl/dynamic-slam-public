#!/usr/bin/env python3
# coding=utf-8

import argparse
import csv
import json
import os


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Export gate-stage summary table from a benchmark_summary.json file.")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    data = load_json(args.summary_json)
    gate_summary = data.get("gate_stage_summary", {})
    rows = gate_summary.get("stages", [])
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stage",
                "instances",
                "filtered_instances",
                "kept_instances",
                "instance_ratio",
                "filtered_ratio_within_stage",
                "filtered_ratio_overall",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
