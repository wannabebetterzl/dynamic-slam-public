#!/usr/bin/env python3
# coding=utf-8

import argparse
import csv
import os
import sys


FIELDNAMES = [
    "phase",
    "experiment_id",
    "experiment_name",
    "objective",
    "input_data",
    "methods",
    "expected_outputs",
    "pass_criteria",
    "status",
]


def load_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_rows(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def append_log(path, experiment_id, status, note):
    if not note:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{experiment_id}\t{status}\t{note}\n")


def main():
    parser = argparse.ArgumentParser(description="Update experiment status in experiment_master_table.csv.")
    parser.add_argument("--csv", default="experiments/experiment_master_table.csv", help="Path to experiment csv.")
    parser.add_argument("--log", default="experiments/experiment_status_log.tsv", help="Path to status log file.")
    parser.add_argument("--experiment-id", required=True, help="Experiment id to update.")
    parser.add_argument("--status", required=True, help="New status value.")
    parser.add_argument("--note", default="", help="Optional note written to the status log.")
    args = parser.parse_args()

    rows = load_rows(args.csv)
    updated = False
    for row in rows:
        if row["experiment_id"] == args.experiment_id:
            row["status"] = args.status
            updated = True
            break

    if not updated:
        raise KeyError(f"Experiment id not found: {args.experiment_id}")

    save_rows(args.csv, rows)
    append_log(args.log, args.experiment_id, args.status, args.note)
    print(f"updated {args.experiment_id} -> {args.status}")
    if args.note:
        print(f"note: {args.note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
