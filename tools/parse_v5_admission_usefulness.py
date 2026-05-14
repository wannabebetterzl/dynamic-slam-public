#!/usr/bin/env python3
"""Parse V5 score-admission candidate and lifecycle diagnostics."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

EVENT_RE = re.compile(r"(\w+)=([^\s]+)")


def to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_events(run_dir: Path) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    stdout = run_dir / "stdout.log"
    candidates: List[Dict[str, str]] = []
    lifecycle: List[Dict[str, str]] = []
    summaries: List[Dict[str, str]] = []
    if not stdout.exists():
        return candidates, lifecycle, summaries

    for line in stdout.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("[STSLAM_DYNAMIC_MAP_ADMISSION_V5_"):
            continue
        tag = line.split("]", 1)[0].strip("[")
        values = dict(EVENT_RE.findall(line))
        values["tag"] = tag
        if tag == "STSLAM_DYNAMIC_MAP_ADMISSION_V5_CANDIDATE":
            candidates.append(values)
        elif tag == "STSLAM_DYNAMIC_MAP_ADMISSION_V5_LIFECYCLE":
            lifecycle.append(values)
        elif tag == "STSLAM_DYNAMIC_MAP_ADMISSION_V5_SUMMARY":
            summaries.append(values)
    return candidates, lifecycle, summaries


def write_csv(path: Path, rows: List[Dict[str, object]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in writer.fieldnames})


def summarize(
    candidates: List[Dict[str, str]],
    lifecycle: List[Dict[str, str]],
    summaries: List[Dict[str, str]],
) -> List[Dict[str, object]]:
    if summaries:
        row: Dict[str, object] = dict(summaries[-1])
        row.pop("tag", None)
        return [row]

    decision_counts = Counter(row.get("decision", "unknown") for row in candidates)
    created = [row for row in candidates if row.get("decision") == "created"]
    geom = [row for row in candidates if row.get("stage") == "geometry"]

    def mean(rows: List[Dict[str, str]], key: str) -> float:
        if not rows:
            return 0.0
        return sum(to_float(row.get(key)) for row in rows) / float(len(rows))

    lifecycle_score_recent = sum(to_float(row.get("score_recent")) for row in lifecycle)
    lifecycle_pose_edges = sum(to_float(row.get("score_pose_use_edges")) for row in lifecycle)
    lifecycle_pose_inliers = sum(to_float(row.get("score_pose_use_inliers")) for row in lifecycle)

    row: Dict[str, object] = {
        "candidate_events": len(candidates),
        "support_candidates": decision_counts.get("support_accepted", 0)
        + decision_counts.get("reject_support", 0),
        "support_accepted": decision_counts.get("support_accepted", 0),
        "reject_support": decision_counts.get("reject_support", 0),
        "geom_events": len(geom),
        "reject_parallax": decision_counts.get("reject_parallax", 0),
        "reject_triangulate": decision_counts.get("reject_triangulate", 0),
        "reject_depth": decision_counts.get("reject_depth", 0),
        "reject_reproj1": decision_counts.get("reject_reproj1", 0),
        "reject_reproj2": decision_counts.get("reject_reproj2", 0),
        "reject_scale": decision_counts.get("reject_scale", 0),
        "reject_score": decision_counts.get("reject_score", 0),
        "created": decision_counts.get("created", 0),
        "created_support_score_mean": mean(created, "support_score"),
        "created_candidate_score_mean": mean(created, "candidate_score"),
        "created_total_score_mean": mean(created, "total_score"),
        "created_reproj_ratio1_mean": mean(created, "reproj_ratio1"),
        "created_reproj_ratio2_mean": mean(created, "reproj_ratio2"),
        "created_parallax_score_mean": mean(created, "parallax_score"),
        "created_scale_score_mean": mean(created, "scale_score"),
        "lifecycle_rows": len(lifecycle),
        "lifecycle_score_recent_sum": lifecycle_score_recent,
        "lifecycle_score_prebad_sum": sum(to_float(row.get("score_prebad")) for row in lifecycle),
        "lifecycle_score_culled_found_ratio_sum": sum(
            to_float(row.get("score_culled_found_ratio")) for row in lifecycle
        ),
        "lifecycle_score_culled_low_obs_sum": sum(
            to_float(row.get("score_culled_low_obs")) for row in lifecycle
        ),
        "lifecycle_score_survived_sum": sum(to_float(row.get("score_survived")) for row in lifecycle),
        "lifecycle_score_matured_sum": sum(to_float(row.get("score_matured")) for row in lifecycle),
        "lifecycle_pose_use_edges_sum": lifecycle_pose_edges,
        "lifecycle_pose_use_inliers_sum": lifecycle_pose_inliers,
        "lifecycle_pose_use_inlier_rate": (
            lifecycle_pose_inliers / lifecycle_pose_edges if lifecycle_pose_edges > 0 else 0.0
        ),
    }
    return [row]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--candidate-out", type=Path, required=True)
    parser.add_argument("--lifecycle-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    candidates, lifecycle, summaries = read_events(args.run_dir)
    candidate_fields = [
        "tag",
        "frame",
        "stage",
        "decision",
        "current_kf",
        "neighbor_kf",
        "idx1",
        "idx2",
        "risk_current",
        "risk_neighbor",
        "raw_support",
        "found_support",
        "frame_support",
        "raw_depth_support",
        "reliable_support",
        "residual_support",
        "depth_support",
        "support_score",
        "binary_support_pass",
        "parallax_score",
        "reproj_ratio1",
        "reproj_ratio2",
        "scale_score",
        "candidate_score",
        "total_score",
        "support_candidate",
        "support_accepted",
        "reject_support",
        "geom_candidate",
        "reject_parallax",
        "reject_triangulate",
        "reject_depth",
        "reject_reproj1",
        "reject_reproj2",
        "reject_scale",
        "reject_score",
        "created",
    ]
    lifecycle_fields = [
        "tag",
        "frame",
        "current_kf",
        "recent_points",
        "remaining_recent_points",
        "score_recent",
        "score_prebad",
        "score_culled_found_ratio",
        "score_culled_low_obs",
        "score_survived",
        "score_matured",
        "score_pose_use_edges",
        "score_pose_use_inliers",
        "score_pose_use_chi2_mean",
    ]
    summary_rows = summarize(candidates, lifecycle, summaries)
    write_csv(args.candidate_out, candidates, candidate_fields)
    write_csv(args.lifecycle_out, lifecycle, lifecycle_fields)
    write_csv(args.summary_out, summary_rows, summary_rows[0].keys())
    print(
        f"candidates={len(candidates)} lifecycle={len(lifecycle)} "
        f"summaries={len(summaries)} summary={args.summary_out}"
    )


if __name__ == "__main__":
    main()
