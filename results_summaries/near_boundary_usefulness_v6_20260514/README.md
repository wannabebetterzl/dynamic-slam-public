# Near-Boundary Usefulness and V6 Divergence Diagnostics

Date: 2026-05-14

This directory contains mechanism diagnostics for the V6 state-aware admission study. The goal is to explain why the same state-aware map admission rule behaves differently on `walking_rpy` and `walking_xyz`.

Important distinction:

- Canonical V6 accuracy is reported in `../state_aware_v6_20260514/`.
- The near-boundary diagnostics here enable extra logging and can perturb the trajectory. Treat them as mechanism probes, not as final precision numbers.

## Core finding

V6 does not simply fail because state-aware admission is wrong. It exposes a quality-coverage tradeoff:

- On `walking_rpy`, stricter V6 gating keeps much cleaner near-boundary points, but removes too much coverage. Scale and SE3 accuracy get worse.
- On `walking_xyz`, stricter V6 gating can slightly improve canonical SE3/scale, suggesting that this sequence has enough alternative static constraints and is less coverage-limited.

This suggests that the next algorithm should not be a pure threshold gate. It should be coverage-aware and residual-usefulness-aware: retain enough spatial/scale support when the system is under tracking/scale/keyframe pressure, then quickly cull or downweight points that fail residual, found-ratio, depth-consistency, or pose-use tests.

## Key numbers

| dataset | mechanism probe | ATE SE3 | Sim3 scale | LM near/clean | admission-near pose-use outlier rate | admission-near chi2 |
|---|---:|---:|---:|---:|---:|---:|
| walking_rpy | V4 near-boundary diagnostics | 0.301582 | 0.323035 | 36/6740 | 0.538 | 51.85 |
| walking_rpy | V6 min_need=2 near-boundary diagnostics | 0.339026 | 0.279586 | 6/6604 | 0.074 | 8.85 |
| walking_xyz | V4 near-boundary diagnostics | 0.017149 | 0.978146 | 10/4820 | 0.061 | 6.11 |
| walking_xyz | V6 min_need=2 near-boundary diagnostics | 0.017701 | 0.972671 | 1/4843 | 0.364 | 62.76 |

Interpretation:

- `walking_rpy`: V6 improves retained point quality, but collapses admitted near-boundary coverage.
- `walking_xyz`: V6 almost removes near-boundary admission; the remaining few admitted-near points are not necessarily better, but the canonical trajectory is less harmed.

## Files

- `near_boundary_usefulness_v6_summary.csv`: per-run near-boundary admission, culling, and pose-use metrics.
- `near_boundary_usefulness_v6_per_frame.csv`: per-frame diagnostics from near-boundary mechanism runs.
- `near_boundary_usefulness_v6_map_events_summary.csv`: parsed map admission event summary.
- `near_boundary_usefulness_v6_map_events_per_frame.csv`: per-frame map admission events.
- `wrpy_v4_vs_v6_need2_bins.csv`: frame-bin divergence audit for `walking_rpy`.
- `wrpy_v4_vs_v6_need2_state_events.csv`: state-aware admission events aligned to `walking_rpy` bins.
- `wrpy_v4_vs_v6_need2_summary.txt`: human-readable divergence summary for `walking_rpy`.
- `wxyz_v4_vs_v6_need2_bins.csv`: frame-bin divergence audit for `walking_xyz`.
- `wxyz_v4_vs_v6_need2_state_events.csv`: state-aware admission events aligned to `walking_xyz` bins.
- `wxyz_v4_vs_v6_need2_summary.txt`: human-readable divergence summary for `walking_xyz`.

## Recommended next step

Use V6 as a mechanism baseline and develop V7 as `coverage-aware residual-usefulness admission`:

- Detect when the system lacks local geometric support using tracking inliers, keyframe pressure, short-term scale/step anomalies, and local BA residual/outlier signals.
- Allow limited near-boundary map admission only when it fills a coverage gap.
- Require admitted points to prove usefulness through short-horizon residual history, found ratio, depth consistency, and pose optimization inlier contribution.
