# Coverage-Aware Residual-Usefulness Admission V7

This snapshot tests V7 as a post-admission map-maintenance probe:

- allow limited near-boundary map-point admission when state or coverage pressure indicates missing constraints
- cap promotions per keyframe and neighbor keyframe
- place score-admitted points on probation and reject them by pose-use residual/inlier evidence or low-use culling

All runs keep the D2MA side-channel isolation protocol: masks are used only as map-admission provenance, while RGB/depth image-level filtering, dynamic feature deletion, and pose-level semantic paths remain disabled.

## Results

| dataset | method | ATE SE3 | ATE Sim3 | scale | matched | final KFs | final MPs | V7 candidates/allowed | probation residual/survived/matured | pose-use edges/inliers/chi2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| walking_rpy | V4 same-build | 0.285427 | 0.126155 | 0.339917 | 906 | 366 | 6056 | 0/0 | 0/0/0 | 0/0/0 |
| walking_rpy | V7 coverage-aware | 0.305616 | 0.131352 | 0.314618 | 906 | 387 | 6506 | 46/46 | 7/40/12 | 185/172/4.14 |
| walking_xyz | V4 same-build | 0.018035 | 0.016368 | 0.975397 | 857 | 174 | 2904 | 0/0 | 0/0/0 | 0/0/0 |
| walking_xyz | V7 coverage-aware | 0.017738 | 0.015810 | 0.973916 | 857 | 178 | 2971 | 33/32 | 4/20/147 | 94335/91076/3.88 |

## Interpretation

- V7 is functional and its coverage/probation loop is observable in logs.
- On walking_rpy, V7 changes map structure but does not extend matched coverage or fix scale; SE3 and scale regress.
- On walking_xyz, V7 gives a small SE3/Sim3 improvement, suggesting the mechanism can act as a local refinement but is not yet a universal robust dynamic-SLAM improvement.
- walking_rpy's failure is now better characterized as a constraint-graph/scale-coverage issue rather than simply low posterior quality of admitted near-boundary points.

## Files

- `coverage_aware_v7_summary.csv`: case-level metrics and event totals.
- `coverage_aware_v7_per_frame.csv`: per-frame event and observability alignment.
- Run root: `/home/lj/dynamic-slam-public/runs/coverage_aware_v7_20260514_001`
