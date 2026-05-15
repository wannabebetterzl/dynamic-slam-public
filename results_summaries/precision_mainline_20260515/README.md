# Precision Mainline Anchor, 2026-05-15

This folder freezes the current same-build precision anchor after the online
diagnostic perturbation tests. The method name is explicit:

`d2ma_b_r5_precision_mainline`

It is algorithmically the D2MA-B r5 side-channel map-admission baseline, but it
forces online diagnostics off:

- `STSLAM_OBSERVABILITY_LOG=0`
- `STSLAM_NEAR_BOUNDARY_DIAGNOSTICS=0`
- `STSLAM_DYNAMIC_MAP_ADMISSION_CONSTRAINT_ROLE_COLLECT=0`
- `STSLAM_DYNAMIC_MAP_ADMISSION_CONSTRAINT_ROLE_LOG=0`
- `STSLAM_DYNAMIC_MAP_ADMISSION_LIFECYCLE_DUMP=0`
- `STSLAM_STATIC_BACKGROUND_POSE_REFINE=0`

## Files

- `precision_mainline_summary.csv`: unified ATE/RPE/scale table.
- `precision_mainline_repeat_check.csv`: byte-level repeat check for
  `CameraTrajectory.txt` and `KeyFrameTimeline.csv`.

## Main Results

| dataset | repeat | ATE SE3 | ATE Sim3 | scale | RPEt | RPER | repeat status |
|---|---|---:|---:|---:|---:|---:|---|
| walking_rpy | r1/r2 | 0.285882 | 0.128593 | 0.336465 | 0.033683 | 0.762219 | byte-identical trajectory and keyframe timeline |
| walking_xyz ablation-A | r1/r2 | 0.017360 | 0.015996 | 0.978029 | 0.011978 | 0.384853 | byte-identical trajectory and keyframe timeline |

## Interpretation

This is the precision branch to use for future main-table comparisons in the
current build. Diagnostic branches may be useful for mechanism evidence, but
their ATE should not be mixed into the main table unless compared against a
same-build, same-wrapper precision baseline.

The next ATE-improvement attempt should be implemented as a separate explicit
method and compared against `d2ma_b_r5_precision_mainline` on the same build.
