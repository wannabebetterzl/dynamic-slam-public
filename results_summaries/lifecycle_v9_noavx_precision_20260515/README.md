# Lifecycle V9 no-AVX precision summary

Date: 2026-05-15

Run root: `/home/lj/dynamic-slam-public/runs/lifecycle_v9_noavx_precision_20260515_162554`

## Why V9

V8 made the score-admitted near-boundary points lifecycle-aware, but still allowed probation points to enter Local BA before they had enough pose-use evidence. V9 keeps the V8 lifecycle and delays Local BA participation for score-admitted probation points until they are old enough and have pose-use support.

Frozen backend note: the normal rebuild caused `g2o`/`PoseOptimization` crashes even for V8. Rebuilding and running the backend from `build_noavx` restored V8 smoke stability, so this round uses the no-AVX backend as the reproducibility baseline.

## Protocol

- Method: `d2ma_lifecycle_v9_precision`
- Base: V8 lifecycle + score-based boundary admission.
- New V9 gate: `STSLAM_DYNAMIC_MAP_ADMISSION_LBA_DELAY_V9=1`
- V9 minimum age: `STSLAM_DYNAMIC_MAP_ADMISSION_LBA_DELAY_V9_MIN_AGE_KFS=2`
- V9 minimum pose-use evidence: `STSLAM_DYNAMIC_MAP_ADMISSION_LBA_DELAY_V9_MIN_POSE_USE=2`
- Diagnostics/logging are kept off except required protocol manifests.
- Repeats: r1/r2 for wrpy and wxyz.

## Results

| dataset | repeat | ATE SE3 | ATE Sim3 | scale | RPEt | RPER | repeat status |
|---|---|---:|---:|---:|---:|---:|---|
| wrpy | r1/r2 | 0.285803 | 0.124701 | 0.341261 | 0.033154 | 0.758053 | byte-identical CameraTrajectory |
| wxyz | r1/r2 | 0.016975 | 0.015439 | 0.977045 | 0.012029 | 0.378443 | byte-identical CameraTrajectory |

## Delta Against Frozen Mainline

| dataset | SE3 delta | Sim3 delta | scale delta | interpretation |
|---|---:|---:|---:|---|
| wrpy | -0.000079 | -0.003892 | +0.004797 | SE3 almost tied but finally slightly positive; Sim3 improves clearly. |
| wxyz | -0.000386 | -0.000557 | -0.000984 | Stable-sequence accuracy also improves slightly. |

## Delta Against V8

| dataset | SE3 delta | Sim3 delta | scale delta | interpretation |
|---|---:|---:|---:|---|
| wrpy | -0.001284 | -0.001559 | +0.003083 | V9 keeps V8's structural benefit and reduces the SE3 penalty. |
| wxyz | -0.000379 | -0.000670 | -0.001906 | V9 improves both SE3 and Sim3 over V8. |

## Residual Notes

- wrpy V9 vs mainline: global SE3 delta is small, but Sim3 improves by `-0.003892`; the biggest remaining SE3 penalty is still concentrated in matches `150-199` and `200-249`.
- wrpy V9 vs V8: early bins `0-149` and several later bins improve, but bins `300-349` and `550-599` get worse, suggesting V9 is not a full trajectory-bias fix.
- wxyz V9 vs mainline/V8: improvement is small but consistent, with no obvious catastrophic segment.

## Current Interpretation

V9 is the first lifecycle variant that improves both tested sequences against the frozen precision mainline while remaining bit-level reproducible. It is still not a strong ATE breakthrough on wrpy; the main unresolved issue remains local SE3 translation bias in specific trajectory segments. The next useful step is not broad admission relaxation, but segment-aware diagnosis and a pose-chain-aware correction/gating mechanism.

## Files

- Summary CSV: `lifecycle_v9_noavx_precision_summary.csv`
- Raw repeat CSV: `lifecycle_v9_noavx_precision_raw.csv`
- Repeat summary CSV: `lifecycle_v9_noavx_precision_repeat_summary.csv`
- Residual comparison: `residual_compare/`
