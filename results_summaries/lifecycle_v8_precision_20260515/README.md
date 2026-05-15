# Lifecycle V8 Precision (2026-05-15)

Run root: `/home/lj/dynamic-slam-public/runs/lifecycle_v8_precision_20260515_154535`

## Mechanism

`d2ma_lifecycle_v8_precision` keeps the V4 score-based boundary admission gate, but changes the lifecycle of newly admitted near-boundary points:

- Score-admitted points are marked even when online diagnostics are disabled.
- A point does not automatically mature at age 3 keyframes.
- It stays in probation until it has both pose-use evidence and Local BA evidence.
- It is culled if pose-use or Local BA evidence is bad, or if it reaches max probation age without enough evidence.
- `STSLAM_DYNAMIC_MAP_ADMISSION_CONSTRAINT_ROLE_COLLECT=1` is used as algorithmic state, while `CONSTRAINT_ROLE_LOG=0` keeps stdout diagnostics off.

## Precision Results

Baseline anchor:

| sequence | method | SE3 ATE | Sim3 ATE | Sim3 scale | RPEt | RPER |
|---|---|---:|---:|---:|---:|---:|
| wrpy | `d2ma_b_r5_precision_mainline` | 0.285882 | 0.128593 | 0.336465 | 0.033683 | 0.762219 |
| wxyz | `d2ma_b_r5_precision_mainline` | 0.017360 | 0.015996 | 0.978029 | 0.011978 | 0.384853 |

V8 repeat-stable result:

| sequence | method | SE3 ATE | Sim3 ATE | Sim3 scale | RPEt | RPER | repeat |
|---|---|---:|---:|---:|---:|---:|---|
| wrpy | `d2ma_lifecycle_v8_precision` | 0.287088 | 0.126260 | 0.338178 | 0.025631 | 0.612865 | r1/r2 byte-identical |
| wxyz | `d2ma_lifecycle_v8_precision` | 0.017354 | 0.016108 | 0.978950 | 0.012044 | 0.367803 | r1/r2 byte-identical |

## Interpretation

- V8 does not yet beat the wrpy SE3 anchor, but it is far better than the rejected supportq/V4 precision candidates.
- wrpy Sim3 ATE improves from `0.128593` to `0.126260`, and RPER improves from `0.762219` to `0.612865`.
- wxyz SE3 is essentially tied and slightly better (`0.017360` to `0.017354`), while Sim3 is slightly worse.
- This suggests lifecycle/BA-aware admission is a valid research direction, but the remaining wrpy error is not just scale-free structure; it is likely tied to absolute SE3 trajectory drift, keyframe pose chain, or residual translation bias.

## Offline Residual Decomposition

Residual files are in `residual_compare/`.

wrpy mainline vs V8:

- Global SE3 delta: `+0.001205586 m`.
- Global Sim3 delta: `-0.002333194 m`.
- Sim3 scale moves from `0.336464571` to `0.338178309`.
- Worst SE3 bins for V8 are match ranges `100-149`, `150-199`, and `200-249`.
- Several later bins improve under V8, including `300-349`, `500-549`, `850-899`, and `900-905`.

wxyz mainline vs V8:

- Global SE3 delta: `-0.000006264 m`.
- Global Sim3 delta: `+0.000112141 m`.
- Most changes are very small; wxyz remains essentially tied.

Working interpretation:

- V8 is not a global regression. It improves scale-aligned structure on wrpy, but introduces or fails to remove a local SE3 translation bias in the early/middle trajectory.
- The next precision iteration should preserve V8 lifecycle evidence, while restricting when admitted points can perturb the early keyframe pose chain.

## Files

- `lifecycle_v8_precision_summary.csv`: ATE/RPE/scale table.
- `lifecycle_v8_precision_repeat_check.csv`: r1/r2 hash and metric repeat check.
- `residual_compare/`: offline per-frame/bin residual comparison against `precision_mainline`.
