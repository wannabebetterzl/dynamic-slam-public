# V7 Lifecycle Shadow Audit 2026-05-15

This folder freezes the P1/P2/P4 follow-up for the V7 coverage-aware diagnostic branch.

## Purpose

- P1: replace event-level `matured` ambiguity with unique score-admitted MapPoint lifecycle rows.
- P2: compare `v7_plain` and `v7_shadow` to decide whether lifecycle/constraint collection is non-intrusive.
- P4: add scale-support proxies for score-admitted points.

## Canonical vs Diagnostic Rule

`d2ma_coverage_aware_lm_only_v7_plain` is the canonical trajectory branch for ATE reporting. `d2ma_coverage_aware_lm_only_v7_shadow` is deterministic and useful for mechanism analysis, but it changes the full-sequence trajectory hash on both `walking_rpy` and `walking_xyz`. Therefore shadow ATE must not be mixed into the main performance table.

## Runs

Run root:

```text
runs/lifecycle_shadow_20260515_133013
```

Valid comparison datasets:

```text
wrpy: external_wrpy_rawrgb_rawdepth_mask
wxyz: ablation_ei_A_raw_rgb_raw_depth_wxyz
```

Invalid/not comparable run:

```text
wxyz_v7_plain_r1 used backend_maskonly_full_wxyz and is kept only as a dataset-entry caution.
```

## Key Results

| case | ATE-SE3 | ATE-Sim3 | scale | trajectory repeat |
|---|---:|---:|---:|---|
| wrpy plain r1/r2 | 0.435072 | 0.152299 | 0.197847 | byte-identical |
| wrpy shadow r1/r2 | 0.283097 | 0.127302 | 0.340826 | byte-identical |
| wxyz plain r1/r2 | 0.018251 | 0.016620 | 0.975493 | byte-identical |
| wxyz shadow r1/r2 | 0.017868 | 0.016038 | 0.974443 | byte-identical |

Interpretation:

- Plain and shadow are each fully reproducible.
- Shadow is not non-intrusive on full sequences, especially on `walking_rpy`.
- Diagnostic metrics are stable, but shadow ATE is diagnostic-only.

## Unique Lifecycle Summary

| sequence | unique points | alive | unique matured | LBA edge points | LBA edges / unique point | LBA inlier rate | local-edge rate | fixed-edge rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wrpy shadow | 38 | 5 | 9 | 12 | 97.71 | 0.936 | 0.830 | 0.170 |
| wxyz shadow | 18 | 1 | 6 | 6 | 400.72 | 0.989 | 0.968 | 0.032 |

Mechanism interpretation:

- `wrpy` score-admitted points can enter Local BA, but they form far fewer LBA edges per unique point, have lower inlier rates, and are connected much more often to fixed-camera edges.
- `wxyz` score-admitted points are fewer, but the surviving/useful ones form much denser and cleaner Local BA constraints.
- This supports a constraint-role/lifecycle explanation more strongly than a simple "points did not enter BA" explanation.

## Scale-Support Proxy

| sequence | parallax mean | baseline/ref mean | parallax mean over LBA points | baseline/ref over LBA points |
|---|---:|---:|---:|---:|
| wrpy | 3.78 deg | 0.0466 | 4.25 deg | 0.0790 |
| wxyz | 1.83 deg | 0.0297 | 2.35 deg | 0.0263 |

This does not support the narrower claim that `wrpy` fails simply because admitted points have weaker raw parallax or baseline. The safer conclusion is that `wrpy` has weaker optimization role and lifecycle stability despite having adequate geometric parallax proxies.

## Files

- `lifecycle_shadow_eval_summary.csv`: ATE, scale, RPE, and run paths for plain/shadow repeats.
- `score_admission_lifecycle_summary.csv`: unique lifecycle summary from shadow runs.
- `score_admission_lifecycle_points.csv`: per-point lifecycle and geometry rows.
- `constraint_scale_support_summary.csv`: compact wrpy-vs-wxyz scale-support/constraint-role summary.
- `constraint_scale_support_points.csv`: per-point scale-support proxy table.
- `*_protocol_validation.json`: side-channel protocol validation for each valid run.
