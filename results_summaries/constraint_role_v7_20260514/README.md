# V7 Constraint-Role Diagnostics

Date: 2026-05-14

Purpose: test whether V7 score/coverage-admitted map points actually become useful optimization constraints, especially on the failing `walking_rpy` sequence.

## Runs

| dataset | repeat | ATE SE3 | ATE Sim3 | Sim3 scale | trajectory hash | note |
|---|---:|---:|---:|---:|---|---|
| `walking_rpy` | r1 | 0.407473 | 0.147121 | 0.221246 | `34f31e36...` | diagnostic logging perturbs canonical V7; use for mechanism only |
| `walking_rpy` | r2 | 0.407473 | 0.147121 | 0.221246 | `34f31e36...` | byte-identical repeat |
| `walking_xyz` | r1 | 0.018134 | 0.015641 | 0.970347 | `692339a3...` | close to canonical V7 |
| `walking_xyz` | r2 | 0.018134 | 0.015641 | 0.970347 | `692339a3...` | byte-identical repeat |

## Mechanism Evidence

The main finding is not that `walking_rpy` admitted points are absent from Local BA. They do enter Local BA. The problem is that their constraint role is much weaker than on `walking_xyz`:

| dataset | V7 allowed | created | matured | score window points | LBA edges | edges/window point | inlier rate | fixed-edge rate | mean chi2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `walking_rpy` | 57 | 28 | 10 | 464 | 5109 | 11.01 | 0.951 | 0.304 | 5.11 |
| `walking_xyz` | 56 | 23 | 206 | 343 | 13150 | 38.34 | 0.993 | 0.044 | 1.95 |

Interpretation:

- `walking_rpy` admitted points provide fewer Local BA edges per score-window point, have higher residuals, and are far more tied to fixed keyframes.
- `walking_xyz` admitted points become denser, cleaner, mostly local BA constraints and mature much more often.
- Therefore the next algorithm should not simply admit more near-boundary points. It should schedule and weight points according to constraint role: local BA leverage, residual history, depth/parallax support, and whether the point contributes to scale-sensitive coverage.

## Caveat

The constraint-role logging path perturbs `walking_rpy` relative to the non-logging canonical V7 run. The diagnostic repeats are deterministic, but their ATE should not be reported as canonical precision. Use this run as mechanism evidence only.

Files:

- `constraint_role_v7_summary.csv`
- `constraint_role_v7_derived.csv`
- `constraint_role_v7_per_frame.csv`
