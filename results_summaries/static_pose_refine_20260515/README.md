# Static-Background Pose Refinement, 2026-05-15

This folder records the first ATE-improvement probe after the V7 lifecycle
audit. The method keeps D2MA-B r5 map admission unchanged, then performs an
extra pose-only optimization on static-background MapPoint matches after
`TrackLocalMap`.

## Files

- `static_pose_refine_summary.csv`: unified SE3/Sim3 ATE, RPE, scale, hashes,
  and protocol status for each run.
- `static_pose_refine_acceptance_summary.csv`: per-run acceptance counts and
  pose-delta statistics from `[STSLAM_STATIC_BG_POSE_REFINE]` logs.

## Runs

Run root:

`/home/lj/dynamic-slam-public/runs/static_pose_refine_20260515_140728`

## Main Observation

Loose refinement accepts almost every frame on `walking_rpy` and hurts global
ATE/scale, even though local RPE improves. A stricter gate reduces harmful
updates and gives a small positive signal on `walking_rpy`, but it still
slightly degrades SE3/Sim3 ATE on the already-good `walking_xyz` ablation-A
sequence.

The result is useful but not final: static-background refinement should not be
used as an unconditional every-frame post-optimizer. The next step is an
adaptive acceptance rule based on residual improvement and correction magnitude,
so the refinement is applied only when it repairs a measurable static-background
pose error without perturbing stable sequences.

## Key Metrics

| case | ATE SE3 | ATE Sim3 | scale | RPEt | RPER | note |
|---|---:|---:|---:|---:|---:|---|
| wrpy D2MA-B r5 baseline | 0.268075 | 0.122732 | 0.361680 | 0.027777 | 0.630599 | comparison baseline |
| wrpy static refine loose | 0.283669 | 0.128115 | 0.339250 | 0.026407 | 0.615647 | RPE improves, ATE worsens |
| wrpy static refine strict | 0.259444 | 0.119157 | 0.374878 | 0.026576 | 0.614130 | small positive signal |
| wxyz D2MA-B r5 baseline | 0.017244 | 0.015866 | 0.978001 | 0.012271 | 0.373766 | comparison baseline |
| wxyz static refine strict | 0.017901 | 0.016600 | 0.978172 | 0.012147 | 0.370648 | RPE improves, ATE slightly worsens |

## Acceptance Summary

| case | accepted / logged | mean accepted translation | mean accepted rotation |
|---|---:|---:|---:|
| smoke30 static refine | 29 / 29 | 0.002089 m | 0.041653 deg |
| wrpy loose | 873 / 908 | 0.003034 m | 0.059155 deg |
| wrpy strict | 447 / 908 | 0.000623 m | 0.013831 deg |
| wxyz strict | 702 / 858 | 0.000429 m | 0.006978 deg |

## Interpretation

This probe supports the diagnosis that the `walking_rpy` failure is not solved
by simply adding or preserving more admitted MapPoints. The local pose can be
improved in RPE terms, but the global keyframe/scale chain remains sensitive.
The ATE-improvement line should therefore move toward adaptive
tracking-and-optimization control rather than unconditional refinement.
