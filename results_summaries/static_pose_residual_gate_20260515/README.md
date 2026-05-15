# Static Pose Residual Gate Negative Control, 2026-05-15

This folder records the follow-up to the static-background pose refinement
probe. The goal was to make frame-level pose refinement more principled by
requiring static-background reprojection residual improvement before accepting
the refined pose.

## Files

- `static_pose_residual_gate_summary.csv`: first residual-gated runs plus
  same-build D2MA-B r5 baselines.
- `static_pose_residual_pregate_summary.csv`: pre-gated residual runs that skip
  the second optimizer when the initial static residual is already low.
- `static_pose_refine_reason_counts.csv`: accepted/rejected reason counts from
  `[STSLAM_STATIC_BG_POSE_REFINE]` logs.

## Key Finding

The online static-background residual is already clean on these sequences, so
frame-level static pose refinement is not addressing the main failure mode.
More importantly, even rejected/no-op residual diagnostics perturb the full
trajectory on `walking_rpy`, which means online diagnostics/refinement must not
be mixed into the main precision table unless they are part of the final
algorithm and evaluated against a same-build baseline.

## Main Metrics

| case | ATE SE3 | ATE Sim3 | scale | accepted | interpretation |
|---|---:|---:|---:|---:|---|
| wrpy same-build D2MA-B r5 | 0.312344 | 0.132797 | 0.306858 | - | same-build baseline after residual code was added |
| wrpy residual gate | 0.272437 | 0.122379 | 0.357374 | 0 / 908 | no accepted refinements; improvement is not a valid algorithmic gain |
| wrpy pre-gate | 0.388081 | 0.144468 | 0.236354 | 0 / 908 | only 5 frames reached second optimizer; still worse |
| wrpy no-optimizer negative control | 0.346037 | 0.133977 | 0.279704 | 0 / 908 | no optimizer calls, residual/log path still perturbs timing |
| wxyz same-build D2MA-B r5 | 0.017851 | 0.016096 | 0.974942 | - | same-build baseline |
| wxyz residual gate | 0.018199 | 0.015786 | 0.970724 | 0 / 858 | mixed SE3/Sim3 change, not algorithmic gain |
| wxyz no-optimizer negative control | 0.017438 | 0.015456 | 0.973815 | 0 / 858 | no-op diagnostics can still shift trajectory slightly |

## Residual Diagnosis

From the residual-gated full logs:

- `walking_rpy`: static residual before refinement had median chi2 about `2.22`,
  p90 about `2.65`, and inlier ratio essentially `1.0`.
- `walking_xyz`: static residual before refinement had median chi2 about `1.59`,
  p90 about `2.10`, and inlier ratio essentially `1.0`.

This means the visible failure is not caused by dirty static-background
frame-level reprojection residuals. The more plausible failure site remains the
keyframe/LocalMapping/Local BA scale chain.

## Consequence

Frame-level online static-background refinement should be paused as a main ATE
improvement path. The next ATE-improvement attempt should move to a same-build,
minimal-logging branch that changes only the intended backend mechanism, likely
at keyframe admission / Local BA weighting / map-point survival rather than
per-frame post-pose refinement.
