# wrpy pose-chain and scale-bias diagnosis

Date: 2026-05-15

Purpose: identify why wrpy still has large absolute-scale / SE3 pose-chain bias after frontend filtering and D2MA lifecycle admission.

## Compared Runs

wrpy:

- Backend-only V9: `runs/lifecycle_v9_noavx_precision_20260515_162554/wrpy_lifecycle_v9_precision_r1`
- D-only frontend filtered: `runs/wrpy_frontend_filter_ablation_20260515_171051/D_filtered_rgb_filtered_depth_wrpy_r2`
- D + V9 default: `runs/wrpy_hybrid_frontend_d2ma_20260515_173605/D_filtered_rgb_depth_plus_v9_r1`
- D + V9 pose1: `runs/wrpy_hybrid_frontend_d2ma_20260515_173605/D_filtered_rgb_depth_plus_v9_pose1_r1`
- D + V9 pose3: `runs/wrpy_hybrid_frontend_d2ma_20260515_173605/D_filtered_rgb_depth_plus_v9_pose3_r1`

wxyz contrast:

- Mainline: `runs/precision_mainline_20260515_151323/wxyz_precision_mainline_r1`
- V8: `runs/lifecycle_v8_precision_20260515_154535/wxyz_lifecycle_v8_precision_r1`
- V9: `runs/lifecycle_v9_noavx_precision_20260515_162554/wxyz_lifecycle_v9_precision_r1`

## Main Finding

The large wrpy bias is not uniform random drift. It is dominated by local over-motion in specific short-displacement dynamic segments, especially matches `150-249`, and then propagates through the pose chain.

The best current configuration, `D + V9 pose1`, improves SE3 mainly by reducing raw over-displacement in those segments. It does not fully solve the global scale/pose-chain problem.

## Key Metrics

| segment | GT chord | backend V9 SE3 | D-only SE3 | D+V9 default SE3 | D+V9 pose1 SE3 | D+V9 pose1 chord ratio | interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| global | 0.311993 | 0.285803 | 0.259914 | 0.210117 | 0.204827 | 2.371284 | Global raw motion remains over-scaled. |
| 100-149 | 0.523733 | 0.305769 | 0.204887 | 0.130127 | 0.123396 | 1.297995 | Hybrid sharply reduces early SE3/chord bias. |
| 150-199 | 0.069034 | 0.612466 | 0.511947 | 0.339353 | 0.320120 | 1.483973 | Critical short-motion segment; D-only chord ratio is 2.657967, pose1 reduces it to 1.483973. |
| 200-249 | 0.273840 | 0.327870 | 0.256077 | 0.169186 | 0.151022 | 1.995670 | Continued over-motion; pose1 improves but still nearly 2x chord. |
| 250-299 | 0.096107 | 0.091971 | 0.073502 | 0.061339 | 0.060811 | 1.346550 | Smaller residual bias. |
| 800-849 | 0.167633 | 0.435625 | 0.394736 | 0.344568 | 0.335379 | 4.830879 | Late large residual remains; likely inherited/second dynamic burst. |
| 850-899 | 0.134233 | 0.736322 | 0.728070 | 0.622732 | 0.619154 | 1.217597 | Late SE3 error remains high although chord ratio is not worst, suggesting accumulated pose-chain offset. |

## Pose1 vs Default

`D + V9 pose1` improves SE3 over `D + V9 default` most clearly in:

- `150-199`: `0.339353 -> 0.320120`
- `200-249`: `0.169186 -> 0.151022`
- `800-849`: `0.344568 -> 0.335379`

This supports the interpretation that default V9 is slightly too conservative after frontend sanitation. Waiting for `min_pose_use=2` lets the local pose chain continue with an already biased structure. Relaxing to `min_pose_use=1` admits some useful constraints earlier and reduces over-motion.

However, `pose3` has the lowest global raw path ratio (`4.812439`) but worse SE3 (`0.218489`) than pose1. Therefore, the problem is not simply total path-length overestimation. The ordering and localization of corrections inside the pose chain matters.

## wxyz Contrast

wxyz does not show the same failure pattern:

| segment | mainline SE3 | V9 SE3 | mainline chord ratio | V9 chord ratio |
|---|---:|---:|---:|---:|
| global | 0.017360 | 0.016975 | 1.040367 | 1.038173 |
| 100-149 | 0.014630 | 0.014191 | 1.017560 | 0.988034 |
| 150-199 | 0.022393 | 0.022708 | 0.925668 | 0.911293 |
| 200-249 | 0.023756 | 0.020542 | 0.973611 | 0.981844 |
| 800-849 | 0.006117 | 0.005973 | 1.024538 | 1.067048 |

wxyz has near-metric chord ratios around 1.0 and SE3 remains at centimeter level. This contrast suggests wrpy's problem is a sequence-condition failure: dynamic/occlusion-heavy short-motion windows create local over-motion and absolute pose-chain offset.

## Working Causal Chain

Current best-supported hypothesis:

1. Dynamic RGB/depth contamination creates local unstable constraints in wrpy.
2. The most damaging windows are short-GT-motion segments, especially `150-199`, where small real displacement makes any wrong feature/depth constraint dominate pose estimation.
3. Frontend filtering reduces the raw contamination but does not eliminate local over-motion.
4. Backend lifecycle V9 reduces admission damage, and `pose1` helps by adding useful constraints earlier under cleaner input.
5. Remaining error is no longer just map-point admission; it is a pose-chain topology and local scale consistency problem.

## Next Mechanism Direction

The next candidate should be pose-chain-aware and scale-consistency-aware, not only support-quality-aware:

- Detect local short-motion windows with abnormal chord/path ratio or keyframe density.
- In those windows, use stricter near-boundary admission and/or delay BA participation for risky new points.
- Conversely, when frontend sanitation has already cleaned the input and pose chain is underconstrained, allow earlier mature/useful constraints, similar to `pose1`.
- Add an online safeguard based on local estimated motion consistency, keyframe density, tracking inliers, and local BA residual, rather than a fixed global threshold.

## Files

- `hybrid_pose1_vs_default_vs_donly_bins.csv`
- `hybrid_pose1_vs_backendv9_vs_donly_bins.csv`
- `wrpy_pose_chain_bias_decomposition.csv`
- `wxyz_backend_v9_contrast_bins.csv`
- `wxyz_pose_chain_bias_contrast.csv`
