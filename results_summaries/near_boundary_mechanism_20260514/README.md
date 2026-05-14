# Near-Boundary Mechanism Diagnostics, 2026-05-14

Purpose: test whether D2MA-B helps `walking_rpy` by preventing static-map admission of dynamic-mask near-boundary points, rather than by generic point pruning.

## Protocol

- Dataset: `external_wrpy_rawrgb_rawdepth_mask`
- Profile: `hybrid_sequential_semantic_only`
- Radius: `STSLAM_NEAR_BOUNDARY_DIAGNOSTIC_RADIUS_PX=5`
- Isolation: `ORB_SLAM3_MASK_MODE=off`, `STSLAM_PANOPTIC_SIDE_CHANNEL_ONLY=1`, dynamic feature deletion / depth invalidation / pose-level dynamic paths disabled.
- Instrumentation: sidecar MapPoint admission table. The first in-object MapPoint diagnostic attempt perturbed the D2MA-B trajectory and is treated as discarded debugging evidence, not a paper result.

## Key Results

| case | ATE-SE3 | scale | CKF near / clean created | LM near / clean created | admission-near pose outlier | near / clean pose outlier | near / clean chi2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sidecar_wrpy_d2ma_min_diag` | 0.492112 | 0.177404 | 6488 / 53621 | 106 / 6568 | 24102 / 39261 = 0.613892 | 0.599953 / 0.436301 | 56.754682 / 34.152051 |
| `sidecar_wrpy_d2ma_support_b_r5_s18_m2_o2_diag` | 0.303604 | 0.317896 | 1703 / 53851 | 29 / 6217 | 7356 / 12136 = 0.606131 | 0.608090 / 0.420792 | 55.586048 / 31.754260 |
| `sidecar_wrpy_d2ma_b_r5_diag` | 0.274261 | 0.353072 | 0 / 54380 | 0 / 6479 | 0 / 0 | 0.626480 / 0.411044 | 58.937938 / 30.747001 |

## Interpretation

- Without boundary veto, D2MA-min admits 6488 CKF near-boundary points and 106 LM near-boundary points.
- Those accepted near-boundary MapPoints later produce 39261 pose-use edges, with 24102 outliers and weighted chi2 57.59.
- D2MA-B suppresses accepted near-boundary MapPoints to zero while preserving clean static admission.
- Near-boundary observations remain high-risk even under D2MA-B, but `admission_near_edges=0`; therefore the mechanism is best stated as preventing risky near-boundary observations from becoming persistent static-map points, not eliminating all near-boundary observations from tracking.
- Culling also supports the mechanism: in D2MA-min, near-boundary recent points have higher removed-event rate than clean points, 0.2085 vs 0.1643.
- The first support-aware probe (`d2ma_support_b_r5_s18_m2_o2`) confirms that soft admission is executable and interpretable: it promotes 1703 CKF near-boundary candidates and 47 LM boundary pairs, but the admitted near-boundary residual remains high. This is an innovation path, not yet a better method than hard D2MA-B.

## Files

- `near_boundary_diagnostics_summary.csv`
- `near_boundary_diagnostics_per_frame.csv`
- `support_near_boundary_diagnostics_summary.csv`
- `support_boundary_event_summary.csv`
