# State-Aware Admission V6 Summary

This snapshot tests whether dynamic near-boundary map admission should be gated by online system state:

- tracking pressure: low pose-optimization inliers
- keyframe pressure: short keyframe frame gap
- scale pressure: abnormal keyframe translation step against EWMA
- local BA pressure: low previous local-BA edges per map point

All runs keep the D2MA side-channel isolation protocol: masks are used only as map-admission provenance, while RGB/depth image-level filtering, dynamic feature deletion, and pose-level semantic paths remain disabled.

## Results

| dataset | method | ATE SE3 | ATE Sim3 | scale | final KFs | final MPs | score candidates/accepted/created | state candidates/allowed/rejected |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| walking_rpy | V4 same-build | 0.285427 | 0.126155 | 0.339917 | 366 | 6056 | 500/57/40 | 0/0/0 |
| walking_rpy | V6 min_need=1 | 0.285427 | 0.126155 | 0.339917 | 366 | 6056 | 500/57/40 | 57/57/0 |
| walking_rpy | V6 min_need=2 | 0.337632 | 0.135054 | 0.283961 | 386 | 6280 | 543/23/16 | 80/23/57 |
| walking_xyz | V4 same-build | 0.018035 | 0.016368 | 0.975397 | 174 | 2904 | 493/53/23 | 0/0/0 |
| walking_xyz | V6 min_need=1 | 0.018311 | 0.016445 | 0.973881 | 181 | 2996 | 463/35/17 | 43/35/8 |
| walking_xyz | V6 min_need=2 | 0.017816 | 0.016637 | 0.979209 | 188 | 3195 | 537/4/3 | 72/4/68 |

## Interpretation

- `min_need=1` is too loose on walking_rpy: every state-aware candidate is allowed, so the result is identical to V4.
- `min_need=2` hurts walking_rpy, especially scale, which suggests that walking_rpy is not mainly suffering from over-admission of dynamic-boundary points. It likely needs better-timed or better-located scale-supporting constraints.
- On walking_xyz, `min_need=2` gives a small SE3/scale improvement but slightly worsens Sim3/RPE and increases map size. The mechanism is sequence-dependent, not a stable universal improvement.
- The local BA pressure signal is inactive in these runs, so future BA-state gating should use residual change, outlier ratio, and accepted near-boundary point usage inside local BA rather than coarse edges-per-map-point.

## Files

- `state_aware_v6_full_summary.csv`: case-level metrics and event totals.
- `state_aware_v6_full_per_frame.csv`: per-frame event and observability alignment.
- Run root: `/home/lj/dynamic-slam-public/runs/state_aware_v6_20260514_001`
