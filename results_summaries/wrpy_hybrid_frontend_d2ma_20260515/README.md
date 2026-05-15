# wrpy hybrid frontend + D2MA backend summary

Date: 2026-05-15

Run root: `/home/lj/dynamic-slam-public/runs/wrpy_hybrid_frontend_d2ma_20260515_173605`

## Protocol

- Dataset: `ablation_ei_D_filtered_rgb_filtered_depth_wrpy`
- Frontend input: filtered RGB + filtered depth from the early-intervention ablation dataset.
- Backend runner: `scripts/run_d2ma_sidechannel_isolated.sh`
- Profile: `hybrid_sequential_semantic_only`
- Backend protocol validation: all runs passed side-channel D2MA protocol validation.
- Important scope note: this is a hybrid system experiment, not a D2MA-only map-admission experiment. The RGB/depth stream is already image-level filtered before ORB-SLAM3 consumes it.

## Results

| case | repeats | ATE SE3 | ATE Sim3 | scale | RPEt | RPER | note |
|---|---:|---:|---:|---:|---:|---:|---|
| D-only frontend filtered | 2 | 0.259914 | 0.123078 | 0.370479 | 0.025121 | 0.606166 | Best repeat from frontend-only ablation. |
| D + precision mainline | 1 | 0.234436 | 0.122839 | 0.403273 | 0.022361 | 0.543398 | Backend mainline already benefits from clean RGB/depth. |
| D + score V4 precision | 1 | 0.226254 | 0.119362 | 0.417970 | 0.021959 | 0.541080 | Positive but below lifecycle variants. |
| D + support quality precision | 1 | 0.234521 | 0.117669 | 0.407416 | 0.025421 | 0.581785 | Sim3 positive, SE3 not competitive. |
| D + static pose residual gate | 1 | 0.228114 | 0.122904 | 0.412417 | 0.022760 | 0.534941 | Not best SE3. |
| D + V8 lifecycle | 1 | 0.218673 | 0.118895 | 0.429967 | 0.022960 | 0.563023 | Strong positive step. |
| D + V9 default | 2 | 0.210117 | 0.115587 | 0.445901 | 0.022965 | 0.536265 | Stable repeat, strong hybrid baseline. |
| D + V9 min_pose_use=3 | 1 | 0.218489 | 0.119302 | 0.429993 | 0.020930 | 0.518752 | Best RPE/RPER, worse SE3. |
| D + V9 min_pose_use=1 | 2 | 0.204827 | 0.116033 | 0.454861 | 0.022505 | 0.537827 | Current best wrpy SE3, byte-identical repeat. |
| D + V9 min_pose_use=1, min_age_kfs=1 | 1 | 0.205976 | 0.115253 | 0.453177 | 0.022654 | 0.543854 | Sim3 slightly better, SE3 slightly worse. |
| D + V9 min_pose_use=1, score_min_total=0.90 | 1 | 0.204827 | 0.116033 | 0.454861 | 0.022505 | 0.537827 | Byte-identical to pose1; score threshold inactive/covered here. |

## Best Candidate

Current best wrpy ATE-SE3 is:

- `D_filtered_rgb_depth_plus_v9_pose1`
- ATE SE3: `0.20482686964522398`
- ATE Sim3: `0.11603301389390526`
- Sim3 scale: `0.454860836631174`
- Repeat status: r1/r2 byte-identical `CameraTrajectory.txt` and keyframe trajectory hashes.

Improvement:

- Versus backend-only V9 (`0.285803`): `28.33%` SE3 reduction.
- Versus frozen precision mainline (`0.285882`): `28.35%` SE3 reduction.
- Versus D-only frontend filtered repeat (`0.259914`): `21.19%` SE3 reduction.
- Versus raw RGB + raw depth frontend baseline (`0.988688`): `79.28%` SE3 reduction.

## Interpretation

- The wrpy bottleneck is not solved by backend map admission alone. Image-level RGB/depth sanitation gives a large upstream correction, especially by reducing the early `150-299` SE3 pose-chain bias identified in the segment analysis.
- Backend lifecycle control remains useful after frontend filtering. D-only reaches about `0.260`, while D + V9 default reaches `0.210`, and D + V9 `min_pose_use=1` reaches `0.205`.
- `min_pose_use=3` reduces local RPE but worsens global SE3, so stricter delayed BA is not monotonic.
- `min_age_kfs=1` slightly improves Sim3 but worsens SE3, suggesting too-early admission still introduces rigid pose-chain bias.
- Lowering `score_min_total` to `0.90` does not change the trajectory under `min_pose_use=1`, indicating that the active bottleneck is not the aggregate score threshold in this filtered-input regime.

## Files

- `wrpy_hybrid_frontend_d2ma_summary.csv`
- `wrpy_hybrid_frontend_d2ma_raw.csv`
- `wrpy_hybrid_frontend_d2ma_repeat_summary.csv`
- `wrpy_hybrid_frontend_d2ma_pose1_hashes.txt`
