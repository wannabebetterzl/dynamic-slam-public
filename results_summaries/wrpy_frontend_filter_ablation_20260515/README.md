# wrpy frontend image-level filtering ablation

Date: 2026-05-15

Run root: `/home/lj/dynamic-slam-public/runs/wrpy_frontend_filter_ablation_20260515_171051`

Protocol:

- Backend runner: `scripts/run_backend_rgbd.sh`
- Profile: `hybrid_sequential_semantic_only`
- `DSLAM_PASS_MASK_ARG=0`
- `ORB_SLAM3_MASK_MODE=off`
- `STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=0`
- This is an image-level RGB/depth filtering ablation, not a D2MA map-admission-only run.

## Results

| case | repeat | ATE SE3 | ATE Sim3 | scale | RPEt | RPER |
|---|---|---:|---:|---:|---:|---:|
| A raw RGB + raw depth | r1 | 0.988688 | 0.160767 | 0.081302 | 0.026849 | 0.638549 |
| B filtered RGB + raw depth | r1 | 0.515822 | 0.151128 | 0.171753 | 0.033311 | 0.748424 |
| C raw RGB + filtered depth | r1 | 0.355289 | 0.135478 | 0.271246 | 0.023897 | 0.582881 |
| D filtered RGB + filtered depth | r1 | 0.262230 | 0.124620 | 0.366184 | 0.027767 | 0.650743 |
| D filtered RGB + filtered depth | r2 | 0.259914 | 0.123078 | 0.370479 | 0.025121 | 0.606166 |

For D, repeat mean:

- ATE SE3: `0.261072 +/- 0.001158`
- ATE Sim3: `0.123849 +/- 0.000771`
- Scale: `0.368331 +/- 0.002148`

## Interpretation

- RGB-only filtering is positive on wrpy: SE3 improves from `0.988688` to `0.515822`.
- Depth-only filtering is stronger: SE3 improves from `0.988688` to `0.355289`.
- Full RGB+depth filtering is strongest and beats the current V9 backend-only precision result: D r2 SE3 `0.259914` vs V9 SE3 `0.285803`.
- D also slightly improves Sim3 over V9: D r2 `0.123078` vs V9 `0.124701`.
- Therefore, wrpy's remaining error is not only a backend map-admission problem. Dynamic-region image/depth contamination before ORB feature/depth usage is a major upstream cause.

## Residual Comparison With V9

Using D r2 vs V9:

- Global SE3 delta: `-0.025889`
- Global Sim3 delta: `-0.001623`
- Scale delta: `+0.029217`

Important segment deltas, D r2 minus V9:

| segment | SE3 delta | Sim3 delta | note |
|---|---:|---:|---|
| 100-149 | -0.100883 | +0.001841 | D greatly reduces SE3 bias but does not improve Sim3 here. |
| 150-199 | -0.100519 | +0.005973 | D fixes much of the worst SE3 segment but keeps a Sim3 cost. |
| 200-249 | -0.071793 | -0.008848 | D improves both SE3 and Sim3. |
| 300-349 | -0.042591 | -0.011122 | D improves both SE3 and Sim3. |
| 550-599 | -0.014021 | +0.006153 | D improves SE3 but not Sim3. |

## Files

- `wrpy_frontend_filter_ablation_summary.csv`
- `wrpy_frontend_filter_ablation_raw.csv`
- `wrpy_frontend_filter_ablation_repeat_summary.csv`
- `residual_compare/`
