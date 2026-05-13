# Deep Thinking Prompt

You are helping diagnose and improve a dynamic Visual SLAM research project. Please read this public repository first, especially:

Repository URL:

```text
https://github.com/wannabebetterzl/dynamic-slam-public
```

- `README.md`
- `docs/CODE_MAP.md`
- `docs/ABANDONED_ROUTES.md`
- `data/datasets.json`
- `docs/动态改进Visual SLAM实验记录.md`
- `docs/动态改进Orb SLAM3.md`
- `backend/orb_slam3_dynamic/src/Tracking.cc`
- `backend/orb_slam3_dynamic/include/Tracking.h`
- `backend/orb_slam3_dynamic/include/Frame.h`
- `backend/orb_slam3_dynamic/src/Frame.cc`
- `backend/orb_slam3_dynamic/src/LocalMapping.cc`
- `backend/orb_slam3_dynamic/src/Optimizer.cc`
- `frontend/basic_model_based_SLAM/scripts/rflysim_slam_nav/world_sam_pipeline.py`
- `frontend/basic_model_based_SLAM/scripts/run_rgbd_slam_benchmark.py`
- `scripts/run_backend_rgbd.sh`
- `scripts/run_d2ma_sidechannel_isolated.sh`
- `scripts/run_frontend_inference.sh`
- `tools/evaluate_trajectory_ate.py`
- `tools/validate_d2ma_sidechannel_protocol.py`
- `tools/summarize_backend_runs.py`
- `results_summaries/`

## Background

The project studies Visual SLAM in dynamic RGB-D / stereo scenes. The current goal is to use foundation models such as YOLOE + SAM3 as high-recall dynamic-object priors while avoiding damage to static tracking and mapping.

The important route correction is:

- The old STSLAM reproduction was abandoned as a failed route.
- The DynoSAM adapter / object frontend was removed from the active public snapshot.
- The current backend is `backend/orb_slam3_dynamic/`, originally from `/home/lj/dynamic_SLAM/stslam_backend`.
- The current frontend is `frontend/basic_model_based_SLAM/`, originally from `/home/lj/d-drive/CODEX/basic_model_based_SLAM`.
- The current local data entry point is `data/datasets.json`.

## Latest Protocol Update

The current paper-facing method is no longer generic backend mask-only filtering. It is:

```text
D²MA-B r5: side-channel-isolated dynamic-depth / boundary-risk map admission.
```

The key experimental correction is that `STSLAM_PANOPTIC_SIDE_CHANNEL_ONLY=1` must be set. Without it, extra panoptic frontend, instance processing, and dynamic split paths are enabled; those runs are invalid for D²MA map-admission-only claims.

The six latest canonical full-sequence runs are summarized in:

```text
results_summaries/canonical_sidechannel_six_20260513/canonical_sidechannel_six_summary.csv
```

All six have `protocol_valid=1`. The headline results are:

| Case | Method | Matched | ATE-SE3 | ATE-Sim3 | Scale |
|---|---|---:|---:|---:|---:|
| `wxyz_d2ma_b_r5` | `d2ma_b_r5` | 857 | 0.017884 | 0.016043 | 0.974357 |
| `wrpy_d2ma_b_r5` | `d2ma_b_r5` | 906 | 0.269999 | 0.120722 | 0.361678 |
| `whalfsphere_d2ma_b_r5` | `d2ma_b_r5` | 1064 | 0.156093 | 0.118844 | 0.823258 |
| `wrpy_d2ma_min` | `d2ma_min` | 906 | 0.474824 | 0.156668 | 0.172691 |
| `wrpy_samecount_nonboundary_r5` | `samecount_nonboundary_r5` | 906 | 0.604425 | 0.156306 | 0.138884 |
| `whalfsphere_raw` | `raw` | 1064 | 0.506439 | 0.290979 | 0.484404 |

Please treat older missing-side-channel-only runs as invalid diagnostics rather than paper evidence.

## Current Metrics

Use `ATE-SE3` as the main RGB-D metric. `ATE-Sim3` and `ATE-origin` are diagnostics.

Strong image-level YOLOE + SAM3 frontend baselines:

- SAM3 box fallback image-level filtering: `ATE-SE3=0.016951 m`, `RPEt-SE3=0.011670 m`
- SAM3 mild dilate image-level filtering: `ATE-SE3=0.016268 m`, `RPEt-SE3=0.012387 m`
- person-v2 dynamic memory image-level filtering: `ATE-SE3=0.018325 m`, `RPEt-SE3=0.013280 m`

Current raw RGB-D + mask-only backend full `walking_xyz`:

- `semantic_only`: `ATE-SE3=0.302858 m`, `ATE-Sim3=0.238408 m`, `Sim3 scale=0.495238`, `RPEt-SE3=0.021200 m`, `RPER=0.571964 deg`
- `geom_framework_noop`: `ATE-SE3=0.330045 m`, `ATE-Sim3=0.261812 m`, `Sim3 scale=0.423952`, `RPEt-SE3=0.021201 m`, `RPER=0.574752 deg`
- `geom_dynamic_reject`: `ATE-SE3=0.314838 m`, `ATE-Sim3=0.203879 m`, `Sim3 scale=0.479517`, `RPEt-SE3=0.018447 m`, `RPER=0.482011 deg`

Sparse-flow gate summary:

- It improves some smoke30 metrics.
- On full sequence it lowers ATE relative to one candidate-gate baseline but reduces coverage and increases local-map tracking failures.
- It should not yet be treated as a final hard-delete strategy.

## What I Need From You

Please do a deep technical diagnosis and propose the next solution route.

First build a code map:

1. Data / sequence input path using `data/datasets.json`.
2. YOLOE + SAM3 frontend output path.
3. Mask/meta side-channel format.
4. ORB-SLAM3 RGB-D entry point.
5. Tracking stages where semantic features are removed, rescued, or re-scored.
6. LocalMapping dynamic logic.
7. Optimizer dynamic-aware logic.
8. Evaluation script and metric definitions.

Then answer these questions:

1. Why is image-level filtering currently far stronger than raw RGB-D + backend mask-only filtering?
2. Why does `geom_dynamic_reject` improve local RPE and Sim3 ATE but not full-sequence `ATE-SE3`?
3. Why do smoke30 improvements fail to generalize to the full `walking_xyz` sequence?
4. Which stage is most likely cutting off tracking support: `before_local_map`, `track_local_map_pre_pose`, or `before_create_keyframe`?
5. Is the full-sequence issue more likely due to over-deletion, wrong rescue, initialization/map contamination, scale/path-length drift, keyframe creation, or evaluation mismatch?
6. What extra logs should be added per frame to diagnose the failure interval?
7. Which current strategy should be kept, modified, or abandoned?

Please propose a prioritized experiment plan:

- Keep it practical.
- Start with full-sequence diagnosis before proposing another complex backend.
- Prefer small, interpretable ablations over broad new architecture.
- Avoid recommending direct DynoSAM / STSLAM joint optimization unless you can justify how it avoids the already observed dynamic pollution and why it should re-enter the active codebase.
- Include expected metrics and decision criteria for each experiment.

Target:

- First make raw RGB-D + mask-only backend route explainable and stable on full `walking_xyz`.
- Then try to move full-sequence `ATE-SE3` substantially below `0.30 m`.
- Long-term target is to approach the image-level frontend baseline around `0.016-0.018 m`, or clearly prove why that route is fundamentally stronger.

Output format:

1. Code map.
2. Current failure hypothesis ranked by likelihood.
3. Minimal instrumentation plan.
4. Next 5 experiments with exact switches/configs.
5. What result would confirm or falsify each hypothesis.
6. One recommended mainline strategy for the next implementation pass.
