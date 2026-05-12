# Code Map For Dynamic SLAM Diagnosis

## Active Code Paths

Latest frontend:

```text
frontend/basic_model_based_SLAM/
```

Original local source:

```text
/home/lj/d-drive/CODEX/basic_model_based_SLAM
```

Latest backend:

```text
backend/orb_slam3_dynamic/
```

Original local source:

```text
/home/lj/dynamic_SLAM/stslam_backend
```

The backend source directory name is historical. It contains ORB-SLAM3 dynamic-filter experiments, not the abandoned STSLAM reproduction route.

## Local Data Layer

Registry:

```text
data/datasets.json
```

Main dataset IDs:

- `frontend_raw_wxyz`
- `backend_maskonly_full_wxyz`
- `backend_maskonly_smoke30_wxyz`
- `frontend_imagelevel_milddilate_full_wxyz`
- `frontend_imagelevel_boxfallback_full_wxyz`

Helpers:

- `scripts/dslam_data.py`
- `scripts/link_local_datasets.sh`
- `scripts/run_frontend_inference.sh`
- `scripts/run_backend_rgbd.sh`

## Abandoned / Background Routes

Not active code paths in this public snapshot:

```text
/home/lj/d-drive/CODEX/STSLAM/workspace/ORB_SLAM3_STSLAM
/home/lj/dynamic_SLAM/third_party/DynOSAM
```

See:

```text
docs/ABANDONED_ROUTES.md
```

## Frontend Flow

Main files:

- `frontend/basic_model_based_SLAM/scripts/rflysim_slam_nav/world_sam_pipeline.py`
- `frontend/basic_model_based_SLAM/scripts/run_rgbd_slam_benchmark.py`
- `frontend/basic_model_based_SLAM/config/world_sam_pipeline_foundation_panoptic_person_v2_milddilate_local.json`
- `frontend/basic_model_based_SLAM/config/world_sam_pipeline_foundation_panoptic_person_v2_local.json`
- `scripts/run_frontend_inference.sh`

Conceptual flow:

```text
RGB-D input
-> YOLOE detector
-> SAM3 segmenter
-> dynamic relevance / mask policy / temporal memory
-> one of:
   - image/depth-level filtered sequence
   - raw image + mask/meta side-channel sequence
```

Important distinction:

- Image-level route modifies input RGB/depth before ORB-SLAM3 and currently gives `ATE-SE3 ~= 0.016-0.018 m`.
- Mask-only route keeps raw RGB/depth and passes masks to the backend, but full-sequence `ATE-SE3` is currently around `0.30 m`.

## Backend Flow

Main files:

- `backend/orb_slam3_dynamic/Examples/RGB-D/rgbd_tum.cc`
- `backend/orb_slam3_dynamic/src/System.cc`
- `backend/orb_slam3_dynamic/src/Tracking.cc`
- `backend/orb_slam3_dynamic/include/Tracking.h`
- `backend/orb_slam3_dynamic/src/Frame.cc`
- `backend/orb_slam3_dynamic/include/Frame.h`
- `backend/orb_slam3_dynamic/src/LocalMapping.cc`
- `backend/orb_slam3_dynamic/src/Optimizer.cc`
- `scripts/run_backend_rgbd.sh`

Conceptual flow:

```text
data/datasets.json
-> scripts/run_backend_rgbd.sh
-> rgbd_tum.cc
-> load RGB, depth, optional panoptic/mask path
-> System::TrackRGBD(...)
-> Tracking::GrabImageRGBD(...)
-> Frame stores per-feature semantic instance / dynamic evidence
-> Tracking stages:
   - before_local_map
   - track_local_map_pre_pose
   - before_create_keyframe
-> LocalMapping may veto or split dynamic observations
-> Optimizer reads feature instance ids for dynamic-aware logic
```

## Most Relevant Backend Switches

Layer 1 / feature extraction:

- `ORB_SLAM3_MASK_MODE=off`
- `ORB_SLAM3_MASK_MODE=premask`
- `ORB_SLAM3_MASK_MODE=postfilter`

Semantic force filtering:

- `STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES`
- `STSLAM_RGBD_DYNAMIC_FRONTEND_SPLIT`

Semantic + geometry candidate logic:

- `STSLAM_SEMANTIC_GEOMETRIC_VERIFICATION`
- `STSLAM_SEMANTIC_GEOMETRIC_RESCUE_STAGES`
- `STSLAM_SEMANTIC_GEOMETRIC_MIN_STATIC_MAP_OBSERVATIONS`
- `STSLAM_SEMANTIC_CANDIDATE_GEOMETRY_GATE`

Geometric dynamic rejection:

- `STSLAM_GEOMETRIC_DYNAMIC_REJECTION`
- `STSLAM_GEOMETRIC_DYNAMIC_REJECTION_STAGES`
- `STSLAM_GEOMETRIC_DYNAMIC_REJECTION_MAX_REPROJ_ERROR_PX`
- `STSLAM_GEOMETRIC_DYNAMIC_REJECTION_MAX_DEPTH_ERROR_M`

Sparse-flow / epipolar gate experiments:

- `STSLAM_SEMANTIC_CANDIDATE_SPARSE_FLOW_GATE`
- `STSLAM_SEMANTIC_SPARSE_FLOW_GATE_STAGES`
- `STSLAM_SEMANTIC_FLOW_MAX_DYNAMIC_REJECT_RATIO`

Latest smoke variants:

- `STSLAM_SEMANTIC_CONSERVATIVE_DYNAMIC_DELETE`
- `STSLAM_SEMANTIC_STRICT_STATIC_KEEP`

## Evaluation

Use:

```text
tools/evaluate_trajectory_ate.py
```

Primary metric:

```text
ATE-SE3
```

Diagnostics:

```text
ATE-Sim3
ATE-origin
RPEt-SE3
RPER-SE3
Sim3 scale
matched poses / coverage
tracking failure intervals
```

## Key Results Directory Map

Feature-layer / layer 2 / layer 3 summaries:

```text
results_summaries/20260505_yoloe_sam3_maskonly_wxyz/
```

Full current-binary controls:

```text
results_summaries/a1_minimal_controls_currentbin_full_wxyz_20260511/
```

Frontend image-level no-mask geometry check:

```text
results_summaries/frontend_nomask_geometry_yoloe_sam3_milddilate_wxyz_20260511/
```

Sparse-flow full failure / coverage tradeoff:

```text
results_summaries/precision_candidate_gate_sparseflow_tracklocal_full_wxyz_20260511/
```

Conservative and strict-static smoke checks:

```text
results_summaries/precision_conservative_candidate_gate_smoke30_wxyz_20260511/
results_summaries/precision_strict_static_keep_smoke30_wxyz_20260511/
```
