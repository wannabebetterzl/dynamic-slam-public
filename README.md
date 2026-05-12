# Dynamic SLAM Public Diagnosis Snapshot

GitHub target:

```text
https://github.com/wannabebetterzl/dynamic-slam-public
```

This repository is a compact public snapshot for diagnosing a dynamic Visual SLAM research project. It contains the current code, experiment notes, key metrics, and run commands needed for external deep analysis. It intentionally does not include datasets, model weights, full image/depth/mask sequences, ORB vocabulary files, compiled binaries, or private Obsidian vault content.

## Research Goal

Improve Visual SLAM robustness in dynamic RGB-D / stereo scenes by using foundation-model perception as a dynamic-object prior without polluting static tracking and mapping.

The current question is no longer simply "can YOLOE/SAM3 remove dynamic objects?" The active research problem is:

```text
When should semantic dynamic information remove, keep, rescue, or re-score visual features
across ORB extraction, tracking/local-map matching, keyframe creation, and optimization?
```

## Repository Layout

```text
frontend/basic_model_based_SLAM/
  Latest YOLOE + SAM3 frontend used to generate image-level filtered sequences,
  mask-only sequences, and mask/meta side channels.

frontend/object_frontend/
  Small object-observation layer for future object-level dynamic SLAM work.

backend/orb_slam3_dynamic/
  Current ORB-SLAM3-derived experimental backend. Despite the historical
  directory name in the source workspace, this is NOT the abandoned STSLAM
  reproduction path. It contains the latest semantic candidate geometry gate,
  geometric dynamic rejection, sparse-flow gate, conservative delete, and
  strict static keep logic.

tools/
  Evaluation and KITTI preparation utilities.

docs/
  Experiment records, route history, code map, and deep-thinking prompt.

results_summaries/
  Small markdown/json/txt summaries only. Full datasets and frame dumps are
  excluded.
```

## Most Important Code

Current backend entry points:

- `backend/orb_slam3_dynamic/src/Tracking.cc`
- `backend/orb_slam3_dynamic/include/Tracking.h`
- `backend/orb_slam3_dynamic/include/Frame.h`
- `backend/orb_slam3_dynamic/src/Frame.cc`
- `backend/orb_slam3_dynamic/src/LocalMapping.cc`
- `backend/orb_slam3_dynamic/src/Optimizer.cc`
- `backend/orb_slam3_dynamic/Examples/RGB-D/rgbd_tum.cc`
- `backend/orb_slam3_dynamic/Examples/RGB-D/TUM3.yaml`

Current frontend entry points:

- `frontend/basic_model_based_SLAM/scripts/rflysim_slam_nav/world_sam_pipeline.py`
- `frontend/basic_model_based_SLAM/scripts/run_rgbd_slam_benchmark.py`
- `frontend/basic_model_based_SLAM/config/world_sam_pipeline_foundation_panoptic_person_v2_milddilate_local.json`
- `frontend/basic_model_based_SLAM/config/world_sam_pipeline_foundation_panoptic_person_v2_local.json`

Unified evaluator:

- `tools/evaluate_trajectory_ate.py`

## Current Metrics To Explain

Use `ATE-SE3` as the main RGB-D metric. `ATE-Sim3` and `ATE-origin` are diagnostic only.

### Strong Image-Level Frontend Baseline

From old YOLOE + SAM3 image/depth-level filtered frontend, evaluated with the unified script:

| Route | ATE-SE3 RMSE | RPEt-SE3 RMSE |
|---|---:|---:|
| SAM3 box fallback image-level filtering | 0.016951 m | 0.011670 m |
| SAM3 mild dilate image-level filtering | 0.016268 m | 0.012387 m |
| person-v2 dynamic memory image-level filtering | 0.018325 m | 0.013280 m |

### Current Raw RGB-D + Mask-Only Backend Route

Full `walking_xyz`, raw RGB-D unchanged, YOLOE + SAM3 mask passed only as backend side-channel:

| Group | Matched | ATE-SE3 RMSE | ATE-Sim3 RMSE | Sim3 scale | RPEt-SE3 RMSE | RPER RMSE |
|---|---:|---:|---:|---:|---:|---:|
| `semantic_only` | 857 | 0.302858 m | 0.238408 m | 0.495238 | 0.021200 m | 0.571964 deg |
| `geom_framework_noop` | 857 | 0.330045 m | 0.261812 m | 0.423952 | 0.021201 m | 0.574752 deg |
| `geom_dynamic_reject` | 857 | 0.314838 m | 0.203879 m | 0.479517 | 0.018447 m | 0.482011 deg |

Interpretation so far:

- The image-level frontend baseline is still much stronger globally.
- `geom_dynamic_reject` improves local relative metrics but does not improve full-sequence `ATE-SE3`.
- Large Sim3 scale corrections indicate global scale/path-length inconsistency in the raw RGB-D + mask-only backend route.
- Smoke30 improvements do not safely extrapolate to the full sequence.

## Important Route Correction

The historical STSLAM reproduction failed and should not be treated as the active code path.

Abandoned / background only:

```text
/home/lj/d-drive/CODEX/STSLAM/workspace/ORB_SLAM3_STSLAM
```

Latest experiment backend code:

```text
/home/lj/dynamic_SLAM/stslam_backend
```

Latest YOLOE + SAM3 frontend code:

```text
/home/lj/d-drive/CODEX/basic_model_based_SLAM
```

This public repo merges those two active code paths.

## Typical Commands

These commands document the experiment protocol. Paths in this public snapshot are relative, but original local data paths are retained in summaries for traceability.

Build backend in the original workspace:

```bash
cd /home/lj/dynamic_SLAM/stslam_backend
./build.sh
```

Run full mask-only RGB-D backend experiment, original workspace style:

```bash
export STSLAM_USE_VIEWER=0
export STSLAM_DISABLE_FRAME_SLEEP=1
export ORB_SLAM3_MASK_MODE=postfilter
export STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=1
export STSLAM_RGBD_DYNAMIC_FRONTEND_SPLIT=0

/home/lj/dynamic_SLAM/stslam_backend/Examples/RGB-D/rgbd_tum \
  /home/lj/dynamic_SLAM/stslam_backend/Vocabulary/ORBvoc.txt \
  /home/lj/dynamic_SLAM/stslam_backend/Examples/RGB-D/TUM3.yaml \
  /home/lj/dynamic_SLAM/results/20260505_yoloe_sam3_maskonly_wxyz/sequence \
  /home/lj/dynamic_SLAM/results/20260505_yoloe_sam3_maskonly_wxyz/sequence/associations.txt \
  /home/lj/dynamic_SLAM/results/20260505_yoloe_sam3_maskonly_wxyz/sequence/mask
```

Evaluate trajectory:

```bash
python /home/lj/dynamic_SLAM/scripts/evaluate_trajectory_ate.py \
  --ground-truth /path/to/groundtruth.txt \
  --estimate /path/to/CameraTrajectory.txt \
  --alignment all \
  --output-json /path/to/eval_unified_all.json
```

## Requested Deep Analysis

Start with `docs/DEEP_THINKING_PROMPT.md`. The desired analysis is:

1. Build a code map of the current frontend/backend data flow.
2. Diagnose why image-level filtering is much stronger than raw RGB-D + mask-only backend filtering.
3. Explain why smoke30 improvements collapse on full `walking_xyz`.
4. Identify likely failure stages around tracking/local-map support, keyframe creation, scale/path-length drift, and over-deletion.
5. Propose a prioritized experiment plan that can realistically move full-sequence `ATE-SE3` toward the image-level baseline.

## Public Snapshot Notes

- This repo is for code review and reasoning, not direct turnkey reproduction.
- Datasets, model weights, ORB vocabulary, compiled binaries, full frame sequences, and private notes are excluded.
- Some configs keep original absolute local paths to document the actual experiment provenance; replace them with your own paths before running.
