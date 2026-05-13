# Dynamic SLAM Public Diagnosis Snapshot

GitHub target:

```text
https://github.com/wannabebetterzl/dynamic-slam-public
```

This repository is a compact public snapshot for diagnosing a dynamic Visual SLAM research project. It contains the current code, experiment notes, key metrics, a local dataset registry, and run wrappers needed for external deep analysis. It intentionally does not include datasets, model weights, full image/depth/mask sequences, ORB vocabulary files, compiled binaries, or private Obsidian vault content.

## Research Goal

Improve Visual SLAM robustness in dynamic RGB-D / stereo scenes by using foundation-model perception as a dynamic-object prior without polluting static tracking and mapping.

The active research question is:

```text
When should semantic dynamic information remove, keep, rescue, or re-score visual features
across ORB extraction, tracking/local-map matching, keyframe creation, and optimization?
```

## Repository Layout

```text
frontend/basic_model_based_SLAM/
  Latest YOLOE + SAM3 frontend used to generate image-level filtered sequences,
  mask-only sequences, and mask/meta side channels.

backend/orb_slam3_dynamic/
  Current ORB-SLAM3-derived experimental backend. Despite the historical
  directory name in the source workspace, this is NOT the abandoned STSLAM
  reproduction path. It contains semantic candidate geometry gate, geometric
  dynamic rejection, sparse-flow gate, conservative delete, and strict static
  keep logic.

data/
  Local dataset registry. This stores paths and metadata only, not frame data.

scripts/
  Thin local wrappers for dataset lookup, frontend inference/export, backend
  RGB-D runs, smoke runs, and trajectory evaluation.

tools/
  Evaluation and KITTI preparation utilities.

docs/
  Experiment records, route decisions, code map, and deep-thinking prompt.
  The current 5.5 Pro follow-up plan is in
  docs/SLAM_5_5_PRO_FEEDBACK_ACTION_PLAN.md, and the full current feedback
  thread is mirrored in docs/5.5_PRO_回馈.md.

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

Local execution helpers:

- `data/datasets.json`
- `scripts/dslam_data.py`
- `scripts/link_local_datasets.sh`
- `scripts/run_backend_rgbd.sh`
- `scripts/run_d2ma_sidechannel_isolated.sh`
- `scripts/run_frontend_inference.sh`
- `tools/evaluate_trajectory_ate.py`
- `tools/check_rgbd_sequence_integrity.py`
- `tools/validate_d2ma_sidechannel_protocol.py`
- `tools/summarize_backend_runs.py`

## Current Metrics To Explain

Use `ATE-SE3` as the main RGB-D metric. `ATE-Sim3` and `ATE-origin` are diagnostic only.

### Canonical Side-Channel Isolation Results

The current paper-facing D²MA protocol is now fixed by
`scripts/run_d2ma_sidechannel_isolated.sh` and audited by
`tools/validate_d2ma_sidechannel_protocol.py`.

All D²MA map-admission-only runs must explicitly isolate the mask as a
side-channel provenance signal:

```text
ORB_SLAM3_MASK_MODE=off
STSLAM_PANOPTIC_SIDE_CHANNEL_ONLY=1
STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=0
STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURE_STAGES=none
STSLAM_DYNAMIC_DEPTH_INVALIDATION=0
STSLAM_RGBD_DYNAMIC_FRONTEND_SPLIT=0
STSLAM_DYNAMIC_MAP_ADMISSION_VETO=0
```

If `STSLAM_PANOPTIC_SIDE_CHANNEL_ONLY=1` is missing, the run enables extra
panoptic frontend / instance processing / dynamic split paths and is not
considered a D²MA map-admission-only experiment.

Latest six-run canonical full-sequence summary:

| Case | Method | Matched | Coverage | ATE-SE3 RMSE | ATE-Sim3 RMSE | Sim3 scale | Protocol |
|---|---|---:|---:|---:|---:|---:|---|
| `wxyz_d2ma_b_r5` | `d2ma_b_r5` | 857 | 0.2972 | 0.017884 m | 0.016043 m | 0.974357 | pass |
| `wrpy_d2ma_b_r5` | `d2ma_b_r5` | 906 | 0.2959 | 0.269999 m | 0.120722 m | 0.361678 | pass |
| `whalfsphere_d2ma_b_r5` | `d2ma_b_r5` | 1064 | 0.2970 | 0.156093 m | 0.118844 m | 0.823258 | pass |
| `wrpy_d2ma_min` | `d2ma_min` | 906 | 0.2959 | 0.474824 m | 0.156668 m | 0.172691 | pass |
| `wrpy_samecount_nonboundary_r5` | `samecount_nonboundary_r5` | 906 | 0.2959 | 0.604425 m | 0.156306 m | 0.138884 | pass |
| `whalfsphere_raw` | `raw` | 1064 | 0.2970 | 0.506439 m | 0.290979 m | 0.484404 | pass |

Detailed CSV and protocol validation artifacts are in:

```text
results_summaries/canonical_sidechannel_six_20260513/
results_summaries/experiments_0512_0517.csv
```

Interpretation:

- The missing-side-channel-only anomaly has been localized and invalidated.
- `D²MA-B r5` is the current main method.
- `samecount_nonboundary_r5` is a negative control showing the gain is not just generic sparsification.
- `walking_rpy` remains the hardest sequence and should be reported with coverage and repeat variance.

### Strong Image-Level Frontend Baseline

From YOLOE + SAM3 image/depth-level filtered frontend, evaluated with the unified script:

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

## Route Decision

The historical STSLAM reproduction failed and should not be treated as the active code path.

The DynoSAM adapter / object-frontend bridge is also removed from the active public snapshot. DynoSAM remains useful as a conceptual reference, but it is not the next implementation base until the current full-sequence mask-only ORB backend failure is explained.

Abandoned / background only:

```text
/home/lj/d-drive/CODEX/STSLAM/workspace/ORB_SLAM3_STSLAM
/home/lj/dynamic_SLAM/third_party/DynOSAM
```

Latest experiment backend code:

```text
/home/lj/dynamic_SLAM/stslam_backend
```

Latest YOLOE + SAM3 frontend code:

```text
/home/lj/d-drive/CODEX/basic_model_based_SLAM
```

This public repo now keeps only the active frontend/backend paths plus a concise abandoned-route note.

## Typical Commands

List registered local datasets:

```bash
python scripts/dslam_data.py list
```

Check local paths:

```bash
python scripts/dslam_data.py check
```

Check full RGB-D/mask/ground-truth sequence integrity:

```bash
DATASET_ID=backend_maskonly_full_wxyz
python tools/check_rgbd_sequence_integrity.py \
  --sequence-root "$(python scripts/dslam_data.py get "$DATASET_ID" sequence_root)" \
  --associations "$(python scripts/dslam_data.py get "$DATASET_ID" associations)" \
  --mask-root "$(python scripts/dslam_data.py get "$DATASET_ID" mask_root)" \
  --groundtruth "$(python scripts/dslam_data.py get "$DATASET_ID" ground_truth)" \
  --out "runs/integrity/${DATASET_ID}.json"
```

Create convenience symlinks under `data/local/`:

```bash
bash scripts/link_local_datasets.sh
```

Build backend in the original workspace:

```bash
cd /home/lj/dynamic_SLAM/stslam_backend
./build.sh
```

Run backend smoke30:

```bash
bash scripts/run_backend_rgbd.sh backend_maskonly_smoke30_wxyz semantic_only
```

Run backend full mask-only RGB-D:

```bash
bash scripts/run_backend_rgbd.sh backend_maskonly_full_wxyz semantic_only
```

Run the first stage-gated hard-delete ablation:

```bash
STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURE_STAGES=before_create_keyframe \
  bash scripts/run_backend_rgbd.sh backend_maskonly_full_wxyz semantic_only
```

Run a protocol-audited D²MA side-channel-only experiment:

```bash
bash scripts/run_d2ma_sidechannel_isolated.sh \
  external_whalfsphere_rawrgb_rawdepth_mask \
  d2ma_b_r5 \
  runs/example_whalfsphere_d2ma_b_r5
```

Validate an existing D²MA run:

```bash
python tools/validate_d2ma_sidechannel_protocol.py \
  --run-dir runs/example_whalfsphere_d2ma_b_r5 \
  --method d2ma_b_r5
```

Run frontend YOLOE/SAM3 export smoke from raw TUM input:

```bash
bash scripts/run_frontend_inference.sh frontend_raw_wxyz runs/frontend_smoke30 30
```

Run frontend YOLOE/SAM3 export full sequence:

```bash
bash scripts/run_frontend_inference.sh frontend_raw_wxyz runs/frontend_full 0
```

Evaluate trajectory:

```bash
python tools/evaluate_trajectory_ate.py \
  --ground-truth /path/to/groundtruth.txt \
  --estimated /path/to/CameraTrajectory.txt \
  --alignment all \
  --json-out /path/to/eval_unified_all.json
```

## Requested Deep Analysis

Start with `docs/DEEP_THINKING_PROMPT.md`. The desired analysis is:

1. Build a code map of the current frontend/backend data flow.
2. Diagnose why image-level filtering is much stronger than raw RGB-D + mask-only backend filtering.
3. Explain why smoke30 improvements collapse on full `walking_xyz`.
4. Identify likely failure stages around tracking/local-map support, keyframe creation, scale/path-length drift, and over-deletion.
5. Propose a prioritized experiment plan that can realistically move full-sequence `ATE-SE3` toward the image-level baseline.

## Public Snapshot Notes

- This repo is for code review, local orchestration, and reasoning, not a self-contained dataset release.
- Datasets, model weights, ORB vocabulary, compiled binaries, full frame sequences, and private notes are excluded.
- Some configs keep original absolute local paths to document actual experiment provenance.
- Failed STSLAM reproduction code and the DynoSAM adapter path are intentionally excluded; see `docs/ABANDONED_ROUTES.md`.
