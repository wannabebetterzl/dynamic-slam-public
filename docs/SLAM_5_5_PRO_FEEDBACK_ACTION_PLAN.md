# 5.5 Pro Feedback Follow-up Plan

This note records the first concrete follow-up after the web-side 5.5 Pro feedback.

## Main Diagnosis To Test

The current failure should not be treated as proof that YOLOE/SAM3 masks are useless.
The more likely failure mode is that dynamic evidence is used too early or too hard,
especially around `track_local_map_pre_pose`, where hard deletion can destroy
tracking support before pose estimation has enough static constraints.

Working direction:

```text
Support-Preserving Dynamic Evidence SLAM
```

## Implemented Follow-up

- Added RGB-D sequence integrity checking:
  - `tools/check_rgbd_sequence_integrity.py`
- Added stage-gated hard deletion:
  - `STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURE_STAGES`
  - Supported values are comma-separated stage names or `*`.
  - Use `none` to disable all stages while keeping the global switch explicit.
- Updated `scripts/run_backend_rgbd.sh` so external `STSLAM_*` overrides are preserved.

## Current Integrity Result

For `backend_maskonly_full_wxyz`:

- associations: 859
- missing RGB/depth/mask files: 0/0/0
- RGB-depth max timestamp difference: about 0.0381 s
- RGB-depth pairs exceeding 0.03 s: 21
- RGB-depth pairs exceeding 0.04 s: 0
- association timestamps matched to ground truth within 0.03 s: 857/859

Interpretation:

The old `depth_files=827` concern is not supported by the current sequence on disk.
The remaining issue is timestamp tolerance rather than missing frame data.

## Verified Smoke Commands

Stage-gated hard delete at keyframe creation only:

```bash
STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURE_STAGES=before_create_keyframe \
  bash scripts/run_backend_rgbd.sh \
  backend_maskonly_smoke30_wxyz \
  semantic_only \
  runs/verify_stagegate_before_create_keyframe
```

Metadata-only / no hard semantic deletion:

```bash
STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=0 \
STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURE_STAGES=none \
  bash scripts/run_backend_rgbd.sh \
  backend_maskonly_smoke30_wxyz \
  semantic_only \
  runs/verify_forcefilter_off
```

## Full-sequence Stage Ablation Result

Dataset:

```text
backend_maskonly_full_wxyz
```

Run root:

```text
runs/full_stage_ablation_20260512
```

Raw run outputs are local-only and ignored by git. The compact public summary is:

```text
results_summaries/full_stage_ablation_20260512/README.md
```

| Variant | Matched | ATE-SE3 RMSE (m) | ATE-Sim3 RMSE (m) | Sim3 scale | RPEt-SE3 RMSE (m) | RPER RMSE (deg) | Local-map failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| `none_metadata_only` | 851 | 0.191482 | 0.167528 | 0.729600 | 0.051918 | 1.136524 | 12 |
| `track_local_map_pre_pose` | 857 | 0.274240 | 0.247344 | 0.590789 | 0.019668 | 0.553184 | 0 |
| `before_local_map` | 857 | 0.388275 | 0.248433 | 0.362142 | 0.026060 | 0.646083 | 0 |
| `before_create_keyframe` | 857 | 0.566043 | 0.265760 | 0.219656 | 0.023246 | 0.593768 | 0 |

Interpretation:

- No hard deletion gives the best global ATE, but the worst local RPE.
- `track_local_map_pre_pose` gives the best local RPE among hard-delete variants and beats the previous full `semantic_only` ATE-SE3 baseline of about 0.303 m.
- `before_local_map` is too early or misplaced.
- `before_create_keyframe` is too late and causes the worst global trajectory/scale behavior.
- The next step should not be another single-stage hard-delete rule. The result points to support-preserving soft/capped dynamic evidence.

## Commands Reproduced

Run these on `backend_maskonly_full_wxyz`:

```bash
STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURE_STAGES=before_local_map \
  bash scripts/run_backend_rgbd.sh backend_maskonly_full_wxyz semantic_only
```

```bash
STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURE_STAGES=track_local_map_pre_pose \
  bash scripts/run_backend_rgbd.sh backend_maskonly_full_wxyz semantic_only
```

```bash
STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURE_STAGES=before_create_keyframe \
  bash scripts/run_backend_rgbd.sh backend_maskonly_full_wxyz semantic_only
```

```bash
STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=0 \
STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURE_STAGES=none \
  bash scripts/run_backend_rgbd.sh backend_maskonly_full_wxyz semantic_only
```

Only after this stage ablation should we move to capped/soft/support-aware
geometric rejection.

## Next Implementation Target

Add a soft/capped/support-aware action around `track_local_map_pre_pose`:

```bash
STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=0
STSLAM_GEOMETRIC_DYNAMIC_REJECTION=1
STSLAM_GEOMETRIC_DYNAMIC_REJECTION_ACTION=soft_weight
STSLAM_GEOMETRIC_DYNAMIC_REJECTION_STAGES=track_local_map_pre_pose
STSLAM_GEOMETRIC_DYNAMIC_REJECTION_MAX_REJECT_RATIO=0.10
STSLAM_GEOMETRIC_DYNAMIC_REJECTION_PROTECT_MIN_INLIERS=45
```

If `STSLAM_GEOMETRIC_DYNAMIC_REJECTION_ACTION=soft_weight` is not implemented yet,
implement that switch before running another full sequence.
