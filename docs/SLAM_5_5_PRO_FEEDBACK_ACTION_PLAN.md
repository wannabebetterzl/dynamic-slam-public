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

## Next Full-sequence Ablation

Run these on `backend_maskonly_full_wxyz`, not only smoke30:

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
