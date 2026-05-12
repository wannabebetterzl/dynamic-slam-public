# Local Data Registry

This directory centralizes the local dataset paths used by active inference and smoke runs.

The registry is:

```text
data/datasets.json
```

It contains paths only. Datasets, RGB/depth/mask frames, model weights, and ORB vocabularies remain outside this public snapshot.

## Quick Checks

List registered datasets:

```bash
python scripts/dslam_data.py list
```

Check whether the local paths exist:

```bash
python scripts/dslam_data.py check
```

Create convenience symlinks under `data/local/`:

```bash
bash scripts/link_local_datasets.sh
```

## Main Dataset IDs

- `frontend_raw_wxyz`: original TUM RGB-D `freiburg3_walking_xyz` input for frontend inference/export.
- `backend_maskonly_full_wxyz`: full raw RGB-D + YOLOE/SAM3 mask side-channel sequence for backend diagnosis.
- `backend_maskonly_smoke30_wxyz`: 30-frame smoke subset from the same backend sequence.
- `frontend_imagelevel_milddilate_full_wxyz`: strong image-level baseline sequence.
- `frontend_imagelevel_boxfallback_full_wxyz`: strong image-level baseline sequence.

## Direct Local Calls

Run backend smoke:

```bash
bash scripts/run_backend_rgbd.sh backend_maskonly_smoke30_wxyz semantic_only
```

Run backend full sequence:

```bash
bash scripts/run_backend_rgbd.sh backend_maskonly_full_wxyz semantic_only
```

Run frontend export/inference smoke from raw TUM input:

```bash
bash scripts/run_frontend_inference.sh frontend_raw_wxyz runs/frontend_smoke30 30
```

Run frontend export/inference full sequence:

```bash
bash scripts/run_frontend_inference.sh frontend_raw_wxyz runs/frontend_full 0
```
