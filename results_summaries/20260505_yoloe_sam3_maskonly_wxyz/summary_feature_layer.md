# walking_xyz: raw image + YOLOE/SAM3 mask only + backend feature-layer comparison

## Protocol

- raw RGB-D source:
  - `/mnt/d/CODEX/basic_model_based_SLAM/experiments/E6-1_full_raw_freiburg3_walking_xyz_20260321/sequence`
- mask/meta donor:
  - `/mnt/d/CODEX/basic_model_based_SLAM/experiments/20260504_yoloe_sam3_full_wxyz/sequence`
- new mask-only sequence:
  - `/home/lj/dynamic_SLAM/results/20260505_yoloe_sam3_maskonly_wxyz/sequence`

This sequence keeps:
- raw `rgb/`
- raw `depth/`
- donor `mask/`
- donor `meta/`

So the frontend no longer modifies image pixels. The only variable is whether the backend uses the mask.

## Results

| mode | meaning | init map points | ATE RMSE (m) | ATE mean (m) | RPE RMSE (m) |
|---|---|---:|---:|---:|---:|
| off | mask exists but backend does not use it | 731 | 0.4357581424 | 0.3772134621 | 0.0218950117 |
| premask | mask enters ORB extractor before feature detection | 610 | 0.3820852059 | 0.3121489810 | 0.0173898132 |
| postfilter | detect first, then remove masked features | 392 | 0.3957635337 | 0.3166855448 | 0.0293173359 |

## Current reading

- This is the first clean answer to the feature-layer question.
- On true raw images, both mask-based backend modes beat `off`.
- `premask` is the best among the three at the feature layer.
- `postfilter` also improves ATE over raw `off`, but harms local stability more than `premask`.
- This experiment still does **not** isolate tracking/map-layer veto and BA-layer rejection as separate factors; it is the feature-layer stage only.
