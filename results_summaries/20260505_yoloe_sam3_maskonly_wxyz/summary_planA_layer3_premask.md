# Walking XYZ: Plan A (Premask, Layer-2 Off, Layer-3 On/Off)

Goal:
- fix layer 1 as `premask`
- fix layer 2 as `off`
- only toggle layer 3 (optimization layer)

Control variables:
- sequence: `/home/lj/dynamic_SLAM/results/20260505_yoloe_sam3_maskonly_wxyz/sequence`
- raw `rgb/depth`
- YOLOE+SAM3 only provides `mask/meta`
- backend: `/home/lj/isolated_eval/orbslam3_clean_frontend/Examples/RGB-D/rgbd_tum`
- env:
  - `ORB_SLAM3_MASK_MODE=premask`
  - `ORB_SLAM3_DYNAMIC_LAYER=off`
  - `ORB_SLAM3_DYNAMIC_OPTIMIZATION=on/off`

## Results

| setting | init map points | ATE RMSE (m) | ATE mean (m) | RPE RMSE (m) |
| --- | ---: | ---: | ---: | ---: |
| premask + layer2 off + layer3 on | 610 | 0.3928 | 0.3209 | 0.0221 |
| premask + layer2 off + layer3 off | 610 | 0.3905 | 0.3225 | 0.0233 |

## Immediate interpretation

1. In this controlled setting, layer 3 does not produce a large effect.
2. Turning layer 3 on:
   - slightly worsens ATE RMSE (`0.3905 -> 0.3928`)
   - slightly improves RPE RMSE (`0.0233 -> 0.0221`)
3. The difference is small enough that we should not over-interpret it as a strong positive or negative result.

## What this means

- Under the `premask` route, the optimization-layer dynamic weighting/gating is not the main source of gain.
- The stronger effects observed earlier are more likely to come from layer 1 and layer 2 interactions, not from layer 3 alone.
- This also means that if we want to understand large performance shifts, the next informative comparison is still `Plan B`:
  - fix `postfilter`
  - keep layer 2 fixed
  - toggle only layer 3

## Caution

- These numbers should only be compared within this controlled pair.
- They should not be mixed directly with earlier runs built from older binaries or different optimization logic.
