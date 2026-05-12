# Walking XYZ: Plan B (Postfilter, Layer-2 Off, Layer-3 On/Off)

Goal:
- fix layer 1 as `postfilter`
- fix layer 2 as `off`
- only toggle layer 3 (optimization layer)

Control variables:
- sequence: `/home/lj/dynamic_SLAM/results/20260505_yoloe_sam3_maskonly_wxyz/sequence`
- raw `rgb/depth`
- YOLOE+SAM3 only provides `mask/meta`
- backend: `/home/lj/isolated_eval/orbslam3_clean_frontend/Examples/RGB-D/rgbd_tum`
- env:
  - `ORB_SLAM3_MASK_MODE=postfilter`
  - `ORB_SLAM3_DYNAMIC_LAYER=off`
  - `ORB_SLAM3_DYNAMIC_OPTIMIZATION=on/off`

## Results

| setting | init map points | ATE RMSE (m) | ATE mean (m) | RPE RMSE (m) |
| --- | ---: | ---: | ---: | ---: |
| postfilter + layer2 off + layer3 on | 392 | 0.3955 | 0.3233 | 0.0190 |
| postfilter + layer2 off + layer3 off | 392 | 0.4005 | 0.3304 | 0.0208 |

## Immediate interpretation

1. In this controlled setting, layer 3 shows a clear positive effect on the `postfilter` route.
2. Turning layer 3 on improves both:
   - ATE RMSE (`0.4005 -> 0.3955`)
   - RPE RMSE (`0.0208 -> 0.0190`)
3. The effect is still moderate, but it is more consistent than what we saw in Plan A (`premask`).

## Comparison with Plan A

- `premask + layer2 off`:
  - layer 3 had almost no net benefit
- `postfilter + layer2 off`:
  - layer 3 improves both global and local trajectory metrics

This supports the hypothesis that:
- layer 3 is more useful when the system keeps more candidate observations alive into the optimization stage
- `postfilter` leaves that room
- `premask` removes many candidates earlier, so there is less for layer 3 to reweight or reject

## What this means next

The next most informative experiment is:
- keep `postfilter`
- compare `layer2 off/on`
- inside each one, toggle `layer3 on/off`

That will answer whether the strong earlier `postfilter + layer2` result is further amplified by layer 3, or whether layer 2 already explains most of the gain.

## Caution

- These numbers should only be compared within this controlled pair, or with Plan A after acknowledging the different fixed layer-1 route.
