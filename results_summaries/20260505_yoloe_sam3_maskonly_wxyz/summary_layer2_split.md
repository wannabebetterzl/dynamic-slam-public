# Walking XYZ: Layer-2 Split Diagnosis

Sequence:
- `/home/lj/dynamic_SLAM/results/20260505_yoloe_sam3_maskonly_wxyz/sequence`

Control:
- Raw `rgb/depth`
- YOLOE+SAM3 only provides `mask/meta`
- Backend executable:
  - `/home/lj/isolated_eval/orbslam3_clean_frontend/Examples/RGB-D/rgbd_tum`

## Meaning of each mode

- `off`: mask exists, backend ignores it
- `premask`: mask enters ORB extractor before feature extraction, and post-mask remains enabled
- `postfilter`: extract first, then remove masked features
- `trackmatch`: layer-2 only vetoes dynamic-candidate features during matching / tracking
- `triangulation`: layer-2 only vetoes dynamic-candidate features during triangulation / map growth
- `feature_trackmap`: layer-2 bundled switch, enables both `trackmatch + triangulation`

## Results

| mode | init map points | ATE RMSE (m) | ATE mean (m) | RPE RMSE (m) |
| --- | ---: | ---: | ---: | ---: |
| off | 731 | 0.4358 | 0.3772 | 0.0219 |
| premask | 610 | 0.3821 | 0.3121 | 0.0174 |
| premask + trackmatch | 610 | 0.4040 | 0.3368 | 0.0294 |
| premask + triangulation | 610 | 0.4002 | 0.3320 | 0.0204 |
| premask + feature_trackmap | 610 | 0.4248 | 0.3622 | 0.0218 |
| postfilter | 392 | 0.3958 | 0.3167 | 0.0293 |
| postfilter + trackmatch | 392 | 0.3917 | 0.3193 | 0.0179 |
| postfilter + triangulation | 392 | 0.3919 | 0.3205 | 0.0214 |
| postfilter + feature_trackmap | 392 | 0.2189 | 0.1804 | 0.0216 |

## Immediate observations

1. Feature-layer conclusion remains stable:
   - `premask` is still the best pure feature-layer setting.

2. For `premask`, both layer-2 sub-vetoes hurt:
   - `trackmatch` worsens ATE and especially worsens RPE.
   - `triangulation` also worsens ATE, though slightly less than `trackmatch`.
   - turning both on together is worst.

3. For `postfilter`, either sub-veto alone gives only mild ATE change:
   - `trackmatch` helps RPE the most.
   - `triangulation` gives a smaller tracking benefit.
   - neither alone explains the very large gain of `postfilter + feature_trackmap`.

4. Therefore the large improvement of `postfilter + feature_trackmap` appears to be a joint effect:
   - likely not from one single veto point
   - more likely from the interaction between matching cleanup and triangulation cleanup

## What this means for the next step

- If our goal is the cleanest backend-only improvement on `walking_xyz`, the current best isolated answer is still:
  - `premask` for layer 1
- But if we continue toward layer 2:
  - `premask` should not inherit the current layer-2 vetoes directly
  - `postfilter` is the branch where layer-2 cooperation seems promising
- Next step should be to isolate layer 3 / BA behavior without mixing image edits.
