# Init-30 Gaussian Blur + Full Feature-Level Chain

Experiment definition:
- first 30 frames: image-level Gaussian blur inside dynamic mask
- after frame 30: backend feature-level pipeline resumes
- backend full chain:
  - layer 1: `premask` or `postfilter`
  - layer 2: `ORB_SLAM3_DYNAMIC_LAYER=feature_trackmap`
  - layer 3: `ORB_SLAM3_DYNAMIC_OPTIMIZATION=on`

Shared init settings:
- `ORB_SLAM3_INIT_BLUR_FRAMES=30`
- `ORB_SLAM3_INIT_BLUR_KERNEL=21`

## Walking XYZ (RGB-D)

Sequence:
- `/home/lj/dynamic_SLAM/results/20260505_yoloe_sam3_maskonly_wxyz/sequence`

Both branches recovered the same initialization map size:
- init map points: `731`

Results:

| setting | init map points | ATE RMSE (m) | ATE mean (m) | RPE RMSE (m) |
| --- | ---: | ---: | ---: | ---: |
| init30 blur + premask + layer2 + layer3 | 731 | 0.3995 | 0.3312 | 0.0194 |
| init30 blur + postfilter + layer2 + layer3 | 731 | 0.4268 | 0.3623 | 0.0243 |

Immediate observation:
- initialization support was clearly improved versus pure masked initialization
- but final trajectory accuracy did **not** surpass the better earlier backend-only controlled runs
- on `walking_xyz`, `premask` remains better than `postfilter` under this init-30 strategy

## KITTI Tracking 0004 (Monocular)

Sequence:
- `/home/lj/dynamic_SLAM/datasets/kitti_tracking/0004/data_sam3_hybrid_base_gmflow_gmstereo`

Both branches recovered the same initialization map size:
- init map points: `386`

For monocular KITTI, `SE3` ATE is dominated by scale drift, so the more meaningful metric here is `Sim3 RMSE`.

Results:

| setting | init map points | matched poses | Sim3 RMSE (m) | Sim3 mean (m) | SE3 RMSE (m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| init30 blur + premask + layer2 + layer3 | 386 | 179 | 1.1078 | 0.9441 | 120.8412 |
| init30 blur + postfilter + layer2 + layer3 | 386 | 192 | 1.0915 | 0.8885 | 118.1154 |

Immediate observation:
- on monocular KITTI, `postfilter` is slightly better than `premask`
- the gain is modest, but consistent in both matched poses and Sim3 error

## Cross-dataset takeaway

1. The init-30 blur policy does affect initialization strongly:
   - `walking_xyz`: init map points restored to `731`
   - `kitti04 mono`: init map points restored to `386`

2. Better initialization does **not** automatically produce the best final trajectory.

3. The likely interpretation is:
   - initialization quality is one bottleneck
   - but later-stage dynamic contamination / feature allocation / observation quality still matter substantially

## Next tuning knob

The next obvious scalar to sweep is:
- init blur window length: `N = 10, 20, 30, 50`

The current `N=30` result suggests:
- the window is strong enough to change initialization
- but may already be too long, or not enough by itself, depending on the dataset and branch
