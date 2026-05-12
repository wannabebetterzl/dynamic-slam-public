# Full walking_xyz minimal controls, current binary, 2026-05-11

Sequence:

- `/home/lj/dynamic_SLAM/results/20260505_yoloe_sam3_maskonly_wxyz/sequence`

Protocol:

- raw RGB-D input
- YOLOE + SAM3 mask passed to backend only
- `ORB_SLAM3_MASK_MODE=postfilter`
- `STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=1`
- unified evaluator: `/home/lj/dynamic_SLAM/scripts/evaluate_trajectory_ate.py`

Binary:

- `/home/lj/dynamic_SLAM/stslam_backend/lib/libORB_SLAM3.so`
- `/home/lj/dynamic_SLAM/stslam_backend/Examples/RGB-D/rgbd_tum`
- timestamp: `2026-05-11 16:19:26 +0800`

## Results

| group | matched | ATE-SE3 RMSE (m) | ATE-Sim3 RMSE (m) | Sim3 scale | ATE-origin RMSE (m) | RPEt-SE3 RMSE (m) | RPER RMSE (deg) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `semantic_only` | 857 | 0.302858 | 0.238408 | 0.495238 | 0.619060 | 0.021200 | 0.571964 |
| `geom_framework_noop` | 857 | 0.330045 | 0.261812 | 0.423952 | 0.663362 | 0.021201 | 0.574752 |
| `geom_dynamic_reject` | 857 | 0.314838 | 0.203879 | 0.479517 | 0.723866 | 0.018447 | 0.482011 |

## Reading

- `semantic_only` is best by the main RGB-D `ATE-SE3` metric.
- `geom_dynamic_reject` is best by `ATE-Sim3`, `RPEt-SE3`, and `RPER`, suggesting better local motion but worse global consistency under fixed scale.
- The large Sim3 scale correction indicates substantial global scale or path-length mismatch on the full sequence, so smoke30 conclusions should not be extrapolated without segment-level diagnosis.
