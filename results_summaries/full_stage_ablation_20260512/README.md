# Full Stage Ablation After 5.5 Pro Feedback

Date: 2026-05-12

Dataset:

```text
backend_maskonly_full_wxyz
```

Run root, local only:

```text
runs/full_stage_ablation_20260512
```

## Result Table

| Variant | Matched | Coverage | ATE-SE3 RMSE (m) | ATE-Sim3 RMSE (m) | Sim3 scale | ATE-origin RMSE (m) | RPEt-SE3 RMSE (m) | RPER RMSE (deg) | Local-map failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `none_metadata_only` | 851 | 0.295076 | 0.191482 | 0.167528 | 0.729600 | 0.563943 | 0.051918 | 1.136524 | 12 |
| `track_local_map_pre_pose` | 857 | 0.297157 | 0.274240 | 0.247344 | 0.590789 | 0.514820 | 0.019668 | 0.553184 | 0 |
| `before_local_map` | 857 | 0.297157 | 0.388275 | 0.248433 | 0.362142 | 0.646846 | 0.026060 | 0.646083 | 0 |
| `before_create_keyframe` | 857 | 0.297157 | 0.566043 | 0.265760 | 0.219656 | 0.829057 | 0.023246 | 0.593768 | 0 |

## Log Aggregates

| Variant | Filter lines | Stages | Removed matches | Tagged outliers |
|---|---:|---|---:|---:|
| `before_local_map` | 613 | `before_local_map` | 3698 | 88711 |
| `track_local_map_pre_pose` | 613 | `track_local_map_pre_pose` | 52043 | 90193 |
| `before_create_keyframe` | 313 | `before_create_keyframe` | 21349 | 20585 |
| `none_metadata_only` | 0 | none | 0 | 0 |

## Main Takeaways

- The best global ATE comes from no hard semantic deletion, but its local relative motion is the worst.
- `track_local_map_pre_pose` is the best hard-delete stage in local RPE and also improves full ATE-SE3 versus the previous all-stage `semantic_only` result of about 0.303 m.
- `before_local_map` is too early or not aligned with the useful support structure.
- `before_create_keyframe` is too late and produces the worst global trajectory behavior.
- The useful next direction is not more single-stage hard deletion. Use dynamic evidence as soft/capped/support-aware weighting around `track_local_map_pre_pose`.

## Next Target

Implement or verify:

```bash
STSLAM_GEOMETRIC_DYNAMIC_REJECTION_ACTION=soft_weight
STSLAM_GEOMETRIC_DYNAMIC_REJECTION_MAX_REJECT_RATIO=0.10
STSLAM_GEOMETRIC_DYNAMIC_REJECTION_PROTECT_MIN_INLIERS=45
```
