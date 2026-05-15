# wrpy segment-aware mechanism analysis

Date: 2026-05-15

Compared runs:

- `mainline`: `/home/lj/dynamic-slam-public/runs/precision_mainline_20260515_151323/wrpy_precision_mainline_r1`
- `v8`: `/home/lj/dynamic-slam-public/runs/lifecycle_v8_precision_20260515_154535/wrpy_lifecycle_v8_precision_r1`
- `v9`: `/home/lj/dynamic-slam-public/runs/lifecycle_v9_noavx_precision_20260515_162554/wrpy_lifecycle_v9_precision_r1`

## Focus Finding

V9 is not failing uniformly on wrpy. Its strongest unresolved SE3 problem is concentrated in the early-middle pose chain.

| segment | frames | V9-mainline SE3 delta | V9-mainline Sim3 delta | V9-v8 SE3 delta | key observation |
|---|---:|---:|---:|---:|---|
| 100-149 | 103-152 | +0.006288 | -0.006641 | -0.052823 | V9 fixes much of V8's local SE3 penalty but still does not beat mainline SE3. |
| 150-199 | 153-202 | +0.069699 | -0.015676 | +0.013302 | Worst contradiction: Sim3 improves but SE3 gets much worse. |
| 200-249 | 203-252 | +0.039158 | +0.010995 | -0.017664 | V9 improves over V8 but remains worse than mainline. |
| 250-299 | 253-302 | +0.008724 | +0.000638 | +0.003797 | Small residual SE3 penalty. |
| 550-599 | 553-602 | +0.005770 | +0.002585 | +0.044282 | Later residual where V9 is worse than both mainline and V8. |

## Mechanism Interpretation

- In matches `150-249`, V9 improves scale-free structure in part of the segment, but the rigid SE3 trajectory gets worse.
- This suggests the remaining wrpy failure is not simply "bad dynamic map points remain"; it is a pose-chain bias problem where keyframe insertion and Local BA constraints move the absolute trajectory unfavorably.
- The `150-299` focus range has more keyframes than mainline: mainline `79`, V8 `94`, V9 `85`. V9 reduces V8's over-fragmentation but still differs from the mainline pose-chain structure.
- In `150-249`, V9 creates `13` score-admitted points from `182` support candidates and delays only `7` probation points across `23` LBA windows. The small delayed count suggests V9's delay gate helps but is too weak to reshape the problematic BA windows.
- In `550-599`, no score-created points occur and delayed count is `0`; this later degradation is likely inherited pose-chain drift or keyframe topology, not direct new-point admission.

## Files

- `wrpy_segment_mechanism_summary.csv`: per-50-match segment table.
- `wrpy_segment_mechanism_summary.txt`: readable summary.
- `wrpy_segment_mechanism_focus_ranges.csv`: merged focus windows.
- `wrpy_segment_mechanism_focus_ranges.txt`: readable focus summary.
