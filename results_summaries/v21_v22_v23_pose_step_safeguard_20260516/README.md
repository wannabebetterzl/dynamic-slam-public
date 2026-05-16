# V21-V23 Pose-Step Safeguard Evidence

This folder contains lightweight public summaries for the current dynamic RGB-D SLAM investigation.

## What To Read First

- `v21_sequential_equivalence_summary.csv`: V21 no-heap equivalence and residual-verification ablation summary.
- `v22_contextual_residual_summary.csv`: V22 contextual residual gate summary.
- `v23_segment_residual_summary.csv`: V23 short-window segment residual controller summary.
- `v23_segment_residual_pose_step_events.csv`: V23 pose-step event log across paired V22/V23 runs.

## Current Interpretation

V21 showed that naive residual hard veto can help one sequence while damaging another, and also exposed that heap allocation in the tracking hot path can break equivalence.

V22 converted residual failure into a contextual dynamic-occlusion hazard test. It protected `wrpy` better than V21 but did not provide a robust segment-level explanation.

V23 added a short-window segment controller. In same-binary paired tests, it improved `wrpy-D` from V22 SE3 ATE `0.133680m` to `0.110625m`, mainly by expanding rejection around frames `818-830`. It slightly hurt the current `whalfsphere` good-basin run, from `0.031078m` to `0.033667m`, suggesting the next step should add a pose-chain-critical escape or soft hold exit.

Important reproducibility caveat: comparisons should be made within the same binary/build batch and dataset condition. Do not compare `external_wrpy_rawrgb_rawdepth_mask` directly with `ablation_ei_D_filtered_rgb_filtered_depth_wrpy`.
