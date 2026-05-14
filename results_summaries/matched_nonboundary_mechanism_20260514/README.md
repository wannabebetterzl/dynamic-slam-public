# Matched Non-Boundary Mechanism Control, 2026-05-14

This run tests whether the `D2MA-B r5` gain on `walking_rpy` can be explained by generic map sparsification.

Protocol:

- Dataset: `external_wrpy_rawrgb_rawdepth_mask`
- Profile: `hybrid_sequential_semantic_only`
- Method under test: `matched_nonboundary_r5`
- Repeats: 3
- Side-channel isolation: enabled
- Mask mode: off
- Matching bin: image grid `4x3` + depth bin + ORB octave, with depth+octave fallback

Main result:

| case | n | valid | ATE-SE3 mean +/- std | ATE-Sim3 mean +/- std | scale mean +/- std | final KFs | final MPs |
|---|---:|---:|---:|---:|---:|---:|---:|
| `wrpy_matched_nonboundary_r5` | 3 | 1 | `0.390071 +/- 0.000000` | `0.145529 +/- 0.000000` | `0.233247 +/- 0.000000` | 412 | 6803 |

Mechanism comparison:

| case | ATE-SE3 | scale | final KFs | final MPs | CKF direct | CKF boundary/control | LM boundary/control | matched-control |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `wrpy_d2ma_min` | 0.474824 | 0.172691 | 430 | 7347 | 102698 | 0/0 | 0/0 | 0 |
| `wrpy_d2ma_b_r5` | 0.269999 | 0.361678 | 370 | 6006 | 103407 | 8179/0 | 587/0 | 0 |
| `wrpy_samecount_nonboundary_r5` | 0.604425 | 0.138884 | 427 | 7389 | 104739 | 0/7988 | 0/209 | 0 |
| `wrpy_matched_nonboundary_r5` | 0.390071 | 0.233247 | 412 | 6803 | 104839 | 0/0 | 0/0 | CKF 7063, LM 107 |

Interpretation:

- Equal-count non-boundary suppression is worse than `d2ma_min`, so the D2MA-B gain is not explained by pruning strength alone.
- Distribution-matched non-boundary suppression improves over `d2ma_min` but remains worse than `D2MA-B r5`.
- This supports the claim that dynamic-boundary-aware map admission carries causal signal beyond generic sparsification.
- The result should still be stated carefully: admission distribution affects scale and stability, so boundary targeting is a major mechanism, not the only mechanism.

Files:

- `matched_nonboundary_repeat_raw.csv`
- `matched_nonboundary_repeat_summary.csv`
- `matched_nonboundary_event_summary.csv`
- `wrpy_mechanism_comparison_event_summary.csv`
- `wrpy_mechanism_comparison_per_frame.csv`
