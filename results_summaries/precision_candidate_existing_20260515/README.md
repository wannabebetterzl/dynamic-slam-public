# Precision-safe Existing Candidate Screening (2026-05-15)

Run root: `/home/lj/dynamic-slam-public/runs/precision_candidate_existing_20260515_152926`

Baseline anchor for comparison:

| sequence | method | SE3 ATE | Sim3 ATE | Sim3 scale |
|---|---|---:|---:|---:|
| wrpy | `d2ma_b_r5_precision_mainline` | 0.285882 | 0.128593 | 0.336465 |
| wxyz | `d2ma_b_r5_precision_mainline` | 0.017360 | 0.015996 | 0.978029 |

Candidate results:

| sequence | method | SE3 ATE | Sim3 ATE | Sim3 scale | decision |
|---|---|---:|---:|---:|---|
| wrpy | `d2ma_supportq_lm_only_b_r5_s18_m2_o2_q2_precision` | 0.359653 | 0.139160 | 0.262499 | reject |
| wxyz | `d2ma_supportq_lm_only_b_r5_s18_m2_o2_q2_precision` | 0.017235 | 0.015260 | 0.974021 | weak positive only on wxyz |
| wrpy | `d2ma_score_lm_only_v4_precision` | 0.319296 | 0.131744 | 0.302709 | reject |
| wxyz | `d2ma_score_lm_only_v4_precision` | 0.018409 | 0.015859 | 0.969809 | reject by SE3 |

Interpretation:

- Existing support-quality and score-based LM-only admission do not provide a reproducible ATE-improvement path for the current strict protocol.
- The mixed behavior is sequence-dependent: supportq can slightly improve already-good wxyz, but it strongly hurts wrpy.
- This supports shifting the next innovation step away from pure point-admission scoring and toward state-aware map-point lifecycle management: when to supplement points, how newly admitted points enter Local BA, and how quickly weak or scale-harming points are culled.

