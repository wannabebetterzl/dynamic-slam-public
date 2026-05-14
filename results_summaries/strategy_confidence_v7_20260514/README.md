# V7 Strategy Confidence Audit

This audit checks whether the next strategy should continue tuning V7 or pivot to constraint-role diagnostics before designing V8.

## What Is Solid

- V7 is reproducible under the fixed side-channel protocol. Re-running V7 produced byte-identical `CameraTrajectory.txt` and `KeyFrameTimeline.csv` for both walking_rpy and walking_xyz.
- The negative walking_rpy result is not caused by a shortened emitted trajectory. Both V4 and V7 output all 909 associated RGB-D frames; the evaluator reports 906 GT matches because ground truth is higher-rate than the RGB-D sequence.
- V7 changes the map and keyframe structure, so it is not a no-op: walking_rpy final KFs/MPs change from `366/6056` to `387/6506`.
- V7 does not fix walking_rpy scale or SE3 ATE in this protocol: walking_rpy changes from `ATE SE3=0.285427, scale=0.339917` to `ATE SE3=0.305616, scale=0.314618`.
- V7 gives only a small deterministic walking_xyz improvement: `ATE SE3=0.018035 -> 0.017738`, `ATE Sim3=0.016368 -> 0.015810`.

## Loopholes Found

- The current V7 probation counters are event aggregates across repeated culling passes, not unique MapPoint lifetime counts.
- `V7 candidates/allowed` are pre-geometry gate events; they do not equal actually triangulated/created MapPoints.
- V7 charges its promotion quota when a candidate is allowed, before geometry confirms a point is actually created. This can under-admit if many allowed candidates fail geometry.
- V7 coverage pressure is computed once per current keyframe and is not updated cell-by-cell after promotions. The gate may admit redundant candidates in already-covered areas while missing true holes.
- Existing pose-use evidence comes from tracking pose optimization, not from Local BA, covisibility graph strength, or long-term scale-support role.
- The current summaries prove reproducible behavior on two dynamic TUM-style sequences, but not generality across dynamic categories, static negative controls under the same V7 code path, or mask-quality perturbations.
- The small walking_xyz gain is deterministic but too small to treat as a publication-grade contribution without broader ablations.

## Proper Fixes Before Claiming A New Algorithm

- Add constraint-role logging before changing the admission rule again: for score/V7-admitted points, log whether they enter tracking pose optimization, Local BA, covisibility edges, and later mature map support.
- Split V7 metrics into pre-geometry eligible, geometry-created, probation-unique, and repeated-event counters.
- Move quota accounting from pre-geometry allow to post-geometry creation, or log both quota-consumed and actual-created counts.
- Replace global keyframe coverage pressure with cell-aware coverage pressure that tracks which near-boundary grid cells remain unsupported after each accepted/created candidate.
- Add frame-bin divergence analysis for V4 vs V7 on walking_rpy: compare keyframe interval, estimated step, inlier count, local map matches, V7 admissions, and scale/path-ratio drift.
- Treat V7 as a mechanism probe, not as the final method. The next defensible strategy is V8 constraint-aware dynamic map maintenance, but only after constraint-role evidence confirms which missing constraint type drives walking_rpy.

## Confidence Statement

I am highly confident that continuing to blindly tune V7 thresholds is the wrong next move.

I am not 100% confident in the full causal explanation yet. The missing evidence is exactly what the next diagnostic step should collect: whether admitted near-boundary points become tracking-only supports, Local-BA constraints, covisibility anchors, or scale-relevant structure.

The revised strategy is therefore:

1. Freeze V7 as a reproducible mechanism probe.
2. Patch logging to measure constraint role and unique lifetime outcomes.
3. Run V4/V7/V8-prototype comparisons only after those logs can explain how map admission affects tracking, Local BA, keyframe chain, and scale.
4. Promote to a new algorithmic claim only if the logs show a repeatable causal path from dynamic-boundary admission to improved constraint quality or coverage.

## Files

- `v7_repeat_check_summary.csv`: original-vs-repeat case metrics and event totals.
- `v7_repeat_check_per_frame.csv`: per-frame repeat-check event alignment.
