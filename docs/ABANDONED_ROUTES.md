# Abandoned Routes

This page keeps the route decision explicit so future analysis does not drift back into stale code paths.

## Direct STSLAM Reproduction

Status: abandoned as an active implementation route.

Reason:

- The reproduction workspace was an earlier attempt, not the source of the 2026-05-11 active results.
- The active backend is the ORB-SLAM3-derived dynamic-filter code under `backend/orb_slam3_dynamic/`.
- Keeping the failed reproduction code in this public package made the repository harder to read and increased the chance that an external model would analyze the wrong code.

What remains useful:

- The paper idea is still background literature.
- The current repository should not use the failed reproduction workspace as a runnable baseline or development base.

## DynoSAM Adapter / Object Frontend

Status: removed from the active public snapshot.

Reason:

- DynoSAM source-style runs on the current `walking_xyz` bundle were unstable without conservative changes.
- The better DynoSAM variants still did not become the latest metric-improving path for this project.
- The object adapter code was mostly a bridge toward DynoSAM-style object observations, while the current bottleneck is already visible in the raw RGB-D + mask-only ORB backend route.

What remains useful:

- DynoSAM is still a useful conceptual reference for object-motion factor graphs.
- It should not be the next implementation dependency unless the current full-sequence mask-only failure is first explained and bounded.

## Current Mainline

The active line is:

```text
YOLOE/SAM3 frontend -> raw RGB-D plus mask/meta side channel -> ORB-SLAM3 dynamic-filter backend
```

The immediate priority is to make the full `walking_xyz` backend route explainable, stable, and measurable before adding another complex backend.
