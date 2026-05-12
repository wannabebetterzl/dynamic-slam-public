# Public Snapshot Scope

This file records what was included and excluded when preparing the public GitHub package.

## Included

- Latest YOLOE + SAM3 frontend code from:
  `/home/lj/d-drive/CODEX/basic_model_based_SLAM`
- Current ORB-SLAM3 dynamic backend code from:
  `/home/lj/dynamic_SLAM/stslam_backend`
- Object frontend scaffold from:
  `/home/lj/dynamic_SLAM/object_frontend`
- Unified evaluator and KITTI helpers from:
  `/home/lj/dynamic_SLAM/scripts`
- Experiment notes from the Obsidian vault:
  `动态改进Visual SLAM实验记录.md`,
  `多维度结合提升.md`,
  `STSLAM复刻.md`,
  `动态改进Orb SLAM3.md`
- Small result summaries under:
  `results_summaries/`

## Excluded

- Raw datasets.
- Full RGB/depth/mask/meta frame sequences.
- Model weights such as YOLOE/SAM3 checkpoints.
- ORB vocabulary file `ORBvoc.txt`.
- Compiled binaries, object files, shared libraries, wheels, and build folders.
- Private Obsidian notes unrelated to this project.
- The abandoned STSLAM reproduction workspace.

## Why This Shape

The goal is to let an external reasoning model inspect the active code and understand the experimental evidence without uploading tens of gigabytes of data or exposing private material. This repository is therefore a diagnosis snapshot, not a fully reproducible artifact.
