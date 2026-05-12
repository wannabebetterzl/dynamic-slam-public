# Public Snapshot Scope

This file records what was included and excluded when preparing the public GitHub package.

## Included

- Latest YOLOE + SAM3 frontend code from:
  `/home/lj/d-drive/CODEX/basic_model_based_SLAM`
- Current ORB-SLAM3 dynamic backend code from:
  `/home/lj/dynamic_SLAM/stslam_backend`
- Local dataset path registry and run wrappers:
  `data/datasets.json`,
  `scripts/dslam_data.py`,
  `scripts/run_backend_rgbd.sh`,
  `scripts/run_frontend_inference.sh`
- Unified evaluator and KITTI helpers from:
  `/home/lj/dynamic_SLAM/scripts`
- Experiment notes from the Obsidian vault:
  `动态改进Visual SLAM实验记录.md`,
  `动态改进Orb SLAM3.md`
- Abandoned route summary:
  `docs/ABANDONED_ROUTES.md`
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
- DynoSAM source trees and adapter/object-frontend code.

## Why This Shape

The goal is to let an external reasoning model inspect the active code and understand the experimental evidence without uploading tens of gigabytes of data or exposing private material. The snapshot is now intentionally biased toward the current mainline and away from failed reproduction branches, while still recording why those branches were removed.
