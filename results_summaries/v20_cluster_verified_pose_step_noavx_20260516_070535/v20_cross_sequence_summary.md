# V20 cluster-verified pose-step cross-sequence summary

| run | SE3 ATE | Sim3 ATE | scale | RPEt | RPER | skipped | skip frames | extreme candidates |
|---|---:|---:|---:|---:|---:|---:|---|---|
| wrpy_v20 | 0.158603 | 0.100050 | 0.553584 | 0.020603 | 0.520047 | 2 | 579;580 | 578;579;580;845 |
| whalfsphere_v20 | 0.115754 | 0.085760 | 0.860235 | 0.014489 | 0.471753 | 1 | 91 | 89;91;239 |
| wxyz_D_v20 | 0.017357 | 0.015269 | 0.973256 | 0.011647 | 0.368518 | 0 | - | - |
| wxyz_A_raw_v20 | 0.018245 | 0.015932 | 0.971238 | 0.011847 | 0.377084 | 0 | - | 217 |
| wstatic_v20 | 0.016487 | 0.011071 | 0.652841 | 0.011624 | 0.250408 | 0 | - | - |

Key observation: V20 does not generalize cleanly. It is stable on wxyz/wstatic, but degrades walking_rpy and walking_halfsphere differently. whalfsphere is damaged by skipping frame 91 after frame 89 forms a low-support cluster; wrpy only skips 579/580 and misses other V19-beneficial skip frames.
