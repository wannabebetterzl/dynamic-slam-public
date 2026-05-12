# 20260504 YOLOE + SAM3 Box Fallback 对比结论

## 实验口径

- 数据集：`TUM RGB-D / freiburg3_walking_xyz`
- 后端：`ORB-SLAM3 RGB-D`
- 当前前端：`YOLOE + SAM3 + person_v2_local gating`
- 本次唯一变量：将 `allow_box_fallback` 从 `false` 改为 `true`

## 三条链路

| 链路 | 前端说明 | 后端说明 | ATE RMSE (m) | RPE RMSE (m) |
| --- | --- | --- | ---: | ---: |
| 旧 best | `E6-9` 旧主线 `YOLO-World + SAM + dynmem` | 仓库历史 benchmark | `0.018325` | `0.013280` |
| SAM3 无 fallback | `YOLOE + SAM3`，`allow_box_fallback=false` | ORB-only 评估 | `0.081587` | `0.027844` |
| SAM3 有 fallback | `YOLOE + SAM3`，`allow_box_fallback=true` | 直接对导出 `sequence` 单跑 ORB | `0.016951` | `0.011670` |
| YOLOE boxmask | `YOLOE` 检测框直接整框删 | 直接对导出 `sequence` 单跑 ORB | `0.192011` | `0.022747` |

## 前端统计对比

| 链路 | mean filtered detections | mean mask ratio | mean foundation |
| --- | ---: | ---: | ---: |
| 旧 best | `0.991851` | `0.112470` | `0.684030` |
| SAM3 无 fallback | `0.827707` | `0.094477` | `0.493020` |
| SAM3 有 fallback | `1.110594` | `0.136234` | `0.493020` |
| YOLOE boxmask | `0.977881` | `0.203930` | `0.393012` |

## 直接结论

1. `SAM3` 在当前本地链路下，如果不启用 `box fallback`，前端对人体动态区域的删除明显偏弱，最终会把 `ATE` 拉高到 `0.0816 m` 左右。
2. 启用 `box fallback` 后，前端删除覆盖显著增强：
   - `mean filtered detections` 从 `0.8277` 提升到 `1.1106`
   - `mean mask ratio` 从 `0.0945` 提升到 `0.1362`
3. 这种增强不是“只是删得更多”，而是切实恢复了最终精度：
   - `ATE RMSE` 从 `0.081587 m` 降到 `0.016951 m`
   - 同时略优于旧 best `0.018325 m`
4. 如果进一步把策略简化成“检测框中的像素全部删除”，速度会大幅提升，但精度明显变差：
   - `mean_runtime_ms = 43.389`
   - `ATE RMSE = 0.192011 m`
   - `mean_mask_ratio = 0.203930`

## 当前判断

- 之前 `YOLOE + SAM3` 版本退化到 `0.09` 左右，核心问题之一确实是：`SAM3` 在当前配置下没有 fallback，导致人体删除不完整。
- 打开 `box fallback` 后，`YOLOE + SAM3` 这套前端已经具备竞争力，至少在 `walking_xyz` 这一条序列上，结果不再劣于旧主线。
- 但“box fallback”不等于“整框全删”。前者是在 `SAM3` 失败或不稳定时用检测框补洞，后者则是把整个检测框都视为动态区域。这两者的物理意义和结果完全不同。
- 从当前结果看，`YOLOE boxmask` 会明显过删，损伤静态支撑，因而不适合作为最终主线。
- 在当前 `walking_xyz` 上，后续的轻度外扩实验进一步表明：保守地把 `dilate_pixels` 从 `5` 提到 `7` 能带来小幅正收益，`ATE RMSE` 可进一步降到 `0.016268 m`。
- “头部/四肢补刀”单独使用或叠加轻度外扩，都没有超过“仅轻度外扩”的结果。
- 但 `mean foundation` 仍然停在 `0.493`，明显低于旧 best 的 `0.684`。这意味着：
  - 当前改进主要来自删除覆盖恢复；
  - `YOLOE + SAM3` 的可靠性打分分布，仍未完全对齐旧主线。

## 备注

- 本实验目录此前通过 benchmark 脚本跑 ORB 时触发 `1800s` 超时，因此最终 ATE 是在**不重跑前端**的前提下，直接对已导出的 `sequence` 单独运行 ORB-SLAM3 后计算得到。
- 因此前端统计与最终 ORB 结果是一一对应的，没有混入新的前端变量。
