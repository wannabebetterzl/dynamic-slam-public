---
type: research_review
tags:
  - 科研
  - 算法
  - slam
entities:
  - 个人研究
related:
---

更新时间：2026-05-07
#实验记录
## 1. 研究目标

本研究主线是：在动态场景中，利用基础模型提升 Visual SLAM 的定位精度、建图质量与动态鲁棒性。

当前已经形成两条并行研究线：

1. **basic-model 前端增强 ORB-SLAM3 / STSLAM 类后端**
   - 核心问题：如何把 YOLO-World / YOLOE、SAM / SAM3 这类基础模型输出，安全接入传统特征点式 SLAM，而不污染静态建图。
2. **DynoSAM / 动态因子图复现与后续融合**
   - 核心问题：动态对象因子图是否能稳定吸收来自基础模型的动态先验，以及源码默认配置为什么在 `walking_xyz` 上不稳定。

当前阶段的总判断是：

- 基础模型前端对动态场景是有价值的；
- 但“动态对象信息”不能粗暴注入后端；
- **何时删点、删到哪一层、初始化阶段如何处理、动态信息是否进入优化**，会显著改变最终 ATE。

---

## 1.1 实验索引表

说明：

- 这张表是总导航，不替代后文的详细记录；
- `最优指标` 一栏优先写当前该组实验最关键的数值；
- `口径` 用来避免把“图像级过滤”“特征级过滤”“DynoSAM 源码复现”混在一起；
- 时间以目录日期或文档更新时间为准。

| 时间 | 实验组 | 数据集 / 序列 | 前端 | 后端 | 口径 / 关键变量 | 最优指标 | 当前结论 |
|---|---|---|---|---|---|---|---|
| 2026-04-17 | mask-to-ORB 隔离控制 | `walking_xyz` RGB-D | 同一导出序列 `rgb/depth/mask/meta` | ORB-SLAM3 clean RGB-D | 只切 `ORB_SLAM3_MASK_MODE=off/postfilter/premask` | `off`: ATE `0.01578 m` | mask 进入 ORB 提点层是可行的，但在该序列上不自动优于 `off` |
| 2026-04-18 | smoke60 geometry_weight | `walking_xyz` RGB-D | WorldSAM 主线 | ORB-SLAM3 RGB-D | geometry-score 后端权重 | ATE `0.01687 m` | 几何权重版未优于既有最优参考 |
| 2026-04-18 | smoke60 backend_weight_lba | `walking_static` RGB-D | WorldSAM 主线 | ORB-SLAM3 RGB-D | backend weight + LBA | ATE `0.00680 m` | 静态场景可运行，但未刷新最佳参考 |
| 2026-04-18 | temporal_guard_probe120 | `walking_xyz` RGB-D | WorldSAM 主线 | ORB-SLAM3 RGB-D | 长窗口时序传播 | ATE `0.01574 m` | 精度尚可，但前端耗时显著膨胀 |
| 2026-04-18 | keyretention / keytrigger 系列 | `walking_xyz` RGB-D | WorldSAM 主线 | ORB-SLAM3 RGB-D | 时序 key memory 数量与触发阈值 | 最好仍未优于 `0.01538 m` 参考 | “多保留历史”没有自然带来收益，反而可能把旧噪声留下 |
| 2026-05-04 ~ 2026-05-05 | YOLOE+SAM3 mask-only feature layer | `walking_xyz` RGB-D | raw 图像 + donor `mask/meta` | ORB-SLAM3 clean RGB-D | Layer 1 only: `off/premask/postfilter` | `premask`: ATE `0.38209 m` | 在真正 raw 图像上，特征层过滤优于 `off`，且 `premask` 最优 |
| 2026-05-05 | Plan A: premask + layer3 | `walking_xyz` RGB-D | YOLOE + SAM3 mask-only | ORB-SLAM3 clean RGB-D | 固定 `premask`、Layer2 off，只切 Layer3 | layer3 off: ATE `0.3905 m` | Layer 3 在 `premask` 路线上影响很小 |
| 2026-05-05 | Plan B: postfilter + layer3 | `walking_xyz` RGB-D | YOLOE + SAM3 mask-only | ORB-SLAM3 clean RGB-D | 固定 `postfilter`、Layer2 off，只切 Layer3 | layer3 on: ATE `0.3955 m` | Layer 3 在 `postfilter` 路线上有稳定正收益 |
| 2026-05-05 | init30 blur + full chain | `walking_xyz` RGB-D | 前 30 帧图像级 blur，后续特征级三层链 | ORB-SLAM3 clean RGB-D | `INIT_BLUR_FRAMES=30`，比较 `premask/postfilter` | `premask`: ATE `0.3995 m` | 初始化被明显改善，但未自动转化为最佳最终精度 |
| 2026-05-05 | init30 blur + full chain | KITTI Tracking `0004` 单目 | 前 30 帧图像级 blur，后续特征级三层链 | ORB-SLAM3 mono | `INIT_BLUR_FRAMES=30`，比较 `premask/postfilter` | `postfilter`: Sim3 `1.0915 m` | KITTI 单目上 `postfilter` 略优于 `premask` |
| 2026-05-05 | DynoSAM 完全源码参数复现 | `walking_xyz` bundle | DynoSAM 默认前端 | DynoSAM hybrid backend | 安装目录源码参数，尽量零改动 | 未得到完整 ATE | 完全源码参数在当前 bundle 上不稳定，中途欠约束崩溃 |
| 2026-05-05 | DynoSAM 近源码稳定版 | `walking_xyz` bundle | 保守前端：`parallel_run=False`、`joint_off`、`GFTT` 等 | DynoSAM hybrid backend | 源码后端 + 保守前端 | SE3 ATE `0.11948 m`，Sim3 `0.11699 m` | 需要一组保守前端修改才能稳定跑完整序列 |
| 2026-05-05 | DynoSAM 单变量崩溃排查 | `walking_xyz` bundle | DynoSAM | DynoSAM | 单独测试 `parallel_false / flow_true / gftt_only / joint_off` | `joint_off` 最晚崩到约 `frame 39` | `joint optical-flow refinement` 是明显风险放大项，但不是唯一根因 |

### 快速查阅建议

如果后面要快速回看历史，建议按下面方式定位：

- 看 **特征级删点值不值得做**：
  先看本表中 `mask-to-ORB 隔离控制` 和 `YOLOE+SAM3 mask-only feature layer`
- 看 **Layer 3 到底有没有用**：
  先看 `Plan A` 与 `Plan B`
- 看 **初始化是不是关键瓶颈**：
  先看 `init30 blur + full chain`
- 看 **DynoSAM 是否已经稳定复现**：
  先看 `DynoSAM 完全源码参数复现` 与 `DynoSAM 近源码稳定版`

---

## 1.2 配置变量字典表

这一节用于统一本项目里高频出现的实验术语、环境变量和配置别名，避免后续记录中同一个概念反复换说法。

### A. ORB-SLAM3 三层动态过滤相关

| 术语 / 变量 | 所属层级 | 含义 | 当前理解 |
|---|---|---|---|
| `ORB_SLAM3_MASK_MODE=off` | Layer 1 | mask 存在，但不参与后端删点 | 用来做 raw 对照组 |
| `ORB_SLAM3_MASK_MODE=premask` | Layer 1 | 在 ORB 提点前把 mask 注入 extractor，动态区域不提点 | 删除更早、更硬；常常更利于控制动态污染 |
| `ORB_SLAM3_MASK_MODE=postfilter` | Layer 1 | 先提点，再删去落在 mask 上的特征点 | 保留更多候选，便于后续 Layer 2/3 再做筛选 |
| `ORB_SLAM3_DYNAMIC_LAYER=off` | Layer 2 | 不启用跟踪/建图层的动态删点逻辑 | 用于隔离 Layer 1 或 Layer 3 作用 |
| `ORB_SLAM3_DYNAMIC_LAYER=trackmatch` | Layer 2 | 在跟踪匹配层进行动态 veto | 单独使用时收益有限，需结合具体 Layer 1 路线判断 |
| `ORB_SLAM3_DYNAMIC_LAYER=triangulation` | Layer 2 | 在三角化 / 建图入口处 veto 动态点 | 单独使用时通常只是轻微影响 |
| `ORB_SLAM3_DYNAMIC_LAYER=feature_trackmap` | Layer 2 | 打包启用 `trackmatch + triangulation` | 在 `postfilter` 路线上曾出现明显收益，在 `premask` 路线上反而可能变差 |
| `ORB_SLAM3_DYNAMIC_OPTIMIZATION=off` | Layer 3 | 不在优化层额外处理动态相关观测 | 用于做 Layer 3 控制变量 |
| `ORB_SLAM3_DYNAMIC_OPTIMIZATION=on` | Layer 3 | 在 BA / 优化阶段对动态观测再降权、拒绝或门控 | 更像后补救机制，在 `postfilter` 路线上更可能有效 |

### B. 初始化阶段相关

| 术语 / 变量 | 含义 | 当前理解 |
|---|---|---|
| `INIT blur` / `图像级 blur 初始化` | 初始化前若干帧，在动态 mask 内做 Gaussian blur | 主要用于保护初始化，不是最终长期策略 |
| `ORB_SLAM3_INIT_BLUR_FRAMES=N` | 前 `N` 帧启用初始化 blur | `N` 太小可能保护不够，太大可能拖累后续真实观测 |
| `ORB_SLAM3_INIT_BLUR_KERNEL=K` | 初始化 blur 的核大小 | 核越大，动态区域被抹得越彻底，但像素结构也改得越重 |
| `init map points` | 初始化成功后图中的初始地图点数 | 是判断初始化质量的重要代理指标，但不等于最终 ATE |

### C. basic-model 前端相关

| 术语 | 含义 | 当前理解 |
|---|---|---|
| `WorldSAM 主线` | `WorldSamFilterPipeline` 为核心的基础模型前端主线 | 负责检测、分割、时序记忆、门控与导出 |
| `YOLO-World` | 早期开放词汇检测器 | 作为基础开放词汇基线使用 |
| `YOLOE` | 后续替换到的更强检测器 | 在当前研究中主要用于构建更强检测前端 |
| `SAM` / `SAM1` | 早期分割模型 | 原始分割基线 |
| `SAM3` | 后续替换到的新分割模型 | 当前重点用于 walking_xyz / KITTI 新链路实验 |
| `mask-only sequence` | 保留 raw 图像，仅替换/附加 `mask/meta` 的序列 | 用于严格区分“图像级过滤”和“特征级过滤” |
| `filtered image sequence` | 直接改写图像像素后的导出序列 | 更接近图像级删除方案 |

### D. DynoSAM 复现相关

| 术语 / 变量 | 含义 | 当前理解 |
|---|---|---|
| `pure_source` | 完全源码参数，尽量不改前端/后端默认值 | 当前在 `walking_xyz bundle` 上不稳定 |
| `parallel_false` | 把 `parallel_run=False` | 只能略微延后崩溃，不是主因 |
| `flow_true` | `prefer_provided_optical_flow=True` | 不是主因，只带来很有限改观 |
| `gftt_only` | 把 `feature_detector_type` 从 `GFFT_CUDA` 改成 `GFTT` | 有帮助，但单独不够稳定 |
| `joint_off` | `refine_camera_pose_with_joint_of=False` 且 `refine_motion_with_joint_of=False` | 当前已知最明显的风险放大项开关；关闭后能显著延后崩溃 |
| `GFFT_CUDA` | DynoSAM 源码默认 CUDA 角点检测器 | 在当前 bundle 上更激进，但未表现出最好鲁棒性 |
| `GFTT` | CPU/传统 GFTT 检测器 | 更保守，常用于稳定版 |
| `prefer_provided_optical_flow` | 是否优先使用外部提供的 dense optical flow | 打开后系统会走“提供光流”而不是 KLT fallback |
| `IndeterminantLinearSystemException` | GTSAM 欠约束异常 | 当前 DynoSAM 崩溃的核心错误类型 |
| `lxxxx` landmark symbol | DynoSAM / GTSAM 日志中的 landmark 变量符号 | 如果异常邻近变量多为 `lxxxx`，通常说明问题更偏向 landmark 欠约束 |

### E. 结果指标相关

| 指标 | 含义 | 当前使用注意 |
|---|---|---|
| `ATE RMSE` | 轨迹绝对误差均方根 | 当前最常用全局精度指标 |
| `ATE mean` | 轨迹绝对误差均值 | 用于辅助理解误差整体水平 |
| `RPE RMSE` | 相邻位姿相对误差均方根 | 更敏感于局部稳定性 |
| `SE3 ATE` | 刚体对齐后的绝对轨迹误差 | 对 RGB-D / stereo 更自然 |
| `Sim3 ATE` | 带尺度对齐的绝对轨迹误差 | 对单目 KITTI 更重要，因为单目会有尺度漂移 |
| `matched poses` | 成功与 GT 对齐的位姿数 | 单目 KITTI 等场景下是重要辅指标 |

### F. 当前推荐的默认说法

为了让后续实验记录更一致，建议默认采用下面这些表达：

- `图像级过滤`
  不说“前端删图”，统一指直接改写输入图像的策略
- `特征级过滤`
  不说“后端删图”，统一指图像保持 raw，仅通过 mask 影响特征/跟踪/优化
- `Layer 1 / Layer 2 / Layer 3`
  分别指提点层、跟踪建图层、优化层
- `初始化保护`
  统一指 `init blur` 这种只在前若干帧做的图像级策略
- `DynoSAM 完全源码口径`
  专指安装目录参数零改动的那套
- `DynoSAM 近源码稳定口径`
  专指保守前端修改后可跑通全序列的那套

---

## 1.3 问题 - 证据 - 当前判断表

这一节不再按“做了什么实验”来组织，而是按“当前研究里真正卡住的问题”来组织。适合后续写论文讨论、开题答辩和阶段汇报时直接引用。

| 核心问题 | 直接证据 | 当前判断 | 还缺什么证据 |
|---|---|---|---|
| 为什么图像级过滤经常比特征级过滤更好？ | `walking_xyz` 上早期最佳结果常来自图像级删除；而特征级三层链虽然可行，但多次未超过图像级路径 | 图像级过滤更容易保护初始化，并且在更早阶段阻断动态污染；特征级过滤如果删点时机不对，会把动态噪声带进跟踪/建图 | 需要逐阶段埋点：初始化图点数、静态/动态特征数、建图进入点数、BA 拒绝点数 |
| 为什么 `walking_xyz` 和 KITTI 对 `premask/postfilter` 的偏好不同？ | `walking_xyz` 中 `premask` 常更优；KITTI 单目里 `postfilter` 在部分链路上略优 | 两个数据集在传感器形态、动态对象类型、初始化难度、尺度约束上差异很大；`walking_xyz` 更像“保护静态背景”，KITTI 更像“保留足够可跟踪观测” | 需要统一口径下的跨数据集对照，包括静态特征密度、mask 覆盖率、初始化帧统计 |
| Layer 3 为什么有时有效、有时无效？ | `Plan A` 中 `premask + layer3` 几乎无增益；`Plan B` 中 `postfilter + layer3` 有稳定正收益 | Layer 3 更像后补救机制。只有当前面还保留较多候选观测时，它才有空间发挥作用 | 需要更多 `Layer1 x Layer2 x Layer3` 的完整正交实验 |
| 初始化阶段是不是决定性瓶颈？ | `init30 blur` 显著恢复 `init map points`，但最终 ATE 不一定最好 | 初始化非常重要，但不是唯一瓶颈；后续跟踪、建图和优化阶段仍会决定最终上限 | 需要扫 `INIT_BLUR_FRAMES`，并配合后续阶段的埋点统计 |
| 动态区域是不是删得越多越好？ | 当静态特征足够时，全删/强删往往更安全；当删得过多导致初始化或跟踪脆弱时，精度反而变差 | 真正的判据不是“删得多不多”，而是“当前静态约束是否足够支撑系统” | 需要建立基于静态特征点数、静态覆盖率的阈值式分析 |
| DynoSAM 为什么源码默认在 `walking_xyz` bundle 上不稳？ | `pure_source` 早期就崩；`joint_off` 可显著延后崩溃；异常邻近变量多为 `lxxxx` | 更像早期静态 landmark 构图不稳，联合光流细化又放大了这些不稳关系 | 需要对崩溃前的 landmark 构造、图约束数量和关键帧质量做更细日志分析 |
| 基础模型前端的价值会不会被“全删动态点”彻底取代？ | 当前很多结果显示，只要动点全删就能明显改善部分场景；但也有场景出现特征不足、初始化脆弱、或跨数据集失配 | 基础模型的长期价值不在“比全删更激进”，而在“按场景、按阶段决定删哪些、删到哪一层、哪些动态信息保留给后端” | 需要用场景分组、静态点密度与动态比例分析来论证“自适应策略”的必要性 |
| SLAMMOT / 动态因子图是否仍然值得做？ | DynoSAM 当前复现不稳，且前端动态信息一旦接入不当会劣化 ATE | 值得做，但更适合定位成“研究线 / 理论线”，而不是当前最稳的主实验线 | 需要先把复现口径站稳，再考虑与基础模型对象观测融合 |

### 现阶段最重要的三条判断

如果只保留当前阶段最值得反复提醒自己的三条判断，我会写成下面这样：

1. **初始化影响非常深远，但初始化不是唯一问题。**
2. **动态信息不是越早、越多、越强地注入后端越好。**
3. **真正值得做的不是“更强地删动态”，而是“更聪明地决定何时删、删到哪一层、保留什么信息给后端”。**

---

## 2. 实验代码总体框架架构

### 2.1 顶层目录

项目主目录：`/home/lj/dynamic_SLAM`

核心子模块如下：

- `basic_frontend/`
  - 从 `basic_model_based_SLAM` 迁移和保留的基础模型前端
  - 负责检测、分割、时序记忆、门控、图像级过滤、基准评测
- `object_frontend/`
  - 将 mask + depth 转成对象级几何观测
  - 面向未来 DynoSAM / 对象因子图接入
- `stslam_backend/`
  - 传统 ORB-SLAM3 / STSLAM 风格后端复现与改造
- `third_party/DynOSAM/`
  - DynoSAM 源码
- `scripts/`
  - 数据集构造、KITTI mask 生成、DynoSAM 评测辅助脚本
- `results/`
  - 结果汇总、对照实验摘要、DynoSAM 复现实验
- `experiments/`
  - 早期 smoke / probe / bundle / object observation 等实验目录

### 2.2 basic-model 前端主链路

主流程可概括为：

```text
RGB / RGB-D 输入
-> 开放词汇检测（YOLO-World / YOLOE）
-> 分割（SAM / SAM3）
-> 时序一致性与动态记忆
-> 可靠性 / 任务相关性门控
-> 输出 mask / meta / filtered image / sequence
-> ORB-SLAM3 或其它后端评测 ATE / RPE
```

对应关键入口：

- 前端流水线：
  [`world_sam_pipeline.py`](/home/lj/dynamic_SLAM/basic_frontend/scripts/rflysim_slam_nav/world_sam_pipeline.py)
- 单次 RGB-D 基准：
  [`run_rgbd_slam_benchmark.py`](/home/lj/dynamic_SLAM/basic_frontend/scripts/run_rgbd_slam_benchmark.py)
- 论文图导出：
  [`generate_paper_figures.py`](/home/lj/dynamic_SLAM/basic_frontend/scripts/generate_paper_figures.py)

### 2.3 传统后端接入的三层删点逻辑

围绕 ORB-SLAM3 后端，目前已形成一个较清晰的“三层动态过滤”理解框架：

1. **Layer 1：特征提取前/提取时**
   - `premask`：mask 直接进入 ORB extractor，动态区域不提点
   - `postfilter`：先提点，再删去落在 mask 上的点
2. **Layer 2：跟踪 / 建图层**
   - 动态点是否进入 track / map / track-map 关联
3. **Layer 3：优化层**
   - 动态点或相关约束是否在 BA / 优化阶段进一步降权、剔除或门控

这三层后来成为多轮 walking_xyz / KITTI 消融的核心坐标系。

### 2.4 图像级过滤与特征级过滤的区别

目前实验里已经严格区分两种思路：

1. **图像级过滤**
   - 直接改写输入图像
   - 典型手段：Gaussian blur、黑块、灰块、inpaint
   - 好处：对原始特征提取影响直接，初始化常常更稳
   - 风险：会改变像素统计，可能引入不自然边缘
2. **特征级过滤**
   - 图像保持 raw，仅把 mask 传入后端
   - 在提点、跟踪、建图、BA 某一层或多层删点
   - 更接近 DynaSLAM / STSLAM 类系统的工程逻辑
   - 风险：如果删点不彻底或时机不对，会把动态噪声留进图里

### 2.5 KITTI / DynoSAM 相关脚本

关键脚本：

- KITTI mask 生成：
  [`build_kitti_sam_motion_masks.py`](/home/lj/dynamic_SLAM/scripts/build_kitti_sam_motion_masks.py)

该脚本支持：

- `sam1 / sam3 / box` 三种分割模式
- `all / nonrigid / rigid / custom` 类别策略
- 膨胀、源 motion 继承、图像级渲染等选项

这也是后续从“只删人”过渡到“KITTI 中车、人、骑行者差异处理”的关键工具。

---

## 3. 当前实验记录的来源

本文件整理自以下主要记录：

- 总实验记录：
  [`basic_frontend/实验记录.md`](/home/lj/dynamic_SLAM/basic_frontend/实验记录.md)
- DynoSAM 源码复现记录：
  [`results/dynosam_wxyz_source_run_20260505/README.md`](/home/lj/dynamic_SLAM/results/dynosam_wxyz_source_run_20260505/README.md)
- ORB mask-to-backend 控制实验：
  [`results/wxyz_maskmode_isolated_rgbd_v2_summary.md`](/home/lj/dynamic_SLAM/results/wxyz_maskmode_isolated_rgbd_v2_summary.md)
- YOLOE + SAM3 feature-layer / layer3 / init30 实验摘要：
  - [`summary_feature_layer.md`](/home/lj/dynamic_SLAM/results/20260505_yoloe_sam3_maskonly_wxyz/summary_feature_layer.md)
  - [`summary_planA_layer3_premask.md`](/home/lj/dynamic_SLAM/results/20260505_yoloe_sam3_maskonly_wxyz/summary_planA_layer3_premask.md)
  - [`summary_planB_layer3_postfilter.md`](/home/lj/dynamic_SLAM/results/20260505_yoloe_sam3_maskonly_wxyz/summary_planB_layer3_postfilter.md)
  - [`summary_init30_blur_fullchain.md`](/home/lj/dynamic_SLAM/results/20260505_init30blur_fullchain/summary_init30_blur_fullchain.md)
- 项目结构说明：
  - [`dynamic_SLAM/README.md`](/home/lj/dynamic_SLAM/README.md)
  - [`basic_frontend/README.md`](/home/lj/dynamic_SLAM/basic_frontend/README.md)

说明：

- 部分早期实验没有统一格式化元数据；
- 这类实验的时间主要依据目录名中的日期，如 `20260418_*`、`20260505_*`；
- 若没有精确运行时间，则记为“按目录日期归档”。

---

## 4. 实验历史时间线

## 4.1 2026-04-17 左右：mask 进入 ORB extractor 的隔离控制实验

### 实验目的

验证在同一份 `walking_xyz` 导出序列上，只改变 mask 接入 ORB-SLAM3 的方式，精度会如何变化。

### 实验架构

- 数据序列：
  `/mnt/d/CODEX/basic_model_based_SLAM/experiments/20260417_joint_maskorb_rebase96_full_wxyz/sequence`
- 后端：
  `/home/lj/isolated_eval/orbslam3_clean_frontend/Examples/RGB-D/rgbd_tum`
- 控制变量：
  - 同一份 `rgb/depth/mask/meta`
  - 只切换 `ORB_SLAM3_MASK_MODE`

### 对比参数

- `off`
- `postfilter`
- `premask`

### 结果

| 模式 | init map points | ATE RMSE (m) | ATE mean (m) | RPE RMSE (m) |
|---|---:|---:|---:|---:|
| off | 522 | 0.0157846753 | 0.0138279689 | 0.0120483913 |
| postfilter | 507 | 0.0160432087 | 0.0138792769 | 0.0122079495 |
| premask | 604 | 0.0161306613 | 0.0139159346 | 0.0118950469 |

### 简单分析

- `off` 仍然是最优 ATE；
- 但 `premask / postfilter` 只略差，没有灾难性退化；
- 说明 **mask 进入 ORB 提点层本身是可行的**；
- 同时也说明：在 `walking_xyz` 这类 RGB-D 场景里，动态区域删点并不一定自动带来更低 ATE。

### 阶段结论

这轮实验奠定了后续研究基线：

- 问题不是“mask 一进入后端就完全不可用”；
- 真正的问题是：**删点位置、删点强度、初始化时机** 会决定最终收益。

---

## 4.2 2026-04-18：basic_frontend smoke / probe 阶段

### 实验目的

围绕 `WorldSAM` 前端主线，验证：

- 不同后端权重策略是否有效；
- 动态记忆保留策略是否有帮助；
- 时序传播的收益与开销是否平衡。

### 实验架构

- 前端：
  `WorldSamFilterPipeline`
- 检测/分割：
  以 `world_sam_pipeline_foundation_panoptic_person_v2_local.json` 为主线
- 后端：
  ORB-SLAM3 RGB-D
- 数据：
  `walking_xyz` / `walking_static`

### 代表性保留结果

#### A. `walking_xyz` smoke60，geometry_weight

- 时间：2026-04-18
- 目录：
  `experiments/20260418_geometry_weight_smoke60_wxyz`
- 参数调整：
  - 几何感知权重版后端
- 结果：
  - 前端总耗时：`26.63 s`
  - 前端平均：`307.65 ms/frame`
  - ORB-SLAM3 总耗时：`8.93 s`
  - ATE RMSE：`0.0168721700 m`

#### B. `walking_static` smoke60，backend_weight_lba

- 时间：2026-04-18
- 目录：
  `experiments/20260418_backend_weight_lba_smoke60_wstatic`
- 结果：
  - 前端总耗时：`26.50 s`
  - 前端平均：`311.26 ms/frame`
  - ORB-SLAM3 总耗时：`9.21 s`
  - ATE RMSE：`0.0068040449 m`

#### C. `walking_xyz` 长窗口 temporal_guard_probe120

- 时间：2026-04-18
- 目录：
  `experiments/20260418_temporal_guard_probe120_wxyz`
- 结果：
  - 前端总耗时：`365.78 s`
  - 前端平均：`2910.90 ms/frame`
  - ORB-SLAM3 总耗时：`16.02 s`
  - ATE RMSE：`0.0157355978 m`

### 同期失败尝试

#### 1. backend_weight_lba（walking_xyz）

- 目录：
  `experiments/20260418_backend_weight_lba_smoke60_wxyz`
- ATE RMSE：`0.0157374033 m`
- 结论：
  没有超过既有最优参考 `0.0153765640 m`

#### 2. geometry_weight（walking_xyz）

- 目录：
  `experiments/20260418_geometry_weight_smoke60_wxyz`
- ATE RMSE：`0.0168721700 m`
- 结论：
  不仅没提升，反而比 `backend_weight_lba` 更差

#### 3. keyretention_v2 / v3 / keytrigger_tight

- 时间：2026-04-18
- 结论概括：
  - 保留更多历史 key memory 并未改善 ATE
  - 收紧 key 触发阈值也收益有限
  - 说明“单纯增加时序记忆强度”会把旧噪声一并保留

### 阶段结论

- 当前瓶颈主要在前端导出阶段，不在 ORB-SLAM3 后端；
- 几何权重、key memory 这类“精细调权”并没有比已有主线更稳；
- **时序机制确实重要，但不是越强越好。**

---

## 4.3 2026-05-04 ~ 2026-05-05：YOLOE + SAM3 替换后，mask-only 序列与后端三层删点研究

这是当前阶段最关键的一批实验。

### 背景

这一阶段开始尝试：

- 检测模型升级为 **YOLOE**
- 分割模型升级为 **SAM3**
- 不再直接改写图像，而是构建 **raw 图像 + mask/meta donor** 的 `mask-only` 序列
- 将“图像级删除”和“特征级删除”拆开验证

---

### 4.3.1 feature-layer 对照：raw 图像 + mask only

#### 实验目的

回答一个非常关键的问题：

> 如果不再修改图像像素，而是只把 mask 传给后端，特征层过滤能不能比 raw 更好？

#### 实验架构

- raw RGB-D 来源：
  `/mnt/d/CODEX/basic_model_based_SLAM/experiments/E6-1_full_raw_freiburg3_walking_xyz_20260321/sequence`
- mask/meta donor：
  `/mnt/d/CODEX/basic_model_based_SLAM/experiments/20260504_yoloe_sam3_full_wxyz/sequence`
- 生成的新序列：
  `/home/lj/dynamic_SLAM/results/20260505_yoloe_sam3_maskonly_wxyz/sequence`
- 后端：
  ORB-SLAM3 clean frontend

#### 参数

- `off`：mask 存在，但后端不用
- `premask`
- `postfilter`

#### 结果

| 模式 | init map points | ATE RMSE (m) | ATE mean (m) | RPE RMSE (m) |
|---|---:|---:|---:|---:|
| off | 731 | 0.4357581424 | 0.3772134621 | 0.0218950117 |
| premask | 610 | 0.3820852059 | 0.3121489810 | 0.0173898132 |
| postfilter | 392 | 0.3957635337 | 0.3166855448 | 0.0293173359 |

#### 简单分析

- 在真正的 raw 图像上，`premask / postfilter` 都优于 `off`
- `premask` 最优
- `postfilter` 虽然改善了 ATE，但局部稳定性更差

#### 阶段结论

- 特征层 mask 过滤不是伪命题，它在 raw 图像链路上确实有效
- 但这轮还只是 **Layer 1**
- 还没有把 Layer 2 / Layer 3 独立出来

---

### 4.3.2 Plan A：固定 premask，只看 Layer 3

#### 实验目的

在 `premask` 路线下，验证优化层（Layer 3）是否单独产生明显收益。

#### 实验架构

- sequence：
  `/home/lj/dynamic_SLAM/results/20260505_yoloe_sam3_maskonly_wxyz/sequence`
- 固定：
  - Layer 1 = `premask`
  - Layer 2 = `off`
- 只切换：
  - Layer 3 = `on / off`

#### 结果

| 设置 | init map points | ATE RMSE (m) | ATE mean (m) | RPE RMSE (m) |
|---|---:|---:|---:|---:|
| premask + layer2 off + layer3 on | 610 | 0.3928 | 0.3209 | 0.0221 |
| premask + layer2 off + layer3 off | 610 | 0.3905 | 0.3225 | 0.0233 |

#### 简单分析

- Layer 3 在 `premask` 路线上影响很小
- 开 Layer 3 略微恶化 ATE，但略微改善 RPE
- 不能把它解释成决定性收益来源

---

### 4.3.3 Plan B：固定 postfilter，只看 Layer 3

#### 实验目的

在 `postfilter` 路线下，验证 Layer 3 是否比 `premask` 路线更有用。

#### 结果

| 设置 | init map points | ATE RMSE (m) | ATE mean (m) | RPE RMSE (m) |
|---|---:|---:|---:|---:|
| postfilter + layer2 off + layer3 on | 392 | 0.3955 | 0.3233 | 0.0190 |
| postfilter + layer2 off + layer3 off | 392 | 0.4005 | 0.3304 | 0.0208 |

#### 简单分析

- 在 `postfilter` 路线上，Layer 3 是正收益
- 开 Layer 3 后，ATE / RPE 都变好
- 这支持一个重要推理：
  - `postfilter` 会保留更多候选观测
  - 因而 Layer 3 才有“进一步重加权 / 拒绝”的空间

#### 阶段结论

- Layer 3 不是在所有 Layer 1 路线上都 equally useful
- **它更像是一个“后补救”机制，而不是在任何前端策略下都直接增益**

---

### 4.3.4 Init-30 Gaussian Blur + Full Chain

#### 实验目的

验证“初始化阶段先做图像级删除，之后再切回特征级删点”是否更优。

#### 实验架构

- 前 30 帧：
  - 在动态 mask 内做 Gaussian blur
- 30 帧后：
  - 恢复后端三层特征级链路

共享初始化设置：

- `ORB_SLAM3_INIT_BLUR_FRAMES=30`
- `ORB_SLAM3_INIT_BLUR_KERNEL=21`

Layer 设置：

- Layer 1：`premask` 或 `postfilter`
- Layer 2：`feature_trackmap`
- Layer 3：`on`

#### Walking XYZ 结果

| 设置 | init map points | ATE RMSE (m) | ATE mean (m) | RPE RMSE (m) |
|---|---:|---:|---:|---:|
| init30 blur + premask + layer2 + layer3 | 731 | 0.3995 | 0.3312 | 0.0194 |
| init30 blur + postfilter + layer2 + layer3 | 731 | 0.4268 | 0.3623 | 0.0243 |

#### KITTI Tracking 0004 单目结果

| 设置 | init map points | matched poses | Sim3 RMSE (m) | Sim3 mean (m) | SE3 RMSE (m) |
|---|---:|---:|---:|---:|---:|
| init30 blur + premask + layer2 + layer3 | 386 | 179 | 1.1078 | 0.9441 | 120.8412 |
| init30 blur + postfilter + layer2 + layer3 | 386 | 192 | 1.0915 | 0.8885 | 118.1154 |

#### 简单分析

- `init30 blur` 明显改善了初始化图点数
- 但更好的初始化并没有自动转化为最佳最终 ATE
- 在 `walking_xyz` 上仍是 `premask` 更优
- 在 KITTI 单目上则 `postfilter` 略优

#### 阶段结论

- 初始化阶段确实极重要
- 但初始化不是唯一瓶颈
- 后续阶段的动态污染、特征分配与优化层处理仍然决定上限

---

## 4.4 2026-05-05：DynoSAM `walking_xyz` 源码复现与稳定性分析

### 实验目的

回答两个问题：

1. DynoSAM 在当前 `walking_xyz bundle` 输入链路上，能否用完全源码参数稳定跑通？
2. 如果不能，是什么类型的参数改动在救活它？

### 实验架构

- DynoSAM 仓库：
  `/home/lj/dynamic_SLAM/third_party/DynOSAM`
- 容器：
  `dyno_sam`
- 输入 bundle：
  `/home/lj/dynamic_SLAM/results/wxyz_full_hybrid075_rawparallel_dynosam_bundle_gmflow`
- GT：
  TUM `walking_xyz groundtruth.txt`

---

### 4.4.1 口径 A：完全源码参数

#### 参数定义

直接使用安装后的源码参数目录：

- `/home/user/dev_ws/install/dynosam/share/dynosam/params/`

只额外传入运行 bundle 所必需的启动参数：

- `--data_provider_type=7`
- `--use_backend=true`
- `--dynosam_bundle_emit_optical_flow=true`
- `--disable_ros_display=true`

#### 关键源码默认项

- `parallel_run: True`
- `refine_camera_pose_with_joint_of: True`
- `object_motion_solver.refine_motion_with_joint_of: True`
- `feature_detector_type: GFFT_CUDA`
- `prefer_provided_optical_flow: false`
- `min_features_per_frame: 200`

#### 结果

- 未跑完整
- 中途抛出：
  `gtsam::IndeterminantLinearSystemException`

#### 简单分析

- 当前 `walking_xyz bundle` 输入下，完全源码前端不稳定
- 因而不能直接作为可比较的完整 DynoSAM 基线

---

### 4.4.2 口径 B：近源码但保守的稳定版本

#### 参数调整

- `parallel_run=False`
- `refine_camera_pose_with_joint_of=False`
- `refine_motion_with_joint_of=False`
- `feature_detector_type=GFTT`
- `min_features_per_frame=800`
- `prefer_provided_optical_flow=true`

#### 结果

- 完整跑通 `859` 帧
- 外部对齐得到：
  - `SE3 ATE RMSE = 0.1194821953 m`
  - `Sim3 ATE RMSE = 0.1169891695 m`

#### 简单分析

- DynoSAM 不是“源码一配环境就稳复现”
- 在这条 bundle 输入上，需要保守前端设置组合才能跑完整

---

### 4.4.3 单变量崩溃排查（ending_frame=500）

#### 实验目的

定位到底是哪类设置在导致源码版早期欠约束崩溃。

#### 单变量结果

| 版本 | 相对源码的唯一改动 | 结果 | 最后明确 frame |
|---|---|---|---:|
| pure_source | 无 | 崩溃 | ~6 |
| parallel_false | `parallel_run=False` | 崩溃 | ~10 |
| flow_true | `prefer_provided_optical_flow=True` | 崩溃 | ~11 |
| gftt_only | `feature_detector_type=GFTT` | 崩溃 | ~17 |
| joint_off | `refine_camera_pose_with_joint_of=False` + `refine_motion_with_joint_of=False` | 崩溃但明显延后 | ~39 |

#### 简单分析

- 所有版本都是 `IndeterminantLinearSystemException`
- 邻近变量符号多为 `lxxxx`
- 说明主要是 **landmark 欠约束**

#### 阶段结论

1. `parallel_run` 不是主因
2. `prefer_provided_optical_flow` 不是主因
3. `GFTT` 有帮助，但单独不够
4. `joint optical-flow refinement` 是最明显的风险放大项

更合理的理解是：

- 源码默认前端在这条 `walking_xyz bundle` 上，早期构图时会生成一批不够稳的静态 landmark；
- 联合光流细化又把这些不稳关系更激进地送入后端；
- 因而更容易触发欠约束。

---

## 5. 当前累计得到的研究认识

### 5.1 关于“动态区域是不是删得越多越好”

当前实验支持如下更细化的判断：

1. **当静态特征点足够支撑初始化、跟踪和建图时**
   - 动态区域删得更干净通常更安全
   - 动态点进入静态地图大概率只会污染结果
2. **当动态删除导致静态可用特征不足时**
   - 过强删除会破坏初始化和局部跟踪
   - 这时才有必要考虑保留一部分动态相关观测，或改成后端分层利用

也就是说，问题不是简单的“全删 vs 不删”，而是：

> **当前帧、当前阶段、当前场景下，静态几何约束是否已经足够。**

### 5.2 关于“图像级过滤 vs 特征级过滤”

当前阶段的经验是：

- **图像级过滤**
  - 对初始化更友好
  - 在 `walking_xyz` 上往往更容易得到低 ATE
- **特征级过滤**
  - 更符合传统动态 SLAM 的逻辑
  - 但如果只做到前两层、不处理初始化或 BA，效果容易不如图像级过滤

### 5.3 关于“为什么 Layer 3 有时有效，有时无效”

- 当 Layer 1 已经很激进地删除了大量候选观测，例如 `premask`
  - Layer 3 的可发挥空间很小
- 当 Layer 1 保留了更多候选信息，例如 `postfilter`
  - Layer 3 更容易表现出增益

### 5.4 关于“为什么 DynoSAM / 动态因子图不一定马上优于全删”

当前认识是：

- 动态对象因子图不是天然增益
- 只有当动态对象观测足够稳定、几何有效、时序一致，并且后端优化设计得足够鲁棒时，它才可能优于全删
- 否则它反而会把不稳定约束注入图优化，导致 ATE 变差

---

## 6. 当前阶段最重要的中间结论

### 6.1 已经基本确认的结论

1. `walking_xyz` 上，mask 接入 ORB 后端是可行的，但收益高度依赖接入层级
2. 初始化阶段对最终精度影响非常深
3. `premask` 与 `postfilter` 的优劣是数据集相关的
4. 更好的初始化图点数不等于更好的最终轨迹
5. DynoSAM 完全源码参数在当前 bundle 接入方式上不稳定
6. DynoSAM 的联合光流细化是明显的风险放大项之一

### 6.2 尚未完全解决的问题

1. 为什么图像级过滤常常优于三层特征级删点
2. 为什么 `walking_xyz` 与 KITTI 对 `premask / postfilter` 的偏好不同
3. 特征级删除是否主要输在初始化阶段，还是输在跟踪/建图/BA 中后期
4. DynoSAM 若结合更强基础模型前端，能否形成真正稳定的对象级优化链路

---

## 6.3 文献启发：Dynamic Feature Rejection Based on Geometric Constraint for Visual SLAM in Autonomous Driving

### 文献信息

- 题目：
  *Dynamic Feature Rejection Based on Geometric Constraint for Visual SLAM in Autonomous Driving*
- 作者：
  Li Huang, Zongyang Wang, Juntong Yun, Du Jiang, Can Gong
- 期刊：
  *IEEE Transactions on Intelligent Transportation Systems*, 2025
- DOI：
  `10.1109/TITS.2025.3550122`

### 通读后的核心内容概括

这篇论文的路线不是“高精度语义分割 + 动态建图”，而是一条更轻量、更工程化的动态特征拒绝路线：

1. 使用改进的 `YOLO-FastestV2` 做动态目标检测，快速定位潜在动态目标区域；
2. 在 ORB-SLAM2 前端中，利用光流匹配、双向极线约束和深度几何约束，对动态候选区域内外的特征点做进一步筛查；
3. 保留更可信的静态特征点做位姿估计；
4. 在 TUM 动态数据集上与 `ORB-SLAM2 / DS-SLAM / DynaSLAM` 对比，得到更好的 ATE / RPE 和更快的处理速度。

这篇论文虽然摘要里强调 `loopback detection / semantic + geometric data association`，但全文真正展开最充分的部分，还是：

- 轻量动态检测
- 基于几何一致性的动态特征剔除
- 前端特征级稳定性提升

### 对当前研究最重要的启发

#### 1. 基础模型最适合作为“高召回候选生成器”，而不是最终裁判

这篇论文没有把检测结果直接当成“整块区域全部删除”的绝对依据，而是：

- 先用检测框找出高风险动态候选区域；
- 再用几何一致性去决定哪些特征点真的该删。

这与我们当前最核心的研究矛盾完全同构：

- 只靠基础模型，误删会伤静态建图；
- 只靠几何，又难以在复杂动态下维持高召回。

因此，这篇论文最重要的理论支持是：

> **语义负责高召回候选生成，几何负责高精度二次裁决。**

#### 2. 问题的关键不是“删整块目标”，而是“删目标区域中真正不满足静态几何的点”

这篇论文的本质动作是：

- 在动态目标区域内筛动态点
- 而不是简单把整个目标区域一刀切抹掉

这对当前实验体系很有启发：

- `premask / postfilter` 都还是区域级别思路；
- 下一步更值得做的是：
  **在语义候选区域内，再做特征点级的几何复核。**

#### 3. 框内删点不够，框外漏检动态也必须补救

论文专门做了两步：

1. 先处理检测框内动态特征；
2. 再对框外潜在漏检动态做几何补刀。

这对我们现在的卡点非常重要，因为我们已经多次怀疑：

- 少量漏掉的动态点
- 尤其是人物边缘、框外动态区域

可能足以污染后端。

因此，后续系统不能只做“语义删点”，还要显式设计：

> **语义候选删点 + 几何补刀**

#### 4. 静态特征的可观测性比“删得绝对干净”更重要

文中一个很关键的工程细节是：

- 在估计基础矩阵时，并不是一味减少动态区域点
- 还会考虑保留足够、分布合理的静态点来保证几何求解稳定

这和我们现在对 `init map points`、初始化脆弱性、强删后建图失稳的观察高度一致。

因此，这篇论文进一步支持我们当前的判断：

> **动态特征处理的目标不是删得最多，而是在最小化动态污染的同时，最大化静态几何可观测性。**

#### 5. 我们的工作可以比它再往前走一步

这篇论文的方法仍然偏：

- 轻量检测器
- 目标框级别
- ORB-SLAM2 前端特征剔除

而我们的研究已经自然推进到了更深的问题：

- mask 比 box 更细
- 不仅关心删点，还关心删点进入哪一层
- 不仅关心动态抑制，还关心初始化保护
- 不仅关心前端，还在观察后端图优化污染

因此，我们并不是要照搬它，而是可以把它作为一个很好的“过渡型参照物”：

> 它证明了“语义候选 + 几何约束”这条路是合理的；
> 而我们要进一步回答的是：这种协同在 SLAM 的不同阶段应如何介入，才不会因为删点过强或过弱而伤害整体系统。

### 对论文写作的直接启发

这篇论文提醒我们，后续论文不要写成：

- “我们用了更强的基础模型”

而应该写成：

- “现有动态 SLAM 中，语义信息常被直接当作二值删点依据，缺乏与几何一致性的分阶段协同”
- “我们关注的是语义候选、几何复核和 SLAM 不同阶段之间的关系”

更具体地说，我们的论文动机可以写成：

> 现有动态特征拒绝方法要么过度依赖语义先验，易造成误删和静态观测损失；要么单独依赖几何一致性，难以在复杂动态环境中获得足够高的动态召回。为此，我们关注基础模型驱动的动态候选生成与几何一致性复核之间的协同，并进一步研究这种协同在初始化、跟踪建图和优化阶段应采用的不同介入强度。

### 当前阶段可直接吸收的原则

如果把这篇文献对我们最有价值的启发压缩成几条原则，我会记成：

1. **基础模型输出不应直接作为最终删点依据，而应作为动态候选先验。**
2. **候选区域内的特征点还需要经过几何一致性复核。**
3. **框外 / mask 外的漏检动态点也必须有补救机制。**
4. **删点策略必须服从静态几何可观测性，而不是单纯追求删除彻底。**

---

## 7. 建议的后续实验组织方式

后续实验建议继续按两条主线推进，并严格分口径记录。

### 7.1 主线 A：basic-model 前端 + ORB-SLAM3

优先级最高，因为目前可重复、可解释、结果最稳。

但从当前阶段开始，**这一主线内部也要再分“主研究线”和“控制线”**：

1. **主研究线**
   - 语义候选 + 几何复核
   - 框内 / 框外两段式补刀
   - 静态可观测性建模与埋点分析
2. **控制线 / 基线线**
   - `premask`
   - `postfilter`
   - Layer 2 on/off
   - Layer 3 on/off
   - 初始化窗口 sweep

这里要特别提醒自己：

> 原有 `premask/postfilter/layer2/layer3` 这条线仍然重要，但它现在的角色主要是
> **基线、对照组、解释性实验平台**，而不是后续论文最值得押注的主创新方向。

### 7.2 主线 B：DynoSAM 复现与后续接入

建议先把它当“复现线”，不要急着当主方法。

建议重点做：

1. `joint_off + GFTT` 等组合验证
2. 分析 landmark 欠约束的构图位置
3. 在稳定近源码版上再考虑接入基础模型对象观测

---

## 7.3 下一轮实验计划表

这一节不是实验结果，而是**计划与控制板**。
后续每做一轮实验，都应该对照这里标记：

- 是否按原计划执行
- 实际结果是否支持原假设
- 中途是否发生路线偏离

建议未来在每条实验后面追加：

- `执行日期`
- `是否按计划完成`
- `是否偏离`
- `偏离原因`
- `新结论`

### 7.3.1 当前阶段的主线重排

基于近期实验和 Huang 等人的启发，当前阶段需要明确：

1. **主问题已经不再是继续微调 `premask/postfilter/layer2/layer3` 的排列组合。**
2. **主问题转为：语义候选应如何与几何复核协同，才能既抑制动态污染，又保住静态可观测性。**
3. 因此，原有三层删点实验线保留，但其定位变为：
   - 基线
   - 控制组
   - 解释性支撑实验

如果某一轮实验不能帮助回答下面三条主问题之一，则优先级应自动下降：

- 语义候选是否只能做“高召回候选生成器”，不能直接做最终删点裁决？
- 框内删点之外，框外 / 掩码外漏检动态是否足以主导最终误差？
- 最优策略是否本质上是在“动态污染最小化”和“静态可观测性最大化”之间找平衡？

### A. 高优先级实验

| 编号 | 实验名称 | 核心改动 | 要记录的关键指标 | 预期回答的问题 | 优先级 |
|---|---|---|---|---|---|
| A1 | 语义候选 + 几何复核（框/掩码内） | 在 YOLOE + SAM3 候选区域内，对特征点增加双向光流、一致性、极线/重投影残差复核 | ATE, RPE, 保留静态点数, 被几何复核救回/剔除点数 | 基础模型是否应只做候选生成器 | 最高 |
| A2 | 框外/掩码外两段式几何补刀 | 第一步处理语义候选区域，第二步对候选区外剩余点做几何异常筛查 | ATE, RPE, 框外动态点估计数, 框外补刀比例 | 漏检动态点是否是当前主要污染源 | 最高 |
| A3 | 静态可观测性埋点 | 增加埋点：静态特征数、动态特征数、track-map 进入数、BA 拒绝数、静态覆盖率 | 分帧统计 CSV, 初始化窗口统计, 空间分布统计 | 特征级方法到底输在初始化、跟踪还是 BA | 最高 |
| A4 | 初始化窗口 sweep（服务于主线） | 固定几何复核主链路，扫 `INIT_BLUR_FRAMES = 10 / 20 / 30 / 50` | ATE, RPE, init map points, 初始化前 50 帧静态点数 | 初始化保护需要多长才能为后续几何复核创造稳定起点 | 高 |
| A5 | 几何复核 + Layer 2/3 联动 | 在几何复核主链路上，再接 Layer 2 / Layer 3，观察后续层是否提供额外收益 | ATE, RPE, 每层删点数, BA 拒绝数 | 三层后端机制是在帮忙，还是在重复/放大前端决策 | 高 |

### B. 中优先级实验

| 编号 | 实验名称 | 核心改动 | 要记录的关键指标 | 预期回答的问题 | 优先级 |
|---|---|---|---|---|---|
| B1 | `premask` 与 `postfilter` 跨数据集对照 | 在 `walking_xyz` 与 KITTI 0004 单目上采用完全统一口径复测 | ATE / Sim3, init map points, mask ratio, 静态点密度 | 为什么两个数据集偏好不同 | 中 |
| B2 | Layer 1 × Layer 2 × Layer 3 正交实验 | 统一初始化策略，完整比较 `premask/postfilter` 与 `layer2/layer3 on/off` | ATE, RPE, init map points, 每层删点数 | 哪一层真正产生主导收益 | 中 |
| B3 | 非刚体补刀策略实验 | 在人物主体 mask 基础上，对头部/四肢仅做辅助补全，不参与主门控 | ATE, RPE, 辅助补刀面积占比, 误删率 | 人物边缘漏检是否需要语义补刀 | 中 |
| B4 | mask 膨胀量 sweep | 对 `tight / mild dilate / larger dilate` 做精细对比 | ATE, RPE, 静态点剩余量, 动态残留点估计 | 适度膨胀是否只是“升点”，还是确有几何意义 | 中 |
| B5 | 图像级过滤方式对照 | 比较 `blur / gray fill / black fill / inpaint` | ATE, RPE, init map points, ORB keypoints before/after | 图像级策略中真正有效的是哪种扰动方式 | 中 |

### C. DynoSAM 复现线实验

| 编号 | 实验名称 | 核心改动 | 要记录的关键指标 | 预期回答的问题 | 优先级 |
|---|---|---|---|---|---|
| C1 | `joint_off + GFTT` 组合验证 | 在源码参数基础上联合关闭 joint OF refinement，并切换 GFTT | 是否完整跑通, 崩溃 frame, 相机 ATE | 稳定版到底最依赖哪些保守改动 | 高 |
| C2 | landmark 欠约束日志分析 | 对崩溃前若干帧的 landmark 构图数量与分布做专项日志 | 崩溃前 active values/factors, lxxxx 分布 | 欠约束发生在什么类型的 landmark 构造上 | 高 |
| C3 | DynoSAM + 基础模型对象观测 smoke | 在稳定近源码版上只接入最简单对象观测，不改主优化结构 | 是否稳定, 是否影响相机轨迹, object logs 是否正常 | DynoSAM 是否值得作为后续研究线底座 | 中 |

### D. 当前暂缓的方向

| 编号 | 方向 | 暂缓原因 | 重新启动条件 |
|---|---|---|---|
| D1 | 直接做复杂 SLAMMOT 联合优化 | 当前 DynoSAM 复现口径还不稳，直接推进风险太大 | DynoSAM 稳定版可重复跑通，并能稳定输出对象轨迹 |
| D2 | 直接写规划闭环系统 | 感知链路本身还没收敛，规划层会把问题进一步放大 | 特征级 / 图像级主线先收敛出较稳结论 |
| D3 | 继续单纯追更强基础模型 | 当前主要瓶颈已不再是模型能力，而是接入机制 | 当接入机制明确后，再回头比较新模型收益 |

### E. 路线偏离检查规则

为了避免后续实验再次“做着做着跑偏”，建议每次执行前先对照下面四条：

1. **本轮唯一想回答的问题是什么？**
   - 不能一轮实验同时回答初始化、Layer 2、Layer 3、模型替换四个问题
2. **控制变量是否真的只改了一项？**
   - 如果改了两项以上，必须在记录中明确写成“联合改动”，不能假装是单变量
3. **这轮实验的结果会改变哪个判断？**
   - 如果结果出来也不会改变任何研究判断，那优先级应下降
4. **是否已经有更基础的问题没解决？**
   - 例如 DynoSAM 还没稳定复现，就不应直接推进复杂联合优化

再补一条当前阶段尤其重要的规则：

5. **这轮实验是否在服务“三个主问题”，还是又回到了旧的排列组合打转？**
   - 如果只是继续比较 `premask/postfilter/layer2/layer3`，但没有引入几何复核、框外补刀或静态可观测性分析，那么它默认只算控制实验，不算主线推进。

### F. 当前推荐的执行顺序

如果按信息增益和现实可执行性排序，我建议下一轮实际执行顺序是：

1. `A3 静态可观测性埋点`
2. `A1 语义候选 + 几何复核`
3. `A2 框外/掩码外两段式几何补刀`
4. `A4 初始化窗口 sweep（服务于几何复核主链路）`
5. `A5 几何复核 + Layer 2/3 联动`
6. `B1 walking_xyz / KITTI 统一口径对照`
7. `B2 Layer 1 × Layer 2 × Layer 3 正交实验`
8. `C1 DynoSAM joint_off + GFTT`
9. `C2 DynoSAM landmark 欠约束分析`

一句话总结当前排序原则：

> 先回答“语义和几何怎么协同”这个主问题，
> 再回答“这种协同在 SLAM 各层怎么落地”，
> 最后再回头用旧的三层删点实验做解释和对照。

---

## 8. 当前短期实验计划

1. 将 YOLOE + SAM3 / 旧前端输出的 `mask / box` 从“图像级最终删除依据”改为后端的候选 side-channel。
   - 前端可以继续导出图像级过滤版本作为基线；
   - 但主线实验中，`mask / box` 应显式传入后端作为“高召回动态嫌疑区域”，而不是直接代表“该区域全部删除”。
2. 后端不再执行“mask 内全删”，而是在候选区域内做逐特征几何裁决。
   - mask 内特征点先进入候选池；
   - 再依据光流一致性、极线/重投影误差、RGB-D 深度误差等几何信息判断保留或剔除；
   - 重点记录：候选区内被救回的静态点数、被确认剔除的动态点数、救回点后续是否进入跟踪/建图/BA。
3. 对 mask 外剩余特征只做轻量几何补刀，避免再次变成全局强删点。
   - 只在低跟踪质量、局部残差异常、静态可观测性不足或疑似漏检窗口中触发；
   - 记录 mask 外补刀点数、补刀触发帧、补刀前后 ATE/RPE 与静态点分布。
4. 明确区分三类几何信息，避免把当前实现误称为完整复现论文：
   - 当前后端几何复核主要是 MapPoint 重投影误差 + RGB-D 深度误差；
   - 旧前端中存在 OpenCV Farneback / PyrLK 光流，用于 mask 传播和运动证据；
   - KITTI 准备脚本中可用 UniMatch/GMFlow 生成学习型光流，但这还没有接入 ORB-SLAM3 后端的逐特征几何裁决。
5. 下一轮实验优先实现“候选 side-channel + mask 内逐点几何裁决”，再做 `semantic candidate only / semantic candidate + geometry / candidate + geometry + mask-out补刀` 三组对照。

参考文献：
[1]Huang L, Wang Z, Yun J, 等. Dynamic Feature Rejection Based on Geometric Constraint for Visual SLAM in Autonomous Driving[J]. IEEE Transactions on Intelligent Transportation Systems, 2025, 26(10): 17879-17888.
### 8.1 2026-05-11 执行进度：A1 第一版已落地并完成 smoke30

本轮严格按本节计划推进，没有再扩线到 BA、框外补刀或新的前端结构。

#### 本轮实现

- 在 `stslam_backend/src/Tracking.cc` 的 `ForceFilterDetectedDynamicFeatureMatches()` 中加入第一版 `semantic candidate + geometric verification`
- 仅作用于当前 `postfilter` 路线
- 语义候选点默认仍进入强制过滤
- 但若满足以下条件，则允许“静态救回”：
  - 当前候选特征有已匹配 `MapPoint`
  - 该 `MapPoint` 不是动态绑定点（`instance_id <= 0`）
  - 当前帧静态重投影误差小于阈值
  - 若有深度，则深度残差也需小于阈值

#### 本轮对照设置

数据：

- `walking_xyz smoke30`
- 序列：`/home/lj/dynamic_SLAM/experiments/basic_filtered_rawdepth_smoke30/sequence`

共同环境：

- `ORB_SLAM3_MASK_MODE=postfilter`
- `STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=1`
- `STSLAM_RGBD_DYNAMIC_FRONTEND_SPLIT=0`
- `STSLAM_USE_VIEWER=0`
- `STSLAM_DISABLE_FRAME_SLEEP=1`

唯一变量：

- `semantic-only`：`STSLAM_SEMANTIC_GEOMETRIC_VERIFICATION=0`
- `semantic+geom`：`STSLAM_SEMANTIC_GEOMETRIC_VERIFICATION=1`

结果目录：

- `semantic-only`：`/home/lj/dynamic_SLAM/results/a1_semantic_only_smoke30_wxyz`
- `semantic+geom`：`/home/lj/dynamic_SLAM/results/a1_semantic_geom_smoke30_wxyz`

#### 结果

| 组别 | matched poses | ATE RMSE (m) | ATE mean (m) | RPEt RMSE (m) | RPER RMSE (deg) |
|---|---:|---:|---:|---:|---:|
| semantic-only | 28 | 0.01179 | 0.01039 | 0.01204 | 0.34671 |
| semantic+geom | 28 | 0.01669 | 0.01409 | 0.01220 | 0.39123 |

几何复核统计：

- `semantic-only`：`geom_checked=0`，`geom_rescued=0`，`removed_matches=442`
- `semantic+geom`：`geom_checked=11619`，`geom_rescued=37`，`removed_matches=479`

#### 当前判断

1. 第一版几何复核**已经真实生效**，不是空开关；日志里出现了稳定的 `geom_rescued`
2. 但在这条 `smoke30` 上，**“救回静态点”并没有转化成更好的 ATE**
3. 这说明当前第一版规则还偏粗：
   - 只看单帧静态重投影/深度一致性，可能仍会救回对轨迹不利的边界点
   - 几何复核插入在 `force filter` 层，尚未和初始化窗口、局部跟踪阶段做更细协同
4. 因此，A1 主线没有被否定，但**第一版判据需要继续收紧或改插入位置**

### 8.2 2026-05-11 策略收紧：几何救回默认只允许在 `track_local_map_pre_pose`

为避免几何复核在不同阶段“既删又救”导致变量混乱，当前决定把“静态救回”默认限制在：

- `track_local_map_pre_pose`

而不在：

- `before_local_map`
- `before_create_keyframe`

执行依据：

1. `before_local_map` 时刻，很多候选点尚未经过 `SearchLocalPoints()` 补充本地地图关联，日志里也显示该阶段的拒绝原因几乎被 `missing_map_point` 主导，因此此时“救回”信息量低、误救风险高。
2. `track_local_map_pre_pose` 位于 `UpdateLocalMap() + SearchLocalPoints()` 之后、`PoseOptimization()` 之前，是当前帧局部静态几何约束最充分、同时又还能直接影响当前位姿估计的阶段。
3. `before_create_keyframe` 已经处于“是否把当前观测写入地图”的入口附近。若在这里救回错误的语义候选点，更容易把人体边界等伪静态观测写进地图结构，污染长期建图；而且它对当前帧位姿的直接帮助反而最弱。

因此当前默认策略是：

> 前两处阶段保持保守强删，
> 只在 `track_local_map_pre_pose` 用几何证据对少量静态点做受控救回。

工程实现上，新增环境变量：

- `STSLAM_SEMANTIC_GEOMETRIC_RESCUE_STAGES`

默认值设为：

- `track_local_map_pre_pose`

这样后续如果要做阶段消融，仍可显式改成：

- `before_local_map,track_local_map_pre_pose`
- `before_create_keyframe`
- `*`

但主线实验先坚持最保守、最容易解释的一版。

### 8.3 2026-05-11 运行确认：只在 `track_local_map_pre_pose` 救回，`Observations >= 1`

本轮目标不是扩展实验，而是确认阶段门控是否真正生效。

#### 源码与二进制确认

- 源码位置：`/home/lj/dynamic_SLAM/stslam_backend/src/Tracking.cc`
- `STSLAM_SEMANTIC_GEOMETRIC_RESCUE_STAGES` 默认值：`track_local_map_pre_pose`
- `STSLAM_SEMANTIC_GEOMETRIC_MIN_STATIC_MAP_OBSERVATIONS` 默认值与下限：`1`
- 已重新编译 `stslam_backend/build_noavx`
- 通过 `strings libORB_SLAM3.so` 确认二进制已包含：
  - `geom_stage_enabled`
  - `STSLAM_SEMANTIC_GEOMETRIC_RESCUE_STAGES`
  - `STSLAM_SEMANTIC_GEOMETRIC_MIN_STATIC_MAP_OBSERVATIONS`

#### 本轮设置

结果目录：

- `/home/lj/dynamic_SLAM/results/a1_semantic_geom_stagegated_obs1_smoke30_wxyz`

关键环境变量：

- `ORB_SLAM3_MASK_MODE=postfilter`
- `STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=1`
- `STSLAM_SEMANTIC_GEOMETRIC_VERIFICATION=1`
- `STSLAM_SEMANTIC_GEOMETRIC_RESCUE_STAGES=track_local_map_pre_pose`
- `STSLAM_SEMANTIC_GEOMETRIC_MIN_STATIC_MAP_OBSERVATIONS=1`

#### 运行时阶段统计

| stage | calls | geom_stage_enabled | geom_checked | geom_rescued | removed_matches | tagged_outliers |
|---|---:|---:|---:|---:|---:|---:|
| `before_local_map` | 29 | 0 | 0 | 0 | 110 | 5172 |
| `track_local_map_pre_pose` | 29 | 29 | 5172 | 27 | 376 | 0 |
| `before_create_keyframe` | 7 | 0 | 0 | 0 | 3 | 3 |

这确认本轮几何复核/静态救回**只在** `track_local_map_pre_pose` 生效；其它阶段仍执行语义强过滤，但不做几何救回。

#### ATE 结果

| 组别 | matched poses | ATE RMSE (m) | ATE mean (m) | RPE RMSE (m) |
|---|---:|---:|---:|---:|
| semantic-only | 28 | 0.011793 | 0.01039 | 0.012044 |
| semantic+geom, all-stage 第一版 | 28 | 0.016688 | 0.01409 | 0.012198 |
| semantic+geom, only `track_local_map_pre_pose`, obs>=1 | 28 | 0.043801 | 0.038649 | 0.012077 |

> 重要校正：本节中的 `semantic-only = 0.011793 m` 来自较早实验目录/较早二进制口径，不能再与 2026-05-11 最新二进制下的几何模块结果直接比较。后续当前二进制对照以 8.4 和 8.5 为准。

#### 当前判断

1. 这次实验已经排除了“二进制没有更新”的问题，阶段门控是真实生效的。
2. 只在 `track_local_map_pre_pose` 救回后，ATE 明显劣化到 `0.043801 m`。
3. 由于与 all-stage 第一版相比，`before_local_map` 本来几乎没有救回点，最值得怀疑的差异阶段是 `before_create_keyframe`。
4. 这说明少量发生在关键帧创建入口前的几何救回，可能对后续地图结构和整体轨迹精度有远超数量表面值的影响。
5. 下一步不应直接下结论，而应做阶段消融：
   - `track_local_map_pre_pose only`
   - `before_create_keyframe only`
   - `track_local_map_pre_pose + before_create_keyframe`
   - `all-stage`

### 8.4 2026-05-11 固定最新二进制的三组最小对照

本轮目的是清理实验口径：固定同一个最新二进制、同一条 `smoke30`、同一套 `postfilter + semantic force filter`，只切几何模块开关。

结果根目录：

- `/home/lj/dynamic_SLAM/results/a1_minimal_controls_currentbin_smoke30`

#### 实验组

| 组别 | 关键设置 | 目的 |
|---|---|---|
| `semantic_only` | `STSLAM_SEMANTIC_GEOMETRIC_VERIFICATION=0`, `STSLAM_GEOMETRIC_DYNAMIC_REJECTION=0` | 当前二进制下的纯语义强过滤基线 |
| `geom_framework_noop` | `STSLAM_SEMANTIC_GEOMETRIC_VERIFICATION=1`, `STSLAM_SEMANTIC_GEOMETRIC_RESCUE_STAGES=none`, `STSLAM_GEOMETRIC_DYNAMIC_REJECTION=0` | 验证几何框架空开关是否改变行为 |
| `geom_dynamic_reject` | `STSLAM_SEMANTIC_GEOMETRIC_VERIFICATION=0`, `STSLAM_GEOMETRIC_DYNAMIC_REJECTION=1`, `STSLAM_GEOMETRIC_DYNAMIC_REJECTION_STAGES=track_local_map_pre_pose` | 不救回，改为在局部地图位姿优化前依据几何异常进一步删点 |

#### ATE 结果

| 组别 | matched poses | ATE RMSE (m) | ATE mean (m) | RPE RMSE (m) |
|---|---:|---:|---:|---:|
| `semantic_only` | 28 | 0.040418 | 0.035879 | 0.012118 |
| `geom_framework_noop` | 28 | 0.040415 | 0.035843 | 0.012425 |
| `geom_dynamic_reject` | 28 | 0.021370 | 0.019266 | 0.011835 |

#### 阶段统计

| 组别 | stage | geom checked | geom rescued | semantic removed | geom dynamic checked | geom dynamic removed |
|---|---|---:|---:|---:|---:|---:|
| `semantic_only` | `track_local_map_pre_pose` | 0 | 0 | 352 | 0 | 0 |
| `geom_framework_noop` | `track_local_map_pre_pose` | 0 | 0 | 347 | 0 | 0 |
| `geom_dynamic_reject` | `track_local_map_pre_pose` | 0 | 0 | 431 | 9298 | 2553 |

#### 当前判断

1. `semantic_only` 与 `geom_framework_noop` 的 ATE 几乎相同，说明本轮“几何框架本身”没有引入额外影响，控制变量是干净的。
2. `geom_dynamic_reject` 将 ATE 从约 `0.0404 m` 降到 `0.02137 m`，说明在 `track_local_map_pre_pose` 阶段**继续依据几何信息过滤疑似漏网动态/不一致点**，相比“救回点”更符合当前系统需求。
3. 这支持一个新的主线判断：对当前 `walking_xyz smoke30`，`track_local_map_pre_pose` 更适合作为“几何动态拒绝”阶段，而不是“几何静态救回”阶段。
4. 后续应围绕 `geom_dynamic_reject` 做阈值扫和统计分析，而不是继续加宽救回条件。

### 8.5 2026-05-11 固定最新二进制重新跑三组最小对照

本轮按用户要求重新跑一次三组最小对照，避免直接复用上一轮结果。

结果根目录：

- `/home/lj/dynamic_SLAM/results/a1_minimal_controls_currentbin_rerun_20260511`

固定二进制：

- `/home/lj/dynamic_SLAM/stslam_backend/lib/libORB_SLAM3.so`
- `/home/lj/dynamic_SLAM/stslam_backend/Examples/RGB-D/rgbd_tum`
- 二者时间戳均为 `2026-05-11 16:19:26 +0800`

固定数据与通路：

- 序列：`/home/lj/dynamic_SLAM/experiments/basic_filtered_rawdepth_smoke30/sequence`
- 配置：`/home/lj/dynamic_SLAM/stslam_backend/Examples/RGB-D/TUM3.yaml`
- `ORB_SLAM3_MASK_MODE=postfilter`
- `STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=1`
- `STSLAM_RGBD_DYNAMIC_FRONTEND_SPLIT=0`
- `STSLAM_USE_VIEWER=0`
- `STSLAM_DISABLE_FRAME_SLEEP=1`

#### ATE 结果

| 组别 | matched poses | ATE RMSE (m) | ATE mean (m) | RPE RMSE (m) |
|---|---:|---:|---:|---:|
| `semantic_only` | 28 | 0.041873 | 0.036952 | 0.012973 |
| `geom_framework_noop` | 28 | 0.040418 | 0.035879 | 0.012118 |
| `geom_dynamic_reject` | 28 | 0.020549 | 0.018467 | 0.011967 |

#### 阶段统计

| 组别 | stage | semantic removed | tagged outliers | geom checked | geom rescued | geom dynamic checked | geom dynamic removed |
|---|---|---:|---:|---:|---:|---:|---:|
| `semantic_only` | `before_local_map` | 117 | 5172 | 0 | 0 | 0 | 0 |
| `semantic_only` | `track_local_map_pre_pose` | 391 | 0 | 0 | 0 | 0 | 0 |
| `geom_framework_noop` | `before_local_map` | 100 | 5172 | 0 | 0 | 0 | 0 |
| `geom_framework_noop` | `track_local_map_pre_pose` | 352 | 0 | 0 | 0 | 0 | 0 |
| `geom_dynamic_reject` | `before_local_map` | 122 | 5172 | 0 | 0 | 0 | 0 |
| `geom_dynamic_reject` | `track_local_map_pre_pose` | 402 | 0 | 0 | 0 | 9131 | 2502 |

#### 本轮判断

1. `semantic_only` 日志确认 `geom_verify_enabled=0`、`geom_stage_enabled=0`、`geom_dyn_reject_enabled=0`，所以它确实是不启用几何复核/几何二次剔除的语义基线。
2. `geom_framework_noop` 中 `geom_verify_enabled=1` 但 `geom_stage_enabled=0`，没有发生 `geom_checked` 或 `geom_rescued`，因此是几何框架空跑对照。
3. 两个语义基线结果不完全 bit-identical，说明 smoke30 上仍存在小幅运行波动；但量级仍在 `0.040-0.042 m` 附近。
4. `geom_dynamic_reject` 在本轮仍把 ATE 降到约 `0.0205 m`，与上一轮 `0.02137 m` 方向一致，说明“几何信息用于进一步拒绝动态/不一致点”是当前最值得推进的主线。

### 8.6 2026-05-11 ATE 评估口径校正：`0.011x` 与 `0.04x` 的来源

复查后确认，`semantic_only` 从 `0.011793 m` 变为 `0.040-0.042 m` 的主要原因不是 SLAM 轨迹突然劣化，而是 ATE 计算口径不同。

旧目录：

- `/home/lj/dynamic_SLAM/results/a1_semantic_only_smoke30_wxyz`

旧 `eval.json` 显示：

- `alignment_method = rigid_umeyama_se3`
- `alignment_scale = 1.0`
- `ATE RMSE = 0.011793 m`

而当前脚本：

- `/home/lj/dynamic_SLAM/basic_frontend/scripts/evaluate_slam_runs.py`

只做了首帧平移归零：

- `gt = gt - gt[0]`
- `est = est - est[0]`

没有做 SE3 旋转/平移刚体对齐。因此它会把初始坐标系方向差异也计入 ATE。

#### 同一评估口径复算

| 轨迹 | 无 SE3 对齐 ATE RMSE (m) | SE3 对齐 ATE RMSE (m) |
|---|---:|---:|
| 旧 `semantic_only` 轨迹 | 0.040417 | 0.011793 |
| 新 `semantic_only` 轨迹 | 0.041873 | 0.013474 |
| 新 `geom_dynamic_reject` 轨迹 | 0.020549 | 0.010389 |

#### 结论

1. `0.011793 m` 与 `0.041873 m` 不能直接比较，因为前者是 SE3 对齐 ATE，后者是首帧平移归零、无旋转对齐 ATE。
2. 用同一个当前无对齐脚本重算旧轨迹，旧 `semantic_only` 也是 `0.040417 m`，与当前二进制下 `0.040-0.042 m` 基本一致。
3. 后续表格必须明确标注 ATE 口径：
   - `ATE-SE3`：刚体 SE3 对齐，适合作为 TUM/ORB-SLAM3 常规报告口径。
   - `ATE-origin`：只做首帧平移归零，更严格，但会混入初始坐标系旋转差异。
4. 为了论文和实验严谨性，后续建议同时记录两列，但主报告使用 `ATE-SE3`，附表保留 `ATE-origin` 用于检查初始化坐标系偏差。

### 8.7 2026-05-11 统一 ATE 评估脚本与刚体对齐复算

为了避免后续继续混用评估口径，已新增统一评估脚本：

- `/home/lj/dynamic_SLAM/scripts/evaluate_trajectory_ate.py`

并将旧入口改为兼容转发：

- `/home/lj/dynamic_SLAM/basic_frontend/scripts/evaluate_slam_runs.py`

后续所有 ATE/RPE 计算应统一调用上述脚本。默认口径为：

- `--alignment se3`
- 输出 `alignment_method = rigid_umeyama_se3`
- `alignment_scale = 1.0`

可选诊断口径：

- `--alignment sim3`：允许估计一个全局尺度，适合单目尺度诊断。
- `--alignment origin`：只做首帧平移归零，用于观察初始化坐标系偏差，不作为主报告口径。
- `--alignment all`：同时输出 `SE3 / Sim3 / origin`。

#### 当前 smoke30 刚体对齐复算

结果目录：

- `/home/lj/dynamic_SLAM/results/a1_minimal_controls_currentbin_rerun_20260511`

| 组别 | ATE-SE3 RMSE (m) | ATE-Sim3 RMSE (m) | Sim3 scale | ATE-origin RMSE (m) |
|---|---:|---:|---:|---:|
| `semantic_only` | 0.013474 | 0.013224 | 1.051470 | 0.041873 |
| `geom_framework_noop` | 0.011472 | 0.011468 | 1.005629 | 0.040418 |
| `geom_dynamic_reject` | 0.010389 | 0.010151 | 1.043166 | 0.020549 |

#### 几何救回相关复算

| 组别 | 原先常引用口径 | ATE-SE3 RMSE (m) | ATE-Sim3 RMSE (m) | ATE-origin RMSE (m) |
|---|---|---:|---:|---:|
| `semantic+geom, all-stage 第一版` | SE3 | 0.016688 | 0.015592 | 0.045082 |
| `semantic+geom, only track_local_map_pre_pose, obs>=1` | origin | 0.014071 | 0.013865 | 0.043801 |

#### 修正后的判断

1. 正常 RGB-D / stereo 轨迹精度报告应优先使用 `ATE-SE3`，因为尺度本身已经可观测，不应再通过 Sim3 免费修正尺度。
2. `ATE-origin` 对初始化坐标系方向非常敏感，适合作为诊断指标，但不适合作为主表结论。
3. 之前“stage-gated 几何救回极差”的说法主要来自 `ATE-origin=0.043801 m`；在 `ATE-SE3` 下它是 `0.014071 m`，并没有那么灾难。
4. 但在同一刚体对齐口径下，当前 `geom_dynamic_reject` 仍是这组 smoke30 中最优：`ATE-SE3=0.010389 m`。
5. 因此主线判断应修正为：几何救回不是绝对不可用，但当前证据更支持“语义候选 + 几何动态二次剔除”。

### 8.8 2026-05-11 三条路线推广到 walking_xyz 全序列

本轮目标是把 smoke30 上的三条路线推广到全序列，观察是否仍然成立。

#### 输入与固定项

全序列输入：

- `/home/lj/dynamic_SLAM/results/20260505_yoloe_sam3_maskonly_wxyz/sequence`

该序列特点：

- 859 条 RGB-D association
- `rgb/` 与 `depth/` 均为原始 TUM 数据
- `mask/` 来自 YOLOE + SAM3 前端
- 前端不直接修改图像像素，mask 只作为后端候选输入

固定二进制：

- `/home/lj/dynamic_SLAM/stslam_backend/lib/libORB_SLAM3.so`
- `/home/lj/dynamic_SLAM/stslam_backend/Examples/RGB-D/rgbd_tum`
- 时间戳：`2026-05-11 16:19:26 +0800`

结果根目录：

- `/home/lj/dynamic_SLAM/results/a1_minimal_controls_currentbin_full_wxyz_20260511`

固定环境变量：

- `STSLAM_USE_VIEWER=0`
- `STSLAM_DISABLE_FRAME_SLEEP=1`
- `ORB_SLAM3_MASK_MODE=postfilter`
- `STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=1`
- `STSLAM_RGBD_DYNAMIC_FRONTEND_SPLIT=0`

统一评估：

- `/home/lj/dynamic_SLAM/scripts/evaluate_trajectory_ate.py`
- 主报告口径：`ATE-SE3`
- 同时记录 `ATE-Sim3` 与 `ATE-origin` 作诊断

#### 全序列结果

| 组别 | matched poses | ATE-SE3 RMSE (m) | ATE-Sim3 RMSE (m) | Sim3 scale | ATE-origin RMSE (m) | RPEt-SE3 RMSE (m) | RPER RMSE (deg) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `semantic_only` | 857 | 0.302858 | 0.238408 | 0.495238 | 0.619060 | 0.021200 | 0.571964 |
| `geom_framework_noop` | 857 | 0.330045 | 0.261812 | 0.423952 | 0.663362 | 0.021201 | 0.574752 |
| `geom_dynamic_reject` | 857 | 0.314838 | 0.203879 | 0.479517 | 0.723866 | 0.018447 | 0.482011 |

#### 阶段统计

| 组别 | stage | calls | semantic removed | tagged outliers | geom dyn checked | geom dyn removed |
|---|---|---:|---:|---:|---:|---:|
| `semantic_only` | `before_local_map` | 613 | 1578 | 90083 | 0 | 0 |
| `semantic_only` | `track_local_map_pre_pose` | 613 | 52358 | 0 | 0 | 0 |
| `geom_framework_noop` | `before_local_map` | 613 | 1568 | 90175 | 0 | 0 |
| `geom_framework_noop` | `track_local_map_pre_pose` | 613 | 53177 | 0 | 0 | 0 |
| `geom_dynamic_reject` | `before_local_map` | 612 | 1521 | 89524 | 0 | 0 |
| `geom_dynamic_reject` | `track_local_map_pre_pose` | 858 | 59010 | 0 | 534626 | 323734 |

#### 当前判断

1. 在全序列 `ATE-SE3` 主口径下，最优是 `semantic_only = 0.302858 m`。
2. `geom_dynamic_reject` 没有延续 smoke30 上的 `ATE-SE3` 最优，得到 `0.314838 m`，略差于 `semantic_only`。
3. 但 `geom_dynamic_reject` 的局部指标更好：
   - `RPEt-SE3`: `0.018447 m`，优于 `semantic_only` 的 `0.021200 m`
   - `RPER`: `0.482011 deg`，优于 `semantic_only` 的 `0.571964 deg`
4. `geom_dynamic_reject` 的 `ATE-Sim3 = 0.203879 m` 是三组最优，说明它可能改善了局部相对运动，但全局尺度/姿态漂移仍然明显。
5. 全序列上三组 Sim3 scale 都远离 `1.0`，尤其 `semantic_only=0.495238`、`geom_dynamic_reject=0.479517`，这提示当前 raw RGB-D + mask 后端链路存在较大的全局尺度/轨迹长度偏差，不能只看短序列 smoke30。
6. 下一步应做分段 ATE/RPE 与尺度漂移分析，找出 `geom_dynamic_reject` 从局部收益转为全局 ATE 不占优的发生区间。

### 8.9 2026-05-11 统一 full / smoke 数据路径口径

为了避免后续再次混用不同图像源，之后 `walking_xyz` 的 mask-only 后端实验统一使用以下路径。

#### 后续默认 full 路径

- `/home/lj/dynamic_SLAM/results/20260505_yoloe_sam3_maskonly_wxyz/sequence`

说明：

- 859 条 RGB-D association
- `rgb/` 与 `depth/` 是原始 TUM RGB-D 数据
- `mask/` 与 `meta/` 来自 YOLOE + SAM3 前端
- 图像像素不被前端直接改写，mask 只作为后端动态候选输入
- 适用于后续全序列 `semantic_only / geom_framework_noop / geom_dynamic_reject` 对照

#### 后续默认 smoke30 路径

- `/home/lj/dynamic_SLAM/results/20260505_yoloe_sam3_maskonly_wxyz/smoke30_sequence`

说明：

- 由上述 full 路径的前 30 条 association 派生
- `rgb/`、`depth/`、`mask/`、`meta/` 均通过符号链接指向 full 路径中的同源文件
- `groundtruth.txt` 也链接到 full 路径
- 用于后续快速 smoke test，保证 smoke 与 full 来自同一图像源和同一 mask 源

#### 历史 smoke 路径

- `/home/lj/dynamic_SLAM/experiments/basic_filtered_rawdepth_smoke30/sequence`

使用限制：

- 这是 8.4 / 8.5 等历史当前二进制 smoke 实验使用的路径。
- 该路径第一帧 RGB 与 full mask-only 路径第一帧 RGB 的 MD5 不一致，因此不能作为后续 full 对应的默认 smoke 路径。
- 以后只有在复现实验记录中已有历史结果时才使用它；新实验默认使用 `smoke30_sequence`。

#### 记录规则

之后所有实验表必须显式写明：

- `sequence_full` 或 `sequence_smoke`
- 评估脚本：`/home/lj/dynamic_SLAM/scripts/evaluate_trajectory_ate.py`
- 主评估口径：`ATE-SE3`
- 如报告 `ATE-origin` 或 `ATE-Sim3`，必须在表头中明确标注，不能再笼统写作 `ATE`

### 8.10 2026-05-11 路径口径校正：为什么 full 结果看起来比旧前端最优差很多

用户指出：旧“只处理前端”的 walking_xyz 最优已经达到约 `0.016-0.018 m`，而 8.8 中 full mask-only 后端实验却得到 `0.30 m` 量级，这看起来不合理。

复查后确认：

1. 旧前端最优是**图像级过滤后再送入 clean ORB-SLAM3**，不是当前的 raw RGB-D + backend mask-only。
2. 旧前端最优与当前统一评估脚本一致，不是评估脚本造成的误差。
3. 当前 full mask-only 实验是**原始 RGB-D 不改图像像素，只把 mask 传给后端删点**，因此与旧前端图像级过滤不是同一路线。

#### 旧前端图像级过滤结果复算

统一脚本：

- `/home/lj/dynamic_SLAM/scripts/evaluate_trajectory_ate.py`

| 路线 | 路径 | ATE-SE3 RMSE (m) | RPEt-SE3 RMSE (m) |
|---|---|---:|---:|
| 旧 SAM3 box fallback 前端图像级过滤 | `/home/lj/d-drive/CODEX/basic_model_based_SLAM/experiments/20260504_yoloe_sam3_boxfallback_wxyz` | 0.016951 | 0.011670 |
| 旧 SAM3 mild dilate 前端图像级过滤 | `/home/lj/d-drive/CODEX/basic_model_based_SLAM/experiments/20260504_yoloe_sam3_milddilate_wxyz` | 0.016268 | 0.012387 |
| 旧 person v2 dynamic memory 前端图像级过滤 | `/home/lj/d-drive/CODEX/basic_model_based_SLAM/experiments/E6-9_full_foundation_panoptic_person_v2_dynamic_memory_freiburg3_walking_xyz_20260321` | 0.018325 | 0.013280 |
| 当前 full mask-only 后端 `semantic_only` | `/home/lj/dynamic_SLAM/results/a1_minimal_controls_currentbin_full_wxyz_20260511/semantic_only` | 0.302858 | 0.021200 |

#### 图像源差异

第一帧 RGB/Depth MD5 对比：

| 数据 | RGB MD5 | Depth MD5 |
|---|---|---|
| 旧 SAM3 box fallback 图像级过滤 | `485296440504bc96cc6eba123bcfd479` | `a5f2e1c23907445cfc3face232d5b077` |
| 当前 mask-only / 原始 TUM | `4b39bbb8ba5916262d4aa2292876abb4` | `04bc548c53cbb537efb21eafbf28c5c3` |

这说明旧前端最优并不是“只生成 mask 但图像不变”，而是图像/深度已经被前端处理过。

#### 同源 smoke30 复查

为避免继续把历史 smoke 与 full 混用，已从当前 full mask-only 序列切出同源 smoke30：

- `/home/lj/dynamic_SLAM/results/20260505_yoloe_sam3_maskonly_wxyz/smoke30_sequence`

结果目录：

- `/home/lj/dynamic_SLAM/results/a1_minimal_controls_currentbin_smoke30_from_full_wxyz_20260511`

| 组别 | matched poses | ATE-SE3 RMSE (m) | ATE-Sim3 RMSE (m) | Sim3 scale | ATE-origin RMSE (m) | RPEt-SE3 RMSE (m) |
|---|---:|---:|---:|---:|---:|---:|
| `semantic_only` | 28 | 0.033042 | 0.012760 | 0.634175 | 0.207599 | 0.011829 |
| `geom_framework_noop` | 28 | 0.035047 | 0.012383 | 0.617488 | 0.212023 | 0.011413 |
| `geom_dynamic_reject` | 28 | 0.037777 | 0.014100 | 0.599658 | 0.220700 | 0.012694 |

#### 修正后的结论

1. 当前 `0.30 m` 量级不是评估脚本错误；统一脚本可以复现旧前端图像级过滤的 `0.016-0.018 m`。
2. 真正差异来自实验路线：
   - 旧最优：前端图像/深度级过滤，clean ORB-SLAM3 直接吃处理后图像。
   - 当前 full：原始图像不变，mask 进入改造后端做特征/匹配/几何过滤。
3. 之前“smoke30 semantic_only 已经优于旧前端最优”的说法来自历史 smoke 路径，和当前 full mask-only 不是同源路径，不能作为全序列推断依据。
4. 当前证据反而强化了一个关键事实：**图像级过滤链路目前显著强于后端 mask-only 特征级链路**；后续若要论证后端几何过滤，需要先解决全序列尺度/全局一致性问题。


### 8.11 2026-05-11 旧 YOLOE+SAM3 图像级前端 + 后端 no-mask 几何模块复查

本轮实验来自用户纠偏：重点不是继续做“旧前端 + 后端再接同一份 mask”的双重语义删点，而是把研究迁移到 **YOLOE+SAM3 的 `basic_model_based_SLAM` 旧前端图像级处理链** 上，并在后端**不接入 mask** 的情况下测试几何过滤/几何拯救是否还能带来增益。

#### 实验目标

验证如下路线：

- 输入图像/深度已经由旧前端完成图像级动态处理；
- 后端命令行不传 `mask` 目录；
- 后端只打开几何相关开关，观察几何过滤/几何拯救/二者组合是否优于旧前端单独运行。

#### 固定条件

- 序列：`/home/lj/d-drive/CODEX/basic_model_based_SLAM/experiments/20260504_yoloe_sam3_milddilate_wxyz/sequence`
- 结果目录：`/home/lj/dynamic_SLAM/results/frontend_nomask_geometry_yoloe_sam3_milddilate_wxyz_20260511`
- ORB-SLAM3 二进制：`/home/lj/dynamic_SLAM/stslam_backend/Examples/RGB-D/rgbd_tum`
- 配置：`/home/lj/dynamic_SLAM/stslam_backend/Examples/RGB-D/TUM3.yaml`
- 评估脚本：`/home/lj/dynamic_SLAM/scripts/evaluate_trajectory_ate.py`
- 主评估口径：`ATE-SE3`
- mask 参数：**不传入**，即命令只到 `sequence associations.txt`，不追加 `sequence/mask`

#### 实验组定义

| 组别 | 后端是否接 mask | 几何拯救 | 几何动态过滤 | 说明 |
|---|---|---|---|---|
| `frontend_only_nomask` | 否 | 关 | 关 | 旧前端图像级处理 + 当前二进制直接跑 |
| `geom_rescue_nomask` | 否 | 开 | 关 | no-mask 下测试几何拯救是否实际生效 |
| `geom_filter_nomask` | 否 | 关 | 开 | no-mask 下只做 MapPoint 几何一致性剔除 |
| `geom_filter_plus_rescue_nomask` | 否 | 开 | 开 | no-mask 下二者同时开启 |

关键代码事实：当前 `STSLAM_SEMANTIC_GEOMETRIC_VERIFICATION` 的“几何拯救”逻辑依赖 `frame.GetFeatureInstanceId(idx) > 0` 的语义候选点。后端不接 mask 时，语义实例候选为空，因此几何拯救在当前实现中应当是 no-op。这个结论必须用日志确认，不能只看 ATE。

#### 结果

| 组别 | matched | ATE-SE3 RMSE (m) | ATE-Sim3 RMSE (m) | Sim3 scale | ATE-origin RMSE (m) | RPEt-SE3 RMSE (m) | RPER-SE3 RMSE (deg) | STSLAM 日志行 | geom rescued | geom dyn removed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `frontend_only_nomask` | 857 | 0.020318 | 0.018261 | 0.971176 | 0.454154 | 0.013073 | 0.371851 | 0 | 0 | 0 |
| `frontend_only_nomask_repeat1` | 857 | 0.017424 | 0.016123 | 0.978474 | 0.453639 | 0.012352 | 0.371422 | 0 | 0 | 0 |
| `geom_rescue_nomask` | 857 | 0.016537 | 0.015126 | 0.978225 | 0.450121 | 0.011922 | 0.368745 | 0 | 0 | 0 |
| `geom_rescue_nomask_repeat1` | 857 | 0.017928 | 0.016654 | 0.978369 | 0.452180 | 0.012642 | 0.365986 | 0 | 0 | 0 |
| `geom_filter_nomask` | 857 | 0.038745 | 0.036843 | 0.961377 | 0.456196 | 0.017810 | 0.399654 | 858 | 0 | 188605 |
| `geom_filter_plus_rescue_nomask` | 857 | 0.157365 | 0.144510 | 0.808907 | 0.615007 | 0.016035 | 0.388159 | 858 | 0 | 218652 |

#### 解释

1. `geom_rescue_nomask` 不能解释为“几何拯救有效”。日志显示 `STSLAM_FORCE_DYNAMIC_FILTER` 行数为 `0`，`geom_rescued=0`，说明 no-mask 下没有语义候选点进入救回逻辑。它的 `0.016537 m` 与重复运行 `0.017928 m`，应被归入 ORB-SLAM3 本身的运行波动范围。
2. 当前二进制下，旧前端图像级处理链路的自然波动大约覆盖 `ATE-SE3=0.0165-0.0203 m`。因此单次 `0.016-0.018 m` 的差异不应过度解释。
3. 真正生效的 no-mask 几何动态过滤是 `geom_filter_nomask`，它删除了 `188605` 个匹配，但 ATE-SE3 变差到 `0.038745 m`，说明当前阈值下的全局几何剔除过强，会删掉对跟踪/建图有用的静态或准静态约束。
4. `geom_filter_plus_rescue_nomask` 更差，`ATE-SE3=0.157365 m`，且 `geom_rescued=0`、`geom_dyn_removed=218652`。这不是“过滤+拯救”的平衡效果，而是“过滤更强但没有实际救回”的结果。
5. 目前结论：在旧 YOLOE+SAM3 图像级前端已经足够强的情况下，继续叠加当前版本的后端几何动态过滤没有收益，反而破坏全局一致性。后续若继续做几何模块，应考虑更保守的触发条件，例如只在 tracking 质量下降、局部残差异常、低静态可观测性窗口或 mask 外漏检补刀场景中启用。

#### 关于历史 smoke30 为什么精度特别好

之前几组 `smoke30` 结果之所以看起来非常好，主要不是因为当前 full mask-only 后端链路本身很强，而是因为使用了另一个历史小序列：

- 历史 smoke30：`/home/lj/dynamic_SLAM/experiments/basic_filtered_rawdepth_smoke30/sequence`
- 当前 full-derived smoke30：`/home/lj/dynamic_SLAM/results/20260505_yoloe_sam3_maskonly_wxyz/smoke30_sequence`

两者 association 时间戳相同，但图像/深度来源不同。第一帧 MD5：

| 数据源 | RGB MD5 | Depth MD5 | 说明 |
|---|---|---|---|
| 历史 `basic_filtered_rawdepth_smoke30` | `b773ca445efd398d0ac6560d51340904` | `9a3497b2c882670aa13b1b58b3861d04` | 30 帧导出的图像/深度级过滤小序列 |
| 当前 full-derived mask-only smoke30 | `4b39bbb8ba5916262d4aa2292876abb4` | `04bc548c53cbb537efb21eafbf28c5c3` | 原始 RGB-D + mask side-channel |
| 旧 SAM3 milddilate 全序列 | `4e982e4ccf639e5336ae8851eab61ff8` | `a309f1ca954e515237ad26339b553ac8` | YOLOE+SAM3 图像级处理全序列 |

历史 smoke30 的 `benchmark_summary.json` 显示：

- 配置：`/home/lj/dynamic_SLAM/basic_frontend/config/world_sam_pipeline_foundation_panoptic_person_v2_local.json`
- `filter_mode`: `filtered`
- `exported_frames`: `30`
- `mean_mask_ratio`: `0.313740`
- `mean_filtered_detections`: `2.0`
- ORB-SLAM3 当时 `attempted=false`，说明它本质上是一个前端导出小序列，不是当前 full 链路的同源截断。

因此，历史 smoke30 的好结果不能推广为“当前 full mask-only 后端实验会好”，也不能和旧 YOLOE+SAM3 full 结果直接混为一条路线。之后记录中如果使用 smoke，必须写明：

- 是历史 `basic_filtered_rawdepth_smoke30`；
- 还是从当前 full 序列派生的 `smoke30_sequence`；
- 是否图像/深度已经被前端改写；
- 后端是否接入 mask。

### 8.12 2026-05-11 精度优先候选几何门控：稀疏光流只在 `track_local_map_pre_pose` 生效

本轮执行用户要求的“精度优先结构”：YOLOE+SAM3 的 mask/box 只作为高召回语义候选 side-channel，后端不再把 mask 内特征默认全删，而是在候选区内按特征进行几何裁决。针对上一轮 full sparse-flow 失败，本轮做了两个收紧：

- 新增 `STSLAM_SEMANTIC_SPARSE_FLOW_GATE_STAGES`，默认只允许稀疏光流/极线门控在 `track_local_map_pre_pose` 阶段参与删点。
- 新增 `STSLAM_SEMANTIC_FLOW_MAX_DYNAMIC_REJECT_RATIO`，默认 `0.15`，限制单帧单阶段由 sparse-flow 动态证据触发的候选区删点比例。

#### 代码修改

- `/home/lj/dynamic_SLAM/stslam_backend/src/Tracking.cc`
- 新增环境变量：
  - `STSLAM_SEMANTIC_SPARSE_FLOW_GATE_STAGES`
  - `STSLAM_SEMANTIC_FLOW_MAX_DYNAMIC_REJECT_RATIO`
- 新增日志字段：
  - `sparse_flow_dynamic_capped`
  - `sparse_flow_max_dynamic_rejects`

#### 固定数据路径

smoke30：

- `/home/lj/dynamic_SLAM/results/20260505_yoloe_sam3_maskonly_wxyz/smoke30_sequence`

full：

- `/home/lj/dynamic_SLAM/results/20260505_yoloe_sam3_maskonly_wxyz/sequence`

本轮结果目录：

- smoke30：`/home/lj/dynamic_SLAM/results/precision_candidate_gate_sparseflow_tracklocal_smoke30_wxyz_20260511`
- full：`/home/lj/dynamic_SLAM/results/precision_candidate_gate_sparseflow_tracklocal_full_wxyz_20260511`

二进制：

- `/home/lj/dynamic_SLAM/stslam_backend/Examples/RGB-D/rgbd_tum`
- SHA256：`7386f47be9022c5b10491551b2a1498ca936a3d89c40ddb31c8e3c1a54c4caf2`

#### smoke30 结果

| 组别 | matched | ATE-SE3 RMSE (m) | ATE-Sim3 RMSE (m) | ATE-origin RMSE (m) | RPEt-SE3 RMSE (m) | 失败次数 |
|---|---:|---:|---:|---:|---:|---:|
| `semantic_strong_delete` | 28 | 0.028063 | 0.013467 | 0.196243 | 0.009720 | 0 |
| `candidate_geometry_gate_all_stages` | 28 | 0.025941 | 0.012320 | 0.194680 | 0.009362 | 0 |
| `candidate_geometry_gate_sparse_flow_tracklocal_cap015` | 28 | 0.023272 | 0.012115 | 0.190742 | 0.010407 | 0 |

smoke30 日志确认：

- `sparse_flow_gate=1` 只出现在 `track_local_map_pre_pose`。
- `before_local_map` 与 `before_create_keyframe` 均为 `sparse_flow_gate=0`。
- `candidate_geometry_gate_sparse_flow_tracklocal_cap015` 聚合：
  - `detected_instance_features=8969`
  - `removed_matches=2128`
  - `geom_rescued=613`
  - `geom_candidate_undecided=6168`
  - `sparse_flow_checked=1204`
  - `sparse_flow_static_kept=1144`
  - `sparse_flow_dynamic_rejected=60`
  - `sparse_flow_dynamic_capped=0`

#### full 结果

| 组别 | matched | coverage | ATE-SE3 RMSE (m) | ATE-Sim3 RMSE (m) | ATE-origin RMSE (m) | RPEt-SE3 RMSE (m) | `Fail to track local map!` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `candidate_geometry_gate_all_stages` | 752 | 0.2607 | 0.453707 | 0.268111 | 0.677403 | 0.045102 | 121 |
| `candidate_geometry_gate_sparse_flow_tracklocal_cap015` | 661 | 0.2292 | 0.314580 | 0.244462 | 0.563793 | 0.042486 | 222 |

full 聚合统计：

| 组别 | removed | geom rescued | undecided | sparse checked | sparse dynamic rejected | sparse capped | 主要失败段 |
|---|---:|---:|---:|---:|---:|---:|---|
| `candidate_geometry_gate_all_stages` | 43494 | 3661 | 131245 | 0 | 0 | 0 | 694-697, 707-732, 748-838 |
| `candidate_geometry_gate_sparse_flow_tracklocal_cap015` | 34952 | 3084 | 129698 | 30668 | 4948 | 2500 | 278-367, 696-736, 763-853 |

#### 本轮结论

1. 在 full-derived smoke30 上，`track_local_map_pre_pose` 限制后的 sparse-flow 版本确实取得三组中最低的 `ATE-SE3`。
2. 但 full 序列上，sparse-flow 版本虽然 `ATE-SE3` 低于同二进制无光流候选门控，却牺牲了覆盖率，并把跟踪失败次数从 `121` 增加到 `222`。这不能作为最终有效策略。
3. 当前证据说明：**稀疏光流/极线证据不适合直接作为候选区内的硬删除条件**。即使限制到 `track_local_map_pre_pose` 并加入 15% 删除上限，仍可能在长序列中提前切断跟踪支撑。
4. 下一步若继续坚持“精度优先”，更合理的改法不是继续调大/调小删除比例，而是把 sparse-flow 改成保守辅助证据：
   - 优先作为“静态救回/不删除”的正证据；
   - 动态证据只记录或提高风险分数；
   - 只有在语义候选、重投影/深度异常、低静态可观测性同时满足时，才允许触发硬删除。

### 8.13 2026-05-11 收紧/反向候选区删点条件 smoke30 验证

本轮根据 8.12 的结论继续测试两条思路：

1. **保守删除**：sparse-flow/极线不再单独触发硬删除，只作为风险证据；候选点只有在语义候选、重投影/深度异常、静态支撑不足共同成立时才删除。动态绑定 MapPoint 仍视为强动态证据。
2. **严格静态保留**：反向处理，候选区内只明确保留几何 rescued 的静态点；对于缺 MapPoint 或低支撑但没有几何矛盾的点，允许 sparse-flow 静态证据辅助保留；其余候选点走删除。

#### 代码修改

- `/home/lj/dynamic_SLAM/stslam_backend/src/Tracking.cc`
- `SemanticGeometricVerificationResult` 新增：
  - `hasMapPoint`
  - `lowStaticSupport`
  - `mapPointObservations`
- 新增环境变量：
  - `STSLAM_SEMANTIC_CONSERVATIVE_DYNAMIC_DELETE`
  - `STSLAM_SEMANTIC_STRICT_STATIC_KEEP`
- 新增日志字段：
  - `conservative_dynamic_delete`
  - `strict_static_keep`
  - `sparse_flow_dynamic_risk_only`

#### 数据与二进制

smoke30：

- `/home/lj/dynamic_SLAM/results/20260505_yoloe_sam3_maskonly_wxyz/smoke30_sequence`

结果目录：

- 保守删除：`/home/lj/dynamic_SLAM/results/precision_conservative_candidate_gate_smoke30_wxyz_20260511`
- 严格静态保留：`/home/lj/dynamic_SLAM/results/precision_strict_static_keep_smoke30_wxyz_20260511`

#### 保守删除 smoke30 结果

| 组别 | matched | ATE-SE3 RMSE (m) | ATE-Sim3 RMSE (m) | ATE-origin RMSE (m) | RPEt-SE3 RMSE (m) | 失败次数 |
|---|---:|---:|---:|---:|---:|---:|
| `semantic_strong_delete` | 28 | 0.030182 | 0.012983 | 0.199994 | 0.009626 | 0 |
| `candidate_geometry_gate_all_stages` | 28 | 0.020741 | 0.012702 | 0.184378 | 0.009629 | 0 |
| `candidate_geometry_gate_conservative_sparse_flow` | 28 | 0.029713 | 0.012946 | 0.199968 | 0.010061 | 0 |

聚合统计：

| 组别 | removed | tagged | rescued | undecided | sparse static kept | sparse dynamic rejected | sparse dynamic risk only |
|---|---:|---:|---:|---:|---:|---:|---:|
| `semantic_strong_delete` | 2200 | 3191 | 0 | 0 | 0 | 0 | 0 |
| `candidate_geometry_gate_all_stages` | 2130 | 1704 | 617 | 6306 | 0 | 0 | 0 |
| `candidate_geometry_gate_conservative_sparse_flow` | 1257 | 857 | 488 | 6934 | 2771 | 7 | 89 |

结论：保守删除明显删得过少，ATE-SE3 接近语义强删，差于普通候选几何门控。因此不推进 full。

#### 严格静态保留 smoke30 结果

| 组别 | matched | ATE-SE3 RMSE (m) | ATE-Sim3 RMSE (m) | ATE-origin RMSE (m) | RPEt-SE3 RMSE (m) |
|---|---:|---:|---:|---:|---:|
| `candidate_geometry_gate_all_stages` | 28 | 0.025326 | 0.012109 | 0.193295 | 0.009946 |
| `candidate_geometry_gate_strict_static_keep_sparse_flow` | 28 | 0.026760 | 0.012384 | 0.196601 | 0.009434 |

结论：严格静态保留比同二进制普通候选门控略差，也没有 smoke 正信号，因此暂不推进 full。

#### 本轮判断

1. “删得更保守”没有解决问题，说明候选区内确实有一批动态/污染点需要被删掉。
2. “严格静态保留”也没有超过普通候选几何门控，说明当前 sparse-flow 静态证据不足以稳定弥补强删带来的信息损失。
3. 目前最稳的候选区策略仍是普通 `candidate_geometry_gate_all_stages`，但它在 full 上仍有覆盖率和跟踪失败问题。
4. 下一步应转向逐帧/阶段诊断，而不是继续调单一删点规则：
   - 对 full 失败段前 20-30 帧统计候选区删点数、rescued 数、undecided 数、当前匹配数、局部地图匹配数；
   - 对比 `before_local_map`、`track_local_map_pre_pose`、`before_create_keyframe` 哪一层最先造成支撑断裂；
   - 将 sparse-flow 从“删点规则”降级为诊断变量，先观察它与失败段的相关性。

### 8.14 2026-05-12 5.5 Pro 反馈后 full 序列阶段消融

本轮根据网页端 5.5 Pro 首次反馈推进，不再继续盲调 sparse-flow 或复杂动态因子，而是先验证一个更基础的问题：

> 当前 mask-only 后端精度崩坏，是不是因为动态证据在错误阶段被过早、过硬地用于删点，尤其在 `track_local_map_pre_pose` 阶段切断了位姿估计所需的静态支撑？

工作方向暂定为：

```text
Support-Preserving Dynamic Evidence SLAM
```

#### 本轮代码与工具修改

已同步到公开仓库：

- `https://github.com/wannabebetterzl/dynamic-slam-public`
- commit：`025f474 Add stage-gated dynamic filtering follow-up`

实际后端源码也已同步修改并编译通过：

- `/home/lj/dynamic_SLAM/stslam_backend/src/Tracking.cc`

新增/修正内容：

- 新增 `STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURE_STAGES`，使硬删除可按阶段启用。
- `scripts/run_backend_rgbd.sh` 改为保留外部传入的 `STSLAM_*` 环境变量，避免消融命令被 wrapper 悄悄覆盖。
- 新增 `tools/check_rgbd_sequence_integrity.py`，用于检查 RGB-D association、mask、GT 对齐和有效深度比例。

#### 序列完整性复查

数据集：

- `backend_maskonly_full_wxyz`
- `/home/lj/dynamic_SLAM/results/20260505_yoloe_sam3_maskonly_wxyz/sequence`

复查结果：

| 项目 | 结果 |
|---|---:|
| associations | 859 |
| missing RGB/depth/mask | 0 / 0 / 0 |
| RGB-depth 最大时间差 | 0.038096 s |
| RGB-depth 超过 0.03 s | 21 |
| RGB-depth 超过 0.04 s | 0 |
| GT 0.03 s 内匹配 | 857 / 859 |

结论：此前 `depth_files=827` 的疑点不被当前磁盘序列支持；当前问题不是缺帧，而是少量时间同步阈值和后端接入策略问题。

#### 本轮 full 消融矩阵

统一使用：

```bash
bash scripts/run_backend_rgbd.sh backend_maskonly_full_wxyz semantic_only <out_dir>
```

结果目录：

- `/home/lj/dynamic-slam-public/runs/full_stage_ablation_20260512`

| 组别 | 环境变量 | 状态 | matched | ATE-SE3 RMSE (m) | ATE-Sim3 RMSE (m) | Sim3 scale | RPEt-SE3 RMSE (m) | `Fail to track local map!` |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `before_local_map` | `STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURE_STAGES=before_local_map` | done | 857 | 0.388275 | 0.248433 | 0.362142 | 0.026060 | 0 |
| `track_local_map_pre_pose` | `STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURE_STAGES=track_local_map_pre_pose` | done | 857 | 0.274240 | 0.247344 | 0.590789 | 0.019668 | 0 |
| `before_create_keyframe` | `STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURE_STAGES=before_create_keyframe` | done | 857 | 0.566043 | 0.265760 | 0.219656 | 0.023246 | 0 |
| `none_metadata_only` | `STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=0`; `STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURE_STAGES=none` | done | 851 | 0.191482 | 0.167528 | 0.729600 | 0.051918 | 12 |

#### 运行中观察

- 2026-05-12 11:23 CST：开始 full 序列阶段消融，先运行 `before_local_map`。
- `before_local_map` 已完成：matched 857，ATE-SE3 0.388275 m，ATE-Sim3 0.248433 m，Sim3 scale 0.362142，RPEt-SE3 0.026060 m；日志中 `Fail to track local map!` 为 0。该组只在 `before_local_map` 触发 613 次动态硬过滤，累计 `removed_matches=3698`、`tagged_outliers=88711`。结果明显差于此前 `semantic_only` 约 0.303 m，说明“只在 local-map 前删”不是正确方向。
- 2026-05-12 约 11:25 CST：开始运行 `track_local_map_pre_pose` full 消融。
- `track_local_map_pre_pose` 已完成：matched 857，ATE-SE3 0.274240 m，ATE-Sim3 0.247344 m，Sim3 scale 0.590789，RPEt-SE3 0.019668 m；日志中 `Fail to track local map!` 为 0。该组触发 613 次动态硬过滤，累计 `removed_matches=52043`、`tagged_outliers=90193`。结果优于此前 `semantic_only` 约 0.303 m，也优于 `before_local_map`，说明“pre-pose 阶段硬删必然破坏支撑”的强假设不成立；更准确的判断应是：该阶段可能改善局部相对运动，但仍存在全局尺度/路径一致性问题，Sim3 scale 仍明显偏离 1。
- 2026-05-12 约 11:26 CST：开始运行 `before_create_keyframe` full 消融。
- `before_create_keyframe` 已完成：matched 857，ATE-SE3 0.566043 m，ATE-Sim3 0.265760 m，Sim3 scale 0.219656，RPEt-SE3 0.023246 m；日志中 `Fail to track local map!` 为 0。该组只触发 313 次动态硬过滤，累计 `removed_matches=21349`、`tagged_outliers=20585`。结果显著恶化，说明“完全延迟到建关键帧前清理”太晚，动态污染已经影响轨迹尺度和路径形状。
- 2026-05-12 约 11:27 CST：开始运行 `none_metadata_only` full 消融，用作“不硬删，仅保留 mask/meta side-channel”的基线。
- `none_metadata_only` 已完成：matched 851，ATE-SE3 0.191482 m，ATE-Sim3 0.167528 m，Sim3 scale 0.729600，RPEt-SE3 0.051918 m，RPER 1.136524 deg；日志中 `Fail to track local map!` 为 12，且没有任何 `[STSLAM_FORCE_DYNAMIC_FILTER]` 日志。该组全局 ATE 最好，但局部 RPE 最差，说明不删动态点可能保留了更多几何支撑并改善全局对齐，却让局部相对运动受动态点污染更严重。

#### 本轮阶段消融结论

按 `ATE-SE3` 排名：

| 排名 | 组别 | ATE-SE3 RMSE (m) | RPEt-SE3 RMSE (m) | 解释 |
|---:|---|---:|---:|---|
| 1 | `none_metadata_only` | 0.191482 | 0.051918 | 全局最好，但局部相对运动最差，动态点仍污染短时运动估计 |
| 2 | `track_local_map_pre_pose` | 0.274240 | 0.019668 | 局部指标最好，说明 pre-pose 删点能压制局部动态污染，但全局尺度仍偏 |
| 3 | `before_local_map` | 0.388275 | 0.026060 | 太早或位置不对，删点收益不足且尺度更差 |
| 4 | `before_create_keyframe` | 0.566043 | 0.023246 | 太晚，动态污染已进入跟踪过程，关键帧前清理无法挽回 |

当前最重要的新事实：

1. **完全不硬删的全局 ATE 反而最好**，这说明当前 hard-delete 规则确实会破坏某些长期几何支撑。
2. **`track_local_map_pre_pose` 的局部 RPE 最好**，说明语义动态信息并非无效，它对短时相对运动有帮助。
3. **全局 ATE 与局部 RPE 出现冲突**：不删更利于全局轨迹形状/尺度，pre-pose 删除更利于局部相对稳定。这正是“support-preserving dynamic evidence”应该解决的核心矛盾。
4. 下一步不应继续做单一阶段 hard-delete，而应改成：
   - 默认不删除或少删除，保留支撑；
   - 在 `track_local_map_pre_pose` 将动态证据转为 soft weight / risk score；
   - 对每帧删除比例设置上限；
   - 当静态 inlier 或局部地图支撑不足时禁止硬删；
   - 评估时同时看 ATE-SE3 与 RPEt/RPER，不再只追单一 ATE。

下一轮建议实验：

```bash
STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=0
STSLAM_GEOMETRIC_DYNAMIC_REJECTION=1
STSLAM_GEOMETRIC_DYNAMIC_REJECTION_ACTION=soft_weight
STSLAM_GEOMETRIC_DYNAMIC_REJECTION_STAGES=track_local_map_pre_pose
STSLAM_GEOMETRIC_DYNAMIC_REJECTION_MAX_REJECT_RATIO=0.10
STSLAM_GEOMETRIC_DYNAMIC_REJECTION_PROTECT_MIN_INLIERS=45
```

如果代码暂时没有 `ACTION=soft_weight`，则下一步先实现这个开关，而不是继续增加硬删除规则。

### 8.15 2026-05-12 soft/capped 动态证据实现与 repeat protocol 修正

本节接续 5.5 Pro 反馈后的路线修正。关键变化不是单纯新增一个 soft weight 开关，而是发现 full `walking_xyz` 的后端结果存在明显运行分叉：同一源码、同一参数下，matched poses、关键帧数量、尺度和 ATE 都会出现大幅波动。因此后续 full 消融必须采用 repeat protocol，不能再用单次结果直接做路线判断。

#### 本轮后端实现

实际后端源码：

- `/home/lj/dynamic_SLAM/stslam_backend/include/Frame.h`
- `/home/lj/dynamic_SLAM/stslam_backend/src/Frame.cc`
- `/home/lj/dynamic_SLAM/stslam_backend/src/Tracking.cc`
- `/home/lj/dynamic_SLAM/stslam_backend/src/Optimizer.cc`

新增能力：

- `STSLAM_GEOMETRIC_DYNAMIC_REJECTION_ACTION=hard_delete|soft_weight|risk_only`
- `STSLAM_GEOMETRIC_DYNAMIC_REJECTION_SOFT_WEIGHT`
- `STSLAM_GEOMETRIC_DYNAMIC_REJECTION_MAX_REJECT_RATIO`
- `STSLAM_GEOMETRIC_DYNAMIC_REJECTION_PROTECT_MIN_INLIERS`
- 几何动态拒绝现在可以在 `STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=0` 时独立运行，避免被语义 hard-delete 总开关误伤。
- soft weight 不再修改 `Frame` 内存布局，而是用 `frame.mnId -> feature weight` 的外部表传入 pose-only optimization；在 no-op 时优化器尽量保持原始 information matrix 路径。

公开仓库本地新增/修正工具：

- `scripts/run_backend_rgbd.sh`：修正 profile 默认值顺序，`geom_dynamic_reject` 现在会真的默认启用 `STSLAM_GEOMETRIC_DYNAMIC_REJECTION=1`。
- `tools/check_rgbd_sequence_integrity.py`：检查 association、mask、GT 最近邻、depth 有效比例。
- `scripts/run_backend_repeat_matrix.sh`：full 序列重复消融并生成 `summary_raw.csv` 与 `summary_stats.csv`。

注意：截至本节记录时，这批新增工具和 active backend soft 实现还没有推送到 GitHub，原因是 full no-op 基线仍未稳定复现，暂不应把不稳定代码当作公开结论。

#### 序列完整性复查

命令输出：

- `/home/lj/dynamic-slam-public/runs/sequence_integrity_20260512_124521.json`

结果摘要：

| 项目 | 结果 |
|---|---:|
| association rows | 859 |
| missing RGB/depth/mask | 0 / 0 / 0 |
| unique RGB paths | 859 |
| unique depth paths | 827 |
| depth duplicate reuse | 32 |
| depth valid ratio median | 0.615540 |
| RGB-depth max abs diff | 0.038096 s |
| RGB-depth diff > 0.03 s | 21 |
| GT nearest diff > 0.03 s | 2 |

判断：

1. 当前不是缺 RGB/depth/mask 文件导致的直接失败。
2. `depth_files=827` 的现象来自 depth 复用，不是缺失文件。
3. 21 行 RGB-depth 时间差超过 0.03 s，是必须记录的数据桥风险项；它不一定解释全部漂移，但足以要求后续做 association 修复/阈值敏感性实验。

#### no-op 基线复现问题

在 active backend 当前代码下，所有新机制关闭：

```bash
STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=0
STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURE_STAGES=none
STSLAM_GEOMETRIC_DYNAMIC_REJECTION=0
STSLAM_RGBD_DYNAMIC_FRONTEND_SPLIT=0
ORB_SLAM3_MASK_MODE=postfilter
```

重复结果：

| run | matched | ATE-SE3 | ATE-Sim3 | Sim3 scale | RPEt-SE3 | RPER |
|---|---:|---:|---:|---:|---:|---:|
| `noop_full_external_weight_store_20260512_122656` | 857 | 0.687759 | 0.279203 | 0.150836 | 0.023517 | 0.621266 |
| `noop_full_external_weight_store_repeat1_20260512_122902` | 767 | 0.280431 | 0.220475 | 0.552003 | 0.057533 | 0.862874 |
| `noop_full_external_weight_store_repeat2_20260512_122935` | 705 | 0.419997 | 0.218183 | 0.346462 | 0.084586 | 1.626898 |
| `noop_full_sleep_repeat_20260512_123007` | 711 | 0.444570 | 0.273202 | 0.309291 | 0.053979 | 1.778426 |

临时构建“公开仓库上传前快照”后，同样不稳定：

| run | matched | ATE-SE3 | ATE-Sim3 | Sim3 scale | RPEt-SE3 | RPER |
|---|---:|---:|---:|---:|---:|---:|
| `noop_full_public_snapshot_rebuild_20260512_124009` | 715 | 0.264150 | 0.215445 | 0.514509 | 0.087643 | 2.054586 |
| `noop_full_public_snapshot_rebuild_repeat1_20260512_124056` | 772 | 0.689153 | 0.265102 | 0.165103 | 0.046712 | 1.336547 |

判断：

1. 基线漂移不完全是 soft weight 新补丁导致，因为上传前快照重新编译后也出现大幅分叉。
2. 当前 full 结果高度受 ORB-SLAM3 前后端线程时序、关键帧插入分支或数据 association 细节影响。
3. 历史 `none_metadata_only=0.191482m` 仍是重要参考，但在当前编译/运行条件下尚未稳定复现，因此不能作为后续 soft 消融的唯一控制值。

#### full repeat matrix，REPEATS=2

命令：

```bash
REPEATS=2 bash scripts/run_backend_repeat_matrix.sh \
  /home/lj/dynamic-slam-public/runs/full_repeat_ablation_20260512_124542
```

汇总文件：

- `/home/lj/dynamic-slam-public/runs/full_repeat_ablation_20260512_124542/summary_raw.csv`
- `/home/lj/dynamic-slam-public/runs/full_repeat_ablation_20260512_124542/summary_stats.csv`

| 组别 | n | matched median | ATE-SE3 median | ATE-SE3 range | ATE-Sim3 median | Sim3 scale median | RPEt median | RPER median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `noop_metadata_only` | 2 | 823.5 | 0.587776 | 0.582992-0.592561 | 0.282076 | 0.178936 | 0.051388 | 1.080141 |
| `geom_riskonly_cap010` | 2 | 857.0 | 0.630016 | 0.602359-0.657674 | 0.273165 | 0.181463 | 0.023803 | 0.614926 |
| `geom_hard_cap010_protect45` | 2 | 819.5 | 0.767787 | 0.735647-0.799927 | 0.281122 | 0.135234 | 0.056210 | 1.097175 |
| `geom_soft_cap010_w025` | 2 | 761.5 | 0.359969 | 0.305325-0.414613 | 0.240567 | 0.387380 | 0.090098 | 1.347623 |
| `geom_soft_cap005_w050` | 2 | 814.5 | 0.794404 | 0.789705-0.799103 | 0.284536 | 0.121076 | 0.040972 | 0.731012 |

日志聚合确认新动作确实触发：

| 组别 | 日志行数 | 动作累计 |
|---|---:|---:|
| `geom_riskonly_cap010` | 1716 | `geom_dyn_risk_only=83965` |
| `geom_hard_cap010_protect45` | 1626 | `geom_dyn_removed_matches=76906` |
| `geom_soft_cap010_w025` | 1483 | `geom_dyn_soft_weighted=61464` |

#### strict association 临时测试

为验证 `RGB-depth diff > 0.03s` 的 21 行是否是主因，临时生成严格 association：

- `/home/lj/dynamic-slam-public/runs/strict_assoc_003/associations_strict003.txt`
- kept 838，dropped 21

单次 no-op 结果：

| run | matched | ATE-SE3 | ATE-Sim3 | Sim3 scale | RPEt-SE3 | RPER |
|---|---:|---:|---:|---:|---:|---:|
| `noop_full_strict_assoc003_20260512_125252` | 766 | 0.596010 | 0.290294 | 0.150050 | 0.042791 | 0.847933 |

判断：简单剔除超阈值 association 没有带来正信号，因此时间同步风险需要继续记录，但它不是当前 full ATE 漂移的单点修复。

#### 本节判断

1. `geom_soft_cap010_w025` 是当前唯一出现正向 SE3 信号的新策略，单次最低达到 `0.305325m`，但 repeat median 只有 `0.359969m`，且 RPE 明显恶化，不能视为已达标。
2. `risk_only` 作为理论 no-op 控制仍改变结果，进一步说明 full pipeline 的线程/时序分叉非常强；后续必须以多次 repeat 和统计区间报告。
3. `hard_cap010_protect45` 与 `soft_cap005_w050` 均明显差，不应继续沿这两个参数方向加码。
4. 当前不建议把“soft 已经有效”作为结论发给 5.5 Pro；更合适的回传问题是：**如何让 ORB-SLAM3-derived backend 的 full evaluation 稳定可复现，以及是否应先修 association/同步和前后端时序，再继续做动态证据策略优化。**

下一步优先级：

1. 为 full evaluation 增加 deterministic/sequential local mapping 或至少固定前后端同步点，降低关键帧插入分叉。
2. 对 `backend_maskonly_full_wxyz` 重新生成更严格的 RGB-depth association，消除 `>0.03s` 的 21 行时间差后复跑 no-op。
3. 若 no-op 稳定后，再保留 `geom_soft_cap010_w025` 作为唯一候选 soft 路线，进行 `REPEATS>=5` 的正式 full 消融。
4. 当前阶段暂不推送 active backend soft 代码到公开仓库主线；等 no-op 控制组稳定后再同步。


### 8.16 可复现性接入进度：deterministic profile + LocalMapping 同步 smoke 验证

日期：2026-05-12

本轮目的不是继续追求单次 ATE，而是先把 8.15 中暴露出的非确定性问题规训起来。当前已接入第一、二层可复现性脚手架：

1. `rgbd_tum` 新增 `STSLAM_SYNC_LOCAL_MAPPING`，每帧 Track 后等待 LocalMapping 队列清空并恢复接受关键帧。
2. 后端新增 `SaveKeyFrameTimeline("KeyFrameTimeline.csv")`，导出关键帧 id、源 frame id、timestamp、map id 与位姿，用于判断关键帧分支是否复现。
3. public wrapper 新增 deterministic profile，默认设置 `STSLAM_SYNC_LOCAL_MAPPING=1`、`STSLAM_DISABLE_FRAME_SLEEP=1`、`OMP_NUM_THREADS=1`、`OPENCV_FOR_THREADS_NUM=1`、`OPENBLAS_NUM_THREADS=1`、`MKL_NUM_THREADS=1`。
4. wrapper 支持 `DSLAM_CPUSET=0` 通过 `taskset` 固定 CPU core，并将这些环境变量写入 `run_manifest.txt`。

编译验证：

```bash
cmake --build build --target rgbd_tum -j4
```

结果：`[100%] Built target rgbd_tum`

单次 deterministic smoke：

- run: `/home/lj/dynamic-slam-public/runs/deterministic_noop_smoke30_20260512_141116`
- returncode: `0`
- `run_manifest.txt` 已记录 `DSLAM_CPUSET=0`、`STSLAM_SYNC_LOCAL_MAPPING=1`、线程数限制等字段。
- `stdout.log` 显示 `Sync local mapping: true`
- 已生成 `KeyFrameTimeline.csv`

| alignment | matched | ATE RMSE | RPEt RMSE | Sim3 scale |
|---|---:|---:|---:|---:|
| SE3 | 28 | 0.029722 | 0.010421 | 1.000000 |
| Sim3 | 28 | 0.013790 | 0.006943 | 0.666322 |
| origin | 28 | 0.198734 | 0.017650 | 1.000000 |

同配置 smoke repeat 两次：

- base: `/home/lj/dynamic-slam-public/runs/deterministic_smoke_repeat_20260512_141209`
- 两次均 matched 28，CameraTrajectory 均 30 行，KeyFrameTimeline 均 23 行。
- 但 `KeyFrameTimeline.csv` 与 `CameraTrajectory.txt` 的 sha256 不一致，说明接入已跑通但尚未实现强确定性。

| run | ATE-SE3 | ATE-Sim3 | Sim3 scale | RPEt-SE3 |
|---|---:|---:|---:|---:|
| run_1 | 0.029394 | 0.014518 | 0.672077 | 0.010338 |
| run_2 | 0.028324 | 0.013685 | 0.679616 | 0.010038 |

关键发现：

1. 当前同步点能固定关键帧数量与关键帧 frame id 序列，但不能固定关键帧位姿数值。
2. 这说明误差不仅来自“是否插入关键帧”的离散分支，还来自 LocalMapping/BA/MapPoint 更新的连续数值路径与异步写入顺序。
3. 第一层运行规训与第二层同步观测已经接入；下一步应进入第三层：实现更强的顺序化 evaluation backend，或者至少在论文实验模式下让 LocalMapping 以固定顺序、固定边界处理关键帧。
4. 在 full 消融继续扩大前，必须先把 no-op 控制组稳定性作为主线，否则动态策略的单次提升仍可能被线程时序噪声淹没。


### 8.17 第三层接入：sequential evaluation backend 初版

日期：2026-05-12

本轮目的：在第一层运行规训和第二层 LocalMapping 同步等待仍无法完全复现的基础上，进一步消除 ORB-SLAM3 后端异步调度。新增 evaluation-only 顺序化模式，优先服务论文实验可复现性。

已实现：

1. `LocalMapping::RunOneStep()`：把原本 `LocalMapping::Run()` 中的一次队列处理拆成可由主线程调用的单步函数。
2. `STSLAM_SEQUENTIAL_LOCAL_MAPPING=1`：打开后 `System` 不启动 LocalMapping 后台线程。
3. `STSLAM_SEQUENTIAL_LOOP_CLOSING=1`：sequential profile 默认不启动 LoopClosing 后台线程，避免异步回环或 GBA 改写地图。
4. `System::ProcessSequentialLocalMappingQueue(maxSteps)`：每帧 Track 后由 `rgbd_tum` 主线程主动处理 LocalMapping 队列，直到队列清空或达到步数上限。
5. public wrapper 新增 `sequential_semantic_only` 与 `sequential_geom_dynamic_reject` profile，继承 deterministic 环境约束并记录 sequential 环境变量。

编译验证：

```bash
cmake --build build --target rgbd_tum -j4
```

结果：`[100%] Built target rgbd_tum`

#### smoke repeat，sequential no-op

命令核心：

```bash
DSLAM_CPUSET=0 \
STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=0 \
STSLAM_GEOMETRIC_DYNAMIC_REJECTION=0 \
bash scripts/run_backend_rgbd.sh \
  backend_maskonly_smoke30_wxyz sequential_semantic_only ...
```

结果目录：

- `/home/lj/dynamic-slam-public/runs/sequential_smoke_repeat_20260512_142942`

两次运行结果完全一致：

- `KeyFrameTimeline.csv` sha256: `3fbb02cad43a1f41291e34d41ba058d12c7bef15810044c63d03651f271cec72`
- `CameraTrajectory.txt` sha256: `142a971220581c2463a70e01c29116f60729ae65404cf275bbc4ae5ea9fa2df4`

| run | matched | ATE-SE3 | ATE-Sim3 | Sim3 scale | RPEt-SE3 |
|---|---:|---:|---:|---:|---:|
| run_1 | 28 | 0.030436 | 0.013154 | 0.657715 | 0.010277 |
| run_2 | 28 | 0.030436 | 0.013154 | 0.657715 | 0.010277 |

#### full repeat，sequential no-op

结果目录：

- `/home/lj/dynamic-slam-public/runs/sequential_full_noop_repeat_20260512_143012`

两次完整序列运行也完全一致：

- `KeyFrameTimeline.csv` sha256: `5c00cc443cbdc8df460e6d9002e61506022934537c77e093305581b20814bd4b`
- `CameraTrajectory.txt` sha256: `d566c362b27ac705fdc58b07defb2154853183c161493f305fd1561f7b2ca9f4`
- `KeyFrameTimeline.csv`: 244 行，即 243 个关键帧。
- `CameraTrajectory.txt`: 859 行。

| run | matched | ATE-SE3 | ATE-Sim3 | Sim3 scale | RPEt-SE3 | RPER-SE3 |
|---|---:|---:|---:|---:|---:|---:|
| run_1 | 857 | 0.814559 | 0.288934 | 0.098576 | 0.042155 | 0.806528 |
| run_2 | 857 | 0.814559 | 0.288934 | 0.098576 | 0.042155 | 0.806528 |

#### 本节判断

1. 第三层初版成功把 smoke 与 full 的 no-op 控制组变成 bit-level 可复现：关键帧时间线、相机轨迹、ATE/RPE 全部一致。
2. 这强力证明此前 full 消融的大幅漂移主要来自后端异步调度，而不是数据集文件随机变化或评估脚本随机性。
3. 当前 sequential no-op 的 SE3 ATE 退化到 `0.814559m`，说明“完全顺序化”改变了原系统的前后端行为，不能直接作为最终精度实验协议。
4. 下一步不应立刻做动态策略 full 消融，而应实现“保持精度的准顺序化”模式：固定 LocalMapping 处理边界，但尽量复刻原 ORB-SLAM3 的关键帧接受节奏与 BA 触发节奏。
5. 对论文来说，当前结果已经可以支撑一个重要方法学结论：必须同时报告 deterministic protocol 与 repeat statistics，否则动态 SLAM 改动的单次精度提升不可解释。

下一步建议：

1. 增加 `STSLAM_SEQUENTIAL_LOCAL_MAPPING_DRAIN_PERIOD` 或 fixed-lag barrier，而不是每帧后立即 drain 队列，以减少对关键帧插入节奏的改变。
2. 记录每次 run 的关键帧 frame-id 序列、BA 次数、关键帧数量、MapPoint 数量，作为可复现性 fingerprint。
3. 在 no-op 精度恢复到接近历史基线后，再重启 `geom_soft_cap010_w025` 等动态策略 full repeat 消融。

#### 8.17 补充试验：准顺序化 drain 策略

为避免 `period=1` 的“每帧立即 drain”过度改变原始系统节奏，又新增了以下控制变量：

- `STSLAM_SEQUENTIAL_LOCAL_MAPPING_DRAIN_PERIOD_FRAMES`
- `STSLAM_SEQUENTIAL_LOCAL_MAPPING_MAX_QUEUE_BEFORE_DRAIN`
- `STSLAM_SEQUENTIAL_LOCAL_MAPPING_HOLD_ACCEPT_WHEN_QUEUED`

其目标是让 Tracking 在队列积压时感知到 LocalMapping 忙碌，但又不必每一帧都立刻执行一次完整 LocalMapping。

1. `period=1, max_queue=3`：
   结果与前述 sequential 初版一致，smoke 仍是 bit-level 可复现，说明这些新控制没有破坏稳定性。
2. `period=0, max_queue=3`：
   full no-op 在初始化后很快出现
   `Fail to track local map! frame=1 current_matches=0 reference_kf=0`
   并进入
   `LM: Active map reset recieved`
   `LM: Active map reset, waiting...`
   说明“完全只靠队列阈值触发 drain”对初始化过于激进，地图建立来不及完成。
3. `period=5, max_queue=3`：
   smoke repeat 可复现，但关键帧数显著下降：
   - `/home/lj/dynamic-slam-public/runs/sequential_period5_smoke_repeat_20260512_145053`
   - `KeyFrameTimeline.csv` 仅 8 行
   - SE3 ATE `0.036551m`
   full no-op 在约 500 帧附近失跟并触发 active map reset：
   `Fail to track local map! frame=497...501`
   随后重新建图并再次进入 reset waiting，不能作为可用实验协议。

补充判断：

1. “降低 drain 频率”确实能让关键帧节奏更接近原系统，但会明显增加初始化和长序列中途失跟风险。
2. 当前最稳的 sequential 配置仍然是 `period=1`，它解决了复现性，却牺牲了全序列精度。
3. 这意味着下一步不应继续盲调 `drain period`，而应改为更细粒度地模拟原系统中 `AcceptKeyFrames`、`InterruptBA` 与 reset 协议的时序边界。

#### 8.18 两阶段 sequential mapper：在保持可复现的同时回收精度

时间：`2026-05-12`

本轮不再继续扫 `drain period`，而是直接修改 LocalMapping 的顺序化方式：

1. `RunOneStep()` 只保留轻量路径：`ProcessNewKeyFrame`、`MapPointCulling`、`CreateNewMapPoints`、`UpdateInstanceStructure`。
2. 将重维护逻辑拆到 `RunMaintenanceStep(force)`：`SearchInNeighbors`、`LBA` / `LocalInertialBA`、`KeyFrameCulling`、后续 inertial maintenance 与 loop closer 插入。
3. 为顺序化模式补充 `QueueReset` / `QueueResetActiveMap` / `ServicePendingResetRequests`，避免没有后台线程时 reset 协议互相等待。
4. 新增 `STSLAM_SEQUENTIAL_LOCAL_MAPPING_MAINTENANCE_PERIOD`，允许“每处理 N 个关键帧才做一次重维护”，从而更接近原始异步 LocalMapping 的节奏。

#### 中间发现：wrapper 的 hybrid profile 一开始并没有真正覆盖 maintenance period

排查时发现 `run_backend_rgbd.sh` 中 `set_hybrid_sequential_defaults()` 先继承了 sequential 默认值，再用 `set_default` 试图改成 `4`，结果 `maintenance_period` 实际仍然是 `1`。

在这一错误 profile 下先得到一个有价值的中间结果：

- 目录：`/home/lj/dynamic-slam-public/runs/hybrid_full_repeat_20260512_1536/r1`
- `STSLAM_SEQUENTIAL_LOCAL_MAPPING_MAINTENANCE_PERIOD=1`
- full no-op：
  - matched `857`
  - ATE-SE3 `0.484819`
  - ATE-Sim3 `0.283585`
  - Sim3 scale `0.202737`
  - RPEt-SE3 `0.012083`
  - RPER-SE3 `0.477890`

这说明即便重维护仍是“每个关键帧都做一次”，仅通过“两阶段拆分 + 顺序化 reset service”，也已经把 full no-op 从旧 sequential 的 `0.814559m` 拉回到了 `0.484819m`。

随后修复 wrapper，使 hybrid profile 在未显式指定时真正使用 `maintenance_period=4`。

#### smoke repeat，true hybrid (`maintenance_period=4`)

目录：

- `/home/lj/dynamic-slam-public/runs/hybrid_true_smoke_repeat_20260512_1540`

`run_manifest.txt` 确认：

- `profile=hybrid_sequential_semantic_only`
- `STSLAM_SEQUENTIAL_LOCAL_MAPPING_MAINTENANCE_PERIOD=4`

两次 smoke 完全一致：

- `CameraTrajectory.txt` sha256: `33312912a7a8d16c0e0f8f0ef579ec46dfff50d3dc0b553f0aa610c6bb33ba0a`
- `KeyFrameTimeline.csv` sha256: `cfb385afae21c1107d6fad1e6b5c4d78c8a25105cc89cdf3ca3e5fdc4da0bbc5`

| run | matched | ATE-SE3 | ATE-Sim3 | Sim3 scale | RPEt-SE3 | RPER-SE3 |
|---|---:|---:|---:|---:|---:|---:|
| run_1 | 28 | 0.036226 | 0.013284 | 0.609968 | 0.012849 | 0.505577 |
| run_2 | 28 | 0.036226 | 0.013284 | 0.609968 | 0.012849 | 0.505577 |

#### full repeat，true hybrid (`maintenance_period=4`)

目录：

- `/home/lj/dynamic-slam-public/runs/hybrid_true_full_repeat_20260512_1541`

两次完整序列运行也完全一致：

- `CameraTrajectory.txt` sha256: `36e0ce12b19076ab5bb761ccc782b9fe9b9552d1d1126cfdee922a6b28d895e5`
- `KeyFrameTimeline.csv` sha256: `cac955e6c1ebf12b0f77b11dc53b3adcc41a09e892eced49209bdec657eb2bac`
- `returncode=0`
- `matched poses=857`

| run | ATE-SE3 | ATE-Sim3 | Sim3 scale | RPEt-SE3 | RPER-SE3 |
|---|---:|---:|---:|---:|---:|
| run_1 | 0.329051 | 0.270971 | 0.411179 | 0.019523 | 0.525301 |
| run_2 | 0.329051 | 0.270971 | 0.411179 | 0.019523 | 0.525301 |

附记：

1. `stderr.log` 中出现过 `virtual int g2o::SparseOptimizer::optimize(int, bool): 0 vertices to optimize...`，但不影响完整运行、轨迹保存与重复一致性。
2. 这条 true hybrid baseline 已经同时满足：
   - full 序列 bit-level 可复现；
   - SE3 ATE 从 `0.814559m` 明显回收到 `0.329051m`；
   - 相比“两阶段但 period=1”的 `0.484819m`，继续提升。

#### 本节结论更新

1. 第三层现在不再只是“稳定控制组”，而是已经形成一条可用的 full-sequence baseline：`hybrid_sequential_semantic_only + maintenance_period=4`。
2. 当前最可信的论文实验基线，应从旧的 `sequential_semantic_only(period=1)` 切换为这条 true hybrid baseline。
3. 下一步终于可以回到动态策略本身，在这个可复现实验协议上重跑 `geom_soft_cap010_w025` 等 full repeat 消融，而不是继续卡在后端时序噪声里。

#### 8.19 迁移到 stable protocol：前端强参考与旧 soft 路线复测

时间：`2026-05-12`

本轮目标是把三类关键对照统一迁移到同一条稳定协议上：

1. 前端图像级强参考：`frontend_imagelevel_milddilate_full_wxyz`
2. 前端图像级次强参考：`frontend_imagelevel_boxfallback_full_wxyz`
3. 旧 backend 候选 soft 路线：`geom_soft_cap010_w025`

其中前端图像级参考采用：

- `profile=hybrid_sequential_semantic_only`
- `STSLAM_SEQUENTIAL_LOCAL_MAPPING_MAINTENANCE_PERIOD=4`
- `DSLAM_PASS_MASK_ARG=0`
- `ORB_SLAM3_MASK_MODE=off`
- `STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=0`

这保证它代表“前端已改输入序列，后端不再额外使用 mask side-channel 删点”的纯前端参考。

#### frontend milddilate，stable no-mask hybrid

目录：

- `/home/lj/dynamic-slam-public/runs/stable_protocol_migration_20260512/frontend_milddilate_nomask_hybrid`

两次 full 运行完全一致：

- `CameraTrajectory.txt` sha256: `91983235c1207bc7169dbbce5a4dcb3f2c3f93d944e63c23efcee1905c8a92a3`
- `KeyFrameTimeline.csv` sha256: `0c133a3c06be511349fa1c19c266467b69d3455efdef49031ae50e247fee276e`

| run | matched | ATE-SE3 | ATE-Sim3 | Sim3 scale | RPEt-SE3 | RPER-SE3 |
|---|---:|---:|---:|---:|---:|---:|
| run_1 | 857 | 0.018459 | 0.016328 | 0.972125 | 0.012051 | 0.365270 |
| run_2 | 857 | 0.018459 | 0.016328 | 0.972125 | 0.012051 | 0.365270 |

与 registry 中的历史前端参考 `0.016268m` 相比，这条 stable-protocol 结果略高，但仍处于同一量级，说明稳定调度没有破坏其“强前端参考”属性。

#### frontend boxfallback，stable no-mask hybrid

目录：

- `/home/lj/dynamic-slam-public/runs/stable_protocol_migration_20260512/frontend_boxfallback_nomask_hybrid`

两次 full 运行完全一致：

- `CameraTrajectory.txt` sha256: `f1153edd018a4c3e5aaa6e61275f2527b4e531e673c83c25d43f4c0773bc84be`
- `KeyFrameTimeline.csv` sha256: `db8cb2490cda85a51b40e0f95c39c928a639cee4b9dff7ab5648f19ddf66c2f9`

| run | matched | ATE-SE3 | ATE-Sim3 | Sim3 scale | RPEt-SE3 | RPER-SE3 |
|---|---:|---:|---:|---:|---:|---:|
| run_1 | 857 | 0.017319 | 0.015419 | 0.974408 | 0.011779 | 0.366396 |
| run_2 | 857 | 0.017319 | 0.015419 | 0.974408 | 0.011779 | 0.366396 |

这说明前端图像级两条旧强参考在 stable protocol 下都保持了 bit-level 可复现，并继续显著优于 backend mask-only 路线。

#### backend `geom_soft_cap010_w025`，stable hybrid repeat

目录：

- `/home/lj/dynamic-slam-public/runs/stable_protocol_migration_20260512/backend_geom_soft_cap010_w025_hybrid`

配置核心：

- `profile=hybrid_sequential_geom_dynamic_reject`
- `STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=0`
- `STSLAM_GEOMETRIC_DYNAMIC_REJECTION_ACTION=soft_weight`
- `STSLAM_GEOMETRIC_DYNAMIC_REJECTION_SOFT_WEIGHT=0.25`
- `STSLAM_GEOMETRIC_DYNAMIC_REJECTION_MAX_REJECT_RATIO=0.10`
- `STSLAM_GEOMETRIC_DYNAMIC_REJECTION_PROTECT_MIN_INLIERS=45`
- `STSLAM_SEQUENTIAL_LOCAL_MAPPING_MAINTENANCE_PERIOD=4`

两次 full 运行完全一致：

- `CameraTrajectory.txt` sha256: `14b37047e9b9ddd8ba6832401cb4ad8738e4f876a2f11912184a3339fa2a12d4`
- `KeyFrameTimeline.csv` sha256: `455447856e924bfafc23adbf867fae7ea24360286b851bbe517abfe577e66f7f`

| run | matched | ATE-SE3 | ATE-Sim3 | Sim3 scale | RPEt-SE3 | RPER-SE3 |
|---|---:|---:|---:|---:|---:|---:|
| run_1 | 772 | 0.531003 | 0.287200 | 0.177044 | 0.057179 | 1.069764 |
| run_2 | 772 | 0.531003 | 0.287200 | 0.177044 | 0.057179 | 1.069764 |

补充观察：

1. 日志显示 `track_local_map_pre_pose` 阶段确实发生了稳定的 soft weighting，但 `geom_dyn_capped` 也长期很高，说明大量候选被 cap 限制。
2. 这条路线在稳定协议下不仅没有优于 no-op baseline，反而明显更差：
   - stable no-op hybrid：`ATE-SE3 = 0.329051m`
   - stable `geom_soft_cap010_w025`：`ATE-SE3 = 0.531003m`
3. `matched poses` 从 `857` 掉到 `772`，`RPEt` 和 `RPER` 也显著恶化，说明它不是“全局精度换局部稳定”的温和退化，而是整体跟踪质量一起下降。

#### 本节结论

1. 前端 image-level 强参考在 stable protocol 下依旧极强，并且现在也具备了 bit-level 可复现性。
2. backend mask-only stable baseline 目前最可信的数字仍然是 `0.329051m`。
3. 旧 `geom_soft_cap010_w025` 在稳定协议下复现的是稳定负结果，因此它不再适合作为当前主候选路线。
4. 这意味着反馈给 5.5 Pro 时，叙事应当更新为：
   - “前端强，后端弱”不是偶然单次现象；
   - stable protocol 已证实这一差距是真实结构性差距；
   - 下一轮候选方法不应继续沿用旧 soft route 原样推进。

#### 8.20 下一步计划板：防止从机制主线跑偏

时间：`2026-05-12`

本节作为下一阶段的执行约束。若与前文旧计划冲突，以本节为准。

核心路线更新：

> 主线从“继续调后端 soft weight / 动态因子”转向
> **前端 image-level dynamic support allocation + 后端 static-observability diagnostics + 轻量 map-admission guard**。

当前最重要的实验事实：

| case | matched | ATE-SE3 | Sim3 scale | 判断 |
|---|---:|---:|---:|---|
| frontend boxfallback no-mask stable hybrid | 857 | 0.017319 | 0.974408 | 当前最强前端参考 |
| frontend milddilate no-mask stable hybrid | 857 | 0.018459 | 0.972125 | 强前端参考 |
| backend mask-only stable hybrid no-op | 857 | 0.329051 | 0.411179 | 当前 mask-only 稳定基线 |
| backend `geom_soft_cap010_w025` stable hybrid | 772 | 0.531003 | 0.177044 | 稳定负结果，停止作为主线 |

##### 执行原则

1. `smoke30` 只用于工程健康检查，不再作为精度判断依据。
2. 所有路线判断必须基于 full `walking_xyz`，优先使用 stable hybrid protocol。
3. 旧 `geom_soft_cap010_w025` 不再继续做参数扫描；若保留，只作为负结果 ablation。
4. 下一步优先解释机制，而不是直接堆新模块。
5. 后端 mask side-channel 暂不参与 `track_local_map_pre_pose` 的强决策；优先做日志、诊断、map admission 层轻量保护。

##### 任务 1：Static Observability Fingerprint

目的：解释 image-level 为什么保持 `Sim3 scale≈0.97`，而 backend mask-only no-op 只有 `0.411`，soft route 更低到 `0.177`。

先实现一个逐帧 CSV，例如：

- `tracking_observability.csv`

每帧至少记录：

- `frame_id`
- `timestamp`
- `tracking_state`
- `num_orb_features`
- `num_features_inside_mask`
- `num_features_outside_mask`
- `num_tracked_map_points`
- `num_local_map_matches_before_pose`
- `num_inliers_after_pose`
- `inlier_margin`
- `num_keyframes`
- `num_mappoints`
- `delta_t_est`
- `delta_t_gt`
- `delta_t_ratio`
- `accum_path_est`
- `accum_path_gt`
- `accum_path_ratio`
- `lost_flag`
- `reset_flag`
- `relocalization_flag`

每个 keyframe 另记录：

- `kf_id`
- `frame_id`
- `num_new_mappoints`
- `num_close_points`
- `num_static_candidate_points`
- `num_dynamic_candidate_points`
- `num_points_entering_local_mapping`
- `num_points_culled_after_lba`

先重跑三组 full：

1. `frontend_imagelevel_boxfallback_full_wxyz`
   - `profile=hybrid_sequential_semantic_only`
   - `DSLAM_PASS_MASK_ARG=0`
   - `ORB_SLAM3_MASK_MODE=off`
   - `STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=0`
2. `backend_maskonly_full_wxyz`
   - `profile=hybrid_sequential_semantic_only`
   - stable no-op baseline
3. `backend_maskonly_full_wxyz`
   - `profile=hybrid_sequential_geom_dynamic_reject`
   - 旧 `geom_soft_cap010_w025`

成功判据：

1. 能定位 backend mask-only 的 `delta_t_ratio / accum_path_ratio` 从哪些 interval 开始偏离。
2. 能定位 soft route 的 `matched poses` 下降和 `inlier_margin` / support collapse 是否对应。
3. 能说明 image-level route 是否通过更稳定的静态支撑保持 `Sim3 scale≈1`。

失败判据：

1. 三组 fingerprint 几乎一致，但 ATE 和 scale 差巨大。
2. 若出现这种情况，优先转查数据桥、timestamp association、evaluator，而不是继续做 SLAM 策略。

##### 任务 2：Early Intervention Causal Ablation

目的：回答 image-level 的优势来自 RGB feature support、depth support，还是二者共同作用。

固定后端条件：

- `profile=hybrid_sequential_semantic_only`
- `maintenance_period=4`
- `DSLAM_PASS_MASK_ARG=0`
- `ORB_SLAM3_MASK_MODE=off`
- `STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=0`

构造四个输入版本：

| 输入版本 | RGB | depth | 用途 |
|---|---|---|---|
| A | raw RGB | raw depth | 原始输入基线 |
| B | filtered RGB | raw depth | 验证 ORB 提点/描述子支撑重分配 |
| C | raw RGB | dynamic depth invalidated/filtered | 验证动态 depth 对路径长度/scale 的影响 |
| D | filtered RGB | filtered/invalidated dynamic depth | 当前 image-level 强参考近似版本 |

成功判据：

1. 若 B 已接近 D，说明核心优势来自 RGB 提点前 feature support reallocation。
2. 若 C 明显改善 scale 但 ATE 仍高，说明 depth 动态污染影响路径长度，但 RGB 支撑仍是主因。
3. 若只有 D 接近 `ATE≈0.017-0.018m` 且 `Sim3 scale≈0.97`，说明 RGB 与 depth 必须同步处理。

失败判据：

1. B/C/D 都无法复现 image-level 强结果。
2. 此时先查前端导出差异、association、depth 有效区域和 evaluator，不急于加后端算法。

##### 任务 3：Frontend Strong + Backend Map-Admission Guard

目的：测试后端 side-channel 是否还能作为“轻量保护”而不是 pose 阶段动态权重。

基线：

- `dataset=frontend_imagelevel_boxfallback_full_wxyz`
- `profile=hybrid_sequential_semantic_only`
- `DSLAM_PASS_MASK_ARG=0`
- `ORB_SLAM3_MASK_MODE=off`
- `STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=0`

guard 版本：

- `dataset=frontend_imagelevel_boxfallback_full_wxyz`
- `profile=hybrid_sequential_semantic_only`
- `DSLAM_PASS_MASK_ARG=1`
- `ORB_SLAM3_MASK_MODE=off`
- `STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=0`
- `STSLAM_GEOMETRIC_DYNAMIC_REJECTION=0`
- 新增或复用：
  - `STSLAM_DYNAMIC_RISK_ONLY=1`
  - `STSLAM_DYNAMIC_MAP_ADMISSION_GUARD=1`
  - `STSLAM_DYNAMIC_GUARD_STAGE=before_create_keyframe`
  - `STSLAM_DYNAMIC_GUARD_MIN_STATIC_INLIERS=80`
  - `STSLAM_DYNAMIC_GUARD_MAX_VETO_RATIO=0.10`

设计原则：

1. tracking pose optimization 不改权重。
2. `track_local_map_pre_pose` 不删点、不降权。
3. 只在 `before_create_keyframe` / map admission 阶段，对高风险 dynamic candidate 做保守 veto。
4. 如果当前静态 inliers 不足，则不 veto。
5. veto ratio 必须 capped。

成功判据：

- `matched=857`
- `ATE-SE3 <= 0.020m`
- `Sim3 scale >= 0.95`
- `RPEt-SE3` 不比 image-level baseline 恶化超过 10%
- 动态区域进入长期 map 的点数下降

失败判据：

- `ATE-SE3 > 0.025m`
- matched poses 下降
- scale 明显偏离 1

若失败，结论不是“后端完全无价值”，而是：当前 ORB-SLAM3 RGB-D pipeline 中，mask side-channel 不应参与 tracking / mapping 的强决策，只适合作为诊断与可视化证据。

##### 暂停投入路线

1. 暂停 `geom_soft_cap010_w025` 及其周边参数扫描。
2. 暂停把 `smoke30` 指标写入论文主结果。
3. 暂缓 DynoSAM / 复杂 SLAMMOT 联合优化。
4. 暂缓单纯更换更强 foundation model；当前主要瓶颈是接入机制，不是 mask 能力本身。
5. `premask/postfilter/layer2/layer3` 排列组合降级为控制实验，除非它服务于 support allocation 机制解释。

##### 下一步开始顺序

1. 先实现 `Static Observability Fingerprint` CSV。
2. 用 full sequence 重跑三组机制诊断。
3. 根据 fingerprint 决定 Early Intervention Ablation 的数据导出方式。
4. 只有当前两步解释清楚后，再实现 map-admission guard。


## 9. 当前最简总括

到 2026-05-07 为止，本研究已经从“基础模型能不能删动态物体”推进到了更具体也更困难的问题：

- **删在哪里最合适？**
- **删多少才不会破坏初始化？**
- **动态信息是应该前端删光，还是后端分层利用？**
- **动态因子图到底是在解决问题，还是在放大前端噪声？**

目前最稳的事实不是“某个复杂动态因子图已经成功”，而是：

> 基础模型前端确实可以改善动态场景 SLAM，但它的收益高度依赖接入方式；
> 初始化、特征层删点时机、以及后端对动态约束的容忍方式，是决定成败的核心。

这份文档后续应作为总实验史持续追加，而不是被新的单次实验摘要替代。

## 10. 2026-05-12 Static Observability Fingerprint 首轮执行

### 10.1 先修复 full 诊断前的可运行性问题

在重编 `stslam_backend` 后，`backend_maskonly_smoke30_wxyz + hybrid_sequential_semantic_only` 一度在第 1 帧 `TrackReferenceKeyFrame -> Optimizer::PoseOptimization` 内崩溃。定位过程：

- `observability` 关闭后仍崩，说明不是 CSV 写盘导致。
- debug breadcrumb 显示已完成 BoW 匹配，`matches=232`，崩在 `optimizer.optimize()` 内部。
- `PoseOptimization` 增加边界/非有限值防护和可选 debug 日志。
- RGB-D pinhole 下的 depthless mono edge 改用 g2o 原生 `EdgeSE3ProjectXYZOnlyPose`，避免自定义 camera Jacobian 路径影响当前稳定基线。
- g2o 重新编译并重新链接后，`smoke30` 与 `STSLAM_OBSERVABILITY_LOG=1` 均恢复可运行。

验证结果：

| run | matched | ATE-SE3 | ATE-Sim3 | Sim3 scale | RPEt | RPER |
|---|---:|---:|---:|---:|---:|---:|
| `observability_sanity_smoke30_20260512_1807` | 28 | 0.033785 | 0.012155 | 0.626960 | 0.010511 | 0.500028 |

### 10.2 三组 full fingerprint 结果

输出目录：

- `/home/lj/dynamic-slam-public/runs/fingerprint_full_20260512_1810/`
- 汇总：`fingerprint_summary_v3.csv`
- 100 帧区间：`fingerprint_intervals_100f.csv`
- matched path 诊断：`fingerprint_path_diagnostics.csv`

| case | matched | ATE-SE3 | ATE-Sim3 | Sim3 scale | RPEt | RPER | has pose | first no pose | final KFs | final MPs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `frontend_boxfallback_nomask_hybrid` | 857 | 0.018163 | 0.016225 | 0.973531 | 0.011928 | 0.361599 | 859 | none | 185 | 4062 |
| `backend_maskonly_hybrid` | 857 | 0.381938 | 0.271534 | 0.324759 | 0.012486 | 0.519099 | 859 | none | 415 | 11155 |
| `backend_geom_soft_cap010_w025_hybrid` | 773 | 0.539582 | 0.286258 | 0.176382 | 0.027629 | 0.681808 | 770 | 770 | 324 | 11067 |

初步判断：

- `frontend_boxfallback_nomask_hybrid` 仍是强基线，scale 接近 1，ATE 和 RPE 都稳定。
- `backend_maskonly_hybrid` 没有 tracking lost，matched 仍是 857，但最终关键帧和地图点显著膨胀，说明失败更像是 map/keyframe admission 与支撑结构漂移，而不是简单丢帧。
- `backend_geom_soft_cap010_w025_hybrid` 在第 770 帧开始无 pose，`recently_lost` 持续到序列末尾，matched 从 857 降到 773，旧 soft route 继续作为负结果成立。
- 原先 `est_gt_accum_ratio` 与 ATE/Sim3 scale 不一致，并不是 ATE 评估被污染，而是该指标本质上测的是轨迹弧长/局部 zigzag，不应直接解释全局尺度。后续全局尺度以 evaluator 的 Umeyama `Sim3 scale` 为准，弧长比例仅作为局部抖动/运动曲折度诊断。

### 10.3 派生 path 指标修正

新增工具：

- `/home/lj/dynamic-slam-public/tools/compute_matched_path_diagnostics.py`

该工具复用 `evaluate_trajectory_ate.py` 的 timestamp association 与 alignment 逻辑，并输出：

- raw matched arc length ratio；
- endpoint displacement ratio；
- per-segment motion ratio 分位数；
- SE3 / Sim3 对齐后的 arc ratio；
- Umeyama Sim3 scale。

三组修正后 path 诊断：

| case | raw arc ratio | endpoint ratio | segment median | Sim3 scale | Sim3 arc ratio |
|---|---:|---:|---:|---:|---:|
| `frontend_boxfallback_nomask_hybrid` | 1.657543 | 1.055940 | 1.551833 | 0.973531 | 1.613670 |
| `backend_maskonly_hybrid` | 1.248811 | 1.796745 | 1.174199 | 0.324759 | 0.405563 |
| `backend_geom_soft_cap010_w025_hybrid` | 1.601890 | 5.305802 | 1.255965 | 0.176382 | 0.282545 |

解释修正：

- ATE / RPE / Sim3 scale 是独立 evaluator 结果，不受 observability CSV 派生指标影响。
- `raw arc ratio` 偏大不能直接推出全局 scale 偏大；它更敏感于局部抖动、zigzag 和轨迹曲折。
- `frontend_boxfallback` 虽然 raw arc ratio 偏大，但 endpoint ratio 接近 1，Sim3 scale 接近 1，说明其全局尺度仍可靠。
- `backend_maskonly` 的 endpoint ratio 明显偏大且 Sim3 scale 很低，说明其问题不是 tracking lost，而是全局轨迹形状/地图结构已经漂移。
- `backend_soft` 同时出现第 770 帧后无 pose、endpoint ratio 极高、Sim3 scale 极低，可作为旧 soft route 的明确负例。

### 10.4 下一步执行顺序

1. 暂停继续扫描 `geom_soft_cap010_w025` 周边参数。
2. 先做 Early Intervention Causal Ablation：A 原始 RGB/原始 depth，B filtered RGB/原始 depth，C 原始 RGB/filtered depth，D filtered RGB/filtered depth。
3. 优先确认 image-level 优势来自 RGB feature support reallocation、depth 动态污染抑制，还是二者共同作用。
4. 在 A/B/C/D 解释清楚之前，不急着实现 map-admission guard。
5. 后续论文中明确区分“全局尺度指标 Sim3 scale”和“轨迹弧长/抖动指标 raw arc ratio”。

### 10.5 Early Intervention Causal Ablation 首轮结果

目的：

- 固定后端为 `hybrid_sequential_semantic_only`、`maintenance_period=4`。
- 关闭 backend mask 接入：`DSLAM_PASS_MASK_ARG=0`、`ORB_SLAM3_MASK_MODE=off`、`STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=0`。
- 只改变输入 RGB / depth，判断 image-level 强基线的收益主要来自 RGB feature support reallocation，还是 depth 动态污染抑制。

数据构造：

- A：raw RGB + raw depth。
- B：filtered RGB + raw depth。
- C：raw RGB + filtered depth。
- D：filtered RGB + filtered depth。

输出位置：

- sequence 构造：`/home/lj/dynamic-slam-public/data/early_intervention_ablation_20260512/`
- run 目录：`/home/lj/dynamic-slam-public/runs/early_intervention_ablation_20260512_2026/`
- 汇总：`/home/lj/dynamic-slam-public/runs/early_intervention_ablation_20260512_2026/early_intervention_ablation_summary.csv`
- 主实验表已追加：`/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/experiments_0512_0517.csv`

首轮 full sequence 结果：

| case | RGB | depth | matched | ATE-SE3 | ATE-Sim3 | Sim3 scale | RPEt | RPER | final KFs | final MPs |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | raw | raw | 857 | 0.593504 | 0.254945 | 0.229283 | 0.019608 | 0.498886 | 478 | 10045 |
| B | filtered | raw | 857 | 0.357958 | 0.267740 | 0.365518 | 0.016928 | 0.489887 | 376 | 6598 |
| C | raw | filtered | 857 | 0.018457 | 0.016026 | 0.970406 | 0.012053 | 0.371801 | 190 | 3302 |
| D | filtered | filtered | 857 | 0.018362 | 0.016258 | 0.972360 | 0.012353 | 0.368723 | 190 | 3750 |

path 诊断补充：

| case | raw arc ratio | endpoint ratio | segment median | Sim3 arc ratio |
|---|---:|---:|---:|---:|
| A | 2.110069 | 4.362093 | 1.818557 | 0.483802 |
| B | 1.846859 | 1.038467 | 1.672290 | 0.675061 |
| C | 1.657860 | 1.036442 | 1.565461 | 1.608798 |
| D | 1.681507 | 1.037561 | 1.559595 | 1.635030 |

初步结论：

- 四组均 matched 857，且 `has_pose_frames=859`、`lost_frames=0`，因此差异不是 tracking lost 造成的。
- A 的 scale 明显塌缩，地图点膨胀到 10045，说明 raw RGB-D 在动态场景中会把错误深度支撑长期写入地图。
- B 只过滤 RGB 有改善，但仍然 ATE-SE3=0.358m、Sim3 scale=0.366，无法恢复可用精度；这说明 RGB feature support reallocation 不是主要矛盾。
- C 只过滤 depth 已经恢复到 ATE-SE3=0.0185m、Sim3 scale=0.970，几乎等价于 D。
- D 相对 C 只带来极小差异，说明当前 `walking_xyz` 上 image-level 强基线的决定性机制更可能是 **动态 depth 污染抑制**，而不是 RGB 纹理过滤本身。
- 这给第三层改动提供了更强支点：后续优先研究 depth-side early intervention / map-admission guard，而不是继续在 RGB mask postfilter 上扫参数。

稳定性补充：

- C / D 各重跑一次，ATE、Sim3 scale、trajectory sha256、KeyFrameTimeline sha256 均与 run_1 完全一致。
- `C_raw_rgb_filtered_depth` trajectory sha256：`505c166a38e8ec7196827acd4980c477bf0fb979f18c21b69299b42f4ad68d91`
- `D_filtered_rgb_filtered_depth` trajectory sha256：`cc72d150ed539e16a4662710f99c7fdeb70068f6e61177891db12adfbee81044`
- repeat 汇总：`/home/lj/dynamic-slam-public/runs/early_intervention_ablation_20260512_2026/early_intervention_cd_repeat_check.csv`

下一步：

1. 统计 filtered depth 的无效化区域、动态 mask 区域与 keyframe / mappoint admission 的对应关系。
2. 将第三层方案收敛为“depth provenance / dynamic-depth-aware map admission”，避免继续扩大到复杂动态因子图。
3. 若需要向 5.5 Pro 回传，优先提交本节 A/B/C/D 因果消融与 C/D bit-level repeat，而不是继续提交旧 soft route 参数扫描。

### 10.6 5.5 Pro 第三次反馈与 depth invalidation 初步统计

5.5 Pro 第三次反馈判断：

- 当前 A/B/C/D 消融结论在 `walking_xyz` 当前稳定协议内可信度高，约 85-90% 支持 “filtered depth 是 image-level 强基线的决定性因素”。
- 论文级还需要排除两个主要混杂：一是 “C/D 好只是因为 MapPoint 数量减少”，二是 “filtered depth 是否改变了数据桥、时间戳、scale 或使用了不可在线获得的信息”。
- 后续主线应收敛为 **foundation-model assisted dynamic-depth suppression and map admission for RGB-D ORB-SLAM3**。
- 最推荐顺序：先做 depth provenance / MapPoint admission 统计，再做 mask-guided depth invalidation vs random dropout 控制，最后实现 backend-side dynamic-depth-aware admission。

本地已先完成不改算法的 depth invalidation 统计：

- 工具：`/home/lj/dynamic-slam-public/tools/compute_depth_invalidation_stats.py`
- CSV：`/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/depth_invalidation_stats_boxfallback.csv`
- JSON summary：`/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/depth_invalidation_stats_boxfallback_summary.json`
- run 目录备份：`/home/lj/dynamic-slam-public/runs/early_intervention_ablation_20260512_2026/`

统计设置：

- raw depth：`/home/lj/dynamic_SLAM/results/20260505_yoloe_sam3_maskonly_wxyz/sequence`
- filtered depth：`/home/lj/d-drive/CODEX/basic_model_based_SLAM/experiments/20260504_yoloe_sam3_boxfallback_wxyz/sequence`
- mask：boxfallback sequence 的 `mask/`
- association：boxfallback sequence 的 `associations.txt`
- boundary 定义：`dilate(mask, r=5) - erode(mask, r=5)`

关键统计：

| item | value |
|---|---:|
| frames | 859 |
| unique depth paths | 827 |
| mean mask area ratio | 0.136234 |
| mean raw valid depth ratio | 0.598216 |
| mean filtered valid depth ratio | 0.477276 |
| total raw-valid depth invalidated ratio | 0.202168 |
| invalidated pixels inside mask share | 0.997409 |
| invalidated pixels outside mask share | 0.002591 |
| invalidated pixels on mask boundary share | 0.134656 |
| nonzero depth value changed but still valid | 0 |

初步解释：

- filtered depth 主要是 **将 raw valid depth 置为 invalid**，没有发现 “raw>0 且 filtered>0 但数值被改写” 的情况。
- 总体上约 20.2% 的 raw-valid depth 被无效化，其中 99.74% 位于动态 mask 内，mask 外仅 0.26%。
- 这初步排除了 “filtered depth 在 mask 外大规模改写数据” 的混杂，也强化了 “C/D 的收益来自定向 dynamic depth invalidation，而不是任意 depth 改动” 的解释。
- 但这还不能排除 “只要随机减少同等数量 depth 也能变好” 的替代解释；下一步必须做 random/static dropout control。

下一步执行顺序更新：

1. 补 A/B 双跑 repeat，形成 A/B/C/D 全组 repeat hash。
2. 构造 `C_random_same_ratio` 与 `C_static_same_ratio`，检验是否只是 depth sparsification 生效。
3. 若 random/static control 不能接近 C，则进入 backend-side V1：在 Frame depth association 阶段用 mask side-channel 无效化 dynamic depth，目标复现 C。
4. V1 成功后，再做 V2：map-admission-only veto，用来证明 “早期 depth association guard 是否必要”。

### 10.7 A/B repeat 与 depth dropout control

#### A/B repeat

目的：补齐 A/B/C/D 全组 repeat，避免只对强结果 C/D 做重复验证。

结果：

| case | repeat | ATE-SE3 | Sim3 scale | trajectory sha256 |
|---|---|---:|---:|---|
| A raw RGB + raw depth | run_2 | 0.593504 | 0.229283 | `602c91e5b1b20b00c881faa0c5fb9c7ece6b115316d24dbec8ade1e854d8a1d7` |
| B filtered RGB + raw depth | run_2 | 0.357958 | 0.365518 | `25fdb2d855c2b461d931e5b2cb835213d8d09f43043f17de0780f6983d9a6884` |

结论：

- A/B 的 run_2 与 run_1 指标、trajectory sha256、KeyFrameTimeline sha256 完全一致。
- 至此 A/B/C/D 四组均有 bit-level repeat，当前稳定协议下的因果消融具备较强可复现性。
- repeat 汇总：`/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/early_intervention_ab_repeat_check.csv`

#### depth dropout controls

目的：排除 “C/D 好只是因为 depth/MapPoint 变少” 的替代解释。

构造：

- `C_random_same_count`：raw RGB + raw depth；按唯一 depth 文件匹配 filtered depth 的无效化像素数量，在 raw-valid depth 全图随机置零。
- `C_static_same_count`：raw RGB + raw depth；按唯一 depth 文件匹配 filtered depth 的无效化像素数量，但只在 dynamic mask union 外置零。
- 两组都使用 `seed=20260512` 的确定性构造；后端仍固定为 `hybrid_sequential_semantic_only`、`maintenance_period=4`、关闭 backend mask。
- 因为原 association 存在 827 个唯一 depth path、859 行 frame association，所以 dropout 构造按唯一 depth 文件匹配 `30,867,270` 个目标无效化像素；上一节 per-frame 统计的 `31,914,213` 包含 depth 复用行的重复计数。

输出：

- sequence：`/home/lj/dynamic-slam-public/data/depth_dropout_controls_20260512/`
- run：`/home/lj/dynamic-slam-public/runs/depth_dropout_controls_20260512_2348/`
- 汇总：`/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/depth_dropout_controls_summary.csv`
- manifest：`/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/dropout_controls_manifest.json`

结果：

| case | matched | ATE-SE3 | ATE-Sim3 | Sim3 scale | RPEt | RPER | final KFs | final MPs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C exact dynamic-depth invalidation | 857 | 0.018457 | 0.016026 | 0.970406 | 0.012053 | 0.371801 | 190 | 3302 |
| random same-count dropout | 857 | 0.585112 | 0.279105 | 0.178698 | 0.018144 | 0.480719 | 494 | 8832 |
| static-region same-count dropout | 857 | 0.628315 | 0.280966 | 0.160108 | 0.019951 | 0.513641 | 523 | 10010 |

path 诊断：

| case | raw arc ratio | endpoint ratio | segment median | Sim3 arc ratio |
|---|---:|---:|---:|---:|
| random same-count dropout | 1.903797 | 2.310337 | 1.731239 | 0.340205 |
| static-region same-count dropout | 2.089340 | 1.274100 | 1.794354 | 0.334519 |

解释：

- 两个 dropout control 都 matched 857、`has_pose_frames=859`、`lost_frames=0`，因此失败仍不是 tracking lost。
- random same-count 与 static same-count 都没有接近 C；ATE-SE3 分别为 0.585m 和 0.628m，Sim3 scale 分别为 0.179 和 0.160。
- static-region same-count 删除了同等数量的 raw-valid depth，但刻意避开 dynamic mask union，结果 final MPs 仍膨胀到 10010，几乎回到 A 的 10045。
- 这说明 C 的收益不是来自 “depth 总量减少 / MapPoint 数量减少”，而是来自 **动态区域 depth 被定向无效化**。
- 当前证据链从 “ABCD 指标现象” 前进一步，已经具备对抗审稿人 “只是稀疏化 depth” 质疑的初步控制实验。

下一步：

1. 进入 backend-side V1：raw RGB + raw depth + mask side-channel，在 Frame depth association 阶段将 dynamic mask 内 depth 置为 invalid，目标复现 C。
2. 若 V1 成功，再做 V2 map-admission-only veto，以证明干预必须早到 depth association，还是 LocalMapping 准入也足够。
3. 在实现 V1/V2 前，先确认 ORB-SLAM3 Frame 中 depth 读取、`mvDepth`、`mvuRight`、`UnprojectStereo`、new MapPoint creation 的具体代码路径，避免误改 pose optimization 或 RGB feature extraction。

### 10.8 Backend-side V1 dynamic depth invalidation

目的：把上一节离线 filtered depth 的因果证据迁移到后端在线机制中。输入保持 raw RGB + raw depth，仅将 mask 作为 side-channel；不删除 RGB keypoints，不启用 postfilter，不改 pose optimization，只在 Frame depth association 后将 dynamic mask 内 keypoint 的 `mvDepth` 和 `mvuRight` 置为 invalid。

实现位置：

- `/home/lj/dynamic_SLAM/stslam_backend/include/Frame.h`
- `/home/lj/dynamic_SLAM/stslam_backend/src/Frame.cc`
- `/home/lj/dynamic_SLAM/stslam_backend/src/Tracking.cc`

核心机制：

- 新增 `Frame::InvalidateDepthInPanopticMask()`。
- 若 keypoint 的 `mvPanopticIds[i] > 0` 且 `mvDepth[i] > 0`，则置 `mvDepth[i] = -1`、`mvuRight[i] = -1`。
- 新增 depth provenance 与计数：`masked_features`、`invalidated_features`、`no_depth_masked_features`。
- 通过环境变量 `STSLAM_DYNAMIC_DEPTH_INVALIDATION=1` 启用。
- 本次运行同时设置 `STSLAM_PANOPTIC_SIDE_CHANNEL_ONLY=1`、`ORB_SLAM3_MASK_MODE=off`、`STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=0`，确保 mask 只作为 depth side-channel，不参与 RGB feature 硬删除。

运行设置：

- dataset：`ablation_ei_A_raw_rgb_raw_depth_wxyz`
- profile：`hybrid_sequential_semantic_only`
- maintenance period：4
- run：`/home/lj/dynamic-slam-public/runs/backend_dynamic_depth_v1_20260513_002527/`
- 汇总 CSV：`/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/backend_dynamic_depth_v1_summary.csv`

结果：

| case | matched | ATE-SE3 | ATE-Sim3 | Sim3 scale | RPEt | RPER | final KFs | final MPs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A raw RGB + raw depth | 857 | 0.593504 | 0.254945 | 0.229283 | 0.019608 | 0.498886 | 478 | 10045 |
| C raw RGB + filtered depth | 857 | 0.018457 | 0.016026 | 0.970406 | 0.012053 | 0.371801 | 190 | 3302 |
| backend V1 raw RGB + raw depth + dynamic depth invalidation | 857 | 0.033068 | 0.030440 | 0.958602 | 0.016215 | 0.429665 | 184 | 3136 |

V1 repeat：

| repeat | trajectory sha256 | KeyFrameTimeline sha256 |
|---|---|---|
| run_1 | `ca11d0300db7c7b4221649e6e73a11fb38fd00e84fd6735d91869bc63693a452` | `e79064d18a7a51eccd302edbc6a4fc755f9bd364e52de0db7121f2089af4a77e` |
| run_2 | `ca11d0300db7c7b4221649e6e73a11fb38fd00e84fd6735d91869bc63693a452` | `e79064d18a7a51eccd302edbc6a4fc755f9bd364e52de0db7121f2089af4a77e` |

depth invalidation 统计：

| item | value |
|---|---:|
| frames with invalidation log | 859 |
| masked keypoint observations | 135406 |
| invalidated depth keypoint observations | 123484 |
| masked but already no-depth observations | 11922 |
| mean invalidated depth features / frame | 143.753 |

path 诊断：

| case | raw arc ratio | endpoint ratio | segment median | Sim3 arc ratio |
|---|---:|---:|---:|---:|
| backend V1 | 1.729306 | 1.320196 | 1.530949 | 1.657716 |

解释：

- V1 从 A 的 `ATE-SE3=0.5935`、`scale=0.2293` 恢复到 `ATE-SE3=0.0331`、`scale=0.9586`，说明主要收益可以由后端在线 dynamic-depth invalidation 复现。
- V1 的 final KFs/MPs 为 184/3136，接近 C 的 190/3302，远低于 A 的 478/10045；这支持 “动态 depth 污染导致地图点和关键帧接纳膨胀” 的机制解释。
- V1 repeat 与 run_1 bit-level 一致，当前稳定协议下可复现。
- V1 尚未完全达到 C/D 的 0.018m 量级，说明离线 filtered depth 与 keypoint-level backend invalidation 之间仍存在细小机制差异；可能来自 mask/keypoint 采样边界、panoptic id assignment 与 pixel-level depth 文件无效化不完全等价，或少量初始化/近点策略差异。

下一步：

1. 做 V1b：对 keypoint depth invalidation 加入可控 mask dilation / boundary sensitivity，对齐离线 filtered depth 的像素级作用范围，检查 ATE 是否继续接近 C。
2. 做 V2：map-admission-only veto，即不改当前帧 pose 的 depth association，只在 new MapPoint / keyframe admission 阶段拒绝 dynamic-mask depth 生成的点；若 V2 明显弱于 V1，则证明干预需要足够早。
3. 追加一列 paper-facing 机制表：`raw depth pollution -> dynamic-depth invalidation -> MapPoint/KF admission contraction -> scale recovery`。

### 10.9 V1b mask boundary sensitivity

目的：解释 V1 与离线 C/D 之间的残余差距。V1 只在 keypoint 中心点落入 mask 时失效 depth；而离线 filtered depth 是像素级 depth 文件无效化，边界采样口径可能略有不同。因此加入可控半径查询：

- 新环境变量：`STSLAM_DYNAMIC_DEPTH_INVALIDATION_DILATION_RADIUS_PX`
- 默认值：0，即完全保持 V1 行为。
- radius > 0 时，在 keypoint 周围正方形窗口内查询 raw panoptic mask，只要邻域存在 `panopticId > 0`，则该 keypoint depth 失效。
- 不改变 RGB feature extraction，不启用 postfilter，不改 pose optimization。

输出：

- run：`/home/lj/dynamic-slam-public/runs/backend_dynamic_depth_v1b_20260513_0039/`
- 汇总：`/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/backend_dynamic_depth_v1b_radius_summary.csv`

结果：

| case | radius | matched | ATE-SE3 | ATE-Sim3 | Sim3 scale | RPEt | RPER | final KFs | final MPs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 | 0 | 857 | 0.033068 | 0.030440 | 0.958602 | 0.016215 | 0.429665 | 184 | 3136 |
| V1b radius=1 | 1 | 857 | 0.019391 | 0.017721 | 0.974438 | 0.012219 | 0.368174 | 170 | 3225 |
| V1b radius=2 | 2 | 857 | 0.019848 | 0.017441 | 0.969405 | 0.012144 | 0.363908 | 147 | 2835 |
| C raw RGB + filtered depth | pixel-level | 857 | 0.018457 | 0.016026 | 0.970406 | 0.012053 | 0.371801 | 190 | 3302 |
| D filtered RGB + filtered depth | pixel-level | 857 | 0.018362 | 0.016258 | 0.972360 | 0.012353 | 0.368723 | 190 | 3750 |

repeat：

| case | trajectory sha256 | KeyFrameTimeline sha256 |
|---|---|---|
| radius=1 run_1 | `3529e44c834a98501efd3590b8ecc2e1996e9424fced4b138a6077aef4d05dbd` | `b530fbb8c42f0992f7a5c449b775a53eea688ac25bd0a8c7cd65b79f3cdb6160` |
| radius=1 run_2 | `3529e44c834a98501efd3590b8ecc2e1996e9424fced4b138a6077aef4d05dbd` | `b530fbb8c42f0992f7a5c449b775a53eea688ac25bd0a8c7cd65b79f3cdb6160` |

depth invalidation 统计：

| case | masked keypoint obs | invalidated depth keypoint obs | mean invalidated / frame |
|---|---:|---:|---:|
| radius=0 | 135406 | 123484 | 143.753 |
| radius=1 | 143700 | 129697 | 150.986 |
| radius=2 | 150408 | 134551 | 156.637 |

path 诊断：

| case | raw arc ratio | endpoint ratio | segment median | Sim3 arc ratio |
|---|---:|---:|---:|---:|
| radius=0 | 1.729306 | 1.320196 | 1.530949 | 1.657716 |
| radius=1 | 1.655721 | 1.056041 | 1.547027 | 1.613397 |
| radius=2 | 1.647851 | 1.069665 | 1.530303 | 1.597435 |

解释：

- radius=1 将 V1 的 SE3 ATE 从 0.0331m 降到 0.0194m，几乎贴近离线 C/D；Sim3 scale 从 0.9586 提升到 0.9744。
- radius=2 的 Sim3 ATE 和 RPER 略优于 radius=1，但 final KFs/MPs 收缩到 147/2835，已经明显比 C/D 更激进；这提示继续扩大 mask 可能进入过度深度失效化区域。
- radius=1 repeat bit-level 一致，因此当前可作为后端在线主候选。
- 这组结果强烈支持：C/D 的收益并非依赖离线改 depth 文件本身，而可以被在线 mask side-channel 的 dynamic-depth-aware association 复现；残差主要来自边界采样口径。

当前推荐论文主线：

1. 主方法先用 `radius=1` 作为 dynamic-depth-aware association 默认设置。
2. radius=0/1/2 作为 boundary sensitivity ablation。
3. 下一步仍需做 V2 map-admission-only veto，回答 “只在建图准入阶段拦截是否足够”。
4. 若时间允许，再做 radius=1 在另一条动态序列上的外推验证；若时间不足，至少保留当前序列的 bit-level repeat 与 dropout control。

### 10.10 V2 map-admission-only veto

目的：做一个更强的因果拆分。V1/V1b 会在 Frame 层把 dynamic-mask keypoint 的 depth 置为无效，因此同时影响当前帧的 tracking / pose association 与后续 MapPoint / keyframe admission。V2 刻意不改当前帧 depth，只把 mask 作为建图准入侧信号，验证 “收益到底来自当前帧 pose depth correction，还是来自阻止动态深度进入地图”。

实现口径：

- 新环境变量：`STSLAM_DYNAMIC_MAP_ADMISSION_VETO=1`
- 显式关闭：`STSLAM_DYNAMIC_DEPTH_INVALIDATION=0`
- 同时设置：`STSLAM_PANOPTIC_SIDE_CHANNEL_ONLY=1`、`ORB_SLAM3_MASK_MODE=off`、`STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=0`、`STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURE_STAGES=none`
- 拦截点：
  - `StereoInitialization()`：动态 mask feature 不生成初始 MapPoint。
  - `NeedNewKeyFrame()`：动态 close depth feature 不参与 close-point 新关键帧触发统计。
  - `CreateNewKeyFrame()`：动态 close depth feature 不进入新关键帧 MapPoint 创建候选。
  - `LocalMapping::CreateNewMapPoints()`：如果匹配对任一端 feature 属于 dynamic instance，则不做三角化建图，关闭 LocalMapping backdoor。

运行：

- dataset：`ablation_ei_A_raw_rgb_raw_depth_wxyz`
- profile：`hybrid_sequential_semantic_only`
- maintenance period：4
- run：`/home/lj/dynamic-slam-public/runs/backend_map_admission_v2_20260513_133243/`
- 汇总：`/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/backend_map_admission_v2_summary.csv`
- 统一因果表：`/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/causal_ablation_unified_summary_20260513.csv`

V2 结果：

| case | matched | ATE-SE3 | ATE-Sim3 | Sim3 scale | RPEt | RPER | final KFs | final MPs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A raw RGB + raw depth | 857 | 0.593504 | 0.254945 | 0.229283 | 0.019608 | 0.498886 | 478 | 10045 |
| C raw RGB + filtered depth | 857 | 0.018457 | 0.016026 | 0.970406 | 0.012053 | 0.371801 | 190 | 3302 |
| D filtered RGB + filtered depth | 857 | 0.018362 | 0.016258 | 0.972360 | 0.012353 | 0.368723 | 190 | 3750 |
| V1b radius=1 dynamic depth invalidation | 857 | 0.019391 | 0.017721 | 0.974438 | 0.012219 | 0.368174 | 170 | 3225 |
| V2 map-admission-only veto | 857 | 0.018192 | 0.016019 | 0.972089 | 0.012130 | 0.372424 | 214 | 3592 |

V2 repeat：

| repeat | trajectory sha256 | KeyFrameTimeline sha256 |
|---|---|---|
| run_1 | `2347b2727a0def47d4ba4ea55bfe205ea6a872a96eddb102c718c0d0d4558d58` | `c1fcba0b72c379975f57fbbf86d22c49b2a974f942084014db300c8a9c0522c5` |
| run_2 | `2347b2727a0def47d4ba4ea55bfe205ea6a872a96eddb102c718c0d0d4558d58` | `c1fcba0b72c379975f57fbbf86d22c49b2a974f942084014db300c8a9c0522c5` |

V2 veto 计数：

| item | value |
|---|---:|
| map-admission veto log lines | 1592 |
| depth invalidation log lines | 0 |
| stereo initialization vetoed candidates | 372 |
| NeedNewKeyFrame dynamic close vetoed | 119960 |
| CreateNewKeyFrame vetoed candidates | 56974 |
| CreateNewKeyFrame accepted depth candidates | 161662 |
| LocalMapping skipped instance pairs | 2091 |
| LocalMapping kept static pairs | 9621 |

path 诊断：

| case | raw arc ratio | endpoint ratio | segment median | Sim3 arc ratio |
|---|---:|---:|---:|---:|
| V2 map-admission-only veto | 1.663761 | 1.030132 | 1.557891 | 1.617324 |

解释：

- V2 在不改当前帧 depth 的前提下达到 `ATE-SE3=0.018192m`、`scale=0.972089`，与 C/D 几乎一致，且 repeat 的轨迹与关键帧时间线 bit-level 一致。
- 这说明当前序列的主要病灶不是 “pose optimization 中短期使用了动态 depth” 本身，而是动态 depth 被允许进入 MapPoint / keyframe admission 后，长期污染地图并放大 scale drift。
- V2 的 final KFs/MPs 为 214/3592，比 C 的 190/3302 略多，但远低于 A 的 478/10045；这更像是 “在线建图准入约束” 与 “离线 depth 文件失效化” 的口径差异，而不是机制失败。
- 与 V1b radius=1 相比，V2 的 SE3 ATE 略好，RPER 略差，整体同量级；这提示论文方法可以优先收敛为 dynamic-depth-aware map admission，而不是必须在 Frame 层修改 depth。
- 需要保持谨慎：目前该强结论只在当前 full sequence 上成立。下一步应做至少一条额外动态序列验证，或在同一序列上做 V2 的关键拦截点 ablation。

下一步建议：

1. 先把 V2 作为当前主方法候选：`dynamic-depth-aware map admission veto`。
2. 做 V2 拦截点消融：`init only`、`NeedNewKeyFrame only`、`CreateNewKeyFrame only`、`LocalMapping only`、`CreateNewKeyFrame + LocalMapping`，确认最小有效机制。
3. 如果时间紧，优先做 `CreateNewKeyFrame + LocalMapping` 对比完整 V2，因为这最贴近 “地图准入” 的论文表述。
4. 将当前 A/C/D/V1b/V2 统一表回传给 5.5 Pro，请其判断论文主线、方法命名和下一组最小消融。

### 10.11 V2 gate-level ablation

依据 5.5 Pro 第五轮反馈，当前 V2 仍是 composite gate stack。为回答 “到底是初始化、keyframe scheduling、直接 MapPoint admission，还是 LocalMapping backdoor prevention 起主要作用”，加入四个细分开关：

- `STSLAM_DYNAMIC_MAP_ADMISSION_VETO_STEREO_INITIALIZATION`
- `STSLAM_DYNAMIC_MAP_ADMISSION_VETO_NEED_NEW_KEYFRAME`
- `STSLAM_DYNAMIC_MAP_ADMISSION_VETO_CREATE_NEW_KEYFRAME`
- `STSLAM_DYNAMIC_MAP_ADMISSION_VETO_LOCAL_MAPPING_CREATE_NEW_MAPPOINTS`

兼容性：若不设置细分开关，则默认继承 `STSLAM_DYNAMIC_MAP_ADMISSION_VETO`，因此原完整 V2 行为不变。

运行共同设置：

- dataset：`ablation_ei_A_raw_rgb_raw_depth_wxyz`
- profile：`hybrid_sequential_semantic_only`
- maintenance period：4
- `STSLAM_DYNAMIC_DEPTH_INVALIDATION=0`
- `STSLAM_PANOPTIC_SIDE_CHANNEL_ONLY=1`
- `ORB_SLAM3_MASK_MODE=off`
- `STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=0`
- `STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURE_STAGES=none`
- run root：`/home/lj/dynamic-slam-public/runs/backend_map_admission_v2_gate_ablation_20260513_144136/`
- 汇总：`/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/backend_map_admission_v2_gate_ablation_summary.csv`

结果：

| case | stereo init | NeedNewKF | CreateNewKF | LocalMapping | matched | ATE-SE3 | ATE-Sim3 | Sim3 scale | RPEt | RPER | final KFs | final MPs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full V2 compat | 1 | 1 | 1 | 1 | 857 | 0.018192 | 0.016019 | 0.972089 | 0.012130 | 0.372424 | 214 | 3592 |
| CreateNewKeyFrame + LocalMapping | 0 | 0 | 1 | 1 | 857 | 0.017934 | 0.016274 | 0.975517 | 0.012161 | 0.367759 | 224 | 3561 |
| StereoInitialization only | 1 | 0 | 0 | 0 | 857 | 0.602691 | 0.280673 | 0.168261 | 0.019679 | 0.509767 | 484 | 9873 |
| Full V2 minus NeedNewKeyFrame | 1 | 0 | 1 | 1 | 857 | 0.017466 | 0.015513 | 0.973968 | 0.012085 | 0.384479 | 223 | 3584 |

repeat：

| case | trajectory sha256 | KeyFrameTimeline sha256 |
|---|---|---|
| CreateNewKeyFrame + LocalMapping run_1 | `963b004699db984fc94ca6c902aeadf3ae7d888cc8e171ba037f15d49c8f58dd` | `ca4a144976a04230e5528c4172e5d906786fe06f6f7f785845565cef8fc51de2` |
| CreateNewKeyFrame + LocalMapping run_2 | `963b004699db984fc94ca6c902aeadf3ae7d888cc8e171ba037f15d49c8f58dd` | `ca4a144976a04230e5528c4172e5d906786fe06f6f7f785845565cef8fc51de2` |
| Full V2 minus NeedNewKeyFrame run_1 | `a9c91a31f069b0147f4cef0c6bcfd1946503b68d96ffbb3984be2c3dafc4e427` | `acd244b0ba32e135055f7dc75d150db46f3037a99ebdf509e0b16b755df8df8f` |
| Full V2 minus NeedNewKeyFrame run_2 | `a9c91a31f069b0147f4cef0c6bcfd1946503b68d96ffbb3984be2c3dafc4e427` | `acd244b0ba32e135055f7dc75d150db46f3037a99ebdf509e0b16b755df8df8f` |

gate 计数：

| case | log lines | depth invalidation log lines | init vetoed | NeedNewKF dynamic close vetoed | CreateNewKF vetoed | CreateNewKF accepted | LocalMapping skipped pairs | LocalMapping kept pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full V2 compat | 1592 | 0 | 372 | 119960 | 56974 | 161662 | 2091 | 9621 |
| CreateNewKeyFrame + LocalMapping | 928 | 0 | 0 | 0 | 67232 | 160457 | 2259 | 9577 |
| StereoInitialization only | 1 | 0 | 372 | 0 | 0 | 0 | 0 | 0 |
| Full V2 minus NeedNewKeyFrame | 951 | 0 | 372 | 0 | 65299 | 163852 | 2437 | 9671 |

解释：

- 兼容性 full V2 与 10.10 旧 V2 完全一致，说明新增细分开关没有改变完整 V2 默认行为。
- `CreateNewKeyFrame + LocalMapping` 单独就达到 `ATE-SE3=0.017934`、`scale=0.975517`，并且 repeat bit-level 一致。这是目前最强的机制证据：核心恢复来自 persistent static-map admission control。
- `StereoInitialization only` 仍然接近 A 的失败 regime：`ATE-SE3=0.602691`、`scale=0.168261`、final KFs/MPs=484/9873。因此不能把 V2 的成功解释为 “只清理了初始地图”。
- `Full V2 minus NeedNewKeyFrame` 仍然强：`ATE-SE3=0.017466`、`scale=0.973968`，repeat bit-level 一致。尽管完整 V2 中 NeedNewKeyFrame dynamic close veto 计数很大，但它不是当前序列恢复的必要 gate。
- 因此当前论文主线可以进一步收敛：D²MA 的核心不是 keyframe scheduling，也不是初始化；而是 `CreateNewKeyFrame` 的 RGB-D close-depth MapPoint 候选准入 + `LocalMapping::CreateNewMapPoints` 的 triangulation backdoor prevention。

当前结论：

1. 方法命名 `D²MA: Dynamic-Depth-Aware Map Admission` 更稳，因为最小有效机制确实落在 MapPoint/static-map admission。
2. NeedNewKeyFrame gate 可以作为辅助稳定项或完整系统配置，但不是论文核心贡献的必要组件。
3. 下一步最缺的是外部动态序列 sanity check，而不是继续做同一序列上的 2^4 全组合扫描。
4. 如果还要做一个同序列补充，优先 `CreateNewKeyFrame only`，用于判断 LocalMapping backdoor 是否必要；但就论文主线而言，现在已经足够支持 “CreateNewKeyFrame + LocalMapping” 作为最小主方法。

### 10.12 六次反馈后的收敛决策

5.5 Pro 对 10.11 gate-level ablation 的判断：

- 正式把 D²MA-min 收敛为 `CreateNewKeyFrame + LocalMapping` 两个 static-map admission gate。
- NeedNewKeyFrame gate 从主方法中移除，作为 optional safety guard / D²MA-full 配置保留。
- `StereoInitialization only` 的失败足以反驳 “只是初始化清理带来收益” 的质疑。
- 停止同一序列上的 2^4 gate 全组合扫描。
- 同序列只补一个 `CreateNewKeyFrame only`，用于判断 LocalMapping triangulation backdoor 是否必要。
- 随后优先转向外部动态序列 sanity check，例如 `fr3/walking_rpy`、`fr3/walking_halfsphere`，再补静态/低动态负例。

当前待执行：

1. 跑 `CreateNewKeyFrame only`：
   - `STSLAM_DYNAMIC_MAP_ADMISSION_VETO_STEREO_INITIALIZATION=0`
   - `STSLAM_DYNAMIC_MAP_ADMISSION_VETO_NEED_NEW_KEYFRAME=0`
   - `STSLAM_DYNAMIC_MAP_ADMISSION_VETO_CREATE_NEW_KEYFRAME=1`
   - `STSLAM_DYNAMIC_MAP_ADMISSION_VETO_LOCAL_MAPPING_CREATE_NEW_MAPPOINTS=0`
2. 判读：
   - 若 `ATE-SE3 <= 0.025` 且 `scale >= 0.95`，说明 direct RGB-D close-depth admission 是主导污染入口，LocalMapping 是安全补丁。
   - 若明显退化，则说明 LocalMapping backdoor 是 D²MA-min 的必要组成。

补充消融结果：

| case | repeat | matched | ATE-SE3 | ATE-Sim3 | Sim3 scale | RPEt | RPER | final KFs | final MPs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CreateNewKeyFrame only | run_1 | 857 | 0.025601 | 0.024892 | 0.980431 | 0.013677 | 0.386654 | 217 | 3957 |
| CreateNewKeyFrame only | run_2 | 857 | 0.025601 | 0.024892 | 0.980431 | 0.013677 | 0.386654 | 217 | 3957 |
| CreateNewKeyFrame + LocalMapping | run_1 | 857 | 0.017934 | 0.016274 | 0.975517 | 0.012161 | 0.367759 | 224 | 3561 |

repeat：

| case | trajectory sha256 | KeyFrameTimeline sha256 |
|---|---|---|
| CreateNewKeyFrame only run_1 | `14f327ed40c766bce77f0030735111bb244275b8536fd843a1ba5b074d20af56` | `5d428e5546cf7ecf6993fadebea1c5186145bb4df4ca57e3ec595e62fc2a25e4` |
| CreateNewKeyFrame only run_2 | `14f327ed40c766bce77f0030735111bb244275b8536fd843a1ba5b074d20af56` | `5d428e5546cf7ecf6993fadebea1c5186145bb4df4ca57e3ec595e62fc2a25e4` |

gate 计数：

| case | log lines | depth invalidation log lines | CreateNewKF vetoed | CreateNewKF accepted | LocalMapping skipped pairs |
|---|---:|---:|---:|---:|---:|
| CreateNewKeyFrame only | 608 | 0 | 104928 | 233832 | 0 |
| CreateNewKeyFrame + LocalMapping | 928 | 0 | 67232 | 160457 | 2259 |

解释：

- `CreateNewKeyFrame only` 已经显著脱离 A 的失败 regime，并保持 `scale=0.980431`，说明 direct RGB-D close-depth MapPoint admission 是主导污染入口。
- 但其 `ATE-SE3=0.025601` 略高于预设 `0.025m` 判据，且 final MPs=3957，高于 `CreateNewKeyFrame + LocalMapping` 的 3561；加入 LocalMapping gate 后 ATE 回到 0.017934，说明 LocalMapping triangulation backdoor 不是纯装饰，而是 D²MA-min 中有实际贡献的后门防护。
- 当前最稳表述：`CreateNewKeyFrame` 是主导 gate，`LocalMapping` 是必要的 backdoor guard；D²MA-min 保持两者共同开启。
- 同序列 gate ablation 至此可以停止，下一步转向外部动态序列 sanity check。

外部序列准备状态：

- 当前 `dynamic-slam-public/data/datasets.json` 只注册了 `walking_xyz` 系列及其 ablation/dropout 派生序列。
- 本地可见原始 TUM 序列包括：
  - `/home/lj/d-drive/CODEX/basic_model_based_SLAM/datasets/tum_rgbd/freiburg3_walking_static/rgbd_dataset_freiburg3_walking_static`
  - `/home/lj/d-drive/CODEX/basic_model_based_SLAM/datasets/tum_rgbd/freiburg3_sitting_xyz/rgbd_dataset_freiburg3_sitting_xyz`
  - `/home/lj/d-drive/CODEX/basic_model_based_SLAM/datasets/tum_rgbd/freiburg3_sitting_static/rgbd_dataset_freiburg3_sitting_static`
- 暂未在本机发现已下载/注册的 `walking_rpy` 或 `walking_halfsphere`。
- 因此外部验证需要先做数据准备：下载或定位 `walking_rpy / walking_halfsphere`，或先用 `walking_static / sitting_xyz` 做低动态/负例 sanity；同时需要生成与当前 D²MA side-channel 兼容的 mask/metadata 和 association。

### 10.13 外部/低动态 sanity：walking_static raw RGB-D + mask side-channel

目的：

- 不再继续在 `walking_xyz` 上做 2^4 gate 扫描。
- 先用本机已有的 `fr3/walking_static` 做一组外部/低动态 sanity，检查 D²MA-min 是否只是在 `walking_xyz` 上偶然有效。
- 该序列不是理想的外部动态运动序列：相机运动很小，GT matched coverage 约 0.298，不能替代 `walking_rpy / walking_halfsphere`。但它可以作为 “动态人物 + 低相机运动/负例风险” 的早期检查。

数据准备：

- raw TUM root：`/home/lj/d-drive/CODEX/basic_model_based_SLAM/datasets/tum_rgbd/freiburg3_walking_static/rgbd_dataset_freiburg3_walking_static`
- frontend mask donor：`/home/lj/dynamic-slam-public/runs/frontend_mask_full_wstatic_20260513_160712/sequence`
- D²MA side-channel sequence：`/home/lj/dynamic-slam-public/data/external_validation_20260513/walking_static_raw_rgb_raw_depth_mask/sequence`
- 新增工具：`/home/lj/dynamic-slam-public/tools/prepare_mask_sidechannel_sequence.py`
- registry id：`external_wstatic_rawrgb_rawdepth_mask`
- 汇总 CSV：`/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/external_wstatic_d2ma_min_summary.csv`
- run root：`/home/lj/dynamic-slam-public/runs/external_wstatic_d2ma_min_20260513_162610/`

integrity：

| item | value |
|---|---:|
| association rows | 743 |
| valid rows | 743 |
| unique RGB paths | 743 |
| unique depth paths | 717 |
| duplicate depth reuse | 26 |
| missing RGB/depth/mask | 0 / 0 / 0 |
| RGB-depth time diff > 0.03s | 17 |
| GT time diff > 0.03s | 3 |

运行设置：

- profile：`hybrid_sequential_semantic_only`
- maintenance period：4
- raw baseline：`DSLAM_PASS_MASK_ARG=0`，D²MA gates off
- D²MA-min：仅开启 `CreateNewKeyFrame + LocalMapping`
- `STSLAM_DYNAMIC_DEPTH_INVALIDATION=0`
- `ORB_SLAM3_MASK_MODE=off`
- `STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=0`
- raw RGB-D 图像不修改，只使用 mask/meta side-channel 做 static-map admission veto

结果：

| case | matched | ATE-SE3 | ATE-Sim3 | Sim3 scale | RPEt | RPER | has pose | lost | final KFs | final MPs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw baseline | 740 | 0.043770 | 0.015914 | 0.328297 | 0.022354 | 0.400347 | 743 | 0 | 251 | 4419 |
| D²MA-min | 740 | 0.017037 | 0.011695 | 0.646556 | 0.012511 | 0.247557 | 743 | 0 | 183 | 2508 |

D²MA-min 计数：

| item | value |
|---|---:|
| CreateNewKeyFrame log rows | 419 |
| CreateNewKeyFrame vetoed candidates | 92041 |
| CreateNewKeyFrame accepted depth candidates | 142400 |
| LocalMapping log rows | 204 |
| LocalMapping skipped instance pairs | 1102 |
| LocalMapping kept static pairs | 3971 |

path 诊断：

| case | raw arc ratio | Sim3 aligned arc ratio | trajectory sha256 | KeyFrameTimeline sha256 |
|---|---:|---:|---|---|
| raw baseline | 14.330887 | 4.704783 | `afb6ad9ef2460c19d950137174e95d0a627d061fa7d01ab54a152b5232189004` | `f0ca9ca657df97a3ae39bc853b0c78b07106cf2a2e2ddd2b07540cc8a6c43e04` |
| D²MA-min | 12.146206 | 7.853203 | `fd1ed211f47dd700fb03f39514a8a5a1e125b54630366e95f96d5aa4078a7c79` | `651234b59dd9a45487ddb7b976612d6f81f897b1c65c291c1ad2edad83888fd8` |

解释：

- `walking_static` 上 D²MA-min 相比 raw baseline 明显改善：`ATE-SE3` 从 0.043770 降到 0.017037，RPEt 从 0.022354 降到 0.012511，RPER 从 0.400347 降到 0.247557，final MPs 从 4419 降到 2508。
- matched poses 与 has-pose frames 完全一致，lost frames 均为 0，因此这次收益不是 “少跑/丢帧” 换来的。
- Sim3 scale 从 0.328 提升到 0.647，但仍没有接近 1；因此该结果支持 D²MA-min 的外部正向趋势，但不能宣称已经完全解决 `walking_static` 的尺度/路径问题。
- 该序列 GT matched coverage 较低，且存在少量 association timing warning；论文中如使用这组结果，应定位为 sanity/negative-control support，而不是主外部动态验证。
- 当前最稳结论：D²MA-min 不只是 `walking_xyz` 单序列上的偶然 gate trick；在 `walking_static` 上也能减少动态深度进入静态地图后的累积污染。但下一步仍必须补真正的外部动态相机运动序列。

下一步：

1. 优先定位或下载 `fr3/walking_rpy` / `fr3/walking_halfsphere`，复用 `prepare_mask_sidechannel_sequence.py` 生成 raw RGB-D + mask side-channel。
2. 若短时间内无法获得上述序列，先对 `sitting_xyz` 或 `sitting_static` 生成 mask side-channel，作为低动态/误伤负例。
3. 对外部序列至少跑 `raw baseline` 与 `D²MA-min`，必要时再补 D²MA-min repeat，确认可复现性。

### 10.14 外部动态相机序列：walking_rpy D²MA-min 验证

目的：

- 回应 5.5 Pro “必须补外部动态序列 sanity check” 的要求。
- 验证 D²MA-min 是否能从 `walking_xyz` 推广到另一条 TUM dynamic camera motion 序列。
- 该序列比 `walking_static` 更重要：相机存在 rpy 运动，动态人物仍持续进入视野，是更接近审稿质疑的外部验证。

数据准备：

- 官方 TUM 包：`rgbd_dataset_freiburg3_walking_rpy.tgz`，约 509 MB。
- raw root：`/home/lj/d-drive/CODEX/basic_model_based_SLAM/datasets/tum_rgbd/freiburg3_walking_rpy/rgbd_dataset_freiburg3_walking_rpy`
- raw count：RGB=910，depth=872，GT=3062。
- frontend donor：`/home/lj/dynamic-slam-public/runs/frontend_mask_full_wrpy_20260513_164700/sequence`
- D²MA side-channel sequence：`/home/lj/dynamic-slam-public/data/external_validation_20260513/walking_rpy_raw_rgb_raw_depth_mask/sequence`
- registry id：`external_wrpy_rawrgb_rawdepth_mask`
- run root：`/home/lj/dynamic-slam-public/runs/external_wrpy_d2ma_min_20260513_170248/`
- 汇总 CSV：`/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/external_wrpy_d2ma_min_summary.csv`

frontend export summary：

| item | value |
|---|---:|
| exported frames | 909 |
| export runtime sec | 539.528 |
| mean runtime ms/frame | 532.285 |
| mean mask ratio | 0.113947 |
| mean filtered detections | 1.042904 |
| total instances | 956 |
| stage5 dynamic memory gate | 777 |
| stage6 motion gate | 154 |
| stage8 high confidence override | 16 |
| stage9 confirmed track gate | 1 |
| kept insufficient temporal support | 8 |

integrity：

| item | value |
|---|---:|
| association rows | 909 |
| valid rows | 909 |
| unique RGB paths | 909 |
| unique depth paths | 866 |
| duplicate depth reuse | 43 |
| missing RGB/depth/mask | 0 / 0 / 0 |
| RGB-depth time diff > 0.03s | 23 |
| GT time diff > 0.03s | 3 |

运行设置：

- profile：`hybrid_sequential_semantic_only`
- maintenance period：4
- raw baseline：`DSLAM_PASS_MASK_ARG=0`，D²MA gates off
- D²MA-min：仅开启 `CreateNewKeyFrame + LocalMapping`
- `STSLAM_DYNAMIC_DEPTH_INVALIDATION=0`
- `ORB_SLAM3_MASK_MODE=off`
- `STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=0`
- raw RGB-D 不修改，只使用 mask/meta side-channel 做 static-map admission veto

结果：

| case | repeat | matched | ATE-SE3 | ATE-Sim3 | Sim3 scale | RPEt | RPER | has pose | lost | final KFs | final MPs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw baseline | run_1 | 906 | 0.997057 | 0.157951 | 0.084939 | 0.026161 | 0.626247 | 909 | 0 | 575 | 13872 |
| D²MA-min | run_1 | 906 | 0.367386 | 0.137205 | 0.260907 | 0.024261 | 0.591429 | 909 | 0 | 426 | 7331 |
| D²MA-min | run_2 | 906 | 0.367386 | 0.137205 | 0.260907 | 0.024261 | 0.591429 | 909 | 0 | 426 | 7331 |

D²MA-min 计数：

| item | value |
|---|---:|
| CreateNewKeyFrame log rows | 585 |
| CreateNewKeyFrame vetoed candidates | 104743 |
| CreateNewKeyFrame accepted depth candidates | 275583 |
| LocalMapping log rows | 994 |
| LocalMapping skipped instance pairs | 4833 |
| LocalMapping kept static pairs | 8465 |

path 诊断：

| case | raw arc ratio | Sim3 aligned arc ratio | trajectory sha256 | KeyFrameTimeline sha256 |
|---|---:|---:|---|---|
| raw baseline | 5.722309 | 0.486049 | `98f24b00e99e63ed42b6f0ff8250cc19654e3a34297611c971bc8d70acaf50e2` | `e2edf16c21f20f81e3ae5c690d47fc79607150f109b417ffe1f48ffbb06a2727` |
| D²MA-min run_1 | 5.292963 | 1.380969 | `956f9f0b5aa75e0c68ac0c6e974441dc77daaa73f1cd113db6a157609a995bbe` | `de8ebd9ad2e07721f1bfa9c1fe6f44b2cd9dc8a4740d78f4ae816572b18fc44d` |
| D²MA-min run_2 | 5.292963 | 1.380969 | `956f9f0b5aa75e0c68ac0c6e974441dc77daaa73f1cd113db6a157609a995bbe` | `de8ebd9ad2e07721f1bfa9c1fe6f44b2cd9dc8a4740d78f4ae816572b18fc44d` |

解释：

- D²MA-min 在真正外部动态相机序列上仍有显著正向作用：`ATE-SE3` 从 0.997 降到 0.367，Sim3 scale 从 0.085 提升到 0.261，final MPs 从 13872 降到 7331。
- D²MA-min repeat 的 trajectory 与 KeyFrameTimeline sha 完全一致，说明该结果不是异步噪声或偶然运行造成。
- 但 D²MA-min 没有把 `walking_rpy` 拉回 `walking_xyz` 的强性能区间；`ATE-SE3=0.367m` 和 `scale=0.261` 仍处于残余失败 regime。
- matched poses 与 has-pose frames 相同，lost frames 均为 0，因此主要问题仍更像 map/path/scale drift，而不是 tracking lost。
- 该结果非常关键：它既支持 D²MA 的机制方向，又暴露 D²MA-min 在更强外部相机运动下仍不足。论文主线需要避免写成 “完全解决动态 RGB-D SLAM”，更稳的说法是 “动态深度静态地图准入显著抑制污染，但外部 rpy 序列提示仍需要与前端 depth support allocation 或更强准入策略结合”。

当前阶段判断：

1. `walking_xyz`：D²MA-min 近似恢复 filtered-depth 强基线，支持核心方法。
2. `walking_static`：D²MA-min 明显改善且不丢帧，但 scale 仍未完全恢复，作为低动态 sanity 有价值。
3. `walking_rpy`：D²MA-min 可复现改善 raw failure，但残余尺度/路径漂移明显；这是下一轮必须回传给 5.5 Pro 的关键信息。

下一步建议：

1. 立即将 `walking_xyz + walking_static + walking_rpy` 的统一表回传 5.5 Pro，请其判断 D²MA 是否仍可作为主方法，还是需要把 full method 改成 “frontend depth support allocation + D²MA backend admission guard”。
2. 本地可继续补 `walking_halfsphere`，但在 `walking_rpy` 出现残余 failure 后，优先级应低于机制复盘；否则容易变成盲目多序列堆实验。
3. 若继续本地实现，下一候选不是更复杂 gate 扫描，而是检查 `walking_rpy` 中被 veto 后仍进入地图/BA 的污染路径，或者测试 D²MA-min + filtered depth / V1b radius 的组合上界。

### 10.15 walking_rpy 机制挖掘：C/D 上界与 V1b 互补性

依据 5.5 Pro 第七轮反馈，当前最重要的问题不是继续堆新序列，而是解释 `walking_rpy` 为什么 D²MA-min 只部分恢复。因此本节做两组机制诊断：

1. Early intervention upper bound：A/B/C/D，判断 filtered depth / filtered RGB 是否能像 `walking_xyz` 一样恢复强性能。
2. V1b / D²MA+V1b：判断 frame-level current-depth invalidation 是否能弥补 admission-only 太晚的问题。

数据与记录：

- A/B/C/D 数据根：`/home/lj/dynamic-slam-public/data/early_intervention_ablation_wrpy_20260513/`
- A/B/C/D run root：`/home/lj/dynamic-slam-public/runs/wrpy_early_intervention_20260513_180140/`
- V1b run root：`/home/lj/dynamic-slam-public/runs/wrpy_v1b_combo_20260513_181105/`
- 汇总 CSV：`/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/wrpy_mechanism_diagnosis_20260513_summary.csv`
- progressive CSV：`/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/wrpy_progressive_diagnostics_20260513.csv`
- progressive summary：`/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/wrpy_progressive_diagnostics_20260513_summary.csv`
- early window stats：`/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/wrpy_early_window_stats_20260513.csv`

共同设置：

- profile：`hybrid_sequential_semantic_only`
- maintenance period：4
- matched poses：906
- estimated poses：909
- has-pose frames：909
- lost frames：0
- 所有 A/B/C/D 均关闭 backend mask 接入：`DSLAM_PASS_MASK_ARG=0`，`ORB_SLAM3_MASK_MODE=off`

核心结果：

| case | intervention | ATE-SE3 | ATE-Sim3 | Sim3 scale | RPEt | RPER | final KFs | final MPs | final path |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | raw RGB + raw depth | 0.997057 | 0.157951 | 0.084939 | 0.026161 | 0.626247 | 575 | 13872 | 18.108613 |
| D²MA-min | raw RGB-D + map admission | 0.367386 | 0.137205 | 0.260907 | 0.024261 | 0.591429 | 426 | 7331 | 15.729182 |
| B | filtered RGB + raw depth | 0.472047 | 0.148503 | 0.191377 | 0.028759 | 0.652902 | 498 | 9623 | 17.491351 |
| C | raw RGB + filtered depth | 0.292579 | 0.125720 | 0.333615 | 0.023897 | 0.583789 | 347 | 6320 | 14.999138 |
| D | filtered RGB + filtered depth | 0.249133 | 0.123608 | 0.382943 | 0.025969 | 0.593776 | 361 | 7067 | 14.914669 |
| V1b radius1 | frame-level dynamic depth invalidation | 0.555564 | 0.151689 | 0.159520 | 0.019056 | 0.551196 | 288 | 5791 | 12.805787 |
| D²MA-min + V1b radius1 | admission + frame-level invalidation | 0.369391 | 0.139738 | 0.255511 | 0.025740 | 0.608669 | 335 | 5983 | 15.427209 |

动态计数：

| case | CreateNewKF vetoed | CreateNewKF accepted | LocalMapping skipped | LocalMapping kept | depth invalidation rows | invalidated features |
|---|---:|---:|---:|---:|---:|---:|
| D²MA-min | 104743 | 275583 | 4833 | 8465 | 0 | 0 |
| V1b radius1 | 0 | 0 | 0 | 0 | 909 | 124623 |
| D²MA-min + V1b radius1 | 102619 | 258546 | 4229 | 7504 | 909 | 124623 |

path diagnostics：

| case | raw arc ratio | Sim3 arc ratio | segment ratio median |
|---|---:|---:|---:|
| A | 5.722309 | 0.486049 | 6.443185 |
| D²MA-min | 5.292963 | 1.380969 | 6.101469 |
| B | 5.960051 | 1.140618 | 6.359554 |
| C | 5.030005 | 1.678085 | 5.219772 |
| D | 5.240289 | 2.006734 | 5.485933 |
| V1b radius1 | 4.166227 | 0.664597 | 4.495097 |
| D²MA-min + V1b radius1 | 5.320524 | 1.359454 | 5.855312 |

progressive 诊断：

| case | first checkpoint arc ratio > 2 | ratio at checkpoint | first checkpoint SE3 RMSE > 0.2 | RMSE at checkpoint | final arc ratio |
|---|---:|---:|---:|---:|---:|
| A | 200 | 2.721472 | 100 | 0.419418 | 5.722309 |
| D²MA-min | 200 | 2.493781 | 100 | 0.230439 | 5.292963 |
| B | 200 | 2.647648 | 100 | 0.238121 | 5.960051 |
| C | 200 | 2.411034 | 200 | 0.362614 | 5.030005 |
| D | 200 | 2.337276 | 200 | 0.237284 | 5.240289 |
| V1b radius1 | 200 | 2.261422 | 100 | 0.305594 | 4.166227 |
| D²MA-min + V1b | 200 | 2.592940 | 100 | 0.212008 | 5.320524 |

第 200 帧窗口状态：

| case | KFs | MPs | tracked MPs | inliers | estimated path | mask ratio |
|---|---:|---:|---:|---:|---:|---:|
| A | 150 | 5517 | 307 | 165 | 3.906259 | 0.000000 |
| D²MA-min | 131 | 3888 | 262 | 191 | 3.565893 | 0.121540 |
| B | 144 | 4928 | 282 | 180 | 3.926579 | 0.000000 |
| C | 129 | 3791 | 221 | 141 | 3.426596 | 0.000000 |
| D | 118 | 3755 | 241 | 141 | 3.270291 | 0.000000 |
| V1b radius1 | 91 | 3512 | 232 | 153 | 3.139116 | 0.121540 |
| D²MA-min + V1b | 118 | 3306 | 240 | 165 | 3.671715 | 0.121540 |

关键解释：

- `walking_rpy` 与 `walking_xyz` 的机制不同：在 `walking_xyz` 上 C/D 几乎恢复到 0.018m；但在 `walking_rpy` 上，C 只有 0.292579m，D 也只有 0.249133m，说明即使 image-level filtered depth/RGB 上界也不能完全恢复。
- filtered depth 仍然是最重要的单一干预：C 优于 B，也优于 D²MA-min；但 C 不够强，说明 residual failure 不是单纯 “admission-only 太晚”。
- filtered RGB 在 raw depth 下有帮助但有限：B 从 A 的 0.997 降到 0.472，说明 RGB feature support 对 rpy 有贡献；但 raw depth 污染仍然压制系统。
- D 比 C 略好，说明 RGB + depth 同时干预存在小幅互补；但 D 仍远离强恢复，提示 mask/association/ORB-SLAM3 在 rpy 下还有更深层问题。
- V1b radius1 改善局部 RPE 和路径长度膨胀，但全局 SE3 仍差，且 D²MA-min + V1b 基本不优于 D²MA-min；因此 “current-frame masked depth invalidation” 不是 rpy 残余失败的充分补丁。
- progressive 诊断显示所有变体在第 200 个 matched pose 时累计路径比已超过 2；这不是后半段突然失败，而是序列早期就进入路径长度膨胀模式，只是不同干预把膨胀程度压低。
- 第 200 帧时 raw 已有 150 个 KFs / 5517 个 MPs，D²MA-min 降到 131/3888，C/D/V1b 进一步压低地图规模；但 estimated path 仍约 3.1-3.9m，说明 “地图规模压缩” 与 “全局路径恢复” 不是一回事。

当前最可信失败原因排序：

1. `walking_rpy` 的失败不是单一路径污染，而是强旋转相机运动下的路径长度膨胀 / map consistency 问题；dynamic-depth admission 是重要通道，但不是唯一通道。
2. mask/depth intervention 对动态人物有效，但仍可能存在边界漏检、深度空洞、遮挡边缘、动态边界处静态/动态混合像素等问题；这些在 rpy 旋转下更容易进入局部地图支持。
3. ORB-SLAM3 RGB-D 在该序列上即使不 lost，也可能通过错误/不稳定的短期匹配持续积累路径尺度错误；这不是简单的 tracking lost。
4. 仅靠更早的 depth invalidation 不够；V1b 能减少部分路径膨胀，但没有恢复全局一致性，且与 D²MA 不互补。

阶段性结论：

- D²MA-min 仍是有效 backend admission guard，但 `walking_rpy` 证明它不是完整解决方案。
- `walking_rpy` 的论文价值在于给出边界条件：当动态深度污染不是唯一主因时，admission-only 可以压制地图膨胀但不能完全恢复尺度一致性。
- 下一步不应继续盲目跑 `walking_halfsphere`，而应先做更细的 failure-path diagnostics：定位第 100-200 帧附近的动态 mask、关键帧、MapPoint admission、局部地图 inliers 和路径膨胀之间的关系。

下一步建议：

1. 导出 `walking_rpy` 第 0-220 帧的关键诊断窗口，重点看第 100/200 pose 附近的 mask ratio、keyframe creation、MapPoint 增长、estimated step/path ratio。
2. 统计 accepted depth candidates 中靠近 mask boundary 的比例，判断是否是边界漏污染。
3. 如果要继续算法尝试，优先做 support-aware admission：动态边界附近 close-depth candidate 降权/延迟 admission，而不是简单扩大 invalidation。

### 10.16 walking_rpy 两项机制诊断：早期路径膨胀同步 + mask-boundary admission

目标：

- 诊断 1：确认 `walking_rpy` 的路径长度膨胀是否与早期 keyframe / MapPoint admission 同步出现，而不是全序列后期才突然崩坏。
- 诊断 2：统计 D²MA-min 已经 veto 直接动态点之后，是否仍有“静态特征但靠近动态 mask 边界”的深度候选进入 keyframe admission，并进一步创建 MapPoint。

代码与运行：

- 后端新增诊断开关：`STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_DIAGNOSTICS=1`，半径参数 `STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_RADIUS_PX=5`。
- 该补丁只输出 `[STSLAM_MAP_ADMISSION_BOUNDARY_DIAG]` 日志，不改变 `ShouldVetoDynamicMapAdmission` 的判断，也不改变默认实验路径。
- 诊断 run：`/home/lj/dynamic-slam-public/runs/wrpy_boundary_diag_d2ma_min_20260513_185107`
- 解析结果已复制到小论文目录：
  - `wrpy_boundary_admission_diag_20260513.csv`
  - `wrpy_boundary_admission_diag_summary_20260513.csv`
  - `wrpy_boundary_path_sync_20260513.csv`

注意：

- 诊断 run 的 ATE-SE3 为 `0.427940m`，Sim3 scale 为 `0.209995`，差于此前稳定 D²MA-min 的 `0.367386m / 0.260907`。
- 由于边界诊断逐 keypoint 扫描 mask 邻域，会增加运行时开销，可能改变线程调度；因此该 run 只用于机制诊断，不作为可比精度 baseline。

边界 admission 汇总：

| window | keyframe rows | valid depth pre-veto | direct dynamic valid ratio | accepted static-near-mask ratio | created static-near-mask ratio |
|---|---:|---:|---:|---:|---:|
| all | 593 | 371123 | 0.245140 | 0.054746 | 0.109338 |
| frame 0-100 | 58 | 49032 | 0.324869 | 0.069480 | 0.097758 |
| frame 0-200 | 148 | 91457 | 0.318587 | 0.080793 | 0.131341 |
| frame 0-220 | 167 | 99613 | 0.298505 | 0.074959 | 0.124942 |
| frame 221-end | 426 | 271510 | 0.225561 | 0.048029 | 0.103191 |

与路径/地图状态同步：

| checkpoint frame | KFs | MPs | estimated path | accepted static-near-mask cumulative | created static-near-mask cumulative | created static-near-mask ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 29 | 1603 | 0.520374 | 1242 | 493 | 0.095321 |
| 100 | 50 | 2606 | 1.263667 | 2300 | 837 | 0.097758 |
| 150 | 84 | 3439 | 2.525301 | 4136 | 1635 | 0.121878 |
| 200 | 131 | 3888 | 3.565893 | 5035 | 2048 | 0.131341 |
| 220 | 149 | 4422 | 4.036887 | 5238 | 2155 | 0.124942 |
| 908 | 422 | 7334 | 15.637800 | 15337 | 6673 | 0.109338 |

诊断解读：

- D²MA-min 对“直接落在动态 mask 内”的有效深度候选是有效的：全序列约 `24.5%` 的有效深度候选被 veto，前 200 帧该比例约 `31.9%`。
- 但 veto 后仍有边界污染通道：前 200 帧 accepted depth candidates 中 `8.08%` 是“静态特征但 5px 内邻近动态 mask”的候选；更关键的是，新建 MapPoint 中这类点占 `13.13%`。
- 这类边界候选在早期已经大量进入地图：第 200 帧前累计创建 `2048` 个 static-near-mask MapPoints，此时系统已达到 `131` 个 keyframes、`3888` 个 MapPoints、估计路径 `3.565893m`。
- 这支持一个更细的失败假设：`walking_rpy` 不是 D²MA 完全无效，而是 D²MA 只堵住了“mask 内直接动态点”，但没有处理 mask 边缘、混合深度、遮挡边界和支持区域不确定性。
- 因此下一步算法尝试应优先从 `support-aware / boundary-aware map admission` 入手，而不是继续扩大普通 depth invalidation 或继续扫 NeedNewKF gate。

下一步收敛方向：

- 设计 `D²MA-boundary`：对 direct dynamic mask 继续 hard veto；对 static-near-mask close-depth candidate 不直接创建 MapPoint，改为延迟 admission、需要跨帧静态支持，或要求更高 inlier/local support。
- 优先在 `walking_rpy` 做小规模验证，再回到 `walking_xyz` 和 `walking_static` 做负面影响检查。
- 新实验必须保持核心表述清楚：D²MA-min 是最小 backend guard，`D²MA-boundary` 是针对 rpy 暴露出的边界污染通道的第二层增强。

### 10.17 D²MA-boundary / support-aware admission 初版验证

目标：

- 基于 10.16 的诊断结果，验证 `walking_rpy` 的残余失败是否确实来自“直接动态 mask 外侧的边界邻域污染”。
- 在不修改 RGB、不 invalid 当前帧 depth、不改 pose optimization 权重的前提下，把动态先验继续限制在 static-map admission 入口。
- 先做一个可控的 hard-veto 初版：直接动态点仍由 D²MA-min veto；对于静态特征但 5px 邻域内存在动态 mask support 的 close-depth / triangulation candidate，暂不允许进入持久静态地图。

实现：

- `KeyFrame` 新增 `mvStaticNearDynamicMask`，在 KeyFrame 从 Frame 构造时缓存每个特征是否属于 static-near-dynamic-mask。
- `CreateNewKeyFrame` 新增 boundary admission veto：若 close-depth 新建 MapPoint 候选是 `static-near-dynamic-mask`，则跳过该新 MapPoint admission。
- `LocalMapping::CreateNewMapPoints` 新增 boundary pair veto：若匹配对任一端特征是 `static-near-dynamic-mask`，则阻止通过 LocalMapping triangulation backdoor 创建新 MapPoint。
- 新开关：
  - `STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_VETO=1`
  - `STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_VETO_CREATE_NEW_KEYFRAME=1`
  - `STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_VETO_LOCAL_MAPPING_CREATE_NEW_MAPPOINTS=1`
  - `STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_RADIUS_PX=5`

注意：

- 该版本是 `D²MA-boundary r=5 hard admission veto`，是 support-aware admission 的第一版风险门控，不是最终的 delayed admission / multi-frame static support 版本。
- 实验仍保持 D²MA-min 的核心边界：raw RGB-D 输入不变，mask 只作为 side-channel，depth invalidation off。

结果文件：

- 汇总表：`/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/d2ma_boundary_support_admission_summary_20260513.csv`
- 总实验表已追加：`/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/experiments_0512_0517.csv`

精度结果：

| sequence | method | repeat | matched | ATE-SE3 | ATE-Sim3 | Sim3 scale | RPEt | RPER |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| walking_rpy | D²MA-boundary r5 | run_1 | 906 | 0.269999 | 0.120722 | 0.361678 | 0.026173 | 0.606005 |
| walking_rpy | D²MA-boundary r5 | run_2 | 906 | 0.269999 | 0.120722 | 0.361678 | 0.026173 | 0.606005 |
| walking_xyz | D²MA-boundary r5 | run_1 | 857 | 0.017884 | 0.016043 | 0.974357 | 0.011772 | 0.375444 |
| walking_static | D²MA-boundary r5 | run_1 | 740 | 0.010972 | 0.008703 | 0.782028 | 0.010462 | 0.219777 |

机制计数：

| sequence | repeat | final KFs | final MPs | skipped new boundary candidates | skipped LM boundary pairs | boundary new skip ratio | boundary LM skip ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| walking_rpy | run_1 | 370 | 6006 | 8179 | 587 | 0.029898 | 0.144084 |
| walking_rpy | run_2 | 370 | 6006 | 8179 | 587 | 0.029898 | 0.144084 |
| walking_xyz | run_1 | 176 | 3155 | 6507 | 521 | 0.039457 | 0.067087 |
| walking_static | run_1 | 141 | 1791 | 3916 | 117 | 0.029891 | 0.096854 |

与既有结果对比：

- `walking_rpy`：raw baseline `ATE-SE3=0.997057 / scale=0.084939`；D²MA-min `0.367386 / 0.260907`；D²MA-boundary r5 达到 `0.269999 / 0.361678`，Sim3 ATE 从 D²MA-min 的 `0.137205` 降到 `0.120722`。
- `walking_rpy` run_1/run_2 trajectory sha 与 KeyFrameTimeline sha 完全一致，说明该结果具备 bit-level repeat stability。
- `walking_xyz`：D²MA-min 为 `ATE-SE3=0.017934 / scale=0.975517`；D²MA-boundary r5 为 `0.017884 / 0.974357`，基本不退化。
- `walking_static`：D²MA-min 为 `ATE-SE3=0.017037 / scale=0.646556`；D²MA-boundary r5 为 `0.010972 / 0.782028`，低动态 sanity 反而继续改善。

阶段性解读：

- 10.16 的边界污染诊断被正向验证：对 static-near-dynamic-mask 的 MapPoint admission 做约束后，`walking_rpy` 从残余失败区进一步恢复，且结果可重复。
- 这说明 `walking_rpy` 不是 D²MA-min 方向错误，而是最小 D²MA 只处理了 mask 内直接动态点，漏掉了动态边界/混合深度/遮挡边缘这一类更隐蔽的静态地图污染通道。
- `walking_xyz` 与 `walking_static` 的 sanity 结果说明 r5 hard-veto 当前没有明显误伤，暂时可以作为 D²MA-boundary 的候选增强机制。
- 但 `walking_rpy` 的 scale 仍只有 `0.361678`，没有恢复到 `walking_xyz` 的强尺度一致性；因此不应宣称已经解决 rpy，只能说 boundary-aware admission 明显缓解残余失败。

下一步：

1. 把本节结果统一回传 5.5 Pro，请其判断是否应将论文方法从 D²MA-min 扩展为 “D²MA-min + boundary-risk admission guard”。
2. 已完成小半径消融：`r=3 / r=5 / r=8`，确认 boundary radius 对 rpy 残余污染存在明确影响。
3. 若继续实现 support-aware 完整版，优先从 hard-veto 升级为 delayed admission：边界候选需要跨关键帧静态支持或更高局部 inlier support 后才允许进入 map。
4. 论文叙事暂定：D²MA-min 是最小核心，D²MA-boundary 是针对 foundation-mask boundary uncertainty 的增强项。

半径敏感性补充：

| sequence | method | radius | matched | ATE-SE3 | ATE-Sim3 | Sim3 scale | RPEt | RPER | skipped new boundary candidates | skipped LM boundary pairs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| walking_rpy | D²MA-boundary | 3 | 906 | 0.317809 | 0.130713 | 0.305339 | 0.027600 | 0.642786 | 5348 | 361 |
| walking_rpy | D²MA-boundary | 5 | 906 | 0.269999 | 0.120722 | 0.361678 | 0.026173 | 0.606005 | 8179 | 587 |
| walking_rpy | D²MA-boundary | 8 | 906 | 0.265845 | 0.120408 | 0.366495 | 0.033081 | 0.740172 | 10999 | 936 |

半径敏感性解读：

- `r=3` 已优于 D²MA-min，但明显弱于 `r=5/8`，说明较窄边界无法覆盖足够的 mask-boundary / mixed-depth 污染。
- `r=8` 的全局 ATE 和 scale 略好于 `r=5`，但 RPEt/RPER 明显变差，说明更宽 hard-veto 可能开始牺牲局部连续性或可用支持点。
- 当前更稳的主配置仍可先用 `r=5`：它在 rpy 上显著优于 D²MA-min，在 xyz/static 上已通过 sanity，且局部 RPE 比 r8 更温和。
- 论文机制上更重要的不是继续扩大半径，而是把 hard boundary veto 升级为真正的 support-aware delayed admission：边界候选先进入 probation，不立即成为长期静态 MapPoint，待跨帧静态支持足够再 admission。

### 10.18 same-count non-boundary veto control

目标：

- 回答 D²MA-B 的收益是否只是“删了更多点 / 让地图更稀疏”，而不是来自 near-mask boundary-risk targeting。
- 构造一个同量删除对照：D²MA-min 的 direct dynamic gates 仍开启，但 boundary targeting 关闭；系统先计算若启用 D²MA-B r5 时会有多少 boundary-risk candidate 被跳过，然后改为跳过同数量的 non-boundary static candidate。

实现：

- 新增 control 开关：
  - `STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_SAME_COUNT_CONTROL=1`
  - `STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_SAME_COUNT_CONTROL_CREATE_NEW_KEYFRAME=1`
  - `STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_SAME_COUNT_CONTROL_LOCAL_MAPPING_CREATE_NEW_MAPPOINTS=1`
- `CreateNewKeyFrame`：按当前 keyframe 的 boundary-risk new-candidate budget，跳过同数量的非 boundary 新 MapPoint 候选。
- `LocalMapping::CreateNewMapPoints`：按匹配对中的 boundary-risk pair budget，跳过同数量的非 boundary pair。
- 该 control 不跳过 boundary-risk candidates，因此可检验“同量删除但不按 boundary targeting”是否能复现 D²MA-B。

结果文件：

- `/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/d2ma_boundary_samecount_control_summary_20260513.csv`
- 总实验表已追加：`experiments_0512_0517.csv`

结果：

| sequence | method | matched | ATE-SE3 | ATE-Sim3 | Sim3 scale | RPEt | RPER | final KFs | final MPs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| walking_rpy | raw RGB-D | 906 | 0.997057 | 0.157951 | 0.084939 | 0.026161 | 0.626247 | 909 | 13872 |
| walking_rpy | D²MA-min | 906 | 0.367386 | 0.137205 | 0.260907 | 0.024261 | 0.591429 | 426 | 7331 |
| walking_rpy | D²MA-B r5 | 906 | 0.269999 | 0.120722 | 0.361678 | 0.026173 | 0.606005 | 370 | 6006 |
| walking_rpy | same-count non-boundary control r5 | 906 | 0.604425 | 0.156306 | 0.138884 | 0.025395 | 0.593053 | 427 | 7389 |

control 计数：

| metric | value |
|---|---:|
| boundary budget new candidates | 7990 |
| skipped non-boundary new candidates | 7988 |
| boundary budget LM pairs | 246 |
| skipped non-boundary LM pairs | 209 |
| accepted depth candidates pre-control | 276790 |

解读：

- same-count control 删除了几乎同等数量的新 MapPoint 候选，但 `ATE-SE3=0.604425`、scale `0.138884`，明显差于 D²MA-min，更远差于 D²MA-B r5。
- 因此 D²MA-B 的收益不能解释为 generic sparsification，也不是单纯减少 MapPoint 数量；关键在于 targeting near-mask boundary-risk candidates。
- control 的 final MPs `7389` 与 D²MA-min `7331` 接近，但精度更差，进一步说明“地图规模压缩”不是充分解释。
- 该结果可作为论文中反驳 “boundary guard 只是多删点 / dilation trick” 的核心因果对照。

阶段性结论：

- D²MA-B r5 现在具备三类证据：机制诊断发现 boundary leakage，targeted boundary veto 改善 rpy，same-count non-boundary veto 不能复现收益。
- 下一步若继续加强方法，不应继续做同量删点或更大 hard-veto，而应进入 D²MA-DA：boundary-risk candidate 的 probationary / delayed static-map admission。

### 10.19 `walking_halfsphere` 外部动态序列验证

目的：

- 按照 5.5 Pro 的建议，补一条不参与前期机制定位的外部动态序列，验证 D²MA-min / D²MA-B r5 是否具备跨序列迁移性。
- 该实验不再使用前端 filtered depth 改写深度，而是保持 raw RGB-D，仅把 YOLOE/SAM3 mask 作为 side-channel 输入后端 admission gate。
- 对比三组：raw RGB-D baseline、D²MA-min、D²MA-B r5。

数据准备：

- TUM RGB-D `fr3/walking_halfsphere` 已下载并解压到 `/home/lj/d-drive/CODEX/basic_model_based_SLAM/datasets/tum_rgbd/freiburg3_walking_halfsphere/rgbd_dataset_freiburg3_walking_halfsphere`。
- 前端 mask 导出目录：`/home/lj/dynamic-slam-public/runs/frontend_mask_full_whalfsphere_20260513_203759/sequence`。
- 生成 raw RGB-D + mask side-channel 序列：`/home/lj/dynamic-slam-public/data/external_validation_20260513/walking_halfsphere_raw_rgb_raw_depth_mask/sequence`。
- `datasets.json` 已注册：`external_whalfsphere_rawrgb_rawdepth_mask`。
- integrity check 文件：`/home/lj/dynamic-slam-public/data/external_validation_20260513/walking_halfsphere_raw_rgb_raw_depth_mask/integrity.json`。

前端导出统计：

| item | value |
|---|---:|
| exported frames | 1067 |
| export runtime sec | 611.488 |
| mean runtime ms | 513.728 |
| mean mask ratio | 0.129538 |
| mean filtered detections | 1.10684 |
| mean motion | 0.43566 |
| mean geometry dynamic | 0.42473 |

integrity note：

- 文件存在性检查通过，`missing_rgb/depth/mask/meta=0`。
- 因 TUM 原始时间戳采样差异，存在 `rgb_depth_time_diff>0.03` 的条目 26 个、`gt_time_diff>0.03` 的条目 3 个；本次结果应记录该注意事项，但三组方法使用完全相同 association，因此组间比较仍有效。

结果文件：

- 汇总表：`/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/whalfsphere_d2ma_b_validation_summary_20260513.csv`
- run root：`/home/lj/dynamic-slam-public/runs/whalfsphere_d2ma_b_validation_20260513_205117`
- 总实验表已追加：`experiments_0512_0517.csv`

结果：

| sequence | method | matched | ATE-SE3 | ATE-Sim3 | Sim3 scale | RPEt | RPER | final KFs | final MPs | final path |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| walking_halfsphere | raw RGB-D | 1064 | 0.418817 | 0.278236 | 0.560136 | 0.020018 | 0.521662 | 610 | 13213 | 18.488295 |
| walking_halfsphere | D²MA-min | 1064 | 0.178466 | 0.133784 | 0.798252 | 0.016987 | 0.498543 | 370 | 5491 | 16.107576 |
| walking_halfsphere | D²MA-B r5 | 1064 | 0.156093 | 0.118844 | 0.823258 | 0.016220 | 0.484402 | 308 | 4804 | 15.225326 |

gate 统计：

| method | direct new veto | LM instance pairs skipped | boundary skipped new | boundary LM pairs skipped |
|---|---:|---:|---:|---:|
| D²MA-min | 131112 | 3054 | 0 | 0 |
| D²MA-B r5 | 133793 | 3516 | 8502 | 529 |

解读：

- `walking_halfsphere` 上，D²MA-min 相比 raw baseline 显著改善：ATE-SE3 从 `0.418817` 降至 `0.178466`，Sim3 scale 从 `0.560136` 提升至 `0.798252`。
- D²MA-B r5 在外部序列上继续优于 D²MA-min：ATE-SE3 进一步降至 `0.156093`，ATE-Sim3 降至 `0.118844`，scale 提升至 `0.823258`，RPEt/RPER 也同步下降。
- 这与 `walking_rpy` 的现象一致：boundary-risk guard 不是只在单个失败序列上过拟合，而是在另一条动态人物序列上也能缓解地图准入污染。
- 同时 D²MA-B r5 final MPs 从 D²MA-min 的 `5491` 降到 `4804`，但 RPE 没有恶化，说明这次不像 r8 那样体现出明显 over-veto / support starvation。

阶段性结论：

- 当前证据链已经明显强于“启发式 mask dilation”：D²MA-B r5 有机制诊断、same-count non-boundary causal control、以及外部动态序列验证三类支撑。
- 下一步可以把 D²MA-B r5 作为论文主方法的稳定版本，同时开始做 D²MA-DA / delayed admission 原型，目标是把 hard boundary veto 升级为 support-aware admission，而不是继续扩大半径。
- 现在值得把 same-count control + `walking_halfsphere` 外部验证结果统一回传给 5.5 Pro，请其判断论文主线是否应正式收敛到 “D²MA-min + Boundary-Risk Admission Guard + Delayed Admission 展望/原型”。

### 10.20 D²MA-DA-lite 分层消融：support-aware delayed admission

目标：

- 在 D²MA-B r5 hard boundary veto 基础上，尝试把边界风险候选从“全部拒绝”推进到“有静态支持才准入”。
- 重点区分两个 admission 层：
  - `CreateNewKeyFrame`：RGB-D 单帧深度直接生成新 MapPoint。
  - `LocalMapping::CreateNewMapPoints`：关键帧之间三角化生成新 MapPoint。
- 这一区分很关键：单帧 RGB-D 边界深度容易受 mixed depth / 遮挡边缘污染；跨关键帧三角化候选至少经过了匹配、视差、重投影、尺度一致性检查。

实现：

- 新增开关：
  - `STSLAM_DYNAMIC_MAP_ADMISSION_DELAYED_BOUNDARY=1`
  - `STSLAM_DYNAMIC_MAP_ADMISSION_DELAYED_BOUNDARY_CREATE_NEW_KEYFRAME=1/0`
  - `STSLAM_DYNAMIC_MAP_ADMISSION_DELAYED_BOUNDARY_LOCAL_MAPPING_CREATE_NEW_MAPPOINTS=1/0`
  - `STSLAM_DYNAMIC_MAP_ADMISSION_DELAYED_BOUNDARY_SUPPORT_RADIUS_PX`
  - `STSLAM_DYNAMIC_MAP_ADMISSION_DELAYED_BOUNDARY_MIN_SUPPORT`
  - `STSLAM_DYNAMIC_MAP_ADMISSION_DELAYED_BOUNDARY_MIN_OBS`
- 支持定义：boundary-risk candidate 周围一定像素半径内，存在足够数量的“干净静态已跟踪 MapPoint”；支持点不能是 direct dynamic feature、不能在 static-near-dynamic-mask 区域，且 MapPoint observations 需达到阈值。
- CKF+LM full DA：两个层都允许 support-confirmed boundary candidate 准入。
- LM-only DA：`CreateNewKeyFrame` 仍 hard veto boundary-risk RGB-D 新点，只在 `LocalMapping::CreateNewMapPoints` 放行 support-confirmed boundary pair。

结果文件：

- 汇总表：`/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/d2ma_da_layered_ablation_summary_20260513.csv`
- 总实验表已追加：`experiments_0512_0517.csv`

结果：

| sequence | method | matched | ATE-SE3 | ATE-Sim3 | Sim3 scale | RPEt | RPER | final KFs | final MPs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| walking_rpy | D²MA-B r5 current ref | 906 | 0.307193 | 0.151482 | 0.275739 | 0.015342 | 0.560025 | 285 | 7393 |
| walking_rpy | D²MA-DA full CKF+LM s18 m2 | 906 | 0.322344 | 0.155957 | 0.251413 | 0.016584 | 0.562662 | 292 | - |
| walking_rpy | D²MA-DA full CKF+LM s18 m3 | 906 | 0.383317 | 0.149283 | 0.229141 | 0.025114 | 0.668378 | 306 | 8067 |
| walking_rpy | D²MA-DA LM-only s18 m2 | 906 | 0.256534 | 0.143501 | 0.346450 | 0.017301 | 0.571184 | 261 | 7180 |
| walking_halfsphere | D²MA-B r5 current ref | 1064 | 0.213132 | 0.168231 | 0.777055 | 0.012444 | 0.451372 | 227 | 5582 |
| walking_halfsphere | D²MA-DA LM-only s18 m2 | 1064 | 0.333637 | 0.238661 | 0.644955 | 0.012844 | 0.464440 | 242 | 5788 |
| walking_halfsphere | D²MA-DA LM-only s18 m3 | 1064 | 0.386641 | 0.263878 | 0.590959 | 0.013968 | 0.468541 | 258 | 6221 |

关键计数：

| sequence | method | CKF skipped new | CKF support-promoted new | LM rejected boundary pairs | LM support-promoted pairs |
|---|---|---:|---:|---:|---:|
| walking_rpy | D²MA-B r5 current ref | 9470 | 0 | 298 | 0 |
| walking_rpy | D²MA-DA full CKF+LM s18 m2 | 7022 | 2067 | 223 | 39 |
| walking_rpy | D²MA-DA full CKF+LM s18 m3 | 7856 | 1372 | 232 | 15 |
| walking_rpy | D²MA-DA LM-only s18 m2 | 9444 | 0 | 310 | 74 |
| walking_halfsphere | D²MA-B r5 current ref | 11183 | 0 | 397 | 0 |
| walking_halfsphere | D²MA-DA LM-only s18 m2 | 11141 | 0 | 317 | 59 |
| walking_halfsphere | D²MA-DA LM-only s18 m3 | 11237 | 0 | 356 | 47 |

解读：

- full CKF+LM DA 在 `walking_rpy` 上没有改善，原因很可能是 CKF 单帧 RGB-D 边界新点被 support 条件大量放回：`m2` 放回 `2067` 个，`m3` 仍放回 `1372` 个，导致全局尺度/SE3 退化。
- LM-only DA 在 `walking_rpy` 上优于当前 hard D²MA-B reference：ATE-SE3 `0.307193 -> 0.256534`，ATE-Sim3 `0.151482 -> 0.143501`，scale `0.275739 -> 0.346450`。这说明跨关键帧几何支持的少量边界 pair 可能有助于恢复 rpy 残余失败。
- 但同样的 LM-only DA 在 `walking_halfsphere` 上明显退化：ATE-SE3 `0.213132 -> 0.333637`，scale `0.777055 -> 0.644955`；提高 `min_support=3` 仍不能救回，说明 fixed-threshold DA 不是跨序列安全主方法。
- 因此，D²MA-DA 的最小原型给出了一个重要边界：`CreateNewKeyFrame` 层必须保持 hard boundary veto；`LocalMapping` 层可以作为 probationary/delayed admission 的候选位置，但需要 sequence/failure-mode-aware guard 或 admission budget。

阶段性结论：

- D²MA-B r5 仍是更稳的主方法候选。
- D²MA-DA-lite 不是当前可直接替代 D²MA-B 的主方法，但它产生了有价值的新方向：support-aware admission 应该只发生在几何更强的 LocalMapping 层，并且需要额外的触发条件。
- 这组结果值得回传 5.5 Pro：它既给出 `walking_rpy` 的正向修复，也给出 `walking_halfsphere` 的负向边界，可请 5.5 Pro 判断是否将 D²MA-DA 定位为 “adaptive/local recovery module” 而不是默认主方法。

下一步建议：

1. 先把本节回传 5.5 Pro，询问 D²MA-DA 是否应改为 conditional module。
2. 若继续本地实现，优先做 `adaptive trigger` 而不是继续扫固定阈值：例如仅当 hard D²MA-B 的局部 tracked static support / scale proxy / inlier margin 低于阈值时，才在 LocalMapping 层允许少量 support-confirmed boundary pairs。
3. 不建议再把 `CreateNewKeyFrame` 边界新点放回主静态地图；当前证据显示单帧 RGB-D boundary admission 是主要污染风险。

### 10.21 可复现性补强 repeat matrix

动机：

- 5.5 Pro 九次回答总体认可 D²MA-B r5 主线，但它的回答偏战略判断，未充分处理当前最危险的问题：ORB-SLAM3 全序列结果存在分支漂移，单次 ATE 不能作为论文主结论。
- 当前二进制下，早期单次结果与重跑结果存在差异，例如 `walking_halfsphere` D²MA-B r5 早期单次 ATE-SE3 `0.156093`，当前 repeat 均值约 `0.223301`；因此必须把主表从“单次最优”切换到 repeat mean/std。
- 本节固定当前二进制、当前数据 association、当前 D²MA-B r5 配置，仅做重复运行，不引入新机制。

实验设置：

- 二进制：`/home/lj/dynamic_SLAM/stslam_backend/Examples/RGB-D/rgbd_tum`
- profile：`hybrid_sequential_semantic_only`
- D²MA-B r5：
  - `STSLAM_DYNAMIC_MAP_ADMISSION_VETO=1`
  - `STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_VETO=1`
  - `STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_RADIUS_PX=5`
  - `STSLAM_DYNAMIC_MAP_ADMISSION_DELAYED_BOUNDARY=0`
- same-count non-boundary control：
  - D²MA-min direct gates on
  - boundary hard veto off
  - `STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_SAME_COUNT_CONTROL=1`
- 所有 run 均开启 `STSLAM_OBSERVABILITY_LOG=1`。

结果文件：

- run root：`/home/lj/dynamic-slam-public/runs/repro_strengthening_20260513_221100`
- raw repeat 表：`/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/repro_strengthening_raw_20260513.csv`
- summary 表：`/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/repro_strengthening_summary_20260513.csv`

repeat summary：

| case | n | matched mean±std | ATE-SE3 mean±std | ATE-Sim3 mean±std | scale mean±std | RPEt mean±std | RPER mean±std |
|---|---:|---:|---:|---:|---:|---:|---:|
| wrpy_d2ma_b_r5 | 3 | 878.0±48.5 | 0.220639±0.040558 | 0.137750±0.005402 | 0.415301±0.082101 | 0.023727±0.008690 | 0.631580±0.086186 |
| wrpy_samecount_nonboundary_r5 | 2 | 864.0±59.4 | 0.448641±0.086496 | 0.147604±0.006255 | 0.194685±0.036772 | 0.032385±0.019759 | 0.802800±0.316604 |
| walking_halfsphere D²MA-B r5 | 3 | 1064.0±0.0 | 0.223301±0.007585 | 0.177781±0.007237 | 0.770038±0.005149 | 0.012491±0.000083 | 0.450718±0.002745 |
| walking_static D²MA-B r5 | 3 | 740.0±0.0 | 0.065779±0.000894 | 0.021985±0.000280 | 0.172437±0.006261 | 0.006611±0.000128 | 0.218065±0.001441 |
| walking_xyz D²MA-B r5 | 3 | 857.0±0.0 | 0.164883±0.010636 | 0.163139±0.008294 | 0.926069±0.055598 | 0.010850±0.000112 | 0.435636±0.002108 |

关键观察：

- D²MA-B r5 在 `walking_halfsphere / walking_static / walking_xyz` 上 repeat 稳定性较好，matched poses 固定，std 较小。
- `walking_rpy` 是主要不稳定来源：3 次中有 1 次 matched poses 从 `906` 掉到 `822`，但 ATE-SE3 反而更低。这说明只看 ATE 会被短轨迹/丢尾轨迹误导，必须同时报告 matched poses / coverage / final KFs / final MPs。
- same-count non-boundary control 的负对照结论仍成立：即使存在 matched 漂移，SE3 ATE 均值 `0.448641` 明显差于 D²MA-B r5 的 `0.220639`，scale 也更差。
- `walking_halfsphere` 的 repeat 均值 `0.223301` 明显保守于早期单次 `0.156093`，因此论文主表不能使用早期单次最优，应使用当前 repeat mean/std。
- `walking_static` 的 Sim3 scale 很低但 SE3/RPE 稳定，可能与该序列相机运动幅度/尺度可观测性有关；不应把 low-dynamic sanity 的 Sim3 scale 作为主要论据。

阶段性结论：

- D²MA-B r5 作为主方法的方向仍成立，但论文写法必须从“单次提升”升级为“repeat mean/std + coverage-aware evaluation”。
- `walking_rpy` 应被明确标注为高敏感序列；任何声称 rpy 最佳的机制都必须重复运行并报告 matched coverage。
- 后续与 5.5 Pro 沟通时，必须要求它按审稿人标准分析：不能只接受 ATE，必须同时检查 coverage、final map size、repeat variance、负对照稳定性和是否混用旧/新二进制结果。

### 10.22 异常大误差原因追踪：漏设 side-channel-only 导致前端/实例路径误开

问题：

- 在尝试补做 canonical map-admission-only repeat 时，出现远大于普通 ORB-SLAM3 分支漂移的退化：
  - `walking_halfsphere D²MA-B r5` 从旧结果 `ATE-SE3=0.156093 / scale=0.823258` 退化到 `0.335829 / 0.642284`。
  - `walking_rpy D²MA-B r5` 从旧结果 `matched=906, ATE-SE3=0.269999` 变成 `matched=822, ATE-SE3=0.319702`。
- 这类差异不应解释为随机波动，必须先作为实验口径事故排查。

排查结论：

- 旧强结果的 manifest 中固定存在：
  - `STSLAM_PANOPTIC_SIDE_CHANNEL_ONLY=1`
  - `ORB_SLAM3_MASK_MODE=off`
  - `STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=0`
  - `STSLAM_DYNAMIC_MAP_ADMISSION_VETO=0`
  - 仅开启 `CREATE_NEW_KEYFRAME + LOCAL_MAPPING_CREATE_NEW_MAPPOINTS` 两个地图准入 gate。
- 异常补跑漏掉了 `STSLAM_PANOPTIC_SIDE_CHANNEL_ONLY=1`。
- 代码确认该开关不是日志开关：当它为 `0/default` 时，系统会额外执行 `ExtractInstanceRegionORB`、panoptic mask refinement、instance processing、panoptic pose optimization、RGB-D dynamic split 等前端/实例路径；当它为 `1` 时，mask 才只作为 D²MA map-admission side-channel。
- 因此异常补跑实际不是 “D²MA map-admission-only”，而是把一套 panoptic/instance 前端机制重新放进了系统。

cause-probe 结果：

- 汇总表：`/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/cause_probe_sidechannel_summary_20260513.csv`
- run root：`/home/lj/dynamic-slam-public/runs/cause_probe_sidechannel_20260513_225300`

| sequence | method | role | matched | ATE-SE3 | ATE-Sim3 | scale | trajectory hash |
|---|---|---|---:|---:|---:|---:|---|
| walking_rpy | D²MA-B r5 | old reference | 906 | 0.269999 | 0.120722 | 0.361678 | `d7db27b3b12f0c13` |
| walking_rpy | D²MA-B r5 | missing side-channel-only | 822 | 0.319702 | 0.161377 | 0.208002 | `9c818f2598d111a3` |
| walking_rpy | D²MA-B r5 | side-channel-only restored | 906 | 0.269999 | 0.120722 | 0.361678 | `d7db27b3b12f0c13` |
| walking_halfsphere | D²MA-B r5 | old reference | 1064 | 0.156093 | 0.118844 | 0.823258 | `aef3dfbd304bed75` |
| walking_halfsphere | D²MA-B r5 | missing side-channel-only | 1064 | 0.335829 | 0.239226 | 0.642284 | `6c2bf2781a887ed2` |
| walking_halfsphere | D²MA-B r5 | side-channel-only restored | 1064 | 0.156093 | 0.118844 | 0.823258 | `aef3dfbd304bed75` |

补充观察：

- `walking_halfsphere D²MA-B r5` 在当前二进制、当前数据、当前评估脚本下，恢复 `STSLAM_PANOPTIC_SIDE_CHANNEL_ONLY=1` 后与旧结果 bit-level 一致，说明数据集和评估脚本不是原因。
- `walking_rpy D²MA-B r5` 同样恢复到旧结果，说明论文主方法 D²MA-B r5 的旧强结果不是偶然单次。
- `walking_rpy D²MA-min` 在恢复 side-channel-only 后从异常 `0.643348` 改善到 `0.474824`，但未完全回到旧 `0.367386`；这说明 D²MA-min 在 `walking_rpy` 上仍存在分支/二进制演化敏感性，不能作为强主结论的唯一支撑。

实验规训：

- 之前 `canonical_maponly_repro_20260513_223534` 中未设置 `STSLAM_PANOPTIC_SIDE_CHANNEL_ONLY=1` 的结果一律标记为 invalid diagnostic，不进入论文主表。
- 后续所有 D²MA map-admission-only 实验必须显式固定：
  - `STSLAM_PANOPTIC_SIDE_CHANNEL_ONLY=1`
  - `ORB_SLAM3_MASK_MODE=off`
  - `STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=0`
  - `STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURE_STAGES=none`
  - `STSLAM_DYNAMIC_DEPTH_INVALIDATION=0`
  - `STSLAM_DYNAMIC_MAP_ADMISSION_VETO=0`
  - `STSLAM_DYNAMIC_MAP_ADMISSION_VETO_STEREO_INITIALIZATION=0`
  - `STSLAM_DYNAMIC_MAP_ADMISSION_VETO_NEED_NEW_KEYFRAME=0`
  - 只通过子开关启用目标 map-admission gate。
- 后续回传 5.5 Pro 时，必须把本节作为“实验口径事故已定位并修正”的证据，要求它不要基于 missing-side-channel-only 的异常矩阵做方法判断。

### 10.23 Side-channel isolation protocol 固化与六条 canonical full-sequence

本轮目标：

- 不再依赖手工记忆环境变量，而是把 D²MA map-admission-only 的实验口径固化为脚本和 validator。
- 按 5.5 Pro 建议重跑 6 条 canonical full-sequence，作为后续论文主表 / 消融表 / 负对照表的干净证据。
- 一整轮完成后同步公开仓库，让后续 5.5 Pro 可以直接阅读整个代码与结果摘要。

新增本地协议资产：

- 运行脚本：`/home/lj/dynamic-slam-public/scripts/run_d2ma_sidechannel_isolated.sh`
- 协议校验器：`/home/lj/dynamic-slam-public/tools/validate_d2ma_sidechannel_protocol.py`
- 结果汇总器：`/home/lj/dynamic-slam-public/tools/summarize_backend_runs.py`
- 本轮 run root：`/home/lj/dynamic-slam-public/runs/canonical_sidechannel_six_20260513_233737`
- 本轮 summary：`/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/canonical_sidechannel_six_summary_20260513.csv`

协议固定项：

- `ORB_SLAM3_MASK_MODE=off`
- `STSLAM_PANOPTIC_SIDE_CHANNEL_ONLY=1`
- `STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES=0`
- `STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURE_STAGES=none`
- `STSLAM_DYNAMIC_DEPTH_INVALIDATION=0`
- `STSLAM_RGBD_DYNAMIC_FRONTEND_SPLIT=0`
- `STSLAM_DYNAMIC_MAP_ADMISSION_VETO=0`
- `STSLAM_DYNAMIC_MAP_ADMISSION_VETO_STEREO_INITIALIZATION=0`
- `STSLAM_DYNAMIC_MAP_ADMISSION_VETO_NEED_NEW_KEYFRAME=0`
- 只允许通过 CKF / LocalMapping 子 gate 与 boundary / same-count 子 gate 启用目标机制。

六条 canonical 结果：

| case | method | matched | coverage | ATE-SE3 | ATE-Sim3 | Sim3 scale | protocol |
|---|---|---:|---:|---:|---:|---:|---:|
| `wxyz_d2ma_b_r5` | `d2ma_b_r5` | 857 | 0.2972 | 0.017884 | 0.016043 | 0.974357 | pass |
| `wrpy_d2ma_b_r5` | `d2ma_b_r5` | 906 | 0.2959 | 0.269999 | 0.120722 | 0.361678 | pass |
| `whalfsphere_d2ma_b_r5` | `d2ma_b_r5` | 1064 | 0.2970 | 0.156093 | 0.118844 | 0.823258 | pass |
| `wrpy_d2ma_min` | `d2ma_min` | 906 | 0.2959 | 0.474824 | 0.156668 | 0.172691 | pass |
| `wrpy_samecount_nonboundary_r5` | `samecount_nonboundary_r5` | 906 | 0.2959 | 0.604425 | 0.156306 | 0.138884 | pass |
| `whalfsphere_raw` | `raw` | 1064 | 0.2970 | 0.506439 | 0.290979 | 0.484404 | pass |

关键结论：

- 六条结果全部通过 `d2ma_protocol_validation.json`，本轮不存在 missing-side-channel-only 口径事故。
- `walking_xyz` 在严格 side-channel-only D²MA-B r5 下达到 `ATE-SE3=0.017884`，说明主方法在该序列上可以接近早期强前端结果。
- `walking_halfsphere` 上 D²MA-B r5 明显优于 raw：`0.156093` vs `0.506439`，说明方法不是只在 xyz 上成立。
- `walking_rpy` 仍是高难序列，但 D²MA-B r5 明显优于 D²MA-min 和 same-count non-boundary control：`0.269999` vs `0.474824` vs `0.604425`。
- same-count non-boundary control 与 D²MA-min 均差于 D²MA-B r5，支持“收益来自 dynamic-depth boundary/support-aware admission，而不是 generic sparsification 或少建同样数量的点”。
- 目前主方法最稳妥的论文表述应收敛为：在 side-channel isolation protocol 下，D²MA-B r5 通过 CKF 与 LocalMapping 两个 admission stage 抑制 direct dynamic-depth 与 near-boundary risk observation 固化到静态地图中。

待继续确认：

- 当前六条是 clean single-run canonical，还需要在论文最终主表前补 repeat mean/std，尤其是 `walking_rpy`。
- 需要进一步统计 CKF / LocalMapping 中 boundary veto 的 frame-level 分布、KF/MP admission 改变量，以及这些量和 ATE/scale 的关系。
- 需要请 5.5 Pro 按审稿人标准检查：当前证据是否足以支持 “targeted boundary/support-aware map admission” 的创新性叙事，以及下一步最小补强实验应优先做 repeat、frame-level causality 还是更多序列。

### 10.24 Canonical repeat matrix：主方法与负对照 repeat mean/std

触发原因：

- 5.5 Pro 十二次回答明确指出：canonical six 仍是 single-run，最容易被审稿人攻击为 cherry-pick。
- 因此先补最小 repeat 证据，而不是继续扩展新模块。

本轮 run root：

- `/home/lj/dynamic-slam-public/runs/canonical_repeat_matrix_20260514_002248`
- raw 表：`/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/canonical_repeat_matrix_raw_20260514.csv`
- summary 表：`/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/canonical_repeat_matrix_summary_20260514.csv`
- 同步到总表：`experiments_0512_0517.csv`，新增 `canonical_repeat_matrix_20260514_*` 共 12 行。

实验矩阵：

- `wxyz_d2ma_b_r5`：3 repeats
- `wrpy_d2ma_b_r5`：3 repeats
- `whalfsphere_d2ma_b_r5`：3 repeats
- `wrpy_samecount_nonboundary_r5`：3 repeats

所有 repeat 均使用 `scripts/run_d2ma_sidechannel_isolated.sh`，且 `d2ma_protocol_validation.json` 全部 pass。

repeat summary：

| case | n | protocol | matched mean±std | ATE-SE3 mean±std | ATE-Sim3 mean±std | Sim3 scale mean±std |
|---|---:|---:|---:|---:|---:|---:|
| `wxyz_d2ma_b_r5` | 3 | pass | 857.0±0.0 | 0.017884±0.000000 | 0.016043±0.000000 | 0.974357±0.000000 |
| `wrpy_d2ma_b_r5` | 3 | pass | 906.0±0.0 | 0.269999±0.000000 | 0.120722±0.000000 | 0.361678±0.000000 |
| `whalfsphere_d2ma_b_r5` | 3 | pass | 1064.0±0.0 | 0.156093±0.000000 | 0.118844±0.000000 | 0.823258±0.000000 |
| `wrpy_samecount_nonboundary_r5` | 3 | pass | 906.0±0.0 | 0.604425±0.000000 | 0.156306±0.000000 | 0.138884±0.000000 |

关键结论：

- 在 side-channel isolation protocol 下，D²MA-B r5 主方法在 `wxyz / wrpy / whalfsphere` 上均为 bit-level repeat stable。
- `wrpy_samecount_nonboundary_r5` 负对照同样 bit-level stable，且显著差于 `wrpy_d2ma_b_r5`：`ATE-SE3=0.604425` vs `0.269999`，`Sim3 scale=0.138884` vs `0.361678`。
- 因此当前最合理的判断是：之前所谓“不稳定/大误差”主要来自实验协议事故，而不是 ORB-SLAM3 后端天然随机波动。
- `walking_rpy` 仍是残余困难序列，但它不是 repeat 随机失败；它是可复现的系统性 residual failure，应进入 frame-level causal logging 分析。
- 论文表述应避免说 D²MA-B “完全解决 wrpy”，而应写成：D²MA-B 稳定压制一个主要 failure channel，但 wrpy 仍暴露旋转主导/边界支撑/尺度路径一致性的剩余问题。

下一步优先级：

1. 先做 `wrpy` frame-level causal logging：比较 raw / D²MA-min / D²MA-B / same-count 的 boundary-risk accepted/vetoed、new MPs、local inliers、path ratio。
2. 再做 association-clean sensitivity：`wrpy` 与 `halfsphere` 的 strict subset 评估。
3. 之后补 low-dynamic/static negative control，确认 D²MA-B 不因 false-positive mask 伤害静态/低动态序列。

### 10.25 `walking_rpy` 零改代码 frame-level causal probe 初版

目的：

- 在正式改后端日志前，先解析现有 `stdout.log` 中的 map-admission event 与 `observability_frame_stats.csv`。
- 检查现有日志是否已经足够支撑 “D²MA-B 的收益不是 generic sparsification，而是 boundary-risk targeted admission control”。

新增解析工具：

- `/home/lj/dynamic-slam-public/tools/parse_map_admission_events.py`

输出文件：

- summary：`/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/wrpy_frame_causal_probe_summary_20260514.csv`
- per-frame：`/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/wrpy_frame_causal_probe_per_frame_20260514.csv`

使用 run：

- raw：`/home/lj/dynamic-slam-public/runs/external_wrpy_d2ma_min_20260513_170248/raw_baseline`
- D²MA-min：`/home/lj/dynamic-slam-public/runs/canonical_sidechannel_six_20260513_233737/wrpy_d2ma_min`
- D²MA-B r5：`/home/lj/dynamic-slam-public/runs/canonical_repeat_matrix_20260514_002248/wrpy_d2ma_b_r5/r1`
- same-count non-boundary r5：`/home/lj/dynamic-slam-public/runs/canonical_repeat_matrix_20260514_002248/wrpy_samecount_nonboundary_r5/r1`

summary：

| case | ATE-SE3 | scale | final KFs | final MPs | CKF direct veto | CKF boundary skip/control | LM boundary/control |
|---|---:|---:|---:|---:|---:|---:|---:|
| `wrpy_raw` | 0.997057 | 0.084939 | 575 | 13872 | 0 | 0/0 | 0/0 |
| `wrpy_d2ma_min` | 0.474824 | 0.172691 | 430 | 7347 | 102698 | 0/0 | 0/0 |
| `wrpy_d2ma_b_r5` | 0.269999 | 0.361678 | 370 | 6006 | 103407 | 8179/0 | 587/0 |
| `wrpy_samecount_nonboundary_r5` | 0.604425 | 0.138884 | 427 | 7389 | 104739 | 0/7988 | 0/209 |

初步结论：

- raw baseline 存在明显 map inflation：`575` KFs / `13872` MPs，ATE-SE3 `0.997057`，Sim3 scale `0.084939`。
- D²MA-min 只做 direct dynamic-depth admission gate，已经大幅压低 map size：`430` KFs / `7347` MPs，ATE-SE3 `0.474824`。
- D²MA-B r5 在 direct gate 基础上额外跳过 CKF boundary-risk new candidates `8179` 个、LM boundary pairs `587` 对，最终 `370` KFs / `6006` MPs，ATE-SE3 `0.269999`，scale `0.361678`。
- same-count non-boundary control 跳过了相近数量的非边界候选：CKF `7988` 个、LM `209` 对，但结果退化到 `427` KFs / `7389` MPs，ATE-SE3 `0.604425`，scale `0.138884`。
- 因此现有日志已经给出一个强中间证据：收益不是“删掉差不多数量的点”，而是边界/支撑位置的 targeted veto 更有效。

仍不足：

- 当前 summary 是事件总量与 final map size，还没有形成真正的 failure interval 图。
- 需要基于 per-frame CSV 继续找出 `estimated_accum_path_m` 跳变、local inliers 下降、boundary veto 高峰之间的时间对应关系。
- 如果要进入论文机制图，仍建议补充或派生：segment-level ATE / path-ratio、per-frame cumulative CKF/LM veto 曲线、new MPs 增量曲线。

候选重点区间：

- 早期 `frame 48-65`：LM boundary pairs 高峰，D²MA-B 在 `frame 57` 跳过 LM boundary pairs `18`，是早期地图污染入口候选。
- 中段 `frame 239-290`：D²MA-B CKF boundary skip 与 LM boundary pairs 均较高，same-count control 在此段也大量跳过非边界点但效果更差。
- 中后段 `frame 574-593`：D²MA-B CKF boundary skip 连续高峰，`frame 574/581/585/593` 均超过 `61`。
- 后段 `frame 812-825`：D²MA-B 与 same-count control 都有高跳过量，但 same-count 的累计路径和 map size 更大，可能是解释 residual scale/path drift 的重点区间。

### 10.26 `walking_rpy` failure interval 图与区间统计

目的：

- 将 `observability_frame_stats.csv` 与 D²MA event logs 对齐，形成可以进入论文机制分析的 failure-interval 图。
- 对比 raw / D²MA-min / D²MA-B r5 / same-count non-boundary r5 在路径长度、地图规模、inlier、boundary/control 累积事件上的差异。

新增工具：

- `/home/lj/dynamic-slam-public/tools/plot_wrpy_failure_intervals.py`

输出文件：

- 图：`/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/wrpy_failure_intervals_20260514.png`
- 时序表：`/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/wrpy_failure_interval_timeseries_20260514.csv`
- 区间统计：`/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/wrpy_failure_interval_summary_20260514.csv`

图的四个子图：

- estimated accumulated path
- MapPoints
- inliers after pose
- cumulative CKF / LM boundary-control events

关键读法：

- raw 的 estimated path 与 MapPoints 全程显著高于 D²MA 系列，尤其后半段 raw MapPoints 接近 `14000`，符合 map inflation / path-length inflation 的失败模式。
- D²MA-min 显著压低 direct dynamic-depth 导致的地图膨胀，但仍保留较多 boundary-risk contamination。
- D²MA-B r5 在 D²MA-min 基础上进一步压低 MapPoints，最终约 `6006`，并使路径曲线更接近 D²MA-min 下方。
- same-count non-boundary r5 的 cumulative CKF/control 曲线与 D²MA-B r5 的 boundary veto 曲线接近，但 MapPoints 与 path 明显更差，说明“删同等数量非边界点”不能替代 targeted boundary-risk veto。

区间统计摘录：

| interval | case | path delta | MP delta | keyframe delta | mean inliers | CKF boundary/control | LM boundary/control |
|---|---|---:|---:|---:|---:|---:|---:|
| `48-65` | raw | 0.205457 | 303 | 12 | 322.22 | 0/0 | 0/0 |
| `48-65` | D²MA-B r5 | 0.279284 | 182 | 10 | 213.56 | 239/0 | 86/0 |
| `48-65` | same-count | 0.288249 | 111 | 8 | 217.00 | 0/172 | 0/4 |
| `239-290` | raw | 1.265742 | 1224 | 42 | 182.40 | 0/0 | 0/0 |
| `239-290` | D²MA-B r5 | 0.930658 | 650 | 35 | 159.23 | 763/0 | 91/0 |
| `239-290` | same-count | 1.075254 | 591 | 37 | 163.10 | 0/927 | 0/20 |
| `574-593` | raw | 0.731099 | 547 | 18 | 162.75 | 0/0 | 0/0 |
| `574-593` | D²MA-B r5 | 0.646738 | 138 | 13 | 166.55 | 621/0 | 25/0 |
| `574-593` | same-count | 1.077322 | 287 | 15 | 147.00 | 0/552 | 0/8 |
| `812-825` | raw | 0.401657 | 853 | 11 | 151.71 | 0/0 | 0/0 |
| `812-825` | D²MA-B r5 | 0.400647 | 484 | -1 | 140.71 | 372/0 | 17/0 |
| `812-825` | same-count | 0.411017 | 580 | 7 | 141.00 | 0/324 | 0/7 |

阶段结论：

- D²MA-B r5 的优势不仅体现在最终 ATE，也能在 failure interval 中看到 map growth 抑制。
- same-count control 在 CKF 跳过量相近的条件下，不能复现 D²MA-B 的路径/地图优势，进一步支持 boundary-risk targeting。
- 但 inlier 曲线不是单调改善，说明 D²MA-B 的收益主要来自抑制错误地图固化，而不是简单提升每帧 tracking inliers。

### 10.27 Association-clean sensitivity

目的：

- 检查 `walking_rpy` 与 `walking_halfsphere` 的 D²MA-B 相对收益是否由 RGB-depth / GT timestamp association outlier 造成。
- 对 current association、strict `rgb_depth_time_diff <= 0.03 && gt_time_diff <= 0.03`、loose `<= 0.05` 三种口径分别评估。

新增工具：

- `/home/lj/dynamic-slam-public/tools/evaluate_association_clean_sensitivity.py`

输出文件：

- `/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/association_clean_sensitivity_20260514.csv`

结果：

| case | variant | matched | ATE-SE3 | ATE-Sim3 | scale | filtered/full |
|---|---|---:|---:|---:|---:|---:|
| `wrpy_raw` | current | 906 | 0.997057 | 0.157951 | 0.084939 | full |
| `wrpy_raw` | strict_003 | 884 | 0.994092 | 0.158237 | 0.085159 | 884/909 |
| `wrpy_d2ma_b_r5` | current | 906 | 0.269999 | 0.120722 | 0.361678 | full |
| `wrpy_d2ma_b_r5` | strict_003 | 884 | 0.269325 | 0.120832 | 0.362819 | 884/909 |
| `halfsphere_raw` | current | 1064 | 0.506439 | 0.290979 | 0.484404 | full |
| `halfsphere_raw` | strict_003 | 1038 | 0.507191 | 0.291515 | 0.484412 | 1038/1067 |
| `halfsphere_d2ma_b_r5` | current | 1064 | 0.156093 | 0.118844 | 0.823258 | full |
| `halfsphere_d2ma_b_r5` | strict_003 | 1038 | 0.155420 | 0.118698 | 0.824752 | 1038/1067 |

补充数据质量：

- `walking_rpy` association rows `909`，strict clean `884`；`rgb_depth_bad=23`，`gt_bad=2`。
- `walking_halfsphere` association rows `1067`，strict clean `1038`；`rgb_depth_bad=26`，`gt_bad=3`。

结论：

- strict association-clean subset 下，raw 与 D²MA-B 的相对差距没有消失。
- `walking_rpy`：raw `0.994092` vs D²MA-B `0.269325`。
- `walking_halfsphere`：raw `0.507191` vs D²MA-B `0.155420`。
- 因此当前主结论不是 timestamp association artifact。

下一步：

- 可以把 association-clean sensitivity 放入 appendix 或 robustness table。
- 下一项建议补 low-dynamic/static negative control，或把当前 repeat + failure interval + association-clean 统一回传给 5.5 Pro 做下一轮审稿式判断。

### 10.28 `walking_static` low-dynamic/static negative control

目的：

- 回应审稿风险：D²MA-B r5 是否会在低动态/近静态序列上误伤静态结构。
- 检查该机制是否只是 generic map sparsification，或者是否仍表现为有协议边界的 map-admission control。
- 全部使用 side-channel isolation protocol，避免混入 RGB masking、feature filtering、depth invalidation、pose-level dynamic weighting。

协议：

- dataset：`external_wstatic_rawrgb_rawdepth_mask`
- profile：`hybrid_sequential_semantic_only`
- methods：`raw`、`d2ma_b_r5`
- repeat：各 3 次
- 关键固定项：`ORB_SLAM3_MASK_MODE=off`，`STSLAM_PANOPTIC_SIDE_CHANNEL_ONLY=1`
- validator：6/6 `protocol_valid=1`

新增工具：

- `/home/lj/dynamic-slam-public/tools/summarize_sidechannel_repeats.py`

输出文件：

- `/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/static_negative_control_raw.csv`
- `/home/lj/d-drive/Obsidian Vault/小论文/小论文2 动态改进Orb SLAM3/static_negative_control_summary.csv`
- repo mirror：`/home/lj/dynamic-slam-public/results_summaries/static_negative_control_20260514/`

结果：

| case | n | valid | matched | ATE-SE3 mean±std | ATE-Sim3 mean±std | scale mean±std | RPEt mean±std | RPER mean±std | final KFs | final MPs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `raw` | 3 | 1 | 740 | 0.073350±0.000000 | 0.014798±0.000000 | 0.224276±0.000000 | 0.017895±0.000000 | 0.340557±0.000000 | 270 | 4430 |
| `d2ma_b_r5` | 3 | 1 | 740 | 0.010972±0.000000 | 0.008703±0.000000 | 0.782028±0.000000 | 0.010462±0.000000 | 0.219777±0.000000 | 141 | 1791 |

事件统计：

| case | CKF direct veto | CKF boundary skip | LM instance skip | LM boundary skip |
|---|---:|---:|---:|---:|
| `raw` | 0 | 0 | 0 | 0 |
| `d2ma_b_r5` | 86402 | 3916 | 675 | 117 |

阶段结论：

- `walking_static` 在 canonical protocol 下 repeat 完全稳定，raw 与 D²MA-B r5 都是 bit-level stable。
- D²MA-B r5 没有造成 low-dynamic/static 序列退化；相反，它显著减少 final MapPoints，并降低 SE3 / Sim3 / RPE。
- 该结果可以作为 sanity check：D²MA-B r5 不是通过随机破坏 tracking 得到收益，也没有在低动态场景中出现明显静态误伤。
- 但不应把该结果过度写成“静态场景也一定受益”。`walking_static` 仍包含被 mask 标记的人体/边界 foreground structure，D²MA-B r5 的收益更可能来自抑制这些 foreground / near-boundary depth observation 固化为静态地图，而不是证明所有静态场景都适用。

论文使用建议：

- 放入 robustness / negative control 表，而不是主性能表。
- 表注中说明：`walking_static` 是 low-dynamic foreground sanity check，不是 pure static no-object sequence。
- 与 same-count non-boundary control、association-clean sensitivity、failure interval probe 联合使用，形成“不是 generic sparsification、不是 timestamp artifact、不是静态误伤”的补强证据链。

下一步：

- 将 `canonical_repeat_matrix_20260514`、`wrpy_failure_intervals_20260514`、`association_clean_sensitivity_20260514`、`static_negative_control_20260514` 与新增工具整理上传 GitHub。
- 回传 5.5 Pro 时重点要求其按审稿人标准检查：static negative control 是否应进主表、是否需要 pure static no-mask 序列、以及 `walking_static` 的 low Sim3 scale 在解释上是否仍需降权。
