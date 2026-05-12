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
