# basic_model_based_SLAM

本仓库的当前主研究方向为：

**利用基础模型提升动态环境下视觉 SLAM 的精度与稳健性。**

当前代码主线并不是简单的“检测器 + 分割器 + SLAM”拼接，而是围绕“**如何把基础模型输出安全接入视觉 SLAM**”构建的前端过滤框架，核心模块包括：

- 任务约束的开放词汇候选生成
- SAM 精细分割
- 面向 SLAM 的任务相关性评估
- 基础模型可靠性评估
- 跨帧动态记忆与多级安全门控
- RGB-D 一致性过滤
- ORB-SLAM3 端到端 ATE / RPE 评估

本 README 的目标是让任何新下载本仓库的人，都可以按照统一步骤稳定复现当前结果。

如果你想先快速把研究主线、创新点、实验结构和配置命名统一起来，建议优先阅读：

- [`docs/统一版研究主线与复现口径_20260322.md`](/home/lj/d-drive/CODEX/basic_model_based_SLAM/docs/统一版研究主线与复现口径_20260322.md)
- [`docs/方法章节公式版_20260322.md`](/home/lj/d-drive/CODEX/basic_model_based_SLAM/docs/方法章节公式版_20260322.md)
- [`docs/变量符号表与公式逻辑图_20260322.md`](/home/lj/d-drive/CODEX/basic_model_based_SLAM/docs/变量符号表与公式逻辑图_20260322.md)

## 1. 仓库结构

- `config/`：主配置文件与实验配置
- `datasets/`：TUM RGB-D 与 Bonn RGB-D Dynamic 数据集
- `experiments/`：实验输出目录，包含导出序列、轨迹、日志、汇总 CSV/JSON
- `docs/`：论文写作草稿、实验总结与阶段性记录
- `scripts/run_rgbd_slam_benchmark.py`：单次 RGB-D SLAM 基准主脚本
- `scripts/run_open_vocab_vs_closed_set_study.py`：开放词汇 vs 闭集检测器对比
- `scripts/run_speed_accuracy_tradeoff_study.py`：`fast / balanced / accurate` 速度-精度研究
- `scripts/generate_paper_figures.py`：论文插图导出
- `scripts/smoke_test_world_sam.py`：单图像过滤可视化测试
- `scripts/rflysim_slam_nav/world_sam_pipeline.py`：核心过滤流水线

## 2. 推荐运行环境

推荐在 **WSL + Conda + NVIDIA GPU** 环境下运行，所有 Python 侧流程统一使用 `openvoc` 环境。

### 2.1 进入环境

```bash
cd ~/d-drive/CODEX/basic_model_based_SLAM
source ~/anaconda3/etc/profile.d/conda.sh
conda activate openvoc
python -c "import sys; print(sys.executable)"
```

### 2.2 检查 GPU

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

若返回 `False`，优先检查：

- WSL 是否正确映射 NVIDIA 驱动
- `openvoc` 中的 PyTorch 是否带 CUDA
- 权重文件是否存在且可读

## 3. 依赖与外部程序

### 3.1 Python 侧核心依赖

建议至少保证以下模块可导入：

```bash
python -c "import cv2, numpy, torch, ultralytics; print('python deps ok')"
python -c "import docx; print('python-docx ok')"
```

### 3.2 ORB-SLAM3

端到端轨迹评估依赖外部 ORB-SLAM3 RGB-D 可执行文件。当前默认使用的路径为：

- ORB 可执行文件：`/home/lj/tools/slam_deps/ORB_SLAM3/Examples/RGB-D/rgbd_tum`
- 词典：`/home/lj/tools/slam_deps/ORB_SLAM3/Vocabulary/ORBvoc.txt`
- 相机配置：`/home/lj/tools/slam_deps/ORB_SLAM3/Examples/RGB-D/TUM3.yaml`

先检查它们是否存在：

```bash
ls -l /home/lj/tools/slam_deps/ORB_SLAM3/Examples/RGB-D/rgbd_tum
ls -l /home/lj/tools/slam_deps/ORB_SLAM3/Vocabulary/ORBvoc.txt
ls -l /home/lj/tools/slam_deps/ORB_SLAM3/Examples/RGB-D/TUM3.yaml
```

如果你的本地路径不同，请在后续命令中手动替换 `--orb-exec`、`--orb-vocab` 和 `--orb-config`。

## 4. 权重文件

推荐将以下权重放在仓库根目录下的 `weights/` 中：

- `weights/yolov8s-world.pt`
- `weights/yolov8n.pt`
- `weights/sam_vit_b_01ec64.pth`
- `weights/best.onnx`

快速检查：

```bash
ls -lh weights
```

## 5. 当前已同步的数据与结果

为了便于直接核查，本仓库当前已经同步了关键公开数据与主要实验结果。

### 5.1 TUM 动态序列

- `datasets/tum_rgbd/freiburg3_walking_xyz/`
- `datasets/tum_rgbd/freiburg3_walking_static/`

快速检查帧数：

```bash
find datasets/tum_rgbd/freiburg3_walking_xyz/rgbd_dataset_freiburg3_walking_xyz/rgb -maxdepth 1 -type f | wc -l
find datasets/tum_rgbd/freiburg3_walking_static/rgbd_dataset_freiburg3_walking_static/rgb -maxdepth 1 -type f | wc -l
```

预期结果：

- `walking_xyz` 约为 `859`
- `walking_static` 约为 `743`

### 5.2 现成结果

当前仓库已包含主实验输出目录，例如：

- `experiments/E6-*`：TUM 全量实验
- `experiments/E7-*`：Bonn 全量实验
- `experiments/E9_*`：开放词汇 vs 闭集对比
- `experiments/E10_*`：多人 crowd 对比
- `experiments/E11_*`：速度-精度权衡
- `experiments/paper_figures_orb_demo_20260321`：ORB 特征点可视化论文图

主汇总表：

- `experiments/experiment_master_table_20260321.csv`

直接查看主结果表：

```bash
sed -n '1,40p' experiments/experiment_master_table_20260321.csv
```

## 6. 稳定复现的推荐顺序

推荐按下面顺序运行，不要一开始就直接跑全套实验：

1. 环境检查
2. 单图像 smoke test
3. 单序列 probe 测试
4. TUM 全量基准
5. Bonn 对比实验
6. 论文图生成

这样最稳，也最容易定位问题。

补充说明：

- 当前仓库的默认主线配置就是 `balanced`，对应 [`config/world_sam_pipeline_foundation_panoptic_person_v2_local.json`](/home/lj/d-drive/CODEX/basic_model_based_SLAM/config/world_sam_pipeline_foundation_panoptic_person_v2_local.json)
- 如果某些脚本里没有显式传 `--config`，核心流水线也会优先回落到这份主线配置，而不是旧工程路径

## 7. 第一步：单图像过滤测试

先验证 `YOLO-World + SAM + 过滤流水线` 是否能正常输出。

```bash
cd ~/d-drive/CODEX/basic_model_based_SLAM
source ~/anaconda3/etc/profile.d/conda.sh
conda activate openvoc
python scripts/smoke_test_world_sam.py \
  --config config/world_sam_pipeline_foundation_panoptic_person_v2_local.json \
  --image /path/to/your/test_image.png \
  --output-dir debug/world_sam_smoke_test_local
```

重点检查输出：

- `filtered_rgb.jpg`
- `overlay.jpg`
- `mask.png`
- `stats.json`

如果这一步就失败，不建议继续跑端到端 SLAM。

## 8. 第二步：单序列 probe 测试

在跑全量实验前，建议先跑短序列 probe。

### 8.1 TUM walking_xyz 短序列测试

```bash
cd ~/d-drive/CODEX/basic_model_based_SLAM
source ~/anaconda3/etc/profile.d/conda.sh
conda activate openvoc
python scripts/run_rgbd_slam_benchmark.py \
  --sequence-root datasets/tum_rgbd/freiburg3_walking_xyz/rgbd_dataset_freiburg3_walking_xyz \
  --output-dir experiments/local_probe_walking_xyz \
  --filter-mode filtered \
  --config config/world_sam_pipeline_foundation_panoptic_person_v2_local.json \
  --max-frames 120 \
  --orb-exec /home/lj/tools/slam_deps/ORB_SLAM3/Examples/RGB-D/rgbd_tum \
  --orb-vocab /home/lj/tools/slam_deps/ORB_SLAM3/Vocabulary/ORBvoc.txt \
  --orb-config /home/lj/tools/slam_deps/ORB_SLAM3/Examples/RGB-D/TUM3.yaml
```

## 9. 第三步：复现 TUM 全量实验

这是最核心的复现部分。

### 9.1 Raw 基线

```bash
python scripts/run_rgbd_slam_benchmark.py \
  --sequence-root datasets/tum_rgbd/freiburg3_walking_xyz/rgbd_dataset_freiburg3_walking_xyz \
  --output-dir experiments/repro_raw_freiburg3_walking_xyz \
  --filter-mode raw \
  --raw-export-mode symlink \
  --orb-exec /home/lj/tools/slam_deps/ORB_SLAM3/Examples/RGB-D/rgbd_tum \
  --orb-vocab /home/lj/tools/slam_deps/ORB_SLAM3/Vocabulary/ORBvoc.txt \
  --orb-config /home/lj/tools/slam_deps/ORB_SLAM3/Examples/RGB-D/TUM3.yaml
```

### 9.2 主方法 `balanced` 默认配置

```bash
python scripts/run_rgbd_slam_benchmark.py \
  --sequence-root datasets/tum_rgbd/freiburg3_walking_xyz/rgbd_dataset_freiburg3_walking_xyz \
  --output-dir experiments/repro_person_v2_dynmem_walking_xyz \
  --filter-mode filtered \
  --config config/world_sam_pipeline_foundation_panoptic_person_v2_local.json \
  --orb-exec /home/lj/tools/slam_deps/ORB_SLAM3/Examples/RGB-D/rgbd_tum \
  --orb-vocab /home/lj/tools/slam_deps/ORB_SLAM3/Vocabulary/ORBvoc.txt \
  --orb-config /home/lj/tools/slam_deps/ORB_SLAM3/Examples/RGB-D/TUM3.yaml
```

### 9.3 无动态记忆版本

```bash
python scripts/run_rgbd_slam_benchmark.py \
  --sequence-root datasets/tum_rgbd/freiburg3_walking_static/rgbd_dataset_freiburg3_walking_static \
  --output-dir experiments/repro_person_v2_no_memory_walking_static \
  --filter-mode filtered \
  --config config/world_sam_pipeline_foundation_panoptic_person_v2_no_dynamic_memory.json \
  --orb-exec /home/lj/tools/slam_deps/ORB_SLAM3/Examples/RGB-D/rgbd_tum \
  --orb-vocab /home/lj/tools/slam_deps/ORB_SLAM3/Vocabulary/ORBvoc.txt \
  --orb-config /home/lj/tools/slam_deps/ORB_SLAM3/Examples/RGB-D/TUM3.yaml
```

### 9.4 强语义删除基线

```bash
python scripts/run_rgbd_slam_benchmark.py \
  --sequence-root datasets/tum_rgbd/freiburg3_walking_static/rgbd_dataset_freiburg3_walking_static \
  --output-dir experiments/repro_semantic_all_delete_walking_static \
  --filter-mode filtered \
  --config config/world_sam_pipeline_semantic_all_delete.json \
  --orb-exec /home/lj/tools/slam_deps/ORB_SLAM3/Examples/RGB-D/rgbd_tum \
  --orb-vocab /home/lj/tools/slam_deps/ORB_SLAM3/Vocabulary/ORBvoc.txt \
  --orb-config /home/lj/tools/slam_deps/ORB_SLAM3/Examples/RGB-D/TUM3.yaml
```

## 10. 第四步：复现开放词汇 vs 闭集对比

该实验用于回答“如果主要做人检测，为什么还要开放词汇”的问题。

### 10.1 person_tracking

```bash
python scripts/run_open_vocab_vs_closed_set_study.py \
  --sequence-root datasets/bonn_rgbd_dynamic/person_tracking/rgbd_bonn_person_tracking \
  --output-root experiments/repro_open_vocab_vs_closed_person_tracking \
  --reuse-existing
```

### 10.2 crowd

```bash
python scripts/run_open_vocab_vs_closed_set_study.py \
  --sequence-root datasets/bonn_rgbd_dynamic/crowd/rgbd_bonn_crowd \
  --output-root experiments/repro_open_vocab_vs_closed_crowd \
  --reuse-existing
```

输出重点：

- `open_vocab_vs_closed_set_summary.csv`
- `open_vocab_vs_closed_set_summary.json`

## 11. 第五步：复现速度-精度权衡

该实验主要用于解释 `fast / balanced / accurate` 三个配置标签。

```bash
python scripts/run_speed_accuracy_tradeoff_study.py \
  --sequence-root datasets/bonn_rgbd_dynamic/person_tracking/rgbd_bonn_person_tracking \
  --output-root experiments/repro_speed_accuracy_tradeoff_person_tracking \
  --include-raw \
  --reuse-existing
```

三个配置的含义如下：

- `fast`：偏向更低前端开销，使用 `world_sam_pipeline_foundation_panoptic_person_v2_fast_local.json`
- `balanced`：默认主线配置，使用 `world_sam_pipeline_foundation_panoptic_person_v2_local.json`
- `accurate`：更激进的高分辨率和更频繁检测设置，使用 `world_sam_pipeline_foundation_panoptic_person_v2_accurate_local.json`

注意：这些名字表示**预设设计目标**，不是事后根据结果起的名字。

## 12. 第六步：生成论文插图

当前论文图已经从“模糊图”改为更有说服力的 **ORB 特征点过滤前后对比**。

### 12.1 直接复用现有结果生成图

```bash
python scripts/generate_paper_figures.py \
  --raw-experiment experiments/E6-1_full_raw_freiburg3_walking_xyz_20260321 \
  --compare-experiment experiments/E6-10_full_foundation_panoptic_person_v2_no_dynamic_memory_freiburg3_walking_xyz_20260321 \
  --label Raw \
  --label PersonV2-NoMem \
  --config config/world_sam_pipeline_foundation_panoptic_person_v2_local.json \
  --output-dir experiments/paper_figures_repro_tum_xyz
```

### 12.2 查看已同步的论文图目录

```bash
ls -la experiments/paper_figures_orb_demo_20260321
```

其中重点文件包括：

- `orb_feature_comparison.png`
- `selected_frame_orb_original.png`
- `selected_frame_orb_filtered.png`
- `selected_frame_orb_mask_split.png`
- `trajectory_comparison.png`
- `orb_feature_summary.json`

## 13. 如何查看结果是否复现成功

每个实验目录下最重要的文件是 `benchmark_summary.json`。

快速查看：

```bash
python - <<'PY'
import json
path = 'experiments/E6-9_full_foundation_panoptic_person_v2_dynamic_memory_freiburg3_walking_xyz_20260321/benchmark_summary.json'
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
print(json.dumps(data['trajectory_metrics'], indent=2, ensure_ascii=False))
print('mean_runtime_ms =', data.get('mean_runtime_ms'))
PY
```

你也可以直接看总表：

```bash
sed -n '1,30p' experiments/experiment_master_table_20260321.csv
```

当前主结果中，TUM 两条序列的代表值为：

- `freiburg3_walking_xyz`：`raw ATE = 0.408588`，`person_v2_dynamic_memory ATE = 0.018325`
- `freiburg3_walking_static`：`raw ATE = 0.118310`，`person_v2_no_dynamic_memory ATE = 0.011088`

## 14. 稳定运行建议

为了让别人下载仓库后更稳定地复现，建议严格遵守以下做法：

- 所有 Python 命令都在仓库根目录执行
- 统一使用 `openvoc` 环境
- 优先使用 `*_local.json` 配置文件
- 先跑 `smoke_test_world_sam.py`，再跑端到端 SLAM
- 先跑 `--max-frames 120` 的短序列，再跑全量序列
- 挂载盘路径下不要使用会强依赖 Linux 时间戳属性的复制方式
- ORB-SLAM3 可执行文件路径必须手动确认，不要盲目沿用别人的绝对路径

## 15. 常见问题

### 15.1 为什么 `filter-mode=filtered` 必须带 `--config`

因为 `scripts/run_rgbd_slam_benchmark.py` 在 `filtered` 模式下会初始化 `WorldSamFilterPipeline`，没有配置文件就无法确定检测器、分割器、门控和时序参数。

### 15.2 为什么 `raw` 模式建议加 `--raw-export-mode symlink`

因为 Raw 基线本身不需要重写图像内容，使用软链接最省空间，也最稳定。

### 15.3 为什么我看到的结果和论文草稿略有差异

常见原因有：

- ORB-SLAM3 版本不同
- GPU / CUDA 环境不同
- 权重版本不同
- 本地绝对路径不同导致实际上用了另一份配置或另一份仓库
- 没有在仓库根目录运行命令

### 15.4 为什么开放词汇在 crowd 上不一定优于闭集

这是当前研究已经明确承认的边界：在多人拥挤场景里，闭集人体检测器当前往往有更强的多人检测召回；开放词汇的优势更多体现在统一框架、类别扩展性与速度-精度折中，而不是在所有场景绝对最优。

## 16. 推荐先做的三件事

如果你是第一次下载仓库，建议按这个最小闭环来：

1. 跑单图像 smoke test，确认基础模型和权重正常
2. 跑 `walking_xyz` 的 120 帧短序列，确认端到端流程正常
3. 查看 `experiment_master_table_20260321.csv` 与现有 `E6-* / E7-*` 结果，确认本仓库中的历史结果可读可复核

做到这三步后，再开始重跑完整论文实验，会稳定很多。

## 17. 批量运行所有序列与统一评估

如果你希望一次性把当前仓库中的 TUM 和 Bonn 核心序列全部重跑，并自动汇总 ATE / RPE / 运行时间，下面这组命令可以直接作为标准复现实验入口。

### 17.1 先设定统一变量

```bash
cd ~/d-drive/CODEX/basic_model_based_SLAM
source ~/anaconda3/etc/profile.d/conda.sh
conda activate openvoc

PYTHON_BIN=/home/lj/anaconda3/envs/openvoc/bin/python
ORB_EXEC=/home/lj/tools/slam_deps/ORB_SLAM3/Examples/RGB-D/rgbd_tum
ORB_VOCAB=/home/lj/tools/slam_deps/ORB_SLAM3/Vocabulary/ORBvoc.txt
ORB_CONFIG=/home/lj/tools/slam_deps/ORB_SLAM3/Examples/RGB-D/TUM3.yaml
FILTER_CONFIG=config/world_sam_pipeline_foundation_panoptic_person_v2_local.json

DATE_TAG=$(date +%Y%m%d_%H%M%S)
RUN_ROOT=experiments/full_benchmark_${DATE_TAG}
mkdir -p "$RUN_ROOT"
echo "$RUN_ROOT"
```

### 17.2 运行主方法 `filtered` 全序列测试

```bash
cd ~/d-drive/CODEX/basic_model_based_SLAM
source ~/anaconda3/etc/profile.d/conda.sh
conda activate openvoc

PYTHON_BIN=/home/lj/anaconda3/envs/openvoc/bin/python
ORB_EXEC=/home/lj/tools/slam_deps/ORB_SLAM3/Examples/RGB-D/rgbd_tum
ORB_VOCAB=/home/lj/tools/slam_deps/ORB_SLAM3/Vocabulary/ORBvoc.txt
ORB_CONFIG=/home/lj/tools/slam_deps/ORB_SLAM3/Examples/RGB-D/TUM3.yaml
FILTER_CONFIG=config/world_sam_pipeline_foundation_panoptic_person_v2_local.json
DATE_TAG=$(date +%Y%m%d_%H%M%S)
RUN_ROOT=experiments/full_benchmark_${DATE_TAG}
mkdir -p "$RUN_ROOT"

while read -r SEQ_NAME SEQ_ROOT; do
  [ -z "$SEQ_NAME" ] && continue
  echo "==== filtered: $SEQ_NAME ===="
  "$PYTHON_BIN" scripts/run_rgbd_slam_benchmark.py \
    --sequence-root "$SEQ_ROOT" \
    --output-dir "$RUN_ROOT/filtered_${SEQ_NAME}" \
    --filter-mode filtered \
    --config "$FILTER_CONFIG" \
    --orb-exec "$ORB_EXEC" \
    --orb-vocab "$ORB_VOCAB" \
    --orb-config "$ORB_CONFIG" \
    --eval-max-diff 0.03
done <<'EOF'
freiburg3_sitting_static datasets/tum_rgbd/freiburg3_sitting_static/rgbd_dataset_freiburg3_sitting_static
freiburg3_sitting_xyz datasets/tum_rgbd/freiburg3_sitting_xyz/rgbd_dataset_freiburg3_sitting_xyz
freiburg3_walking_static datasets/tum_rgbd/freiburg3_walking_static/rgbd_dataset_freiburg3_walking_static
freiburg3_walking_xyz datasets/tum_rgbd/freiburg3_walking_xyz/rgbd_dataset_freiburg3_walking_xyz
bonn_static datasets/bonn_rgbd_dynamic/static/rgbd_bonn_static
bonn_person_tracking datasets/bonn_rgbd_dynamic/person_tracking/rgbd_bonn_person_tracking
bonn_crowd datasets/bonn_rgbd_dynamic/crowd/rgbd_bonn_crowd
bonn_kidnapping_box datasets/bonn_rgbd_dynamic/kidnapping_box/rgbd_bonn_kidnapping_box
EOF
```

### 17.3 运行 `raw` 基线全序列测试

如果你需要和未过滤输入做严格对照，再追加跑下面这一组命令。

```bash
cd ~/d-drive/CODEX/basic_model_based_SLAM
source ~/anaconda3/etc/profile.d/conda.sh
conda activate openvoc

PYTHON_BIN=/home/lj/anaconda3/envs/openvoc/bin/python
ORB_EXEC=/home/lj/tools/slam_deps/ORB_SLAM3/Examples/RGB-D/rgbd_tum
ORB_VOCAB=/home/lj/tools/slam_deps/ORB_SLAM3/Vocabulary/ORBvoc.txt
ORB_CONFIG=/home/lj/tools/slam_deps/ORB_SLAM3/Examples/RGB-D/TUM3.yaml

RUN_ROOT=experiments/full_benchmark_20260415_000000

while read -r SEQ_NAME SEQ_ROOT; do
  [ -z "$SEQ_NAME" ] && continue
  echo "==== raw: $SEQ_NAME ===="
  "$PYTHON_BIN" scripts/run_rgbd_slam_benchmark.py \
    --sequence-root "$SEQ_ROOT" \
    --output-dir "$RUN_ROOT/raw_${SEQ_NAME}" \
    --filter-mode raw \
    --raw-export-mode symlink \
    --orb-exec "$ORB_EXEC" \
    --orb-vocab "$ORB_VOCAB" \
    --orb-config "$ORB_CONFIG" \
    --eval-max-diff 0.03
done <<'EOF'
freiburg3_sitting_static datasets/tum_rgbd/freiburg3_sitting_static/rgbd_dataset_freiburg3_sitting_static
freiburg3_sitting_xyz datasets/tum_rgbd/freiburg3_sitting_xyz/rgbd_dataset_freiburg3_sitting_xyz
freiburg3_walking_static datasets/tum_rgbd/freiburg3_walking_static/rgbd_dataset_freiburg3_walking_static
freiburg3_walking_xyz datasets/tum_rgbd/freiburg3_walking_xyz/rgbd_dataset_freiburg3_walking_xyz
bonn_static datasets/bonn_rgbd_dynamic/static/rgbd_bonn_static
bonn_person_tracking datasets/bonn_rgbd_dynamic/person_tracking/rgbd_bonn_person_tracking
bonn_crowd datasets/bonn_rgbd_dynamic/crowd/rgbd_bonn_crowd
bonn_kidnapping_box datasets/bonn_rgbd_dynamic/kidnapping_box/rgbd_bonn_kidnapping_box
EOF
```

说明：

- `17.2` 里会创建新的 `RUN_ROOT`
- `17.3` 里的 `RUN_ROOT` 要改成你在 `17.2` 实际生成的目录
- 如果你只想复现主方法，不做原始基线对照，可以只执行 `17.2`

### 17.4 汇总所有运行结果为总表

下面这段命令会扫描 `RUN_ROOT` 下所有 `benchmark_summary.json`，并输出总表 `batch_metrics_summary.csv`。

```bash
cd ~/d-drive/CODEX/basic_model_based_SLAM
source ~/anaconda3/etc/profile.d/conda.sh
conda activate openvoc

PYTHON_BIN=/home/lj/anaconda3/envs/openvoc/bin/python
RUN_ROOT=experiments/full_benchmark_20260415_000000

"$PYTHON_BIN" - <<'PY'
import csv
import json
from pathlib import Path

run_root = Path("experiments/full_benchmark_20260415_000000")
rows = []

for summary_path in sorted(run_root.glob("*/benchmark_summary.json")):
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    traj = data.get("trajectory_metrics", {})
    orb = data.get("orb_slam3", {})
    rows.append(
        {
            "run_dir": summary_path.parent.name,
            "sequence_root": data.get("sequence_root", ""),
            "filter_mode": data.get("filter_mode", ""),
            "orb_status": orb.get("status", ""),
            "exported_frames": data.get("exported_frames", ""),
            "mean_runtime_ms": data.get("mean_runtime_ms", ""),
            "mean_mask_ratio": data.get("mean_mask_ratio", ""),
            "mean_filtered_detections": data.get("mean_filtered_detections", ""),
            "trajectory_coverage": traj.get("trajectory_coverage", ""),
            "matched_poses": traj.get("matched_poses", ""),
            "ate_rmse_m": traj.get("ate_rmse_m", ""),
            "ate_mean_m": traj.get("ate_mean_m", ""),
            "rpe_rmse_m": traj.get("rpe_rmse_m", ""),
        }
    )

out_path = run_root / "batch_metrics_summary.csv"
with open(out_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "run_dir",
            "sequence_root",
            "filter_mode",
            "orb_status",
            "exported_frames",
            "mean_runtime_ms",
            "mean_mask_ratio",
            "mean_filtered_detections",
            "trajectory_coverage",
            "matched_poses",
            "ate_rmse_m",
            "ate_mean_m",
            "rpe_rmse_m",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(out_path)
PY
```

### 17.5 生成 `raw` vs `filtered` 配对对比表

如果你同时跑了 `raw_*` 和 `filtered_*`，建议再生成一份按序列配对的比较表，直接看 ATE / RPE 是提升还是下降。

```bash
cd ~/d-drive/CODEX/basic_model_based_SLAM
source ~/anaconda3/etc/profile.d/conda.sh
conda activate openvoc

PYTHON_BIN=/home/lj/anaconda3/envs/openvoc/bin/python

"$PYTHON_BIN" - <<'PY'
import csv
import json
from pathlib import Path

run_root = Path("experiments/full_benchmark_20260415_000000")
raw_runs = {}
filtered_runs = {}

for summary_path in sorted(run_root.glob("*/benchmark_summary.json")):
    run_name = summary_path.parent.name
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    key = run_name
    if run_name.startswith("raw_"):
        raw_runs[run_name[len("raw_"):]] = data
    elif run_name.startswith("filtered_"):
        filtered_runs[run_name[len("filtered_"):]] = data

rows = []
for seq_name in sorted(set(raw_runs) & set(filtered_runs)):
    raw = raw_runs[seq_name].get("trajectory_metrics", {})
    filt = filtered_runs[seq_name].get("trajectory_metrics", {})
    raw_ate = float(raw.get("ate_rmse_m", 0.0))
    filt_ate = float(filt.get("ate_rmse_m", 0.0))
    raw_rpe = float(raw.get("rpe_rmse_m", 0.0))
    filt_rpe = float(filt.get("rpe_rmse_m", 0.0))
    rows.append(
        {
            "sequence": seq_name,
            "raw_ate_rmse_m": raw_ate,
            "filtered_ate_rmse_m": filt_ate,
            "ate_delta_m": filt_ate - raw_ate,
            "ate_improve_percent": ((raw_ate - filt_ate) / raw_ate * 100.0) if raw_ate > 0 else 0.0,
            "raw_rpe_rmse_m": raw_rpe,
            "filtered_rpe_rmse_m": filt_rpe,
            "rpe_delta_m": filt_rpe - raw_rpe,
            "raw_coverage": raw.get("trajectory_coverage", ""),
            "filtered_coverage": filt.get("trajectory_coverage", ""),
        }
    )

out_path = run_root / "pairwise_raw_vs_filtered.csv"
with open(out_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "sequence",
            "raw_ate_rmse_m",
            "filtered_ate_rmse_m",
            "ate_delta_m",
            "ate_improve_percent",
            "raw_rpe_rmse_m",
            "filtered_rpe_rmse_m",
            "rpe_delta_m",
            "raw_coverage",
            "filtered_coverage",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(out_path)
PY
```

### 17.6 如何判断结果是否正常

批量实验结束后，优先看下面三份文件：

- `RUN_ROOT/batch_metrics_summary.csv`：全部序列的 ATE / RPE / 速度总表
- `RUN_ROOT/pairwise_raw_vs_filtered.csv`：主方法相对 Raw 基线的提升或退化
- `RUN_ROOT/*/benchmark_summary.json`：某一条序列的完整细节

建议的快速判读口径：

- 动态场景如 `freiburg3_walking_xyz`、`bonn_person_tracking`、`bonn_crowd`，重点看 `filtered_ate_rmse_m` 是否低于 `raw_ate_rmse_m`
- 静态或弱动态场景如 `freiburg3_sitting_static`、`bonn_static`，重点看 `ate_delta_m` 是否接近 `0`，以验证“不过滤静态背景”
- `mean_runtime_ms` 用来衡量前端过滤开销
- `trajectory_coverage` 过低时，即使 ATE 数值看起来较小，也不能直接说明方法更好
