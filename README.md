# ResNet 2D 轻量化/标准模型

## 概述

本项目是本科毕设“基于昇腾 AI 架构的高效化无人机射频信号识别”的训练端代码实现。仓库围绕 2D `.npy` 数据集构建了六阶段主线：

- `base_model`：基座模型训练、验证、测试与 UMAP/混淆矩阵可视化
- `pruning`：基于 `torch-pruning` 的 iterative structured pruning + 微调
- `qat`：基于 Torch 原生 FX graph mode 的保守单路径 QAT
- `onnx`：导出 `pruning_fp16` 或 `qat_convert` ONNX，并用 ONNX Runtime 评估
- `amct`：消费 `qat_convert` ONNX，生成 Ascend 侧可继续下游处理的 deploy/fakequant ONNX
- `atc`：消费 `pruning_fp16` 或 `amct_deploy` ONNX，编译固定 batch=1 的 `.om`

工程主线为：
```text
base_model checkpoint
  -> pruning checkpoint
  -> QAT prepare checkpoint
  -> ONNX 导出 / ORT 评估
  -> AMCT 转换
  -> ATC 编译
  -> 后续部署 / 推理验证
```

## 项目状态

- `base_model`：已实现，作为训练主线上游稳定使用
- `pruning`：已实现，支持多轮剪枝、微调、拓扑导出与实验摘要
- `qat`：已实现，支持按剪枝拓扑严格恢复并导出 prepare 后 QAT checkpoint
- `onnx`：已实现，支持 `pruning_fp16` / `qat_convert` 双分支导出、动态 batch、ORT 精度评估
- `amct`：已实现并接入主线；运行该阶段前需要额外准备仓库附带的 `amct_onnx` wheel 与算子包
- `atc`：已实现并接入主线；真实编译与运行验证依赖 Ascend 宿主机环境

说明：

- “已实现 / 已接入主线”表示代码、入口脚本和产物契约已经落地。
- 文档统一以代码实现与标准产物契约为准，不把“已实现”误写成“已充分验证”。

## 核心能力

- 5 种 2D ResNet 架构：`resnet6_2d`、`resnet10_2d`、`resnet14_2d`、`resnet18_2d`、`resnet34_2d`
- `base_model / pruning` 支持 `fp16` 或 `fp32` 数据加载
- `qat` 固定纯 `fp32`
- 稳定的数据集切分与 `output/splits/` manifest 复用
- 基座、剪枝、QAT 三类结构化 checkpoint
- 剪枝后完整拓扑导出：`channel_cfg + architecture_signature`
- 所有消费 `.pth` checkpoint 的步骤统一强校验 `architecture_signature`
- ONNX / deploy ONNX 阶段统一通过同目录 summary 读取并校验上游 `architecture_signature` 与来源信息
- Torch 原生 FX graph mode QAT
- `pruning_fp16` / `qat_convert` 双分支 ONNX 导出
- QAT ONNX rewrite + validate，用于 CANN/AMCT/ATC 兼容约束
- AMCT deploy / fakequant ONNX 转换
- pruning FP16 / AMCT deploy ONNX 的 ATC 编译
- TensorBoard、混淆矩阵、UMAP 等辅助分析能力

## 环境与安装

### 需要用户手动安装的项目

项目只把以下项目视为“用户独立手动安装的前置项”：

- `git`：用于克隆仓库
- `pixi`：负责系统工具链与运行时环境
- `uv`：负责 Python 依赖同步与 `uv run ...`
- `direnv`（可选）：用于自动激活项目公共环境层

### `pixi install` / `uv sync` 会自动安装的内容

以下内容不应再作为“手动环境依赖”单独列出，因为它们由项目命令自动安装：

- `pixi install`：
  - Python 3.12 运行时
  - GCC / G++ / Make / CMake 等工具链
  - `cuda-runtime`、`cudnn`
  - `ascend-cann-toolkit`、`ascend-cann-310b-ops`
- `uv sync`：
  - `torch`
  - `onnx`
  - `onnxruntime-gpu`
  - `torch-pruning`
  - `tensorboard`
  - `matplotlib`
  - `umap-learn`
  - 以及 `pyproject.toml` 中声明的其余 Python 包

### 条件性宿主机要求

- 若要使用 CUDA 加速训练，宿主机需要可用的 NVIDIA GPU 与驱动。
- 若要进行真实的 Ascend 编译或部署验证，宿主机需要对应的 Ascend 设备/驱动环境。
- 上述宿主机硬件/驱动要求是条件说明，不属于仓库自身的可自动安装依赖。

### 安装步骤

1. 克隆项目
   ```bash
   git clone git@github.com:wh-wang132/ResNet.git
   cd ResNet
   ```
2. 安装 Pixi 环境
   ```bash
   pixi install
   ```
3. 同步 Python 依赖
   ```bash
   uv sync
   ```
4. 启用 `direnv`（推荐）
   ```bash
   direnv allow
   ```
   当前项目根目录的 [`.envrc`](.envrc) 提供仓库级公共变量：
   - `REPO_ROOT`
   - `PYTHONPATH=$REPO_ROOT/src`

   说明：
   - `direnv` 为推荐方案；若不使用 `direnv` 自动激活，也必须手动提供与 `.envrc` 等价的环境变量
   - 所有脚本统一通过 `.envrc` 提供的 `REPO_ROOT` 识别仓库根
5. 若需要运行 AMCT 阶段，额外准备仓库附带的 AMCT 组件
   - `amct_onnx/amct_onnx-0.23.2-py3-none-linux_x86_64.whl`
   - `amct_onnx/amct_onnx_op.tar.gz`
   说明：
   - 上述 wheel 与算子包已经随仓库提供
   - 它们不在 `pyproject.toml` 或 `pixi.toml` 中声明
   - 因此它们不是 `uv sync` / `pixi install` 自动安装的公共依赖，而是 AMCT 阶段专用的手动准备项
   - 若宿主机已预装系统全局 CUDA，AMCT 在安装 `amct_onnx/amct_onnx_op.tar.gz` 或运行阶段可能优先搜索到系统 CUDA，从而引发 CUDA 版本不匹配错误；此场景建议在 Docker 环境中部署和运行 AMCT 相关流程
6. 准备数据集
   - 将 `.npy` 数据集放入 `Data/`
   - 目录结构见 [数据准备指南](docs/DATA_PREPARATION.md)

## 环境层次

项目采用“公共层 + 阶段增量层”的环境结构：

| 阶段 | 公共层 | 阶段增量层 |
| --- | --- | --- |
| `base_model` | `.envrc` | `source scripts/load_base_model_env.sh` |
| `pruning` | `.envrc` | 无 |
| `qat` | `.envrc` | 无 |
| `onnx` | `.envrc` | `source scripts/load_onnx_env.sh` |
| `amct` | `.envrc` | `source scripts/load_amct_env.sh` |
| `atc` | `.envrc` | `source scripts/load_atc_env.sh` |

说明：

- `.envrc` 是唯一公共入口，只负责仓库级变量。
- `scripts/load_*_env.sh` 只负责补齐阶段增量环境。
- 阶段脚本默认在已加载 `.envrc` 的 shell 中运行。

## 基本使用

### 基座模型训练

```bash
# 完整训练 + 测试
uv run python -m base_model --epochs 20 --model resnet6_2d

# 仅训练
uv run python -m base_model --epochs 20 --Test False

# 仅测试 + UMAP
uv run python -m base_model --Train False --UMAP True
```

### 剪枝 + 微调

```bash
# 最小剪枝命令
uv run python -m pruning --model resnet6_2d

# 指定总剪枝率与轮数
uv run python -m pruning \
  --model resnet18_2d \
  --pruning_ratio 0.30 \
  --pruning_steps 5 \
  --global_pruning True \
  --finetune_epochs 10

# 不做微调，只保存最终剪枝结果
uv run python -m pruning \
  --model resnet14_2d \
  --finetune_epochs 0 \
  --evaluate_test False
```

### QAT

```bash
# 最小 QAT 命令
uv run python -m qat \
  --pruning_checkpoint output/pruning/resnet14_2d/ratio0.60_steps8_global_ft10_bs64/best_pruned_model.pth

# 指定保守 QAT 微调参数
uv run python -m qat \
  --pruning_checkpoint output/pruning/resnet34_2d/ratio0.80_steps8_global_ft10_bs64/best_pruned_model.pth \
  --qat_epochs 10 \
  --lr 1e-5 \
  --batch_size 64
```

### ONNX 导出

```bash
# pruning checkpoint -> FP16 ONNX
uv run python -m onnx_export \
  --branch pruning_fp16 \
  --checkpoint output/pruning/resnet10_2d/ratio0.40_steps5_global_ft10_bs64/best_pruned_model.pth \
  --eval_batch_size 64

# QAT checkpoint -> convert 后量化 ONNX
uv run python -m onnx_export \
  --branch qat_convert \
  --checkpoint output/qat/resnet10_2d/from_ratio0.40_steps5_global_ft10_bs64/best_qat_prepare_model.pth \
  --eval_batch_size 64
```

说明：

- ONNX 导出当前统一使用动态 batch。
- `onnx_summary.json.example_input_shape` 中的 `batch=1` 只表示导出样例输入。
- `--eval_batch_size` 只影响 Torch / ORT 精度评估，不影响导出图结构。

### AMCT 转换

运行前请先确认仓库附带的 `amct_onnx` wheel 与算子包已经按目标环境自行安装或部署。

```bash
. scripts/load_amct_env.sh

uv run python -m amct \
  --onnx_model output/onnx/qat_convert/resnet6_2d/from_ratio0.60_steps8_global_ft10_bs64/model_quant.onnx
```

### ATC 编译

```bash
. scripts/load_atc_env.sh

# pruning_fp16 ONNX -> ATC
pixi run python -m atc \
  --branch pruning_fp16 \
  --onnx_model output/onnx/pruning_fp16/resnet10_2d/from_ratio0.40_steps5_global_ft10_bs64/model_fp16.onnx

# AMCT deploy ONNX -> ATC
pixi run python -m atc \
  --branch amct_deploy \
  --onnx_model output/amct/resnet6_2d/from_ratio0.60_steps8_global_ft10_bs64/deploy_model.onnx
```

默认：

- `soc_version=Ascend310B4`
- `input_format=NCHW`
- `input_shape` 默认从上游摘要中的输入接口派生，并将 batch 固定为 `1`
- 若用户显式传入 `--input_shape`，其输入名与各维度必须与自动派生结果完全一致，否则直接报错

## 基座模型自动选择约定

剪枝入口不会手动接收基座 checkpoint 路径，而是自动扫描：

```text
output/base_model/<model>/<experiment_dir>/best_model.pth
```

程序会遍历 `output/base_model/<model>/` 下各实验目录，读取每个目录 `best_val_acc_info.txt` 的最后一条有效记录，并按 `val acc` 优先、`val loss` 次优选择最佳实验权重进入剪枝链路。

## 自动化脚本

`autorun/` 目录提供 6 份顺序执行脚本，并在 `pixi.toml` 中一一映射为独立 task：

- [autorun/autorun_base_model.sh](autorun/autorun_base_model.sh) -> `pixi run autorun-base-model`
- [autorun/autorun_pruning.sh](autorun/autorun_pruning.sh) -> `pixi run autorun-pruning`
- [autorun/autorun_qat.sh](autorun/autorun_qat.sh) -> `pixi run autorun-qat`
- [autorun/autorun_onnx.sh](autorun/autorun_onnx.sh) -> `pixi run autorun-onnx`
- [autorun/autorun_amct.sh](autorun/autorun_amct.sh) -> `pixi run autorun-amct`
- [autorun/autorun_atc.sh](autorun/autorun_atc.sh) -> `pixi run autorun-atc`

这些脚本整体仍服务于顺序批处理执行；其中 `onnx` / `amct` / `atc` autorun 已包含 shell 函数、`mktemp`、`trap`、`find` 遍历与临时文件清理等基础控制逻辑，适合直接在服务器终端监视运行并安全清理中间状态。推荐优先通过 `pixi run <task>` 调用；对应 shell 脚本仍保留作为底层实现。

运行前提：

- 使用 `pixi run autorun-*` 时，task 会自动注入 `REPO_ROOT=$PIXI_PROJECT_ROOT` 与 `PYTHONPATH=$PIXI_PROJECT_ROOT/src`
- 若手动执行 `bash autorun/*.sh`，仍要求 shell 已自动或手动激活项目根目录的 `.envrc`

脚本行为概览：

- `autorun-base-model`
  - 批量训练 5 个基座模型
  - 搜索维度：模型与 `batch_size`
  - 固定显式传入：`--full_load True`
- `autorun-pruning`
  - 批量运行 pruning 实验
  - 搜索维度：模型、`pruning_ratio`、`pruning_steps`
  - 固定显式传入：`--full_load True`
- `autorun-qat`
  - 批量消费 pruning checkpoint
  - 固定显式传入：`--full_load True`
- `autorun-onnx`
  - 遍历 `output/pruning/**/best_pruned_model.pth`
  - 遍历 `output/qat/**/best_qat_prepare_model.pth`
  - 默认参数与 ONNX CLI 保持一致：`evaluate_test=True`、`eval_batch_size=64`
- `autorun-amct`
  - 遍历 `output/onnx/qat_convert/**/model_quant.onnx`
- `autorun-atc`
  - 遍历 `output/onnx/pruning_fp16/**/model_fp16.onnx`
  - 遍历 `output/amct/**/deploy_model.onnx`
  - 默认参数与 ATC CLI 保持一致：`soc_version=Ascend310B4`

## 数据划分清单

`base_model.dataset.data_set_split()` 会优先读取：

```text
output/splits/
```

中的数据集划分 manifest。若 manifest 不存在或与当前配置不匹配，则会重新划分并重新落盘。

manifest 保存：

- 训练 / 验证 / 测试的相对路径清单
- `class_names` 与 `class_to_idx`
- 划分比例与随机种子
- 相对路径 `data_dir=Data`

## 文档索引

- [项目架构分析](docs/PROJECT_ARCHITECTURE.md)
- [模块说明](docs/MODULES.md)
- [命令行参数详解](docs/CLI_ARGUMENTS.md)
- [自动化脚本说明](docs/AUTOMATED_TRAINING.md)
- [数据准备指南](docs/DATA_PREPARATION.md)
- [模型架构说明](docs/MODEL_ARCHITECTURE.md)
- [训练参数调优指南](docs/TRAINING_GUIDE.md)
- [剪枝指南](docs/PRUNING_GUIDE.md)
- [贡献指南](docs/CONTRIBUTING.md)
