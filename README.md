# ResNet 2D 轻量化/标准模型

## 概述

本项目是本科毕设“基于昇腾 AI 架构的高效化无人机射频信号识别”的训练代码实现，围绕 2D `.npy` 数据集构建了三阶段工程主线：

- `base_model`：基座模型训练、验证、测试与可视化
- `pruning`：基于 `torch-pruning` 的 iterative structured pruning + 微调
- `qat`：基于 Torch 原生 FX graph mode 的保守单路径 QAT，消费 pruning checkpoint 并导出 prepare 后 QAT checkpoint

当前代码主线已经完整覆盖：

```text
基座训练 checkpoint -> pruning checkpoint -> QAT prepare checkpoint -> （后续）ONNX / 部署恢复
```

## 当前完成度

- `base_model`：已稳定，可直接用于训练、测试、混淆矩阵与 UMAP 可视化
- `pruning`：已收敛，可直接从基座模型符号链接出发执行多轮剪枝、微调并导出 pruning checkpoint
- `qat`：已落地，负责消费 pruning checkpoint、恢复剪枝结构并执行保守 QAT 微调

## 核心能力

- 多种 2D ResNet 架构：轻量级与标准版
- FP16 AMP 混合精度训练
- Warmup + Cosine Annealing 学习率调度
- 稳定的数据集划分与多线程加载
- 基座 checkpoint 的结构化保存
- iterative structured pruning
- 剪枝后完整拓扑导出：`channel_cfg` + `architecture_signature`
- Torch 原生 FX graph mode QAT
- QAT prepare checkpoint 导出
- 最终测试混淆矩阵生成
- TensorBoard 日志记录

## 环境与安装

### 环境要求

- Python 3.12+
- CUDA 13.0+（如需 GPU 加速）
- NVIDIA GPU（推荐 8GB+ 显存）
- `pixi`（项目默认必需：负责 GCC / Make / CMake 等系统工具链环境）
- `uv`（负责 Python 依赖与 `uv run ...` 入口）
- `direnv`（推荐：自动激活 `pixi shell-hook` 并设置 `PYTHONPATH`）

### 安装步骤

1. 克隆项目
   ```bash
   git clone git@github.com:wh-wang132/ResNet.git
   cd ResNet
   ```
2. 安装 Pixi 工具链环境
   ```bash
   pixi install
   ```
3. 安装 Python 依赖
   ```bash
   uv sync
   ```
4. 启用 `direnv`（推荐）
   ```bash
   direnv allow
   ```
   当前项目根目录的 [`.envrc`](/root/ResNet/.envrc) 会：
   - 注入 `pixi shell-hook`
   - 将 `PYTHONPATH` 固定为项目根目录下的 `src`
5. 准备数据集
   - 将 `.npy` 数据集放入 `Data/`
   - 目录结构说明见 [数据准备指南](docs/DATA_PREPARATION.md)

## 基本使用

默认前提：当前 shell 已处于项目标准 `pixi + uv` 环境中。推荐直接进入项目根目录并通过 `.envrc` 自动激活；若未使用 `direnv`，则应先手动进入 `pixi` 环境后再执行 `uv run ...`。

### 基座模型训练

```bash
# 完整训练 + 测试
uv run src/base_model_main.py --epochs 20 --model resnet6_2d

# 仅训练
uv run src/base_model_main.py --epochs 20 --Test False

# 仅测试 + UMAP
uv run src/base_model_main.py --Train False --UMAP True
```

### 剪枝 + 微调

```bash
# 最小剪枝命令
uv run src/pruning_main.py --model resnet6_2d

# 指定总剪枝率与轮数
uv run src/pruning_main.py \
  --model resnet18_2d \
  --pruning_ratio 0.30 \
  --pruning_steps 5 \
  --global_pruning True \
  --finetune_epochs 10

# 不做微调，只保存最终剪枝结果
uv run src/pruning_main.py \
  --model resnet14_2d \
  --finetune_epochs 0 \
  --evaluate_test False
```

### QAT

```bash
# 最小 QAT 命令
uv run src/qat_main.py \
  --pruning_checkpoint output/pruning/resnet14_2d/ratio0.60_steps8_global_ft10_bs64/best_pruned_model.pth

# 指定保守 QAT 微调参数
uv run src/qat_main.py \
  --pruning_checkpoint output/pruning/resnet34_2d/ratio0.80_steps8_global_ft10_bs64/best_pruned_model.pth \
  --qat_epochs 10 \
  --lr 1e-5 \
  --batch_size 64
```

### 基座模型符号链接约定

剪枝入口不会手动接收基座 checkpoint 路径，而是固定读取：

```text
output/base_model/<model>/best_model.pth
```

这里的 `best_model.pth` 由你在对应基座模型根目录下维护为指向最佳实验权重的符号链接。

## 自动化脚本

两份自动化脚本同样默认运行在项目标准 `pixi + uv` 环境中。

项目根目录当前提供两份顺序执行脚本：

- [autorun_base_model.sh](/root/ResNet/autorun_base_model.sh)
  - 批量训练全部 5 个基座模型
  - 主要搜索模型与 `batch_size`
- [autorun_pruning.sh](/root/ResNet/autorun_pruning.sh)
  - 批量运行 pruning 实验
  - 主要搜索模型、`pruning_ratio` 与 `pruning_steps`

两份脚本都采用“逐行命令、顺序执行、无复杂控制流”的风格，适合在服务器终端直接监控。

## 项目结构

```text
ResNet/
├── src/
│   ├── base_model_main.py      # 基座模型训练入口
│   ├── pruning_main.py         # 剪枝 + 微调入口
│   ├── qat_main.py             # QAT 入口
│   ├── base_model/
│   │   ├── args.py
│   │   ├── dataset.py
│   │   ├── utils.py
│   │   ├── trainer.py
│   │   ├── tester.py
│   │   ├── visualizer.py
│   │   ├── confusionMatrix.py
│   │   ├── lr_scheduler.py
│   │   ├── resnet_lightweight.py
│   │   └── resnet_standard.py
│   ├── pruning/
│   │   ├── args.py
│   │   ├── checkpoint.py
│   │   ├── evaluator.py
│   │   ├── output.py
│   │   ├── pruner.py
│   │   ├── topology.py
│   │   ├── trainer.py
│   │   ├── utils.py
│   │   └── README.md
│   └── qat/
│       ├── args.py
│       ├── checkpoint.py
│       ├── evaluator.py
│       ├── output.py
│       ├── quantization.py
│       ├── trainer.py
│       ├── utils.py
│       └── README.md
├── docs/
├── Data/
├── output/
├── autorun_base_model.sh
├── autorun_pruning.sh
├── .envrc
├── pixi.toml
├── pixi.lock
├── pyproject.toml
├── uv.lock
└── README.md
```

## 模型概览

### 轻量级模型

| 模型          | 参数量     | 结构特点                           |
| ------------- | ---------- | ---------------------------------- |
| `resnet6_2d`  | 约 310,392 | 3 个残差层，`init_channels=32`     |
| `resnet10_2d` | 约 694,440 | 3 个残差层，`init_channels=48`     |
| `resnet14_2d` | 约 902,376 | 3 个残差层，残差块配置 `[2, 2, 1]` |

### 标准模型

| 模型          | 参数量   | 残差块       |
| ------------- | -------- | ------------ |
| `resnet18_2d` | 约 11.2M | `BasicBlock` |
| `resnet34_2d` | 约 21.3M | `BasicBlock` |

详细说明见 [模型架构说明](docs/MODEL_ARCHITECTURE.md)。

## 输出文件

### 基座训练输出

```text
output/base_model/<model>/epochs<epochs>_bs<batch_size>/
├── best_model.pth
├── best_val_acc_info.txt
├── lr_schedule.png
├── training_curves.png
├── Confusion_matrix.png
├── umap_plot.png            # 仅启用 UMAP 时生成
└── runs/
```

另外，若你希望将某个实验指定为 pruning 上游输入，还会在：

```text
output/base_model/<model>/best_model.pth
```

维护一个指向最佳实验 checkpoint 的符号链接。

### 剪枝输出

```text
output/pruning/<model>/ratio<ratio>_steps<steps>_<global|local>_ft<epochs>_bs<batch_size>/
├── best_pruned_model.pth
├── best_pruned_info.txt
├── pruning_summary.json
├── Confusion_matrix.png     # 仅最终测试阶段生成
└── runs/
    ├── round_1/
    ├── round_2/
    └── ...
```

其中：

- `best_pruned_model.pth`：仅最终轮保存的 pruning checkpoint
- `best_pruned_info.txt`：每轮一行，记录该轮最佳验证结果
- `pruning_summary.json`：记录 `baseline / rounds / pruning_meta / final / final_topology` 等摘要信息

### QAT 输出

```text
output/qat/<model>/from_<pruning_exp>/
├── best_qat_prepare_model.pth
├── best_qat_info.txt
├── qat_summary.json
├── Confusion_matrix.png     # 仅最终测试阶段生成
└── runs/
```

其中：

- `best_qat_prepare_model.pth`：prepare 后的 QAT checkpoint
- `best_qat_info.txt`：每次 best 刷新时追加一行
- `qat_summary.json`：记录 `baseline / quantization_meta / finetune_summary / final / final_topology`

## 文档导航

- [数据准备指南](docs/DATA_PREPARATION.md)
- [命令行参数详解](docs/CLI_ARGUMENTS.md)
- [模型架构说明](docs/MODEL_ARCHITECTURE.md)
- [训练参数调优](docs/TRAINING_GUIDE.md)
- [自动化脚本说明](docs/AUTOMATED_TRAINING.md)
- [剪枝指南](docs/PRUNING_GUIDE.md)
- [QAT 说明](src/qat/README.md)
- [模块说明](docs/MODULES.md)
- [项目架构分析](docs/PROJECT_ARCHITECTURE.md)

## 当前阶段边界

- `base_model` 负责产出稳定的基座 checkpoint
- `pruning` 负责读取基座 checkpoint，执行剪枝与微调，并导出 pruning checkpoint
- `qat` 负责读取 pruning checkpoint，恢复剪枝结构并导出 QAT prepare checkpoint
- 后续 ONNX/部署模块将负责消费 QAT checkpoint 恢复接口并继续恢复或导出

也就是说：

- pruning 只负责产出 pruning checkpoint
- QAT 只负责产出 QAT prepare checkpoint
- ONNX/部署恢复链留待后续阶段单独实现

## 贡献与许可证

- 贡献方式与开发规范见 [贡献指南](docs/CONTRIBUTING.md)
- 项目许可证见 [LICENSE](LICENSE)
