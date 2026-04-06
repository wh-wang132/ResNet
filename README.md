# ResNet 2D 轻量化/标准模型

## 概述

本项目是本科毕设“基于昇腾 AI 架构的高效化无人机射频信号识别”的训练代码实现，围绕 2D `.npy` 数据集构建了六阶段工程主线：

- `base_model`：基座模型训练、验证、测试与可视化
- `pruning`：基于 `torch-pruning` 的 iterative structured pruning + 微调
- `qat`：基于 Torch 原生 FX graph mode 的保守单路径 QAT，消费 pruning checkpoint 并导出 prepare 后 QAT checkpoint
- `onnx`：导出 pruning FP16 ONNX 或 QAT convert ONNX，并用 ONNX Runtime 评估精度
- `amct`：消费 `qat_convert` ONNX，转换为 ATC 可接受的 deploy/fakequant ONNX 产物
- `atc`：消费 pruning FP16 ONNX 或 AMCT deploy ONNX，编译固定 batch=1 的 `.om` 产物

当前代码主线已经完整覆盖：

```text
基座训练 checkpoint -> pruning checkpoint -> QAT prepare checkpoint -> ONNX 导出 / 评估 -> AMCT 转换 -> 后续 ATC / 部署
```

## 当前完成度

- `base_model`：已稳定，可直接用于训练、测试、混淆矩阵与 UMAP 可视化
- `pruning`：已收敛，可直接从基座模型符号链接出发执行多轮剪枝、微调并导出 pruning checkpoint
- `qat`：已落地，负责消费 pruning checkpoint、恢复剪枝结构并执行保守 QAT 微调
- `onnx`：已落地，负责导出 `pruning_fp16` 与 `qat_convert` 两条标准 ONNX 分支并执行 ORT 精度评估
- `amct`：已落地，负责消费仓库 `qat_convert` ONNX 产物并生成 ATC 可接受的 `deploy_model.onnx`
- `atc`：已落地，负责消费 pruning FP16 ONNX 与 AMCT deploy ONNX，并编译 `Ascend310B4` 目标 `.om`

## 核心能力

- 多种 2D ResNet 架构：轻量级与标准版
- `base_model / pruning` 支持 FP16 数据与训练链
- `qat` 固定纯 FP32
- Warmup + Cosine Annealing 学习率调度
- 稳定的数据集划分与多线程加载
- 基座 checkpoint 的结构化保存
- iterative structured pruning
- 剪枝后完整拓扑导出：`channel_cfg` + `architecture_signature`
- Torch 原生 FX graph mode QAT
- QAT prepare checkpoint 导出
- ONNX opset 16 双分支导出
- ONNX Runtime 精度评估
- AMCT deploy / fakequant ONNX 转换
- pruning FP16 / AMCT deploy ONNX 的 ATC 编译
- 最终测试混淆矩阵生成
- TensorBoard 日志记录

## 环境与安装

### 环境要求

- Python 3.12+
- CUDA 13.0+（如需 GPU 加速）
- NVIDIA GPU（推荐 8GB+ 显存）
- `pixi`（项目默认必需：负责 GCC / Make / CMake 等系统工具链环境）
- `uv`（负责 Python 依赖与 `uv run ...` 入口）
- `direnv`（推荐：自动激活项目公共环境层）

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
   当前项目根目录的 [`.envrc`](.envrc) 会：
   - 导出 `REPO_ROOT`
   - 导出 `PYTHONPATH=$REPO_ROOT/src`
   - 不再负责注入 `pixi shell-hook`
5. 准备数据集
   - 将 `.npy` 数据集放入 `Data/`
   - 目录结构说明见 [数据准备指南](docs/DATA_PREPARATION.md)

## 基本使用

默认前提：当前 shell 已进入项目根目录，并由 `direnv` 自动加载 [`.envrc`](.envrc) 公共环境层。各阶段如需额外环境变量，再按需 source 对应 `scripts/load_*_env.sh`。

### 阶段环境映射

| 阶段 | 公共层 | 额外脚本 |
| --- | --- | --- |
| `base_model` | `.envrc` | `source scripts/load_base_model_env.sh` |
| `pruning` | `.envrc` | 无 |
| `qat` | `.envrc` | 无 |
| `onnx` | `.envrc` | `source scripts/load_onnx_env.sh` |
| `amct` | `.envrc` | `source scripts/load_amct_env.sh` |
| `atc` | `.envrc` | `source scripts/load_atc_env.sh` |

说明：

- `.envrc` 是唯一公共入口，只负责仓库级公共变量。
- `scripts/load_*_env.sh` 只补各阶段增量环境，不重复激活公共变量。
- 当前不再保证“脱离 `.envrc` 直接 source `scripts/load_*_env.sh`”这一用法。

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

### ONNX 导出

```bash
# pruning checkpoint -> FP16 ONNX
uv run src/onnx_main.py \
  --branch pruning_fp16 \
  --checkpoint output/pruning/resnet10_2d/ratio0.40_steps5_global_ft10_bs64/best_pruned_model.pth \
  --eval_batch_size 64

# QAT checkpoint -> convert 后量化 ONNX
uv run src/onnx_main.py \
  --branch qat_convert \
  --checkpoint output/qat/resnet10_2d/from_ratio0.40_steps5_global_ft10_bs64/best_qat_prepare_model.pth \
  --eval_batch_size 64
```

说明：

- ONNX 导出当前统一使用动态 batch；`onnx_summary.json.example_input_shape` 中的 `batch=1` 仅表示导出样例输入。
- 当前 `--eval_batch_size` 只影响 Torch / ORT 精度评估批次，不影响导出图结构。
- 若后续部署链需要静态 batch，可在 ATC 阶段通过 `--input_shape="input:1,1,543,512"` 固化输入形状。

### AMCT 转换

AMCT 阶段只接受本仓库 `qat_convert` 导出的 `model_quant.onnx`。建议先加载 AMCT 环境脚本，再执行主入口：

```bash
. scripts/load_amct_env.sh

uv run src/amct_main.py \
  --onnx_model output/onnx/qat_convert/resnet6_2d/from_ratio0.60_steps8_global_ft10_bs64/model_quant.onnx
```

### ATC 编译

ATC 阶段当前支持两条输入分支：

- `pruning_fp16`：消费 `output/onnx/pruning_fp16/.../model_fp16.onnx`
- `amct_deploy`：消费 `output/amct/.../deploy_model.onnx`

默认固定：

- `soc_version=Ascend310B4`
- `input_format=NCHW`
- `input_shape="input:1,1,543,512"`

```bash
. scripts/load_atc_env.sh

# pruning_fp16 ONNX -> ATC
pixi run python src/atc_main.py \
  --branch pruning_fp16 \
  --onnx_model output/onnx/pruning_fp16/resnet10_2d/from_ratio0.40_steps5_global_ft10_bs64/model_fp16.onnx

# AMCT deploy ONNX -> ATC
pixi run python src/atc_main.py \
  --branch amct_deploy \
  --onnx_model output/amct/resnet6_2d/from_ratio0.60_steps8_global_ft10_bs64/deploy_model.onnx
```

### 基座模型符号链接约定

剪枝入口不会手动接收基座 checkpoint 路径，而是固定读取：

```text
output/base_model/<model>/best_model.pth
```

这里的 `best_model.pth` 由你在对应基座模型根目录下维护为指向最佳实验权重的符号链接。

## 自动化脚本

自动化脚本默认要求当前 shell 已由 `direnv` 激活 [`.envrc`](.envrc) 公共环境层；未加载时会直接报错。需要阶段增量环境的脚本会在内部 source 对应 `load_*_env.sh`。

项目根目录当前提供六份顺序执行脚本：

- [autorun_base_model.sh](autorun_base_model.sh)
  - 批量训练全部 5 个基座模型
  - 主要搜索模型与 `batch_size`
- [autorun_pruning.sh](autorun_pruning.sh)
  - 批量运行 pruning 实验
  - 主要搜索模型、`pruning_ratio` 与 `pruning_steps`
- [autorun_qat.sh](autorun_qat.sh)
  - 批量消费 pruning 产物并顺序执行 QAT
  - 主要搜索 pruning 实验组合对应的 QAT 恢复与微调
- [autorun_onnx.sh](autorun_onnx.sh)
  - 批量消费 pruning / QAT checkpoint 并顺序执行 ONNX 导出
  - 主要搜索 `pruning_fp16` 与 `qat_convert` 两条导出分支
- [autorun_amct.sh](autorun_amct.sh)
  - 批量消费 `output/onnx/qat_convert` 下的 `model_quant.onnx`
  - 顺序执行 AMCT 转换并生成 deploy / fakequant ONNX
- [autorun_atc.sh](autorun_atc.sh)
  - 批量消费 `output/onnx/pruning_fp16` 下的 `model_fp16.onnx` 与 `output/amct` 下的 `deploy_model.onnx`
  - 顺序执行 ATC 编译并生成 `.om`

这些脚本都采用“逐行命令、顺序执行、无复杂控制流”的风格，适合在服务器终端直接监控。

## 数据划分清单

当前 `base_model.dataset.data_set_split()` 会优先读取：

```text
output/splits/
```

中已落盘的数据集划分清单；若不存在或与当前配置不匹配，则按原有规则重新划分并重新落盘。

这份 split manifest 当前作为训练端与后续推理端共享的数据划分真值，其中 `data_dir` 以相对路径 `Data` 保存。

## 项目结构

```text
ResNet/
├── src/
│   ├── base_model_main.py      # 基座模型训练入口
│   ├── pruning_main.py         # 剪枝 + 微调入口
│   ├── qat_main.py             # QAT 入口
│   ├── onnx_main.py            # ONNX 导出与评估入口
│   ├── amct_main.py            # AMCT 转换入口
│   ├── atc_main.py             # ATC 编译入口
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
│   └── onnx_export/
│       ├── args.py
│       ├── evaluator.py
│       ├── exporter.py
│       └── output.py
│   └── amct/
│       ├── args.py
│       ├── converter.py
│       └── output.py
│   └── atc/
│       ├── args.py
│       ├── converter.py
│       └── output.py
├── docs/
├── Data/
├── output/
├── autorun_amct.sh
├── autorun_atc.sh
├── autorun_base_model.sh
├── autorun_onnx.sh
├── autorun_pruning.sh
├── autorun_qat.sh
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
- ONNX 阶段当前已经消费 QAT checkpoint 恢复接口完成 `qat_convert` 导出

也就是说：

- pruning 只负责产出 pruning checkpoint
- QAT 只负责产出 QAT prepare checkpoint
- 后续部署阶段将在当前 ONNX 导出产物基础上继续衔接推理与编译链

## 贡献与许可证

- 贡献方式与开发规范见 [贡献指南](docs/CONTRIBUTING.md)
- 项目许可证见 [LICENSE](LICENSE)
