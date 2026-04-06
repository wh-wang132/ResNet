# 命令行参数详解

## 概述

当前项目包含 6 套独立 CLI：

- 基座训练：`uv run src/base_model_main.py ...`
- 剪枝：`uv run src/pruning_main.py ...`
- QAT：`uv run src/qat_main.py ...`
- ONNX 导出：`uv run src/onnx_main.py ...`
- AMCT 转换：`uv run src/amct_main.py ...`
- ATC 编译：`pixi run python src/atc_main.py ...`

各入口参数并不完全相同，阅读时需要严格区分阶段。

## 环境前提

### 需要用户手动安装的项目

- `git`
- `pixi`
- `uv`
- `direnv`（可选）

### `pixi install` / `uv sync` 会自动安装的内容

- `pixi install`
  - Python 3.12 运行时
  - GCC / G++ / Make / CMake
  - `cuda-runtime`、`cudnn`
  - `ascend-cann-toolkit`、`ascend-cann-310b-ops`
- `uv sync`
  - `torch`
  - `onnx`
  - `onnxruntime-gpu`
  - `torch-pruning`
  - 以及 `pyproject.toml` 中声明的其余 Python 依赖

### 条件性宿主机要求

- 若要使用 CUDA 加速训练，宿主机需要可用的 NVIDIA GPU 与驱动。
- 若要做真实的 Ascend 编译或部署验证，宿主机需要对应的 Ascend 设备/驱动环境。

### 公共环境层

推荐先在项目根目录执行：

```bash
pixi install
uv sync
direnv allow
```

其中 [`.envrc`](../.envrc) 当前只负责：

- `REPO_ROOT`
- `PYTHONPATH=$REPO_ROOT/src`

各阶段如需额外环境变量，再按需 source `scripts/load_*_env.sh`。

### 阶段专用手动准备

AMCT CLI 额外依赖仓库自带组件：

- `amct_onnx/amct_onnx-0.23.2-py3-none-linux_x86_64.whl`
- `amct_onnx/amct_onnx_op.tar.gz`

这两项不在 `uv sync` / `pixi install` 自动安装范围内，运行 AMCT 前需按目标环境自行安装或部署。

## 基座模型 CLI

### 入口

```bash
uv run src/base_model_main.py --help
```

### 核心参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--epochs` | `60` | 训练轮数 |
| `--lr` | `0.0003` | 学习率 |
| `--batch_size` | `64` | 批次大小 |
| `--model_path` | `best_model.pth` | 模型保存文件名 |
| `--class_num` | `24` | 分类数 |
| `--model` | `resnet6_2d` | 模型名 |
| `--data_dir` | `Data` | 数据集路径 |
| `--data_dtype` | `fp16` | 数据集输出 tensor 精度 |

### 数据加载与性能参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--full_load` | `False` | 是否全量加载数据集 |
| `--num_workers` | `None` | DataLoader 工作线程数 |
| `--prefetch_factor` | `2` | DataLoader 预取因子 |
| `--persistent_workers` | `True` | 是否保持工作线程 |
| `--pin_memory` | `True` | 是否启用 `pin_memory` |
| `--cudnn_benchmark` | `True` | 是否启用 cuDNN benchmark |
| `--cudnn_deterministic` | `False` | 是否启用确定性算法 |
| `--compile_model` | `True` | 是否启用 `torch.compile` |
| `--compile_mode` | `default` | 编译模式 |

### 训练行为参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--Train` | `True` | 是否执行训练 |
| `--Test` | `True` | 是否执行测试 |
| `--UMAP` | `False` | 是否执行 UMAP |
| `--dropout_p` | `0.3` | Dropout 概率 |
| `--weight_decay` | `1e-4` | 权重衰减 |
| `--warmup_ratio` | `0.05` | Warmup 占总步数比例 |
| `--warmup_steps` | `0` | Warmup 步数 |
| `--min_lr` | `1e-6` | 最小学习率 |
| `--plot_lr_schedule` | `True` | 是否绘制学习率曲线 |

### 示例

```bash
uv run src/base_model_main.py --epochs 100 --batch_size 64 --model resnet18_2d
```

## 剪枝 CLI

### 入口

```bash
uv run src/pruning_main.py --help
```

完整实现见 [src/pruning/args.py](../src/pruning/args.py)。

### 核心参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | 必填 | 基座模型名，自动解析 `output/base_model/<model>/best_model.pth` |
| `--model_path` | `best_pruned_model.pth` | 最终剪枝模型文件名 |
| `--data_dir` | `Data` | 数据集路径 |
| `--data_dtype` | `fp16` | 数据集输出 tensor 精度 |
| `--full_load` | `False` | 是否全量加载数据集 |
| `--num_workers` | `None` | DataLoader 工作线程数 |
| `--prefetch_factor` | `2` | DataLoader 预取因子 |
| `--persistent_workers` | `True` | 是否保持 DataLoader 工作线程 |
| `--pin_memory` | `True` | 是否启用 `pin_memory` |
| `--pruning_ratio` | `0.30` | 最终总剪枝率，统一规范到 2 位小数 |
| `--pruning_steps` | `5` | iterative pruning 剪枝轮数 |
| `--global_pruning` | `True` | 是否启用全局剪枝 |
| `--ignore_fc` | `True` | 是否忽略分类头 |
| `--finetune_epochs` | `10` | 每轮剪枝后微调轮数 |
| `--batch_size` | `64` | 批次大小 |
| `--lr` | `1e-4` | 微调学习率 |
| `--weight_decay` | `1e-4` | 权重衰减 |
| `--warmup_ratio` | `0.05` | Warmup 占总步数比例 |
| `--warmup_steps` | `0` | Warmup 步数 |
| `--min_lr` | `1e-7` | 最小学习率 |
| `--cudnn_benchmark` | `True` | 是否启用 cuDNN benchmark |
| `--cudnn_deterministic` | `False` | 是否启用 cuDNN 确定性算法 |
| `--evaluate_test` | `True` | 是否在最终阶段评估测试集 |

### 特点

- pruning 不手动接收基座 checkpoint 路径，而是依赖 `best_model.pth` 符号链接
- 产物用于后续 QAT / ONNX 恢复

### 示例

```bash
uv run src/pruning_main.py --model resnet34_2d --pruning_ratio 0.80 --pruning_steps 8
```

## QAT CLI

### 入口

```bash
uv run src/qat_main.py --help
```

完整实现见 [src/qat/args.py](../src/qat/args.py)。

### 核心参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--pruning_checkpoint` | 必填 | 输入 pruning checkpoint 路径 |
| `--model_path` | `best_qat_prepare_model.pth` | QAT prepare 模型文件名 |
| `--data_dir` | `Data` | 数据集路径 |
| `--full_load` | `False` | 是否全量加载数据集 |
| `--num_workers` | `None` | DataLoader 工作线程数 |
| `--prefetch_factor` | `2` | DataLoader 预取因子 |
| `--persistent_workers` | `True` | 是否保持 DataLoader 工作线程 |
| `--pin_memory` | `True` | 是否启用 `pin_memory` |
| `--qat_epochs` | `10` | QAT 微调轮数 |
| `--batch_size` | `64` | 批次大小 |
| `--lr` | `1e-5` | QAT 微调学习率 |
| `--weight_decay` | `1e-4` | 权重衰减 |
| `--warmup_ratio` | `0.05` | Warmup 占总步数比例 |
| `--warmup_steps` | `0` | Warmup 步数 |
| `--min_lr` | `1e-7` | 最小学习率 |
| `--cudnn_benchmark` | `True` | 是否启用 cuDNN benchmark |
| `--cudnn_deterministic` | `False` | 是否启用 cuDNN 确定性算法 |
| `--evaluate_test` | `True` | 是否在最终阶段评估测试集 |

### 特点

- QAT 固定纯 `fp32`
- 当前只落 prepare checkpoint，不在本阶段做 `torch.convert`
- QAT checkpoint 使用最小 `quantization_meta` 契约，供后续 ONNX 恢复

### 示例

```bash
uv run src/qat_main.py \
  --pruning_checkpoint output/pruning/resnet18_2d/ratio0.60_steps8_global_ft10_bs64/best_pruned_model.pth
```

## ONNX CLI

### 入口

```bash
uv run src/onnx_main.py --help
```

### 核心参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--branch` | 必填 | `pruning_fp16` 或 `qat_convert` |
| `--checkpoint` | 必填 | pruning 分支传 pruning checkpoint；QAT 分支传 QAT checkpoint |
| `--data_dir` | `Data` | 数据集路径 |
| `--full_load` | `False` | 是否全量加载数据集 |
| `--num_workers` | `None` | 数据加载工作线程数 |
| `--evaluate_test` | `True` | 是否在导出后执行测试集精度评估 |
| `--eval_batch_size` | `64` | Torch / ORT 评估 batch size，不影响导出图结构 |
| `--opset_version` | `16` | 当前固定 ONNX opset 16 |

### 分支说明

- `pruning_fp16`
  - pruning checkpoint -> FP16 ONNX
- `qat_convert`
  - QAT checkpoint -> `convert_fx` -> quantized ONNX

### 当前约束

- 导出使用动态 batch
- `--eval_batch_size` 仅影响评估
- `qat_convert` 导出后还会执行 graph rewrite 与 validate

### 示例

```bash
uv run src/onnx_main.py \
  --branch pruning_fp16 \
  --checkpoint output/pruning/resnet10_2d/ratio0.40_steps5_global_ft10_bs64/best_pruned_model.pth \
  --eval_batch_size 64

uv run src/onnx_main.py \
  --branch qat_convert \
  --checkpoint output/qat/resnet10_2d/from_ratio0.40_steps5_global_ft10_bs64/best_qat_prepare_model.pth \
  --eval_batch_size 64
```

## AMCT CLI

### 入口

```bash
uv run src/amct_main.py --help
```

### 核心参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--onnx_model` | 必填 | 输入的 `qat_convert` ONNX 路径，固定为仓库导出的 `model_quant.onnx` |

### 阶段附加准备

运行前请先按目标环境自行安装或部署仓库附带的：

- `amct_onnx/amct_onnx-0.23.2-py3-none-linux_x86_64.whl`
- `amct_onnx/amct_onnx_op.tar.gz`

### 输入契约

- 只接受仓库 `onnx_main.py --branch qat_convert` 导出的 `model_quant.onnx`
- 同目录必须存在 `onnx_summary.json`
- `onnx_summary.json.branch` 必须为 `qat_convert`

### 输出产物

- `deploy_model.onnx`
- `fake_quant_model.onnx`
- `scale_offset_record.txt`
- `amct_summary.json`

### 示例

```bash
uv run src/amct_main.py \
  --onnx_model output/onnx/qat_convert/resnet6_2d/from_ratio0.60_steps8_global_ft10_bs64/model_quant.onnx
```

## ATC CLI

### 入口

```bash
pixi run python src/atc_main.py --help
```

### 核心参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--branch` | 必填 | `pruning_fp16` 或 `amct_deploy` |
| `--onnx_model` | 必填 | pruning 分支传 `model_fp16.onnx`，AMCT 分支传 `deploy_model.onnx` |
| `--soc_version` | `Ascend310B4` | 目标芯片版本 |
| `--input_shape` | `input:1,1,543,512` | 静态输入形状 |
| `--input_format` | `NCHW` | 输入格式 |

### 输入契约

- `pruning_fp16`：
  - 输入 `model_fp16.onnx`
  - 同目录必须存在 `onnx_summary.json`
- `amct_deploy`：
  - 输入 `deploy_model.onnx`
  - 同目录必须存在 `amct_summary.json`

### 输出产物

- `.om`
- `atc_summary.json`
- `check_result.json` / `fusion_result.json`（若 ATC 生成）

### 示例

```bash
pixi run python src/atc_main.py \
  --branch pruning_fp16 \
  --onnx_model output/onnx/pruning_fp16/resnet10_2d/from_ratio0.40_steps5_global_ft10_bs64/model_fp16.onnx

pixi run python src/atc_main.py \
  --branch amct_deploy \
  --onnx_model output/amct/resnet6_2d/from_ratio0.60_steps8_global_ft10_bs64/deploy_model.onnx
```
