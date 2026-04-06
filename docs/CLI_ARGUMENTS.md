# 命令行参数详解

## 概述

当前项目包含六套独立 CLI：

- 基座训练入口：`uv run src/base_model_main.py ...`
- 剪枝入口：`uv run src/pruning_main.py ...`
- QAT 入口：`uv run src/qat_main.py ...`
- ONNX 导出入口：`uv run src/onnx_main.py ...`
- AMCT 转换入口：`uv run src/amct_main.py ...`
- ATC 编译入口：`pixi run python src/atc_main.py ...`

各套入口的参数并不完全相同，阅读文档时需要区分阶段。

## 环境前提

本文档中的所有 `uv run ...` / `pixi run ...` 命令都默认建立在项目公共环境层已激活的前提下：

- `pixi`：系统/工具链环境
- `uv`：Python 依赖与运行入口
- `direnv`：推荐自动加载 [`.envrc`](../.envrc)

推荐先在项目根目录完成：

```bash
pixi install
uv sync
direnv allow
```

其中 [`.envrc`](../.envrc) 当前只负责：

- `REPO_ROOT`
- `PYTHONPATH=$REPO_ROOT/src`

若某一阶段需要额外环境变量，再按需 source 对应 `scripts/load_*_env.sh`。当前不再保证脱离 `.envrc` 直接 source 这些脚本。

## 基座模型 CLI

### 入口

```bash
uv run src/base_model_main.py --help
```

### 核心参数

| 参数           | 默认值           | 说明                                    |
| -------------- | ---------------- | --------------------------------------- |
| `--epochs`     | `60`             | 训练轮数                                |
| `--lr`         | `0.0003`         | 学习率                                  |
| `--batch_size` | `64`             | 批次大小                                |
| `--model_path` | `best_model.pth` | 模型保存文件名                          |
| `--class_num`  | `24`             | 分类数                                  |
| `--model`      | `resnet6_2d`     | 模型名                                  |
| `--data_dir`   | `Data`           | 数据集路径                              |
| `--data_dtype` | `fp16`           | 数据集输出 tensor 精度                   |

### 数据加载参数

| 参数                   | 默认值  | 说明                         |
| ---------------------- | ------- | ---------------------------- |
| `--full_load`          | `False` | 是否全量加载数据集           |
| `--num_workers`        | `None`  | DataLoader 工作线程数        |
| `--prefetch_factor`    | `2`     | DataLoader 预取因子          |
| `--persistent_workers` | `True`  | 是否保持 DataLoader 工作线程 |
| `--pin_memory`         | `True`  | 是否启用 `pin_memory`        |

### 性能与训练行为参数

| 参数                    | 默认值    | 说明                     |
| ----------------------- | --------- | ------------------------ |
| `--cudnn_benchmark`     | `True`    | 是否启用 cuDNN benchmark |
| `--cudnn_deterministic` | `False`   | 是否启用确定性算法       |
| `--compile_model`       | `True`    | 是否启用 `torch.compile` |
| `--compile_mode`        | `default` | 编译模式                 |
| `--Train`               | `True`    | 是否执行训练             |
| `--Test`                | `True`    | 是否执行测试             |
| `--UMAP`                | `False`   | 是否执行 UMAP 可视化     |
| `--dropout_p`           | `0.3`     | Dropout 概率             |
| `--weight_decay`        | `1e-4`    | 权重衰减                 |
| `--warmup_ratio`        | `0.05`    | Warmup 占总步数比例      |
| `--warmup_steps`        | `0`       | Warmup 步数              |
| `--min_lr`              | `1e-6`    | 最小学习率               |
| `--plot_lr_schedule`    | `True`    | 是否绘制学习率曲线       |

### 示例

```bash
uv run src/base_model_main.py --epochs 100 --batch_size 64 --model resnet18_2d
```

## 剪枝 CLI

### 入口

```bash
uv run src/pruning_main.py --help
```

当前 pruning 参数定义以 [src/pruning/args.py](../src/pruning/args.py) 为准，完整流程说明见 [剪枝指南](PRUNING_GUIDE.md)。

### 核心参数概览

| 参数                | 默认值                  | 说明                                                              |
| ------------------- | ----------------------- | ----------------------------------------------------------------- |
| `--model`           | 必填                    | 基座模型名，将自动解析 `output/base_model/<model>/best_model.pth` |
| `--model_path`      | `best_pruned_model.pth` | 最终剪枝模型文件名                                                |
| `--data_dir`        | `Data`                  | 数据集路径                                                        |
| `--data_dtype`      | `fp16`                  | 数据集输出 tensor 精度                                            |
| `--full_load`       | `False`                 | 是否全量加载数据集                                                |
| `--pruning_ratio`   | `0.30`                  | 最终总剪枝率，入口会规范到 2 位小数                               |
| `--pruning_steps`   | `5`                     | iterative pruning 的剪枝轮数                                      |
| `--global_pruning`  | `True`                  | 是否启用全局剪枝                                                  |
| `--ignore_fc`       | `True`                  | 是否忽略分类头                                                    |
| `--finetune_epochs` | `10`                    | 每轮剪枝后的微调轮数                                              |
| `--batch_size`      | `64`                    | 批次大小                                                          |
| `--lr`              | `1e-4`                  | 微调学习率                                                        |
| `--weight_decay`    | `1e-4`                  | 权重衰减                                                          |
| `--warmup_ratio`    | `0.05`                  | Warmup 占总步数比例                                               |
| `--warmup_steps`    | `0`                     | Warmup 步数                                                       |
| `--min_lr`          | `1e-7`                  | 最小学习率                                                        |
| `--evaluate_test`   | `True`                  | 是否在最终阶段评估测试集                                          |

### 与基座 CLI 的差异

- pruning 不提供 `--class_num`
- pruning 不提供 `--Train / --Test / --UMAP`
- pruning 不手动接收基座 checkpoint 路径，而是按 `--model` 自动解析符号链接
- pruning 的 `--pruning_ratio` 是 2 位小数权威值，并会同步体现在输出目录、summary 和 checkpoint 中

### 示例

```bash
uv run src/pruning_main.py --model resnet34_2d --pruning_ratio 0.80 --pruning_steps 8
```

## QAT CLI

### 入口

```bash
uv run src/qat_main.py --help
```

当前 QAT 参数定义以 [src/qat/args.py](../src/qat/args.py) 为准，完整流程说明见 [src/qat/README.md](../src/qat/README.md)。

### 核心参数概览

| 参数                   | 默认值                       | 说明                         |
| ---------------------- | ---------------------------- | ---------------------------- |
| `--pruning_checkpoint` | 必填                         | 输入 pruning checkpoint 路径 |
| `--model_path`         | `best_qat_prepare_model.pth` | QAT prepare 模型文件名       |
| `--data_dir`           | `Data`                       | 数据集路径                   |
| `--full_load`          | `False`                      | 是否全量加载数据集           |
| `--qat_epochs`         | `10`                         | QAT 微调轮数                 |
| `--batch_size`         | `64`                         | 批次大小                     |
| `--lr`                 | `1e-5`                       | 保守 QAT 微调学习率          |
| `--weight_decay`       | `1e-4`                       | 权重衰减                     |
| `--warmup_ratio`       | `0.05`                       | Warmup 占总步数比例          |
| `--warmup_steps`       | `0`                          | Warmup 步数                  |
| `--min_lr`             | `1e-7`                       | 最小学习率                   |
| `--evaluate_test`      | `True`                       | 是否在最终阶段评估测试集     |

### 与 pruning CLI 的差异

- QAT 不再接收 `--model`，而是直接接收 `--pruning_checkpoint`
- QAT 不负责结构化剪枝，只负责 `prepare_qat_fx` 后的保守单路径微调
- QAT 当前固定纯 `fp32` 数据链，不再暴露 `data_dtype`
- QAT 当前不暴露 qconfig / observer / quant scheme 为 CLI 参数
- QAT 当前只导出 prepare checkpoint，不执行 `torch.convert`
- 未来 ONNX/导出阶段应消费 QAT checkpoint 恢复接口，而不是重新读取 pruning checkpoint
- 新版 QAT checkpoint 使用最小 `quantization_meta` 契约；旧版 QAT checkpoint 需要重新跑新的 QAT
- `source_pruning_checkpoint_path` 仅用于溯源，不参与 QAT 对象恢复

### 示例

```bash
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet18_2d/ratio0.60_steps8_global_ft10_bs64/best_pruned_model.pth
```

## ONNX CLI

### 入口

```bash
uv run src/onnx_main.py --help
```

### 核心参数概览

| 参数              | 默认值    | 说明                                                        |
| ----------------- | --------- | ----------------------------------------------------------- |
| `--branch`        | 必填      | `pruning_fp16` 或 `qat_convert`                             |
| `--checkpoint`    | 必填      | 输入 checkpoint；pruning 分支传 pruning checkpoint，QAT 分支传 QAT checkpoint |
| `--data_dir`      | `Data`    | 数据集路径                                                  |
| `--full_load`     | `False`   | 是否全量加载数据集                                          |
| `--num_workers`   | `None`    | DataLoader 工作线程数                                       |
| `--evaluate_test` | `True`    | 是否在导出后执行 ORT 测试集精度评估                         |
| `--eval_batch_size` | `64`    | Torch / ORT 精度评估批次大小，仅影响评估，不影响导出图结构  |
| `--opset_version` | `16`      | ONNX opset 版本，当前固定为 16                              |

### 分支说明

- `pruning_fp16`
  - 输入 pruning checkpoint
  - 恢复剪枝后浮点模型
  - 由 Torch 直接导出 FP16 ONNX
- `qat_convert`
  - 输入 QAT checkpoint
  - 通过 `load_qat_checkpoint(...)` 恢复 prepared model
  - `convert_fx` 后导出量化 ONNX

当前 ONNX 导出默认使用动态 batch：

- `onnx_summary.json.example_input_shape` 中的 `batch=1` 仅表示导出样例输入
- 精度评估可通过 `--eval_batch_size` 使用更大的 batch
- 若后续部署需要静态 batch，可在 ATC 阶段使用 `--input_shape="input:1,1,543,512"` 固化

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

### 核心参数概览

| 参数           | 默认值 | 说明 |
| -------------- | ------ | ---- |
| `--onnx_model` | 必填   | 输入的 `qat_convert` ONNX 路径，固定为仓库导出的 `model_quant.onnx` |

### 输入契约说明

- 只接受仓库 `onnx_main.py --branch qat_convert` 产出的 `model_quant.onnx`
- 同目录必须存在 `onnx_summary.json`
- `onnx_summary.json` 中：
  - `branch` 必须为 `qat_convert`
  - `onnx_path` 必须能回指当前输入文件

### 输出产物

- `deploy_model.onnx`
- `fake_quant_model.onnx`
- `scale_offset_record.txt`
- `amct_summary.json`

### 示例

```bash
. scripts/load_amct_env.sh

uv run src/amct_main.py \
  --onnx_model output/onnx/qat_convert/resnet6_2d/from_ratio0.60_steps8_global_ft10_bs64/model_quant.onnx
```

## ATC CLI

### 入口

```bash
pixi run python src/atc_main.py --help
```

### 核心参数概览

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--branch` | 必填 | `pruning_fp16` 或 `amct_deploy` |
| `--onnx_model` | 必填 | 输入 ONNX 路径；pruning 分支传 `model_fp16.onnx`，AMCT 分支传 `deploy_model.onnx` |
| `--soc_version` | `Ascend310B4` | 目标芯片版本 |
| `--input_shape` | `input:1,1,543,512` | 静态输入形状 |
| `--input_format` | `NCHW` | 输入格式 |

### 输入契约说明

- `pruning_fp16`
  - 只接受仓库 `onnx_main.py --branch pruning_fp16` 产出的 `model_fp16.onnx`
  - 同目录必须存在 `onnx_summary.json`
  - `onnx_summary.json.branch` 必须为 `pruning_fp16`
- `amct_deploy`
  - 只接受仓库 `amct_main.py` 产出的 `deploy_model.onnx`
  - 同目录必须存在 `amct_summary.json`
  - `amct_summary.json.deploy_model_path` 必须能回指当前输入文件

### 输出产物

- `model_fp16.om` 或 `deploy_model.om`
- `atc_summary.json`
- `check_result.json` / `fusion_result.json`（若 ATC 生成）

### 示例

```bash
. scripts/load_atc_env.sh

pixi run python src/atc_main.py \
  --branch pruning_fp16 \
  --onnx_model output/onnx/pruning_fp16/resnet10_2d/from_ratio0.40_steps5_global_ft10_bs64/model_fp16.onnx

pixi run python src/atc_main.py \
  --branch amct_deploy \
  --onnx_model output/amct/resnet6_2d/from_ratio0.60_steps8_global_ft10_bs64/deploy_model.onnx
```
