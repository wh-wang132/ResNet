# 命令行参数详解

## 概述

当前项目包含三套独立 CLI：

- 基座训练入口：`uv run src/base_model_main.py ...`
- 剪枝入口：`uv run src/pruning_main.py ...`
- QAT 入口：`uv run src/qat_main.py ...`

两套入口的参数并不完全相同，阅读文档时需要区分阶段。

## 环境前提

本文档中的所有 `uv run ...` 命令都默认建立在项目标准环境已激活的前提下：

- `pixi`：系统/工具链环境
- `uv`：Python 依赖与运行入口
- `direnv`：推荐自动注入 `pixi shell-hook` 与 `PYTHONPATH`

推荐先在项目根目录完成：

```bash
pixi install
uv sync
direnv allow
```

之后再执行本文中的命令。

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
| `--data_dtype` | `fp32`           | 数据集输出 tensor 精度（QAT 固定 fp32） |

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

当前 pruning 参数定义以 [src/pruning/args.py](/root/ResNet/src/pruning/args.py) 为准，完整流程说明见 [剪枝指南](PRUNING_GUIDE.md)。

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

当前 QAT 参数定义以 [src/qat/args.py](/root/ResNet/src/qat/args.py) 为准，完整流程说明见 [src/qat/README.md](/root/ResNet/src/qat/README.md)。

### 核心参数概览

| 参数                   | 默认值                       | 说明                         |
| ---------------------- | ---------------------------- | ---------------------------- |
| `--pruning_checkpoint` | 必填                         | 输入 pruning checkpoint 路径 |
| `--model_path`         | `best_qat_prepare_model.pth` | QAT prepare 模型文件名       |
| `--data_dir`           | `Data`                       | 数据集路径                   |
| `--data_dtype`         | `fp16`                       | 数据集输出 tensor 精度       |
| `--full_load`          | `False`                      | 是否全量加载数据集           |
| `--qat_epochs`         | `20`                         | QAT 微调轮数                 |
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
- QAT 当前不暴露 qconfig / observer / quant scheme 为 CLI 参数
- QAT 当前只导出 prepare checkpoint，不执行 `torch.convert`

### 示例

```bash
uv run src/qat_main.py --pruning_checkpoint output/pruning/resnet18_2d/ratio0.60_steps8_global_ft10_bs64/best_pruned_model.pth
```
