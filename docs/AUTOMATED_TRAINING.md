# 自动化训练使用指南

## 概述

当前仓库的自动化训练入口是项目根目录的 `autorun.sh`，用于批量运行不同模型和批次大小组合。

> 说明：历史文档中提到的 `automated_training.py`、`quick_test.py`、`run_automated_training.sh` 当前仓库未提供。

## 快速开始

在项目根目录执行：

```bash
bash autorun.sh
```

脚本内部调用统一训练入口：

```bash
uv run src/base_model_main.py ...
```

## 当前脚本覆盖范围

- 模型：`resnet6_2d` / `resnet10_2d` / `resnet14_2d` / `resnet18_2d` / `resnet34_2d` / `resnet50_2d`
- 批次大小：`32` / `64` / `128`
- 每个模型对应预设训练轮数（见 `autorun.sh`）

## 如何自定义

直接编辑 `autorun.sh` 中的参数组合：

- 调整 `--epochs`
- 调整 `--batch_size`
- 调整 `--model`
- 增加或移除命令行参数（如 `--full_load`、`--compile_model`）

## 输出位置

训练输出默认写入：

```text
output/base_model/<model>_<dataset>/<config>/
```

典型文件包括：

- `best_model.pth`
- `best_val_acc_info.txt`
- `training_curves.png`
- `lr_schedule.png`
- `Confusion matrix.png`
- `runs/`（TensorBoard 日志）

## 注意事项

1. 批量实验耗时较长，请先用单条命令验证环境。
2. 如遇显存不足，优先降低 `--batch_size` 或切换更轻量模型。
3. 如仅需训练可设置 `--Test False`，减少评估开销。
