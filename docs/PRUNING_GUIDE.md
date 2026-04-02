# 剪枝指南

## 概述

当前项目已提供基于 `torch-pruning` 的 iterative structured pruning + 微调框架。该阶段以基座模型 checkpoint 为输入，输出 pruning checkpoint，供后续 QAT / ONNX 阶段恢复使用。

当前 pruning 阶段负责：

1. 按模型名读取基座模型符号链接
2. 恢复默认模型并严格加载基座权重
3. 执行多轮 iterative pruning
4. 每轮进行验证与可选微调
5. 仅最终轮保存 pruning checkpoint

当前 pruning 阶段**不负责**读取 pruning checkpoint 并恢复模型；该职责留给后续 QAT / ONNX 模块。

## 工作流

```text
output/base_model/<model>/best_model.pth
  -> 恢复基座模型
  -> iterative structured pruning
  -> 每轮提取 topology(channel_cfg + architecture_signature)
  -> 每轮微调恢复（可选）
  -> 仅最终轮保存 pruning checkpoint
```

## 基座模型来源约定

剪枝入口固定读取：

```text
output/base_model/<model>/best_model.pth
```

要求：

- 路径存在
- 若为符号链接，必须能正常解析
- checkpoint 内的 `model_structure.model_name` 必须与命令行 `--model` 一致

## CLI 参数总览

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | 必填 | 基座模型名 |
| `--model_path` | `best_pruned_model.pth` | 最终剪枝模型文件名 |
| `--data_dir` | `Data` | 数据集路径 |
| `--data_dtype` | `fp16` | 数据集输出 tensor 精度 |
| `--full_load` | `False` | 是否全量加载数据集 |
| `--num_workers` | `None` | DataLoader 工作线程数 |
| `--prefetch_factor` | `2` | DataLoader 预取因子 |
| `--persistent_workers` | `True` | 是否保持 DataLoader 工作线程 |
| `--pin_memory` | `True` | 是否启用 `pin_memory` |
| `--pruning_ratio` | `0.30` | 最终总剪枝率，统一规范到 2 位小数 |
| `--pruning_steps` | `5` | iterative pruning 的剪枝轮数 |
| `--global_pruning` | `True` | 是否启用全局剪枝 |
| `--ignore_fc` | `True` | 是否忽略分类头 |
| `--finetune_epochs` | `10` | 每轮剪枝后的微调轮数 |
| `--batch_size` | `64` | 批次大小 |
| `--lr` | `1e-4` | 微调学习率 |
| `--weight_decay` | `1e-4` | 权重衰减 |
| `--warmup_ratio` | `0.05` | Warmup 占总步数比例 |
| `--warmup_steps` | `0` | Warmup 步数 |
| `--min_lr` | `1e-7` | 最小学习率 |
| `--cudnn_benchmark` | `True` | 是否启用 cuDNN benchmark |
| `--cudnn_deterministic` | `False` | 是否启用 cuDNN 确定性算法 |
| `--evaluate_test` | `True` | 是否在最终阶段评估测试集 |

## 命令示例

### 最小剪枝命令

```bash
uv run src/pruning_main.py --model resnet6_2d
```

### 指定总剪枝率与轮数

```bash
uv run src/pruning_main.py \
  --model resnet18_2d \
  --pruning_ratio 0.30 \
  --pruning_steps 5 \
  --global_pruning True \
  --finetune_epochs 10
```

### 不做微调，仅保存最终剪枝结果

```bash
uv run src/pruning_main.py \
  --model resnet14_2d \
  --finetune_epochs 0 \
  --evaluate_test False
```

## 输出目录

```text
output/pruning/<model>/ratio<ratio>_steps<steps>_<global|local>_ft<epochs>_bs<batch_size>/
```

典型产物：

- `best_pruned_model.pth`
- `best_pruned_info.txt`
- `pruning_summary.json`
- `Confusion_matrix.png`（仅最终测试阶段生成）
- `runs/round_<n>/`

## `best_pruned_info.txt`

该文件不是“只记录最终轮”，而是：

- 每轮微调结束后追加一行
- 每行记录该轮的：
  - `round`
  - `best_val_acc`
  - `best_val_loss`
  - `best_epoch`

若 `finetune_epochs=0`，则不会生成该文件。

## `pruning_summary.json` 当前结构

顶层摘要当前包括：

- `model_name`
- `pruning_steps`
- `labels`
- `baseline`
  - `val`
  - `test`
  - `stats`
- `rounds`
- `pruning_meta`
- `finetune_summary`
- `final`
  - `val`
  - `test`
  - `stats`
- `final_topology`
- `checkpoint_link_path`
- `resolved_checkpoint_path`

其中：

- `rounds[*].before_finetune.topology` 保留每轮剪枝后的过程拓扑
- `final_topology` 保留最终模型拓扑快照
- 顶层 `pruning_meta` 是最终轮的紧凑摘要

## pruning checkpoint 当前结构

`best_pruned_model.pth` 目前保存的主要字段为：

- `model_state_dict`
- `epoch`
- `best_acc`
- `best_val_loss`
- `train_context`
- `model_structure`
- `pruning_meta`

### `model_structure`

- `model_name`
- `model_class`
- `model_kwargs`
- `include_top`
- `in_channels`
- `init_channels`
- `input_tensor_meta`
- `channel_cfg`
- `architecture_signature`

### `pruning_meta`

- `step_index`
- `pruning_steps`
- `step_ratio`
- `target_total_ratio`
- `global_pruning`
- `ignored_layers`
- `example_input_shape`
- `torch_pruning_version`
- `params_before`
- `params_after`
- `macs_before`
- `macs_after`

### `train_context`

- `stage`
- `checkpoint_link_path`
- `resolved_checkpoint_path`
- `model_name`
- `round_index`
- `class_num`
- `finetune_epochs`
- `batch_size`
- `lr`
- `weight_decay`
- `warmup_ratio`
- `warmup_steps`
- `min_lr`
- `data_dtype`
- `full_load`

## 当前阶段边界

- pruning 模块只负责产出 pruning checkpoint
- pruning checkpoint 的恢复入口将由后续 QAT / ONNX 阶段实现
- 因此当前 pruning checkpoint 已经包含：
  - 可定位基座类定义的 `model_name / model_kwargs`
  - 可恢复剪枝后拓扑的 `channel_cfg`
  - 可校验结构一致性的 `architecture_signature`
  - 最终权重 `model_state_dict`

## 补充说明

- `--pruning_ratio` 的有效精度固定为 2 位小数；输出目录、summary 与 checkpoint 使用同一规范值。
- pruning 当前复用基座模型的 Warmup + Cosine 调度器实现，但默认学习率已下调到更适合微调恢复的 `1e-4`。
- 只有最终测试阶段才会生成 `Confusion_matrix.png`；baseline 和中间轮验证不会生成混淆矩阵。
