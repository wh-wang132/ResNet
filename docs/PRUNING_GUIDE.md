# 剪枝指南

## 概述

当前项目已提供基于 `torch-pruning` 的结构化通道剪枝 + 微调框架，支持从基座模型 checkpoint 出发，完成：

1. 按模型名读取基座模型根目录下的 `best_model.pth` 符号链接
2. 恢复默认模型结构并严格加载权重
3. 执行多轮 iterative structured pruning
4. 每轮剪枝后进行微调恢复（可选）
5. 仅在最终轮保存包含权重、拓扑与元数据的剪枝 checkpoint

当前阶段暂不包含 ONNX/ATC 导出，也不包含 QAT。

剪枝阶段当前复用基座模型的 Warmup + Cosine 调度器实现，但默认超参已针对“剪枝后微调恢复”做了下调。

## 工作流

```text
base checkpoint
  -> 恢复基座模型
  -> iterative structured pruning
  -> 每轮提取 channel_cfg / architecture_signature
  -> 每轮微调恢复（可选）
  -> 仅最终轮保存 pruning checkpoint
```

## 当前支持范围

- `torch-pruning` 结构化通道剪枝
- iterative pruning 多轮剪枝
- 基座模型按 `output/base_model/<model>/best_model.pth` 自动解析
- 剪枝后拓扑通过 `channel_cfg` 显式保存
- 剪枝 checkpoint 作为后续 QAT 的上游输入

## 入口命令

项目根目录执行：

```bash
uv run src/pruning_main.py --help
```

## CLI 参数总览

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | 必填 | 基座模型名，将自动解析 `output/base_model/<model>/best_model.pth` |
| `--model_path` | `best_pruned_model.pth` | 剪枝后最佳模型文件名 |
| `--data_dir` | `Data` | 数据集路径 |
| `--data_dtype` | `fp16` | 数据集输出 tensor 精度，可选 `fp16` / `fp32` |
| `--full_load` | `False` | 是否全量加载数据集 |
| `--num_workers` | `None` | DataLoader 工作线程数 |
| `--prefetch_factor` | `2` | DataLoader 预取因子 |
| `--persistent_workers` | `True` | 是否保持 DataLoader 工作线程 |
| `--pin_memory` | `True` | 是否启用 `pin_memory` |
| `--pruning_ratio` | `0.3` | 最终总剪枝率，会按十进制四舍五入规范到 2 位小数 |
| `--pruning_steps` | `5` | iterative pruning 的剪枝轮数 |
| `--global_pruning` | `True` | 是否启用全局剪枝 |
| `--ignore_fc` | `True` | 是否默认忽略分类头 |
| `--finetune_epochs` | `10` | 每轮剪枝后的微调轮数 |
| `--batch_size` | `64` | 批次大小 |
| `--lr` | `1e-4` | 微调学习率 |
| `--weight_decay` | `1e-4` | 权重衰减 |
| `--warmup_ratio` | `0.05` | Warmup 占总步数比例 |
| `--warmup_steps` | `0` | Warmup 步数，`0` 表示使用 `warmup_ratio` |
| `--min_lr` | `1e-7` | 最小学习率 |
| `--cudnn_benchmark` | `True` | 是否启用 cuDNN benchmark |
| `--cudnn_deterministic` | `False` | 是否启用 cuDNN 确定性算法 |
| `--evaluate_test` | `True` | 微调结束后是否评估测试集 |

## 命令示例

### 最小剪枝命令

```bash
uv run src/pruning_main.py --model resnet6_2d
```

### 关闭全局剪枝

```bash
uv run src/pruning_main.py \
  --model resnet18_2d \
  --pruning_ratio 0.25 \
  --pruning_steps 5 \
  --global_pruning False \
  --finetune_epochs 12
```

### 不做微调，只保存剪枝结果

```bash
uv run src/pruning_main.py \
  --model resnet14_2d \
  --finetune_epochs 0 \
  --evaluate_test False
```

## 基座模型符号链接约定

剪枝入口不会手动接收外部 checkpoint 路径，而是固定读取：

```text
output/base_model/<model>/best_model.pth
```

要求：

- 该路径存在
- 若为符号链接，则链接必须可正常解析
- 加载后的 checkpoint 中 `model_structure.model_name` 必须与 `--model` 一致

若上述任一条件不满足，剪枝入口会直接报错退出。

## 剪枝输出目录

```text
output/pruning/<model>/ratio<ratio>_steps<steps>_<global|local>_ft<epochs>_bs<batch_size>/
```

典型产物包括：

- `best_pruned_model.pth`
- `best_pruned_info.txt`（每轮一行最佳摘要）
- `pruning_summary.json`
- `runs/`

## 剪枝 checkpoint 主要字段

- `model_state_dict`
- `model_structure`
  - `model_name`
  - `model_kwargs`
  - `channel_cfg`
  - `architecture_signature`
- `pruning_meta`
  - `pruning_steps`
  - `target_total_ratio`
  - `step_ratio`
  - `global_pruning`
  - `ignored_layers`
  - `example_input_shape`
  - `torch_pruning_version`
  - `params_before / params_after`
  - `macs_before / macs_after`
- `train_context`
  - `checkpoint_link_path`
  - `resolved_checkpoint_path`
- `best_acc`
- `best_val_loss`

## 说明

- 剪枝阶段当前读取的是基座模型 checkpoint，因此模型恢复入口优先使用默认 `load_model_map()`。
- `--pruning_ratio` 的有效精度固定为 2 位小数；summary、checkpoint 和输出目录都会使用同一个规范值。
- 剪枝后的完整拓扑通过实际模型提取，不依赖默认模板反推。
- 多轮剪枝过程中，中间轮的最佳权重只保留在内存中作为下一轮输入，不落盘。
- 后续 QAT 可直接以剪枝 checkpoint 中保存的 `channel_cfg` 和权重为输入继续恢复。
