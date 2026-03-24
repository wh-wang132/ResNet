# 剪枝指南

## 概述

当前项目已提供基于 `torch-pruning` 的结构化通道剪枝 + 微调框架，支持从基座模型 checkpoint 出发，完成：

1. 读取基座模型 checkpoint
2. 恢复默认模型结构并严格加载权重
3. 执行一次结构化通道剪枝
4. 评估剪枝后的验证集表现
5. 进行微调恢复（可选）
6. 保存包含权重、拓扑与元数据的剪枝 checkpoint

当前阶段暂不包含 ONNX/ATC 导出，也不包含 QAT。

## 工作流

```text
base checkpoint
  -> 恢复基座模型
  -> torch-pruning 结构化通道剪枝
  -> 提取剪枝后 channel_cfg / architecture_signature
  -> 微调恢复（可选）
  -> 保存 pruning checkpoint
```

## 当前支持范围

- `torch-pruning` 结构化通道剪枝
- 基座 checkpoint 作为输入
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
| `--base_checkpoint` | 必填 | 基座模型 checkpoint 路径 |
| `--model` | None | 可选：显式指定模型名，用于与 checkpoint 中记录值做一致性校验 |
| `--model_path` | `best_pruned_model.pth` | 剪枝后最佳模型文件名 |
| `--data_dir` | `Data` | 数据集路径 |
| `--class_num` | `24` | 分类数 |
| `--data_dtype` | `fp16` | 数据集输出 tensor 精度，可选 `fp16` / `fp32` |
| `--full_load` | `False` | 是否全量加载数据集 |
| `--num_workers` | `None` | DataLoader 工作线程数 |
| `--prefetch_factor` | `2` | DataLoader 预取因子 |
| `--persistent_workers` | `True` | 是否保持 DataLoader 工作线程 |
| `--pin_memory` | `True` | 是否启用 `pin_memory` |
| `--pruning_ratio` | `0.3` | 结构化通道剪枝比例 |
| `--global_pruning` | `True` | 是否启用全局剪枝 |
| `--ignore_fc` | `True` | 是否默认忽略分类头 |
| `--finetune_epochs` | `10` | 剪枝后微调轮数 |
| `--batch_size` | `64` | 批次大小 |
| `--lr` | `3e-4` | 微调学习率 |
| `--weight_decay` | `1e-4` | 权重衰减 |
| `--warmup_ratio` | `0.05` | Warmup 占总步数比例 |
| `--warmup_steps` | `0` | Warmup 步数，`0` 表示使用 `warmup_ratio` |
| `--min_lr` | `1e-6` | 最小学习率 |
| `--cudnn_benchmark` | `True` | 是否启用 cuDNN benchmark |
| `--cudnn_deterministic` | `False` | 是否启用 cuDNN 确定性算法 |
| `--evaluate_test` | `True` | 微调结束后是否评估测试集 |

## 命令示例

### 最小剪枝命令

```bash
uv run src/pruning_main.py \
  --base_checkpoint output/base_model/resnet6_2d/epochs20_bs64/best_model.pth
```

### 关闭全局剪枝

```bash
uv run src/pruning_main.py \
  --base_checkpoint output/base_model/resnet18_2d/epochs20_bs64/best_model.pth \
  --pruning_ratio 0.25 \
  --global_pruning False \
  --finetune_epochs 12
```

### 不做微调，只保存剪枝结果

```bash
uv run src/pruning_main.py \
  --base_checkpoint output/base_model/resnet14_2d/epochs20_bs64/best_model.pth \
  --finetune_epochs 0 \
  --evaluate_test False
```

## 剪枝输出目录

```text
output/pruning/<model>/ratio<ratio>_<global|local>_ft<epochs>_bs<batch_size>/
```

典型产物包括：

- `best_pruned_model.pth`
- `best_pruned_info.txt`
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
  - `pruning_ratio`
  - `global_pruning`
  - `ignored_layers`
  - `example_input_shape`
  - `torch_pruning_version`
  - `params_before / params_after`
  - `macs_before / macs_after`
- `train_context`
- `best_acc`
- `best_val_loss`

## 说明

- 剪枝阶段当前读取的是基座模型 checkpoint，因此模型恢复入口优先使用默认 `load_model_map()`。
- 剪枝后的完整拓扑通过实际模型提取，不依赖默认模板反推。
- 后续 QAT 可直接以剪枝 checkpoint 中保存的 `channel_cfg` 和权重为输入继续恢复。
