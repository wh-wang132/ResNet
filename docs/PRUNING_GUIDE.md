# 剪枝指南

## 概述

当前项目已提供基于 `torch-pruning` 的 iterative structured pruning + 微调框架。该阶段以基座模型 checkpoint 为输入，输出 pruning checkpoint，供后续 QAT / ONNX 阶段恢复使用。

当前 pruning 阶段负责：

1. 按模型名读取基座模型符号链接
2. 恢复默认模型并严格加载基座权重
3. 执行多轮 iterative pruning
4. 每轮进行验证与可选微调
5. 仅最终轮保存 pruning checkpoint

当前 pruning 阶段不负责读取 pruning checkpoint 并恢复模型；该职责由后续 QAT / ONNX 模块承担。

## 环境前提

### 需要用户手动安装的项目

- `git`
- `pixi`
- `uv`
- `direnv`（可选）

### 自动安装的内容

- `pixi install`
  - Python 3.12 运行时
  - GCC / G++ / Make / CMake
  - CUDA runtime、cuDNN、CANN toolkit 等工具链内容
- `uv sync`
  - `torch`
  - `torch-pruning`
  - 以及其余 Python 包依赖

### 推荐初始化顺序

```bash
pixi install
uv sync
direnv allow
```

其中 [`.envrc`](../.envrc) 当前只负责导出：

- `REPO_ROOT`
- `PYTHONPATH=$REPO_ROOT/src`

pruning 阶段只依赖这层公共环境，不需要额外 `load_*_env.sh`。

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
- checkpoint 中的 `model_structure.model_name` 必须与命令行 `--model` 一致

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

## `pruning_summary.json` 当前结构

顶层摘要包括：

- `model_name`
- `pruning_steps`
- `labels`
- `baseline`
- `rounds`
- `pruning_meta`
- `finetune_summary`
- `final`
- `final_topology`
- `checkpoint_link_path`
- `resolved_checkpoint_path`

其中：

- `rounds[*].before_finetune.topology` 保留每轮剪枝过程拓扑
- `final_topology` 保留最终模型拓扑快照
- 顶层 `pruning_meta` 是最终轮的紧凑摘要

## pruning checkpoint 当前结构

`best_pruned_model.pth` 主要字段包括：

- `model_state_dict`
- `epoch`
- `best_acc`
- `best_val_loss`
- `train_context`
- `model_structure`
- `pruning_meta`

其中 `model_structure` 保存：

- `model_name`
- `model_class`
- `model_kwargs`
- `include_top`
- `in_channels`
- `init_channels`
- `input_tensor_meta`
- `channel_cfg`
- `architecture_signature`
