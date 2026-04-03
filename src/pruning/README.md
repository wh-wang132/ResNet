# Pruning

该目录包含基于 `torch-pruning` 的结构化剪枝 + 微调实现。

推荐从项目根目录执行：

```bash
uv run src/pruning_main.py --help
```

默认前提：当前 shell 已处于项目标准 `pixi + uv` 环境中。推荐直接在项目根目录通过 `.envrc` 自动激活；若未使用 `direnv`，则应先手动进入 `pixi` 环境后再执行 `uv run ...`。

## 当前阶段定位

当前 pruning 阶段负责：

- 从 `output/base_model/<model>/best_model.pth` 恢复基座模型
- 执行 iterative structured pruning
- 每轮进行验证与可选微调
- 仅最终轮保存 pruning checkpoint

当前 pruning 阶段**不负责**恢复 pruning checkpoint；后续恢复入口将由 QAT / ONNX 阶段负责。

## 当前输入约定

基座模型来源固定为：

```text
output/base_model/<model>/best_model.pth
```

这里的 `best_model.pth` 应是你在对应基座模型根目录下维护的最佳权重符号链接。

## 当前输出约定

```text
output/pruning/<model>/ratio<ratio>_steps<steps>_<global|local>_ft<epochs>_bs<batch_size>/
```

典型产物：

- `best_pruned_model.pth`
- `best_pruned_info.txt`
- `pruning_summary.json`
- `Confusion_matrix.png`（仅最终测试阶段生成）
- `runs/round_<n>/`

## 关键产物语义

### `best_pruned_info.txt`

- 每轮微调结束后追加一行
- 每行记录该轮最佳验证结果

### `pruning_summary.json`

当前顶层结构为：

- `baseline`
- `rounds`
- `pruning_meta`
- `finetune_summary`
- `final`
- `final_topology`
- `checkpoint_link_path`
- `resolved_checkpoint_path`

### pruning checkpoint

当前主要字段为：

- `model_state_dict`
- `model_structure`
- `pruning_meta`
- `train_context`
- `best_acc`
- `best_val_loss`

其中 `model_structure` 会保存未来恢复所需的：

- `model_name`
- `model_kwargs`
- `channel_cfg`
- `architecture_signature`
