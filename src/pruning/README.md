# Pruning

该目录包含基于 `torch-pruning` 的结构化剪枝 + 微调实现。

推荐从项目根目录执行：

```bash
uv run src/pruning_main.py --help
```

## 环境说明

运行 pruning 阶段前，用户只需要独立手动安装：

- `git`
- `pixi`
- `uv`
- `direnv`（可选）

随后执行：

```bash
pixi install
uv sync
direnv allow
```

说明：

- Python 运行时、工具链、CUDA runtime 等由 `pixi install` 自动提供
- Python 包依赖由 `uv sync` 自动提供
- pruning 阶段本身只依赖公共环境层，不需要额外 `load_*_env.sh`
- README 默认从仓库根执行；若结合 `autorun/autorun_pruning.sh` 使用，必须先激活 `.envrc` 以提供 `REPO_ROOT`
- `direnv` 为推荐方案；若不使用 `direnv` 自动激活，也必须手动提供与 `.envrc` 等价的环境变量

## 阶段定位

pruning 阶段负责：

- 自动扫描 `output/base_model/<model>/` 下实验目录并选择最佳 `best_model.pth` 恢复基座模型
- 执行 iterative structured pruning
- 每轮进行验证与可选微调
- 仅最终轮保存 pruning checkpoint

不负责：

- 恢复 pruning checkpoint
- 导出 ONNX
- 执行量化训练

## 输入约定

基座模型来源为：

```text
output/base_model/<model>/<experiment_dir>/best_model.pth
```

pruning 会遍历 `output/base_model/<model>/` 下所有直接子目录，读取每个目录 `best_val_acc_info.txt` 的最后一条有效记录，并按 `val acc` 优先、`val loss` 次优自动选择最佳实验权重。

## 输出约定

```text
output/pruning/<model>/ratio<ratio>_steps<steps>_<global|local>_ft<epochs>_bs<batch_size>/
```

典型产物：

- `best_pruned_model.pth`
- `best_pruned_info.txt`
- `pruning_summary.json`
- `Confusion_matrix.png`
- `runs/round_<n>/`

## 关键产物语义

### `best_pruned_info.txt`

- 每轮微调结束后追加一行
- 每行记录该轮最佳验证结果

### `pruning_summary.json`

顶层结构包括：

- `baseline`
- `rounds`
- `pruning_meta`
- `finetune_summary`
- `final`
- `final_topology`
- `checkpoint_link_path`
- `resolved_checkpoint_path`

### pruning checkpoint

主要字段包括：

- `model_state_dict`
- `model_structure`
- `pruning_meta`
- `train_context`
- `best_acc`
- `best_val_loss`

其中 `model_structure` 保存未来恢复所需的：

- `model_name`
- `model_kwargs`
- `channel_cfg`
- `architecture_signature`

说明：

- 后续所有直接消费 pruning checkpoint 的 `.pth` 链路，都会对 `architecture_signature` 执行强校验
- 后续 ONNX / AMCT / ATC 阶段统一通过对应 summary 读取并校验上游签名与来源信息
