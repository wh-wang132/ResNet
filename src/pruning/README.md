# Pruning

该目录包含基于 `torch-pruning` 的结构化剪枝 + 微调实现。

推荐从项目根目录执行：

```bash
uv run src/pruning_main.py --help
```

当前入口按 `--model` 自动读取：

```text
output/base_model/<model>/best_model.pth
```

这里的 `best_model.pth` 应是你在对应基座模型根目录下建立的最佳权重符号链接。
`--pruning_ratio` 会在入口按十进制四舍五入规范到 2 位小数，并贯穿输出目录、summary 和 checkpoint。

## 模块说明

- `args.py`
  - pruning CLI 参数解析
- `checkpoint.py`
  - 基座模型符号链接解析、checkpoint 读取与默认模型恢复
- `evaluator.py`
  - 剪枝前后指标与参数量/MACs 统计
- `output.py`
  - 输出目录与摘要保存
- `pruner.py`
  - `torch-pruning` 封装
- `topology.py`
  - 剪枝后 `channel_cfg` 与结构签名提取
- `trainer.py`
  - 剪枝后微调训练与最佳模型保存
- `utils.py`
  - 剪枝阶段公共工具统一入口

## 当前流程

```text
基座模型根目录下的 best_model.pth 符号链接
  -> 恢复默认模型
  -> iterative structured pruning
  -> 每轮评估 / 微调恢复（可选）
  -> 仅最终轮保存 pruning checkpoint
```
