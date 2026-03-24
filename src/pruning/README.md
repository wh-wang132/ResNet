# Pruning

该目录包含基于 `torch-pruning` 的结构化剪枝 + 微调实现。

推荐从项目根目录执行：

```bash
uv run src/pruning_main.py --help
```

## 模块说明

- `args.py`
  - pruning CLI 参数解析
- `checkpoint.py`
  - 基座 checkpoint 读取与默认模型恢复
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
基座 checkpoint
  -> 恢复默认模型
  -> 结构化通道剪枝
  -> 评估
  -> 微调恢复（可选）
  -> 保存 pruning checkpoint
```
