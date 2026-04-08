# 训练参数调优指南

## 适用范围

本指南主要适用于：

- `base_model` 基座训练
- `pruning` 阶段的剪枝后微调

不直接适用于：

- `qat`：QAT 采用保守固定方案，重点不在大范围超参数搜索
- `onnx / amct / atc`：这些阶段属于导出与部署链，不属于训练调参阶段

## 基座训练的默认起点

基座训练默认参数为：

- `epochs=60`
- `lr=3e-4`
- `batch_size=64`
- `weight_decay=1e-4`
- `warmup_ratio=0.05`
- `min_lr=1e-6`

对于自动化脚本中的大规模实验，常用起点是：

- `resnet6_2d`：250 epoch
- `resnet10_2d`：200 epoch
- `resnet14_2d`：160 epoch
- `resnet18_2d`：130 epoch
- `resnet34_2d`：100 epoch

## 学习率建议

### 基座训练

| 模型 | 推荐初始学习率 |
| --- | --- |
| `resnet6_2d` / `resnet10_2d` / `resnet14_2d` | `3e-4` 到 `1e-3` |
| `resnet18_2d` / `resnet34_2d` | `1e-4` 到 `3e-4` |

### 剪枝后微调

pruning 默认使用更保守的学习率：

- `lr=1e-4`
- `finetune_epochs=10`

若剪枝率较高，通常优先增加微调轮数，再考虑升学习率。

### QAT

QAT 默认：

- `lr=1e-5`
- `qat_epochs=10`

QAT 不建议沿用基座训练的大学习率调参习惯。

## Batch Size 建议

| 显存条件 | 推荐 batch size |
| --- | --- |
| 资源较紧 | `16` - `32` |
| 中等资源 | `32` - `64` |
| 资源充足 | `64` - `128` |

说明：

- 基座自动化脚本主要搜索 `32 / 64 / 128`
- pruning / QAT 默认使用 `64`
- ONNX 阶段的 `eval_batch_size` 只影响评估，不属于训练 batch

## 训练轮数建议

| 模型 | 常见起点 |
| --- | --- |
| 轻量模型 | 更高 epoch，换取收敛稳定性 |
| 标准模型 | 可相对减少 epoch，但需要更多算力 |

经验上：

- 小模型更适合长时间训练
- 大模型更适合在较强算力下配合合理正则化

## 正则化建议

### Dropout

常见经验是：

- 轻量模型可使用较明显的 Dropout
- 标准模型可使用更保守的 Dropout

### Weight Decay

默认：

- 基座训练：`1e-4`
- pruning 微调：`1e-4`
- QAT：`1e-4`

建议优先在学习率和 epoch 之间调，再动 `weight_decay`。

## Warmup 与学习率调度

训练链统一使用：

- Warmup
- Cosine Annealing

默认参数：

- `warmup_ratio=0.05`
- `warmup_steps=0`

说明：

- 若 `warmup_steps=0`，则内部使用 `warmup_ratio`
- 训练入口会按 step 更新学习率，并输出学习率曲线

## 监控指标

建议优先关注：

1. 训练损失
2. 验证损失
3. 验证准确率
4. 学习率曲线
5. 剪枝前后参数量 / MACs 变化

TensorBoard 示例：

```bash
tensorboard --logdir output/
```

## 常见问题

### 基座训练不收敛

可能原因：

- 学习率过高或过低
- 数据范围不合适
- 模型容量与数据复杂度不匹配

### 剪枝后精度下降过快

优先检查：

- `pruning_ratio` 是否过高
- `pruning_steps` 是否过少
- `finetune_epochs` 是否不足

### QAT 后精度退化明显

优先检查：

- 输入 checkpoint 是否与给定的 `quantization_meta` 契约兼容
- 是否仍使用过大的训练学习率
- 是否直接复用了不适合 QAT 的上游实验

## 调优建议顺序

推荐顺序：

1. 先拿到稳定基座模型
2. 再做 pruning sweep，寻找压缩甜点区
3. 只对少数优选 pruning 结果进入 QAT
4. 最后再进入 ONNX / AMCT / ATC 导出与部署链

这比一开始就把所有实验都推进到部署链更节省时间。
