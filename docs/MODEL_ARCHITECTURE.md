# 模型架构说明

## 概述

当前项目提供两类 2D ResNet：

- 轻量级模型：`resnet6_2d`、`resnet10_2d`、`resnet14_2d`
- 标准模型：`resnet18_2d`、`resnet34_2d`

所有模型都面向单通道 2D `.npy` 特征图，默认输入形状为 `(1, 543, 512)`。

说明：

- 当前仓库只保留以上 5 个模型
- 不再包含 `resnet50` 或 bottleneck 结构
- 两类模型都支持 `*_from_cfg()`，用于恢复剪枝后的不规则拓扑

## 轻量级模型

### ResNet-6 2D

- 约 `310,392` 参数
- 适合快速实验、资源受限场景
- 3 个残差层
- 初始通道数：32
- block 配置：`[1, 1, 1]`

### ResNet-10 2D

- 约 `694,440` 参数
- 适合平衡精度与速度
- 3 个残差层
- 初始通道数：48
- block 配置：`[1, 1, 1]`

### ResNet-14 2D

- 约 `902,376` 参数
- 适合更高精度的轻量模型实验
- 3 个残差层
- 初始通道数：48
- block 配置：`[2, 2, 1]`

## 标准模型

### ResNet-18 2D

- 约 `11.2M` 参数
- 4 个残差层
- 使用 `BasicBlock`
- 适合中高精度需求

### ResNet-34 2D

- 约 `21.3M` 参数
- 更深的 4 stage 结构
- 使用 `BasicBlock`
- 适合对容量更敏感的场景

## 共同特性

所有当前支持的模型都具备：

1. 单通道输入
2. 可配置 Dropout
3. Kaiming 初始化
4. `get_features()` 中间特征提取接口
5. `channel_cfg` 驱动的拓扑恢复能力

这意味着模型不仅用于基座训练，也用于：

- 剪枝后拓扑导出
- QAT 阶段按剪枝拓扑重建浮点模型
- 维持跨阶段 checkpoint 恢复一致性

## 轻量级模型特性

- 初始卷积核为 `5x5`
- 默认不含第 4 个残差 stage
- 初始通道数更小
- 额外使用 Dropout，适合较小模型容量下的正则化

## 标准模型特性

- 初始卷积核为 `7x7`
- 保留 4 个残差 stage
- 更接近经典 ResNet-18 / 34 结构
- 更适合高容量基座或深度剪枝实验

## 模型选择建议

| 需求 | 推荐模型 |
| --- | --- |
| 快速原型验证 | `resnet6_2d` |
| 平衡速度与精度 | `resnet10_2d` / `resnet14_2d` |
| 追求更高基座容量 | `resnet18_2d` / `resnet34_2d` |
| 计划做大幅度结构化剪枝 | 优先从 `resnet18_2d` / `resnet34_2d` 开始 |

## 使用示例

```python
from base_model.resnet_lightweight import resnet6_2d
from base_model.resnet_standard import resnet18_2d

model_light = resnet6_2d(num_classes=24, dropout_p=0.3)
model_standard = resnet18_2d(num_classes=24, dropout_p=0.3)
```

## 拓扑恢复示例

剪枝后模型并不是靠“猜测通道数”恢复，而是通过 `channel_cfg` 明确重建：

```python
from base_model.resnet_lightweight import resnet6_2d_from_cfg

model = resnet6_2d_from_cfg(
    channel_cfg=channel_cfg,
    num_classes=24,
    dropout_p=0.3,
)
```

这套机制是 pruning -> QAT -> ONNX 主线的基础。
