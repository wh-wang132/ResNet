# 模型架构说明

## 概述

项目提供两类 2D ResNet 模型族：

- 轻量级模型：`resnet6_2d`、`resnet10_2d`、`resnet14_2d`
- 标准模型：`resnet18_2d`、`resnet34_2d`

所有模型都面向 `.npy` 特征图；`_2d` 表示网络主体采用 2D 卷积，并不限制原始样本只能是二维数组。训练期实际输入 shape 由数据集样本推断得到：

- 若样本为 2D `H,W`，加载后自动补成 `CHW=(1,H,W)`
- 若样本为 3D `C,H,W`，则直接沿用其通道数

说明：

- 仓库包含以上 5 个模型
- 训练入口不会要求显式传入类别数；类别数会先从 `Data/<class>/` 一级子目录推断，再传给模型构造逻辑
- 模型结构均采用 `BasicBlock` 路线，不包含 `resnet50` 或 bottleneck 结构
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

所有支持的模型都具备：

1. 输入通道数与数据集样本一致
2. 分类头输出维度与数据集动态推断出的类别数一致
3. 可配置 Dropout
4. Kaiming 初始化
5. `get_features()` 中间特征提取接口
6. `channel_cfg` 驱动的拓扑恢复能力

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
from base_model.dataset import discover_dataset_classes
from base_model.resnet_lightweight import resnet6_2d
from base_model.resnet_standard import resnet18_2d

class_names, _ = discover_dataset_classes("Data")

model_light = resnet6_2d(num_classes=len(class_names), dropout_p=0.3, in_channels=1)
model_standard = resnet18_2d(num_classes=len(class_names), dropout_p=0.3, in_channels=3)
```

## 拓扑恢复示例

剪枝后模型并不是靠“猜测通道数”恢复，而是通过 `channel_cfg` 明确重建；若 `channel_cfg` 已携带分类头信息，则无需再额外手动传入类别数：

```python
from base_model.resnet_lightweight import resnet6_2d_from_cfg

in_channels = 3

model = resnet6_2d_from_cfg(
    channel_cfg=channel_cfg,
    dropout_p=0.3,
    in_channels=in_channels,
)
```

这套机制是 pruning -> QAT -> ONNX 主线的基础。
