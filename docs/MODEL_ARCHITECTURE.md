# 模型架构说明

## 概述

本项目提供两种类型的 ResNet 架构：轻量级和标准。每种架构都针对 2D .npy 格式数据进行了优化。

## 轻量级模型

轻量级模型专为资源受限环境设计，在保持较高准确率的同时显著减少参数量。

### ResNet-6 2D

- **参数量**: 约 310,392
- **适用场景**: 快速实验、资源受限环境、原型验证
- **特点**:
  - 只有 3 个残差层（无第 4 层）
  - 初始通道数：32
  - 残差块配置：[1, 1, 1]

### ResNet-10 2D

- **参数量**: 约 694,440
- **适用场景**: 平衡精度与速度
- **特点**:
  - 3 个残差层
  - 初始通道数：48
  - 残差块配置：[1, 1, 1]

### ResNet-14 2D

- **参数量**: 约 902,376
- **适用场景**: 更高精度、轻量级架构
- **特点**:
  - 3 个残差层
  - 初始通道数：48
  - 残差块配置：[2, 2, 1]

## 标准模型

标准模型基于经典的 ResNet 架构，提供更高的表达能力。

### ResNet-18 2D

- **参数量**: 约 11.2M
- **残差块**: BasicBlock
- **特点**:
  - 4 个残差层
  - 经典 ResNet-18 架构
  - 适合中等规模数据集

### ResNet-34 2D

- **参数量**: 约 21.3M
- **残差块**: BasicBlock
- **特点**:
  - 更深的网络结构
  - 更强的特征提取能力
  - 需要更多计算资源

### ResNet-50 2D

- **参数量**: 约 23.6M
- **残差块**: Bottleneck
- **特点**:
  - 使用瓶颈块设计
  - 更高的参数效率
  - 适合复杂任务

## 架构特性

### 共同特性

所有模型都包含以下优化：

1. **FP16 兼容**: 自动混合精度训练支持
2. **Dropout 正则化**: 可配置的 Dropout 层缓解过拟合
3. **Kaiming 初始化**: 优化的权重初始化策略
4. **单通道输入**: 专为单通道数据设计（输入形状 `(1, H, W)`）
5. **特征提取**: 支持中间层特征提取

### 轻量级模型特有特性

- 更小的初始卷积核（5x5 而非 7x7）
- 更少的初始通道数
- 移除第 4 残差层
- 额外的 Dropout 层

## 模型选择指南

| 需求 | 推荐模型 |
|------|----------|
| 快速原型验证 | ResNet-6 2D |
| 平衡精度和速度 | ResNet-10 2D 或 ResNet-14 2D |
| 资源充足，追求高精度 | ResNet-18 2D 或 ResNet-34 2D |
| 复杂任务 | ResNet-50 2D |

## 使用示例

```python
from resnet_lightweight import resnet6_2d
from resnet_standard import resnet18_2d

# 轻量级模型
model_light = resnet6_2d(num_classes=24, dropout_p=0.3)

# 标准模型
model_standard = resnet18_2d(num_classes=24, dropout_p=0.3)
```

## 模型架构图

### 轻量级 ResNet 架构

```
输入 (1, H, W)
    ↓
Conv2d(5x5, stride=2) + BatchNorm + ReLU + MaxPool
    ↓
Layer 1 (残差块 × n)
    ↓
Layer 2 (残差块 × n)
    ↓
Layer 3 (残差块 × n)
    ↓
AdaptiveAvgPool2d
    ↓
Dropout
    ↓
FC (全连接层)
    ↓
输出 (num_classes)
```

### 标准 ResNet 架构

```
输入 (1, H, W)
    ↓
Conv2d(7x7, stride=2) + BatchNorm + ReLU + MaxPool
    ↓
Layer 1 (残差块 × n)
    ↓
Layer 2 (残差块 × n)
    ↓
Layer 3 (残差块 × n)
    ↓
Layer 4 (残差块 × n)
    ↓
AdaptiveAvgPool2d
    ↓
FC (全连接层)
    ↓
输出 (num_classes)
```
