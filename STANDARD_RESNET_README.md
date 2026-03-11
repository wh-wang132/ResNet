# 标准 ResNet 模型实现说明

## 概述

本项目在现有轻量级 ResNet 架构的基础上，新增了标准的 ResNet-18、ResNet-34 和 ResNet-50 模型实现。这些模型完全遵循标准 ResNet 架构规范，并与现有代码框架无缝集成。

## 文件结构

```
src/
├── resnet_lightweight.py   # 现有轻量级 ResNet 模型（未修改）
├── resnet_standard.py      # 新增标准 ResNet 模型实现
├── test_resnet_standard.py # 标准 ResNet 模型单元测试
└── main_lightweight.py      # 已更新以支持新模型
```

## 新模型架构详情

### 1. ResNet-18

- **残差块类型**：BasicBlock
- **层数配置**：[2, 2, 2, 2] (layer1-layer4)
- **初始通道数**：64
- **参数数量**：约 11.2M
- **特点**：
  - 使用基础残差块（2个 3x3 卷积层）
  - 四个标准残差层
  - 初始卷积使用 7x7 卷积核
  - 支持 FP16 混合精度训练

### 2. ResNet-34

- **残差块类型**：BasicBlock
- **层数配置**：[3, 4, 6, 3] (layer1-layer4)
- **初始通道数**：64
- **参数数量**：约 21.3M
- **特点**：
  - 与 ResNet-18 相同的基础残差块
  - 更深的网络结构
  - 更强的表达能力
  - 支持 FP16 混合精度训练

### 3. ResNet-50

- **残差块类型**：Bottleneck
- **层数配置**：[3, 4, 6, 3] (layer1-layer4)
- **初始通道数**：64
- **参数数量**：约 23.6M
- **特点**：
  - 使用瓶颈残差块（1x1 -> 3x3 -> 1x1 卷积）
  - expansion = 4（通道数扩展 4 倍）
  - 更高效的参数利用
  - 支持 FP16 混合精度训练

## 编码规范与接口设计

新实现严格遵循以下规范：

### 1. 与现有代码的一致性

- **类命名约定**：保持与 `LightweightBasicBlock2D` 相似的风格
- **接口设计**：`ResNet2D` 类与 `LightweightResNet2D` 保持一致的接口
  - `__init__` 参数兼容
  - `forward` 方法相同
  - `get_features` 方法支持中间层特征提取
- **FP16 支持**：完整保留 `autocast` 装饰器，支持混合精度训练

### 2. 标准 ResNet 架构规范

- **初始卷积**：7x7 卷积核，步长 2，填充 3
- **最大池化**：3x3 池化核，步长 2，填充 1
- **通道数倍增**：每层通道数按 64 -> 128 -> 256 -> 512 递增
- **步长配置**：layer2-layer4 第一个块使用 stride=2 进行下采样
- **残差连接**：使用 1x1 卷积进行维度匹配的下采样

## 模型参数量对比

| 模型 | 类型 | 参数量 | 相对比例 |
|------|------|--------|----------|
| ResNet-6 | 轻量 | 310,392 | 1.0x |
| ResNet-10 | 轻量 | 694,440 | 2.2x |
| ResNet-14 | 轻量 | 902,376 | 2.9x |
| ResNet-18 | 标准 | 11,182,552 | 36.0x |
| ResNet-34 | 标准 | 21,290,712 | 68.6x |
| ResNet-50 | 标准 | 23,550,936 | 75.9x |

## 使用方法

### 1. 导入模型

```python
from resnet_standard import resnet18_2d, resnet34_2d, resnet50_2d

# 创建模型
model18 = resnet18_2d(num_classes=24, dropout_p=0.0)
model34 = resnet34_2d(num_classes=24, dropout_p=0.0)
model50 = resnet50_2d(num_classes=24, dropout_p=0.0)
```

### 2. 命令行使用

在 `main_lightweight.py` 中，通过 `--model` 参数选择新模型：

```bash
# 使用 ResNet-18
uv run python src/main_lightweight.py --model resnet18_2d

# 使用 ResNet-34
uv run python src/main_lightweight.py --model resnet34_2d

# 使用 ResNet-50
uv run python src/main_lightweight.py --model resnet50_2d
```

### 3. 中间层特征提取

与现有模型一致，支持中间层特征提取：

```python
features = model.get_features(x, layer=["layer1", "layer2", "layer3", "layer4"])
```

## 单元测试

运行单元测试验证新模型：

```bash
uv run python src/test_resnet_standard.py
```

测试内容包括：
- ✓ 模型创建
- ✓ 参数量验证（在预期范围内）
- ✓ 前向传播正确性
- ✓ 输出形状验证
- ✓ get_features 方法功能
- ✓ 所有模型参数量对比

## 关键设计决策

### 1. 不修改现有代码

严格遵守要求，未对 `resnet_lightweight.py` 进行任何修改，确保现有功能完全不受影响。

### 2. Dropout 配置

标准 ResNet 默认不使用 Dropout，因此新模型的 `dropout_p` 默认值设为 0.0，但仍支持通过参数配置。

### 3. 输入通道数

保持与现有模型一致，默认输入通道数为 1（单通道灰度/特征图），可通过 `in_channels` 参数调整。

### 4. 四个残差层

标准 ResNet 包含四个残差层，因此在 `ResNet2D` 类中恢复了 `layer4`，并相应更新了 `get_features` 方法。

## 内存与计算资源建议

| 模型 | 建议显存 | 建议批次大小 |
|------|----------|--------------|
| ResNet-18 | 4GB+ | 32-64 |
| ResNet-34 | 6GB+ | 16-32 |
| ResNet-50 | 8GB+ | 16-32 |

注意：使用 FP16 混合精度训练可显著减少显存占用。

## 与现有模型的关系

- **轻量级模型**（ResNet-6/10/14）：适用于资源受限环境，快速实验
- **标准模型**（ResNet-18/34/50）：适用于追求更高精度，有充足计算资源的场景

两类模型共享相同的训练、评估和可视化流程，可无缝切换使用。

## 版本历史

- **v1.0** (2026-03-11)：初始版本，实现 ResNet-18/34/50，通过所有单元测试

