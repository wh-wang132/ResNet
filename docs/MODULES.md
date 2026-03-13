# 模块说明

## 概述

本文档详细说明项目中各个代码模块的功能和使用方法。

## 项目结构

```
src/
├── main.py              # 主训练脚本
├── utils.py             # 通用工具函数
├── trainer.py           # 训练模块
├── tester.py            # 测试模块
├── visualizer.py        # UMAP 可视化模块
├── resnet_lightweight.py # 轻量级 ResNet 模型
├── resnet_standard.py  # 标准 ResNet 模型
├── dataset.py           # .npy 数据集加载器
├── confusionMatrix.py    # 混淆矩阵工具
└── lr_scheduler.py      # 学习率调度器
```

## 模块详解

### main.py

**功能**: 主训练脚本，协调整个训练流程

**主要职责**:
- 解析命令行参数
- 配置计算设备
- 加载数据集
- 初始化模型
- 调用训练、测试和可视化模块

**关键函数**:
- `main()`: 主函数，协调整个流程

### utils.py

**功能**: 通用工具函数模块

**主要函数**:
- `release_gpu_memory()`: 释放 GPU 内存
- `get_gpu_memory_info()`: 获取 GPU 内存使用信息
- `setup_device()`: 设置计算设备
- `parse_args()`: 解析命令行参数
- `create_output_directory(args)`: 创建输出目录
- `load_model_map()`: 加载模型映射
- `print_model_info(model, device)`: 打印模型信息
- `print_training_summary(...)`: 打印训练摘要

### trainer.py

**功能**: 训练模块，包含完整的训练和验证流程

**主要函数**:
- `train_model(...)`: 训练模型的主函数
  - 支持 FP16 和 AMP 混合精度
  - 包含 Warmup + Cosine Annealing 学习率调度
  - 自动保存最佳模型
  - TensorBoard 日志记录

- `plot_training_curves(...)`: 绘制训练曲线（损失、准确率、学习率）

**训练流程**:
1. 初始化损失函数和优化器
2. 设置 GradScaler 用于自动混合精度
3. 初始化学习率调度器
4. 训练循环：
   - 前向传播（autocast）
   - 反向传播（scaler）
   - 梯度裁剪
   - 参数更新
   - 学习率调度
5. 验证循环
6. 保存最佳模型
7. 绘制训练曲线

### tester.py

**功能**: 测试模块，用于评估模型性能

**主要函数**:
- `test_model(...)`: 测试模型并生成混淆矩阵
  - 在测试集上评估模型
  - 计算准确率
  - 生成混淆矩阵图
  - 打印分类报告

### visualizer.py

**功能**: UMAP 可视化模块

**主要函数**:
- `visualize_umap(...)`: UMAP 可视化
  - 提取模型特征
  - 使用 UMAP 降维到 2D
  - 绘制可视化图
  - 内存优化版本

### resnet_lightweight.py

**功能**: 轻量级 2D ResNet 架构

**主要类**:
- `LightweightBasicBlock2D`: 轻量级基础残差块
- `LightweightResNet2D`: 轻量级 ResNet 主类

**主要函数**:
- `resnet6_2d(num_classes, dropout_p)`: ResNet-6 2D 模型
- `resnet10_2d(num_classes, dropout_p)`: ResNet-10 2D 模型
- `resnet14_2d(num_classes, dropout_p)`: ResNet-14 2D 模型

**特性**:
- FP16 兼容
- Dropout 正则化
- Kaiming 初始化
- 特征提取支持

### resnet_standard.py

**功能**: 标准 2D ResNet 架构

**主要类**:
- `BasicBlock2D`: 基础残差块
- `Bottleneck2D`: 瓶颈残差块
- `ResNet2D`: 标准 ResNet 主类

**主要函数**:
- `resnet18_2d(num_classes, dropout_p)`: ResNet-18 2D 模型
- `resnet34_2d(num_classes, dropout_p)`: ResNet-34 2D 模型
- `resnet50_2d(num_classes, dropout_p)`: ResNet-50 2D 模型

### dataset.py

**功能**: .npy 数据集加载模块

**主要类**:
- `NPYDataset`: 自定义 Dataset 类
  - 加载 .npy 文件
  - 自动转换为 float16
  - 添加通道维度

**主要函数**:
- `data_set_split(...)`: 划分数据集
  - 分层抽样
  - 可配置划分比例
  - 返回训练/验证/测试集

### confusionMatrix.py

**功能**: 混淆矩阵生成工具

**主要功能**:
- 计算混淆矩阵
- 绘制混淆矩阵图
- 生成分类报告

### lr_scheduler.py

**功能**: 学习率调度器

**主要类**:
- `WarmupCosineAnnealingLR`: Warmup + Cosine Annealing 学习率调度器

**主要函数**:
- `plot_lr_schedule(...)`: 绘制学习率调度曲线

## 模块依赖关系

```
main.py
├── utils.py
├── dataset.py
├── resnet_lightweight.py
├── resnet_standard.py
├── trainer.py
│   ├── lr_scheduler.py
│   └── utils.py
├── tester.py
│   └── confusionMatrix.py
└── visualizer.py
```

## 扩展开发

### 添加新的模型

1. 在 `resnet_lightweight.py` 或 `resnet_standard.py` 中定义新模型
2. 在 `utils.py` 的 `load_model_map()` 中添加映射
3. 在 `parse_args()` 的 `--model` 参数中添加选项

### 添加新的数据增强

1. 修改 `dataset.py` 中的 `NPYDataset` 类
2. 添加 `transform` 参数支持

### 添加新的可视化方法

1. 创建新的可视化模块或扩展 `visualizer.py`
2. 在 `main.py` 中添加相应的调用
