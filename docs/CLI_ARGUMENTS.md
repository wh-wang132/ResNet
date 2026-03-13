# 命令行参数详解

## 概述

本文档详细说明所有可用的命令行参数及其用法。

## 基本参数

### --epochs

- **类型**: int
- **默认值**: 20
- **说明**: 训练轮数
- **示例**:
  ```bash
  uv run src/main.py --epochs 60
  ```

### --lr

- **类型**: float
- **默认值**: 0.0003
- **说明**: 学习率
- **示例**:
  ```bash
  uv run src/main.py --lr 0.001
  ```

### --batch_size

- **类型**: int
- **默认值**: 64
- **说明**: 批次大小
- **示例**:
  ```bash
  uv run src/main.py --batch_size 32
  ```

### --model_path

- **类型**: str
- **默认值**: "best_model.pth"
- **说明**: 模型保存路径（相对于输出目录）
- **示例**:
  ```bash
  uv run src/main.py --model_path checkpoint.pth
  ```

### --class_num

- **类型**: int
- **默认值**: 24
- **说明**: 分类数
- **示例**:
  ```bash
  uv run src/main.py --class_num 10
  ```

### --model

- **类型**: str
- **默认值**: "resnet6_2d"
- **可选值**: "resnet6_2d", "resnet10_2d", "resnet14_2d", "resnet18_2d", "resnet34_2d", "resnet50_2d"
- **说明**: 选择模型架构
- **示例**:
  ```bash
  uv run src/main.py --model resnet18_2d
  ```

### --data_dir

- **类型**: str
- **默认值**: "Data"
- **说明**: 数据集路径
- **示例**:
  ```bash
  uv run src/main.py --data_dir ./my_dataset
  ```

## 功能开关

### --Train / --no-Train

- **类型**: boolean
- **默认值**: True
- **说明**: 启用/禁用训练
- **示例**:
  ```bash
  # 启用训练（默认）
  uv run src/main.py --Train

  # 禁用训练
  uv run src/main.py --no-Train
  ```

### --Test / --no-Test

- **类型**: boolean
- **默认值**: True
- **说明**: 启用/禁用测试
- **示例**:
  ```bash
  # 启用测试（默认）
  uv run src/main.py --Test

  # 禁用测试
  uv run src/main.py --no-Test
  ```

### --UMAP

- **类型**: boolean
- **默认值**: False
- **说明**: 启用 UMAP 可视化
- **示例**:
  ```bash
  uv run src/main.py --UMAP
  ```

## 正则化参数

### --dropout_p

- **类型**: float
- **默认值**: 0.3
- **说明**: Dropout 概率
- **示例**:
  ```bash
  uv run src/main.py --dropout_p 0.5
  ```

### --weight_decay

- **类型**: float
- **默认值**: 0.0001
- **说明**: 权重衰减（L2 正则化）
- **示例**:
  ```bash
  uv run src/main.py --weight_decay 0.001
  ```

## 学习率调度器参数

### --warmup_ratio

- **类型**: float
- **默认值**: 0.05
- **说明**: Warmup 占总步数的比例
- **示例**:
  ```bash
  uv run src/main.py --warmup_ratio 0.1
  ```

### --warmup_steps

- **类型**: int
- **默认值**: 0
- **说明**: Warmup 步数（如果为 0，则使用 warmup_ratio）
- **示例**:
  ```bash
  uv run src/main.py --warmup_steps 1000
  ```

### --min_lr

- **类型**: float
- **默认值**: 1e-6
- **说明**: 最小学习率
- **示例**:
  ```bash
  uv run src/main.py --min_lr 1e-7
  ```

### --plot_lr_schedule / --no-plot_lr_schedule

- **类型**: boolean
- **默认值**: True
- **说明**: 是否绘制学习率调度曲线
- **示例**:
  ```bash
  # 绘制学习率曲线（默认）
  uv run src/main.py --plot_lr_schedule

  # 禁用学习率曲线
  uv run src/main.py --no-plot_lr_schedule
  ```

## 常用组合示例

### 完整训练流程

```bash
uv run src/main.py --epochs 60 --model resnet10_2d
```

### 仅训练

```bash
uv run src/main.py --epochs 60 --no-Test
```

### 仅测试和可视化

```bash
uv run src/main.py --no-Train --UMAP
```

### 自定义超参数

```bash
uv run src/main.py \
  --epochs 60 \
  --lr 0.001 \
  --batch_size 128 \
  --model resnet18_2d \
  --dropout_p 0.5 \
  --weight_decay 0.0001
```

## 参数调优建议

详细的参数调优建议请参考 [训练参数调优指南](TRAINING_GUIDE.md)。
