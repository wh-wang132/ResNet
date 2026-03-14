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
  uv run python src/main.py --data_dir ./my_dataset
  ```

### --full_load / --no-full_load

- **类型**: boolean
- **默认值**: False
- **说明**: 启用/禁用全量加载数据集到内存
  - 启用：将整个数据集一次性加载到内存中，优化数据访问速度（适用于内存充足的情况）
  - 禁用：保持原有的增量加载机制，按需从磁盘加载数据（适用于内存受限的情况）
- **使用场景**:
  - 内存充足且追求训练速度：使用 `--full_load`
  - 内存受限或数据集过大：使用默认（不添加该参数）或 `--no-full_load`
- **示例**:
  ```bash
  # 启用全量加载
  uv run python src/main.py --full_load

  # 禁用全量加载（默认）
  uv run python src/main.py --no-full_load
  ```

### --num_workers

- **类型**: int
- **默认值**: None
- **说明**: 数据加载工作线程数
  - None：自动检测CPU核心数（推荐）
  - 0：单线程加载（主线程）
  - >0：指定工作线程数
- **使用场景**:
  - 自动检测（推荐）：不添加该参数
  - 手动控制：根据系统资源调整
- **示例**:
  ```bash
  # 自动检测（推荐）
  uv run python src/main.py

  # 指定4个工作线程
  uv run python src/main.py --num_workers 4
  ```

### --prefetch_factor

- **类型**: int
- **默认值**: 2
- **说明**: DataLoader预取因子（每个工作线程预取的样本数）
  - 较高值：提高吞吐量但增加内存使用
  - 较低值：减少内存使用但可能降低吞吐量
- **使用场景**:
  - 内存充足：使用3-4
  - 内存受限：使用1-2
- **示例**:
  ```bash
  # 预取因子设为4（内存充足）
  uv run python src/main.py --prefetch_factor 4
  ```

### --persistent_workers / --no-persistent_workers

- **类型**: boolean
- **默认值**: True
- **说明**: 保持DataLoader工作线程活跃
  - 启用：训练过程中保持工作线程存活，减少线程创建开销
  - 禁用：每个epoch后销毁工作线程，节省资源
- **使用场景**:
  - 长时间训练：保持启用（默认）
  - 快速测试或资源受限：可以禁用
- **示例**:
  ```bash
  # 禁用持久化工作线程
  uv run python src/main.py --no-persistent_workers
  ```

### --pin_memory / --no-pin_memory

- **类型**: boolean
- **默认值**: True
- **说明**: 启用CUDA内存钉住（GPU训练时推荐）
  - 启用：将数据固定在页锁定内存中，加速CPU到GPU的数据传输
  - 禁用：使用常规内存
- **注意**: 仅在使用GPU训练时有效
- **示例**:
  ```bash
  # 禁用内存钉住（CPU训练时）
  uv run python src/main.py --no-pin_memory
  ```

### --cudnn_benchmark / --no-cudnn_benchmark

- **类型**: boolean
- **默认值**: True
- **说明**: 启用/禁用cuDNN自动调优
  - 启用：自动为每个卷积层选择最优算法，显著提升卷积操作性能
  - 禁用：使用固定算法，减少启动时间但可能降低性能
- **注意**: 当启用确定性算法时，此选项会被自动禁用
- **使用场景**:
  - 追求最佳性能：保持启用（默认）
  - 需要快速启动或确定性：可以禁用
- **示例**:
  ```bash
  # 禁用cuDNN自动调优
  uv run python src/main.py --no-cudnn_benchmark
  ```

### --cudnn_deterministic / --no-cudnn_deterministic

- **类型**: boolean
- **默认值**: False
- **说明**: 启用/禁用cuDNN确定性算法
  - 启用：使用确定性算法，确保结果可复现，但会降低训练速度
  - 禁用：使用非确定性算法，提高训练速度（推荐用于训练）
- **使用场景**:
  - 训练阶段：禁用（默认）以获得最佳性能
  - 调试/研究/需要可复现结果：启用
- **示例**:
  ```bash
  # 启用确定性算法（结果可复现）
  uv run python src/main.py --cudnn_deterministic
  ```

### --compile_model / --no-compile_model

- **类型**: boolean
- **默认值**: True
- **说明**: 启用/禁用模型编译
  - 启用：使用torch.compile编译模型，优化前向和反向传播
  - 禁用：不编译模型，直接使用原始模型
- **注意**: 仅在CUDA可用时生效
- **使用场景**:
  - 追求最佳性能：保持启用（默认）
  - 调试或快速测试：可以禁用
- **示例**:
  ```bash
  # 禁用模型编译
  uv run python src/main.py --no-compile_model
  ```

### --compile_mode

- **类型**: str
- **默认值**: "default"
- **可选值**: "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
- **说明**: 模型编译模式
  - default: 平衡编译时间和性能
  - reduce-overhead: 优先减少CPU开销
  - max-autotune: 最大程度自动调优（编译时间最长，性能最佳）
  - max-autotune-no-cudagraphs: 最大自动调优但不使用CUDA Graphs
- **使用场景**:
  - 常规使用：default
  - 追求极致性能：max-autotune
  - 内存受限：max-autotune-no-cudagraphs
- **示例**:
  ```bash
  # 使用最大自动调优模式
  uv run python src/main.py --compile_mode max-autotune
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
