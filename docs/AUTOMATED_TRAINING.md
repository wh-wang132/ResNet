# 自动化训练使用指南

## 概述

本项目提供了自动化训练脚本，用于系统地验证不同网络结构、批次大小和训练轮数对模型性能的影响。

## 文件说明

| 文件 | 说明 |
|------|------|
| `automated_training.py` | 完整的自动化训练脚本（包含所有参数组合） |
| `quick_test.py` | 快速测试脚本（使用少量参数组合） |
| `run_automated_training.sh` | Shell包装器脚本 |

## 功能特性

✅ **强制参数设置**
- `Train=True`：始终启用训练
- `Test=True`：始终启用测试
- `UMAP=False`：始终禁用UMAP可视化

✅ **ResNet50显存限制**
- 当使用ResNet50时，最大批次大小限制为128（适配48GB显存）

✅ **自动遍历参数**
- 支持多种网络结构
- 支持多种批次大小
- 支持多种训练轮数

✅ **性能指标记录**
- 训练时间
- 最佳验证准确率
- 测试准确率
- 错误信息

✅ **结构化报告**
- JSON格式结果
- CSV格式表格
- Markdown实验报告

✅ **错误处理**
- 单组参数失败不影响整体流程
- 详细的错误日志记录

✅ **日志记录**
- 实时控制台输出
- 文件日志保存

## 快速开始

### 1. 快速测试（推荐先运行）

使用快速测试脚本验证环境和脚本功能：

```bash
uv run python quick_test.py
```

这个脚本只会运行：
- 1种模型：resnet6_2d
- 1种批次大小：8
- 1种训练轮数：1

### 2. 完整实验

运行完整的自动化训练实验：

```bash
# 方式1：直接运行Python脚本
uv run python automated_training.py

# 方式2：使用Shell包装器
./run_automated_training.sh
```

## 配置说明

### automated_training.py 配置

在 `automated_training.py` 的 `__init__` 方法中可以修改实验配置：

```python
# 实验配置
self.models = [
    "resnet6_2d",
    "resnet10_2d", 
    "resnet14_2d",
    "resnet18_2d",
    "resnet34_2d",
    "resnet50_2d"
]

self.batch_sizes = [8, 16, 32, 64, 128]
self.epochs_list = [10, 20, 30]

# ResNet50的最大批次大小限制
self.resnet50_max_batch_size = 128
```

### quick_test.py 配置

在 `quick_test.py` 中可以修改快速测试配置：

```python
# 快速测试配置
self.models = ["resnet6_2d"]
self.batch_sizes = [8]
self.epochs_list = [1]
```

## 输出目录结构

实验完成后，会生成以下目录和文件：

```
experiments/
└── <timestamp>/
    ├── results.json              # JSON格式结果
    ├── results.csv               # CSV格式结果
    ├── experiment_report.md      # 实验报告
    ├── output_<model>_bs<bs>_ep<ep>.txt  # 每次训练的输出
    └── ...

logs/
└── training_<timestamp>.log    # 训练日志
```

## 实验报告内容

生成的 `experiment_report.md` 包含：

1. **实验配置**
   - 模型列表
   - 批次大小
   - 训练轮数
   - ResNet50限制

2. **实验结果汇总**
   - 总实验次数
   - 成功/失败统计

3. **详细结果表格**
   - 模型
   - 批次大小
   - 训练轮数
   - 状态
   - 最佳验证准确率
   - 测试准确率
   - 训练时间

4. **性能分析**
   - 最佳性能模型

5. **结论**（需要手动填写）

## 自定义配置示例

### 示例1：只测试轻量级模型

```python
self.models = ["resnet6_2d", "resnet10_2d", "resnet14_2d"]
self.batch_sizes = [32, 64, 128]
self.epochs_list = [20, 40]
```

### 示例2：只测试标准模型

```python
self.models = ["resnet18_2d", "resnet34_2d", "resnet50_2d"]
self.batch_sizes = [16, 32, 64]
self.epochs_list = [30]
```

### 示例3：调整ResNet50批次限制

```python
self.resnet50_max_batch_size = 64  # 如果显存较小
# 或
self.resnet50_max_batch_size = 256  # 如果显存更大
```

## 注意事项

1. **显存限制**：ResNet50批次大小已限制为128，可根据实际显存调整
2. **实验时间**：完整实验可能需要较长时间，建议先运行快速测试
3. **磁盘空间**：确保有足够的磁盘空间保存实验结果和模型
4. **错误处理**：单个实验失败不会中断整体流程
5. **资源释放**：每次实验间隔2秒，用于释放GPU资源

## 故障排除

### 问题1：uv命令未找到

**解决方案**：
```bash
# 安装uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 问题2：数据集不存在

**解决方案**：
确保 `Data/` 目录存在并包含正确的数据集结构。

### 问题3：GPU显存不足

**解决方案**：
- 减小批次大小
- 调整 `resnet50_max_batch_size`
- 使用更小的模型

### 问题4：训练输出解析失败

**解决方案**：
检查 `extract_best_val_acc` 和 `extract_test_acc` 方法中的正则表达式，确保与实际输出格式匹配。
