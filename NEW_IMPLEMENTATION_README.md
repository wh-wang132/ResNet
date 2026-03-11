# 1D_lw_ResNet 新实现：轻量级 2D ResNet 训练脚本

## 概述

本项目为 ./Data 目录下的 .npy 格式数据集开发了一个完整的轻量级 2D ResNet 训练系统，专门解决了 ResNet-18 存在的内存溢出（OOM）和严重过拟合问题。

## 新创建的文件

| 文件名 | 位置 | 说明 |
|--------|------|------|
| `dataset_npy.py` | `./test/` | .npy 数据集加载模块 |
| `resnet_lightweight_2d.py` | `./test/` | 轻量级 2D ResNet 模型架构 |
| `main_grey_npy.py` | `./test/` | 完整的训练/测试主脚本 |
| `verify_new_implementation.py` | `./` | 验证脚本 |

## 主要改进和优化

### 1. 轻量级 2D ResNet 架构

#### 针对 ResNet-18 问题的优化：
- **减少初始通道数**：从 64 减少到 32/48
- **移除 Layer4**：减少深度，降低复杂度
- **使用更小的卷积核**：从 7×7 改为 5×5
- **可选分组卷积**：进一步减少参数量

#### 提供的模型选项：

| 模型 | 参数量 | 说明 |
|------|--------|------|
| `resnet6_2d` | ~310 K | 超轻量级，适合资源受限场景 |
| `resnet10_2d` | ~694 K | 推荐使用，平衡性能和复杂度 |
| `resnet14_2d` | ~902 K | 中等复杂度，性能更好 |

### 2. 全面的过拟合缓解策略

- **Dropout 正则化**：p=0.2-0.3
- **Dropout2d**：在残差块内部应用
- **L2 权重衰减**：通过 `weight_decay` 参数
- **学习率调度**：ReduceLROnPlateau，自动调整学习率
- **早停**：保存最佳验证模型

### 3. GPU 内存优化

- **自动内存监控**：实时跟踪 GPU 使用情况
- **定期内存释放**：`release_gpu_memory()` 函数
- **峰值内存记录**：识别内存使用峰值
- **TensorBoard 记录**：内存使用曲线可视化

### 4. .npy 数据集支持

- **自定义 Dataset 类**：专门处理 .npy 文件
- **自动 6:2:2 划分**：训练/验证/测试集
- **float16 → float32 转换**：提高数值精度
- **分层抽样**：保持类别比例

## 数据集信息

### 数据集统计

| 属性 | 数值 |
|------|------|
| 类别数量 | 24 类 |
| 总样本数 | 14,484 个 |
| 数据形状 | (543, 512) 2D 矩阵 |
| 数据类型 | float16 (自动转换为 float32) |
| 值域 | [0.0, 1.0] (已归一化) |

### 数据集目录结构

```
/root/1D_lw_ResNet/Data/
├── 0/
│   ├── 000-a.npy
│   ├── 001-a.npy
│   └── ...
├── 1/
│   └── ...
├── 2/
│   └── ...
...
└── 23/
    └── ...
```

## 使用方法

### 基本训练命令

```bash
cd /root/1D_lw_ResNet/test

# 完整训练（推荐）
python main_grey_npy.py \
    --Train True \
    --Test True \
    --epochs 60 \
    --model resnet10_2d \
    --batch_size 32 \
    --lr 0.001 \
    --dropout_p 0.3
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 60 | 训练轮数 |
| `--lr` | 0.001 | 学习率 |
| `--batch_size` | 32 | 批次大小（建议 16-64） |
| `--class_num` | 24 | 分类类别数 |
| `--model` | resnet10_2d | 模型选择（resnet6_2d/resnet10_2d/resnet14_2d） |
| `--data_dir` | ./Data | 数据集路径 |
| `--Train` | True | 是否进行训练 |
| `--Test` | True | 是否进行测试 |
| `--TSNE` | False | 是否进行 t-SNE 可视化 |
| `--dropout_p` | 0.3 | Dropout 概率（0.1-0.5） |
| `--weight_decay` | 1e-4 | L2 权重衰减 |

### 训练和测试分离

```bash
# 仅训练（不测试）
uv run ./test/main_grey_npy.py --Train True --Test False --epochs 60

# 仅测试（使用已训练模型）
uv run ./test/main_grey_npy.py --Train False --Test True
```

### 使用不同模型

```bash
# 超轻量模型
uv run ./test/main_grey_npy.py --model resnet6_2d --batch_size 64

# 推荐模型
uv run ./test/main_grey_npy.py --model resnet10_2d --batch_size 32

# 中等复杂度模型
uv run ./test/main_grey_npy.py --model resnet14_2d --batch_size 16
```

### 调整正则化参数

```bash
# 更强的正则化（用于严重过拟合）
uv run ./test/main_grey_npy.py --dropout_p 0.4 --weight_decay 5e-4

# 更弱的正则化（欠拟合时）
uv run ./test/main_grey_npy.py --dropout_p 0.2 --weight_decay 1e-5
```

## 输出内容

训练完成后，输出目录包含：

```
{model_name}_{dataset_name}_bs{batch_size}_lr{lr}_drop{dropout_p}/
├── best_model.pth              # 最佳模型权重
├── training_curves.png         # 训练曲线（Loss, Acc）
├── {folder} Confusion matrix.png  # 混淆矩阵
├── tsne_plot.png               # t-SNE 可视化（可选）
└── runs/                       # TensorBoard 日志
    └── [TensorBoard 文件]
```

## TensorBoard 可视化

```bash
tensorboard --logdir ./test/{output_folder}/runs
```

可以监控：
- 训练/验证 Loss
- 验证准确率
- GPU 内存使用情况
- 峰值内存记录

## 内存监控指标

训练过程中会自动记录和显示：

- **Allocated**：当前分配的 GPU 内存
- **Cached**：缓存的 GPU 内存
- **Peak**：历史峰值内存

## 训练监控指标

- **Train Loss**：训练集损失
- **Val Loss**：验证集损失（过拟合警告：Val Loss 上升）
- **Val Acc**：验证集准确率
- **Learning Rate**：当前学习率

## 验证脚本

运行完整验证：

```bash
cd /root/1D_lw_ResNet
uv run ./verify_new_implementation.py
```

验证内容包括：
- 依赖项检查
- 文件完整性检查
- 数据集加载测试
- 模型前向传播测试
- 参数数量统计

## 与现有项目的兼容性

- **保留原有文件**：所有原有的 test/ 目录下的文件都保留
- **独立实现**：新实现不影响现有代码
- **共享依赖**：使用相同的 pyproject.toml 依赖
- **共享评估**：使用相同的 confusionMatrix.py

## 技术要点总结

### 关键改进

| 问题 | 解决方案 |
|------|----------|
| ResNet-18 内存溢出 | 使用轻量级架构（减少 60-70% 参数） |
| 严重过拟合（Epoch 2 开始） | Dropout + Dropout2d + 权重衰减 + 学习率调度 |
| GPU 内存无法监控 | 自动内存统计 + TensorBoard 记录 |
| .npy 数据不支持 | 自定义 Dataset 类，支持 (543,512) 形状 |

### 推荐配置

**默认配置（平衡性能和内存）**：
```bash
uv run ./test/main_grey_npy.py \
    --model resnet10_2d \
    --batch_size 32 \
    --dropout_p 0.3 \
    --lr 0.001 \
    --weight_decay 1e-4
```

**内存受限配置**：
```bash
uv run ./test/main_grey_npy.py \
    --model resnet6_2d \
    --batch_size 16 \
    --dropout_p 0.2
```

**追求性能配置**：
```bash
uv run ./test/main_grey_npy.py \
    --model resnet14_2d \
    --batch_size 32 \
    --dropout_p 0.4 \
    --weight_decay 5e-4
```

## 常见问题

### Q: 如何解决 OOM 错误？
A: 1. 使用更小的模型（resnet6_2d） 2. 减小 batch_size 3. 减小输入尺寸（需要修改代码）

### Q: 如何判断是否过拟合？
A: 1. 验证损失开始上升 2. 训练损失继续下降但验证准确率停滞 3. 使用 TensorBoard 监控曲线

### Q: 如何修改为支持 GPU？
A: 代码已自动检测 GPU，无需修改。确保安装 CUDA 版本的 PyTorch。

### Q: 如何调整输入数据形状？
A: 修改 `dataset_npy.py` 中的预处理，或修改模型的初始层。

## 依赖项

确保已安装所有依赖：

```bash
cd /root/1D_lw_ResNet
uv sync
```

主要依赖：
- PyTorch 2.4.1 (CUDA 12.4)
- TorchVision 0.19.1
- NumPy, SciPy, Scikit-learn
- Matplotlib, TensorBoard, tqdm, PrettyTable

---

**作者**：AI Assistant  
**日期**：2026-03-09  
**版本**：1.0.0
