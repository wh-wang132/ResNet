# ResNet 2D 轻量化模型

## 概述

本项目是本科毕设"基于昇腾AI架构的高效化无人机射频信号识别"的训练代码实现，提供轻量级和标准的 ResNet 模型，用于 2D .npy 格式数据集的训练、评估和可视化。

## 核心功能

- ✅ 多种 ResNet 架构（轻量级和标准）
- ✅ FP16 混合精度训练（AMP）
- ✅ Warmup + Cosine Annealing 学习率调度
- ✅ 完整的训练、验证和测试流程
- ✅ 混淆矩阵生成和统计
- ✅ UMAP 可视化（内存优化版本）
- ✅ TensorBoard 日志记录
- ✅ GPU 内存监控和管理
- ✅ 模块化代码架构，易于扩展

## 快速开始

### 环境要求

- Python 3.12+
- CUDA 13.0+（如需 GPU 加速）
- NVIDIA GPU（推荐 8GB+ 显存）

### 安装部署

1. 克隆项目：
   ```bash
   git clone git@github.com:wh-wang132/ResNet.git
   cd ResNet
   ```

2. 安装依赖（使用 uv）：
   ```bash
   uv sync
   ```

3. 准备数据集：
   - 将 .npy 格式数据集放入 `Data/` 目录
   - 数据集结构详见 [`数据准备`](docs/DATA_PREPARATION.md)

### 基本使用

```bash
# 完整训练流程（训练 + 测试）
uv run src/main.py --epochs 20 --model resnet6_2d

# 仅训练
uv run src/main.py --epochs 20 --no-Test

# 仅测试和可视化
uv run src/main.py --no-Train --UMAP

# 使用不同的模型
uv run src/main.py --model resnet18_2d
```

## 技术栈选型

| 技术 | 版本 | 用途 |
|------|------|------|
| Python | 3.12+ | 开发语言 |
| PyTorch | 2.10.0+ | 深度学习框架 |
| NumPy | 2.4.3+ | 数值计算 |
| Matplotlib | 3.10.8+ | 数据可视化 |
| Scikit-learn | 1.8.0+ | 机器学习工具 |
| UMAP-learn | 0.5.11+ | 降维可视化 |
| TensorBoard | 2.20.0+ | 训练日志记录 |
| uv | - | 包管理工具 |

## 项目结构

```
ResNet/
├── src/
│   ├── main.py              # 主训练脚本
│   ├── utils.py             # 通用工具函数
│   ├── trainer.py           # 训练模块
│   ├── tester.py            # 测试模块
│   ├── visualizer.py        # UMAP 可视化模块
│   ├── resnet_lightweight.py # 轻量级 ResNet 模型
│   ├── resnet_standard.py  # 标准 ResNet 模型
│   ├── dataset_npy.py       # .npy 数据集加载器
│   ├── confusionMatrix.py    # 混淆矩阵工具
│   └── lr_scheduler.py      # 学习率调度器
├── docs/                    # 文档目录
├── Data/                    # 数据集目录
├── output/                  # 训练输出目录
├── pyproject.toml           # 项目依赖配置
├── uv.lock                  # 锁定依赖版本
├── README.md                # 本文件
└── LICENSE                  # 许可证
```

## 模型架构

### 轻量级模型

| 模型 | 参数量 | 适用场景 |
|------|--------|----------|
| ResNet-6 | 310,392 | 快速实验，资源受限环境 |
| ResNet-10 | 694,440 | 平衡精度与速度 |
| ResNet-14 | 902,376 | 更高精度，轻量级架构 |

### 标准模型

| 模型 | 参数量 | 残差块 |
|------|--------|--------|
| ResNet-18 | 11.2M | BasicBlock |
| ResNet-34 | 21.3M | BasicBlock |
| ResNet-50 | 23.6M | Bottleneck |

详细模型说明请参考 [`模型架构`](docs/MODEL_ARCHITECTURE.md)。

## 命令行参数

### 基本参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 1 | 训练轮数 |
| `--lr` | 0.0003 | 学习率 |
| `--batch_size` | 64 | 批次大小 |
| `--model_path` | best_model.pth | 模型保存路径 |
| `--class_num` | 24 | 分类数 |
| `--model` | resnet6_2d | 选择模型架构 |
| `--data_dir` | Data | 数据集路径 |

### 功能开关

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--Train` | True | 启用训练 |
| `--no-Train` | - | 禁用训练 |
| `--Test` | True | 启用测试 |
| `--no-Test` | - | 禁用测试 |
| `--UMAP` | False | 启用 UMAP 可视化 |

### 正则化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dropout_p` | 0.3 | Dropout 概率 |
| `--weight_decay` | 0.0001 | 权重衰减 |

### 学习率调度器

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--warmup_ratio` | 0.05 | Warmup 占总步数的比例 |
| `--warmup_steps` | 0 | Warmup 步数（优先使用） |
| `--min_lr` | 1e-6 | 最小学习率 |
| `--plot_lr_schedule` | True | 绘制学习率曲线 |
| `--no-plot_lr_schedule` | - | 禁用学习率曲线绘制 |

详细参数说明请参考 [`命令行参数`](docs/CLI_ARGUMENTS.md)。

## 输出文件

训练完成后，输出目录会包含以下文件：

```
output/<model>_Data/<config>/
├── best_model.pth           # 最佳模型权重
├── lr_schedule.png           # 学习率调度曲线
├── training_curves.png        # 训练曲线（损失、准确率、学习率）
├── Confusion matrix.png      # 混淆矩阵图
├── umap_plot.png            # UMAP 可视化图（如启用）
└── runs/                    # TensorBoard 日志目录
```

## 文档

- [`数据准备指南`](docs/DATA_PREPARATION.md) - 如何准备和组织数据集
- [`模型架构说明`](docs/MODEL_ARCHITECTURE.md) - 各种 ResNet 架构的详细说明
- [`训练参数调优`](docs/TRAINING_GUIDE.md) - 训练参数调优建议
- [`命令行参数详解`](docs/CLI_ARGUMENTS.md) - 完整的命令行参数说明
- [`模块说明`](docs/MODULES.md) - 代码模块结构和功能说明

## 贡献规范

欢迎提交 Issue 和 Pull Request！请遵循以下规范：

1. 代码风格遵循 PEP 8
2. 提交前运行测试
3. 新功能请添加相应文档
4. 提交信息清晰明确

详细规范请参考 [`贡献指南`](docs/CONTRIBUTING.md)。

## 许可证

本项目采用 [`GPLv3 许可证`](LICENSE)。

## 联系方式

如有问题或建议，请通过 Issue 联系。

---

**项目维护**: 持续更新中

