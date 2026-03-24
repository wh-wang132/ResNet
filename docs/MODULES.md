# 模块说明

## 概述

本文档说明当前代码结构与各模块职责。默认执行方式为在项目根目录运行：

```bash
uv run src/base_model_main.py ...
```

## 项目结构

```text
src/
├── base_model_main.py          # 基座模型训练入口
├── pruning_main.py             # 剪枝 + 微调入口
├── base_model/
│   ├── dataset.py              # .npy 数据集加载与划分
│   ├── utils.py                # 参数解析/设备/数据加载器/权重加载工具
│   ├── trainer.py              # 训练与验证流程
│   ├── tester.py               # 测试与混淆矩阵
│   ├── visualizer.py           # UMAP 可视化
│   ├── resnet_lightweight.py   # 轻量级 ResNet 模型定义
│   ├── resnet_standard.py      # 标准 ResNet 模型定义
│   ├── confusionMatrix.py      # 混淆矩阵工具
│   ├── lr_scheduler.py         # Warmup + Cosine 调度器
│   └── fix_compiled_weights.py # 修复编译模型权重前缀工具
├── pruning/                    # 剪枝阶段目录
│   ├── args.py                 # 剪枝阶段参数解析
│   ├── checkpoint.py           # 基座 checkpoint 读取与模型恢复
│   ├── evaluator.py            # 剪枝前后评估与统计
│   ├── output.py               # 剪枝输出目录与摘要保存
│   ├── pruner.py               # torch-pruning 封装
│   ├── topology.py             # 剪枝后拓扑提取与签名导出
│   ├── trainer.py              # 剪枝后微调训练
│   ├── utils.py                # 剪枝阶段公共工具入口
│   └── README.md               # 剪枝模块说明
└── qat/                        # QAT 阶段目录（待实现）
```

## 入口与调用关系

- 入口脚本：`src/base_model_main.py`
- 剪枝入口：`src/pruning_main.py`
- 入口职责：
  - 解析参数与准备环境
  - 构建 `DataLoader`
  - 初始化模型
  - 调用 `train_model` / `test_model` / `visualize_umap`

## 关键模块

### base_model_main.py

- 主入口，组织完整训练流程。

### base_model/utils.py

- `parse_args()`：命令行参数定义
- `load_model_map()`：模型名称到构造函数映射
- `create_optimized_dataloader()`：统一 DataLoader 配置
- `compile_model()`：`torch.compile` 编译与验证
- `load_state_dict_safely()`：处理 `_orig_mod.` 前缀的安全加载

### base_model/trainer.py

- `train_model(...)`：训练与验证主循环
- `plot_training_curves(...)`：训练曲线绘制
- 训练产物：最佳模型 checkpoint、TensorBoard 日志、曲线图
- 默认输出目录：`output/base_model/<model>/epochs<epochs>_bs<batch_size>/`

### base_model/tester.py

- `test_model(...)`：加载权重并执行测试
- 生成混淆矩阵与统计摘要

### base_model/resnet_standard.py

- `BasicBlock`、`Bottleneck`、`ResNet2D`
- 导出函数：`resnet18_2d` / `resnet34_2d` / `resnet50_2d`

### base_model/resnet_lightweight.py

- `LightweightBasicBlock2D`、`LightweightResNet2D`
- 导出函数：`resnet6_2d` / `resnet10_2d` / `resnet14_2d`

### base_model/dataset.py

- `NPYDataset`：加载 `.npy` 样本并转换为 `(1, H, W)`
- `data_set_split(...)`：分层划分训练/验证/测试集

### pruning_main.py

- 剪枝 + 微调入口
- 负责组织：
  - 基座 checkpoint 恢复
  - torch-pruning 结构化剪枝
  - 剪枝后评估
  - 微调恢复
  - 剪枝 checkpoint 与摘要保存

### pruning/checkpoint.py

- 读取基座 checkpoint
- 用 `load_model_map()` 恢复默认模型并严格加载权重

### pruning/pruner.py

- 封装 `torch-pruning`
- 执行结构化通道剪枝
- 输出剪枝统计信息

### pruning/topology.py

- 从剪枝后的实际模型提取 `channel_cfg`
- 生成 `architecture_signature`

### pruning/trainer.py

- 剪枝后微调训练
- 复用基座阶段的 AMP、优化器和学习率调度思路
- 保存最优剪枝模型 checkpoint

### pruning/evaluator.py

- 评估剪枝前后验证/测试指标
- 统计参数量与可选 MACs

### pruning/args.py

- 定义 pruning CLI 参数

### pruning/output.py

- 统一 pruning 输出目录与摘要保存逻辑

## 扩展建议

### 添加新模型

1. 在 `resnet_lightweight.py` 或 `resnet_standard.py` 实现模型
2. 在 `utils.py` 的 `load_model_map()` 增加映射
3. 在 `parse_args()` 的 `--model` 选项中注册名称

### 添加新阶段（剪枝/QAT）

1. 复用 `base_model` 中的模型定义和权重加载工具
2. 保持 checkpoint 字段兼容，便于跨阶段衔接
3. 参考 `pruning` 阶段的入口、输出和 checkpoint 组织方式扩展后续阶段
