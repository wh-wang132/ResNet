# 模块说明

## 概述

本文档说明当前代码结构与各模块职责。当前项目已经形成两条已实现主线：

- `base_model`：基座训练、测试与可视化
- `pruning`：结构化剪枝、微调与 pruning checkpoint 导出

`qat` 当前已经具备独立主链，负责消费 pruning checkpoint 并导出 QAT prepare checkpoint。

## 项目结构

```text
src/
├── base_model_main.py
├── pruning_main.py
├── qat_main.py
├── base_model/
│   ├── args.py
│   ├── dataset.py
│   ├── utils.py
│   ├── trainer.py
│   ├── tester.py
│   ├── visualizer.py
│   ├── confusionMatrix.py
│   ├── lr_scheduler.py
│   ├── resnet_lightweight.py
│   └── resnet_standard.py
├── pruning/
│   ├── args.py
│   ├── checkpoint.py
│   ├── evaluator.py
│   ├── output.py
│   ├── pruner.py
│   ├── topology.py
│   ├── trainer.py
│   ├── utils.py
│   └── README.md
└── qat/
    ├── args.py
    ├── checkpoint.py
    ├── evaluator.py
    ├── output.py
    ├── quantization.py
    ├── trainer.py
    ├── utils.py
    └── README.md
```

## 入口脚本

### `src/base_model_main.py`

负责：

- 解析基座训练参数
- 构建数据集与 DataLoader
- 初始化模型
- 调用训练、测试、UMAP 可视化主流程

### `src/pruning_main.py`

负责：

- 解析 pruning 参数
- 读取基座模型符号链接
- 恢复基座 checkpoint
- 执行 iterative pruning
- 每轮评估与可选微调
- 导出 pruning checkpoint 与 summary

### `src/qat_main.py`

负责：

- 解析 QAT 参数
- 读取 pruning checkpoint
- 恢复剪枝后的浮点模型
- 执行 `prepare_qat_fx`
- 执行保守单路径 QAT 微调
- 导出 QAT prepare checkpoint 与 summary

## `base_model` 模块职责

### `base_model/args.py`

- 定义基座训练 CLI 参数
- 与 pruning CLI 解耦

### `base_model/dataset.py`

- `.npy` 数据集加载
- 稳定的训练/验证/测试划分
- 自然排序保证切分可复现

### `base_model/utils.py`

提供跨阶段可复用的稳定公共工具：

- `load_model_map()`
- `create_optimized_dataloader()`
- `load_state_dict_safely()`
- `get_raw_model()`
- `build_architecture_signature()`
- 设备、显存与 `torch.compile` 相关辅助函数

### `base_model/trainer.py`

- 基座训练主循环
- AMP、优化器、学习率调度
- 最优模型判定与 checkpoint 保存
- 保存 `model_structure`、`train_context`、`architecture_signature`

### `base_model/tester.py`

- 加载基座模型权重
- 测试集评估
- 生成 `Confusion_matrix.png`

### `base_model/visualizer.py`

- UMAP 特征可视化

### `base_model/resnet_lightweight.py`

- 轻量级 ResNet 定义
- 默认工厂函数：`resnet6_2d / resnet10_2d / resnet14_2d`
- `*_from_cfg()` 恢复入口
- 逐层 `channel_cfg` 支持

### `base_model/resnet_standard.py`

- 标准 ResNet 定义
- 默认工厂函数：`resnet18_2d / resnet34_2d / resnet50_2d`
- `*_from_cfg()` 恢复入口
- 逐层 `channel_cfg` 支持

## `pruning` 模块职责

### `pruning/args.py`

- 定义 pruning CLI 参数
- `--pruning_ratio` 在入口统一规范到 2 位小数

### `pruning/checkpoint.py`

- 解析 `output/base_model/<model>/best_model.pth`
- 恢复基座 checkpoint
- 用默认模型工厂严格加载基座权重

### `pruning/pruner.py`

- 封装 `torch-pruning`
- 执行单轮结构化通道剪枝
- 返回该轮剪枝统计

### `pruning/topology.py`

- 从剪枝后的实际模型提取 `channel_cfg`
- 生成 `architecture_signature`
- 保证 pruning 产物与 `base_model` 的逐层配置模型定义对齐

### `pruning/trainer.py`

- 剪枝后微调
- 每轮仅保留内存中的最佳权重
- 仅最终轮保存 pruning checkpoint

### `pruning/evaluator.py`

- 验证/测试评估
- 参数量与 MACs 统计
- 最终测试阶段生成混淆矩阵

### `pruning/output.py`

- pruning 输出目录命名
- `pruning_summary.json` 保存

### `pruning/utils.py`

- 复用 `base_model` 中稳定的公共函数
- 提供 pruning 阶段自己的路径与 `pruning_meta` 收敛工具

## `qat` 模块职责

### `qat/README.md`

- 说明当前 QAT 目标、输出产物与阶段边界

### `qat/args.py`

- 定义 QAT CLI 参数

### `qat/checkpoint.py`

- 读取 pruning checkpoint
- 按 `*_from_cfg()` 重建剪枝后的浮点模型
- 严格加载 pruning 权重

### `qat/quantization.py`

- 构建保守量化约束下的 `QConfigMapping`
- 执行 `prepare_qat_fx`
- 管理 observer / BN 冻结策略

### `qat/trainer.py`

- QAT 微调主循环
- 最优模型判定与 prepare checkpoint 保存

### `qat/evaluator.py`

- 复用 pruning 阶段稳定的验证、测试与混淆矩阵逻辑

### `qat/output.py`

- QAT 输出目录命名
- `qat_summary.json` 保存

### `qat/utils.py`

- 复用 `base_model` 中稳定的公共工具
- 提供 QAT 阶段自己的路径工具

## 当前阶段边界

- `base_model`：产出稳定的基座 checkpoint
- `pruning`：消费基座 checkpoint，产出 pruning checkpoint
- `qat`：消费 pruning checkpoint，负责恢复与量化训练，并产出 QAT prepare checkpoint
