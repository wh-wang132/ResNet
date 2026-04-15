# 模块说明

## 概述

项目采用完整六阶段训练端结构：

```text
base_model -> pruning -> qat -> onnx -> amct -> atc
```

本文档按代码实际结构说明各模块职责。

## 项目结构

```text
src/
├── base_model/
│   ├── __main__.py
│   ├── args.py
│   ├── confusionMatrix.py
│   ├── dataset.py
│   ├── lr_scheduler.py
│   ├── plotting.py
│   ├── resnet_lightweight.py
│   ├── resnet_standard.py
│   ├── tester.py
│   ├── trainer.py
│   ├── utils.py
│   └── visualizer.py
├── pruning/
│   ├── __main__.py
│   ├── args.py
│   ├── checkpoint.py
│   ├── evaluator.py
│   ├── output.py
│   ├── pruner.py
│   ├── topology.py
│   ├── trainer.py
│   ├── utils.py
│   └── README.md
├── qat/
│   ├── __main__.py
│   ├── args.py
│   ├── checkpoint.py
│   ├── evaluator.py
│   ├── output.py
│   ├── quantization.py
│   ├── trainer.py
│   ├── utils.py
│   └── README.md
├── onnx_export/
│   ├── __main__.py
│   ├── args.py
│   ├── evaluator.py
│   ├── exporter.py
│   ├── output.py
│   ├── rewrite.py
│   ├── utils.py
│   └── validate.py
├── amct/
│   ├── __main__.py
│   ├── __init__.py
│   ├── args.py
│   ├── converter.py
│   ├── output.py
│   └── utils.py
└── atc/
    ├── __main__.py
    ├── __init__.py
    ├── args.py
    ├── converter.py
    ├── output.py
    └── utils.py
```

## 入口模块

### `src/base_model/__main__.py`

负责：

- 解析基座训练参数
- 调用数据切分与 DataLoader
- 初始化模型
- 执行训练 / 测试 / UMAP

### `src/pruning/__main__.py`

负责：

- 解析 pruning 参数
- 读取基座模型符号链接
- 恢复基座 checkpoint
- 执行 iterative pruning
- 每轮评估与可选微调
- 导出 pruning checkpoint 与 summary

### `src/qat/__main__.py`

负责：

- 解析 QAT 参数
- 读取 pruning checkpoint
- 恢复剪枝后的浮点模型
- `prepare_qat_fx`
- 执行保守 QAT 微调
- 导出 QAT prepare checkpoint 与 summary

### `src/onnx_export/__main__.py`

负责：

- 解析 ONNX 导出参数
- 调度 `pruning_fp16` / `qat_convert` 两条导出分支
- 构建测试 DataLoader
- 做 Torch / ORT 精度对照
- 输出 `onnx_summary.json`

### `src/amct/__main__.py`

负责：

- 解析 AMCT 参数
- 调用 AMCT 转换逻辑
- 保存 `amct_summary.json`

### `src/atc/__main__.py`

负责：

- 解析 ATC 参数
- 调用 ATC 编译逻辑
- 保存 `atc_summary.json`

## `base_model` 模块职责

### `base_model/args.py`

- 定义基座训练 CLI 参数
- 管理训练、测试、UMAP、性能选项

### `base_model/dataset.py`

- `.npy` 数据集加载
- 自然排序扫描
- 稳定 train / val / test 划分
- `output/splits` manifest 落盘与复用

### `base_model/utils.py`

提供跨阶段复用的公共函数：

- `load_model_map()`
- `create_optimized_dataloader()`
- `load_state_dict_safely()`
- `get_raw_model()`
- `build_architecture_signature()`
- 设备、显存、`torch.compile` 相关辅助函数

### `base_model/trainer.py`

- 基座训练主循环
- AMP、优化器、学习率调度
- TensorBoard
- 最优模型判定与结构化 checkpoint 保存

### `base_model/tester.py`

- 加载基座权重
- 测试集评估
- 生成混淆矩阵

### `base_model/visualizer.py`

- 提取特征
- 执行 PCA + UMAP 可视化

### `base_model/resnet_lightweight.py`

- 轻量级 ResNet 定义
- 默认工厂：
  - `resnet6_2d`
  - `resnet10_2d`
  - `resnet14_2d`
- `*_from_cfg()` 恢复入口

### `base_model/resnet_standard.py`

- 标准 ResNet 定义
- 默认工厂：
  - `resnet18_2d`
  - `resnet34_2d`
- `*_from_cfg()` 恢复入口

## `pruning` 模块职责

### `pruning/args.py`

- 定义 pruning CLI 参数
- `--pruning_ratio` 在入口统一规范到 2 位小数

### `pruning/checkpoint.py`

- 扫描 `output/base_model/<model>/` 下实验目录并选择最佳 `best_model.pth`
- 严格恢复基座 checkpoint
- 对 base checkpoint 执行 `architecture_signature` 强校验

### `pruning/pruner.py`

- 封装 `torch-pruning`
- 执行单轮结构化通道剪枝
- 计算 step ratio 与剪枝统计

### `pruning/topology.py`

- 从剪枝后的真实模型提取 `channel_cfg`
- 生成 `architecture_signature`
- 保证 pruning 产物与 `*_from_cfg()` 恢复入口对齐

### `pruning/trainer.py`

- 剪枝后微调
- 每轮仅保留内存中的最佳权重
- 最终轮保存 pruning checkpoint

### `pruning/evaluator.py`

- 验证 / 测试评估
- 参数量与 MACs 统计
- 最终测试阶段生成混淆矩阵

### `pruning/output.py`

- pruning 输出目录命名
- 保存 `pruning_summary.json`

### `pruning/utils.py`

- 复用 `base_model` 中稳定公共函数
- 提供 pruning 阶段的路径与元数据辅助函数

## `qat` 模块职责

### `qat/args.py`

- 定义 QAT CLI 参数

### `qat/checkpoint.py`

- 读取 pruning / QAT checkpoint
- 用 `*_from_cfg()` 恢复剪枝结构
- 对 pruning / QAT checkpoint 执行 `architecture_signature` 强校验
- 严格加载权重

### `qat/quantization.py`

- 构建 canonical QAT 方案
- 执行 `prepare_qat_fx`
- 管理 `quantization_meta`
- 管理 observer / BN 冻结策略

### `qat/trainer.py`

- QAT 微调主循环
- 最优模型判定与 prepare checkpoint 保存

### `qat/evaluator.py`

- 复用 pruning 阶段的验证 / 测试逻辑
- 明确在 QAT 阶段禁用 AMP

### `qat/output.py`

- QAT 输出目录命名
- 保存 `qat_summary.json`

### `qat/utils.py`

- 复用公共工具
- 提供 QAT 阶段路径与相对路径辅助函数

## `onnx_export` 模块职责

### `onnx_export/args.py`

- 定义 ONNX 导出相关参数

### `onnx_export/exporter.py`

- 按分支构建导出产物
- 管理导出设备、输入 dtype、动态 batch

### `onnx_export/evaluator.py`

- 创建 ORT session
- 执行 Torch / ORT 评估
- 生成导出后混淆矩阵

### `onnx_export/rewrite.py`

- 对 `qat_convert` ONNX 做 graph rewrite
- 调整量化节点连接模式
- 收敛到 CANN/AMCT/ATC 兼容图形态

### `onnx_export/validate.py`

- 校验 ONNX opset、输入输出 dtype、量化节点模式
- 验证 QAT 图是否满足当前约束

### `onnx_export/output.py`

- ONNX 输出目录命名
- 保存 `onnx_summary.json`
- 在 ONNX 无法可靠嵌入 `architecture_signature` 时，通过 summary 传递上游签名引用

### `onnx_export/utils.py`

- ONNX 阶段通用路径与 DataLoader 工具

## `amct` 模块职责

### `amct/args.py`

- 定义 AMCT CLI 参数

### `amct/converter.py`

- 校验输入 `model_quant.onnx`
- 校验 `onnx_summary.json`
- 在 ONNX 实体无法可靠承载签名时，消费 summary 中的 `architecture_signature` 引用
- 调用 `amct_onnx.convert_qat_model(...)`
- 校验 deploy / fakequant ONNX 输出
- 在 `amct_summary.json` 中继续传递上游签名引用

### `amct/output.py`

- AMCT 输出目录命名
- 保存 `amct_summary.json`

### `amct/utils.py`

- 仓库根路径解析
- JSON 读取
- 路径解析与文件存在性检查

## `atc` 模块职责

### `atc/args.py`

- 定义 ATC CLI 参数

### `atc/converter.py`

- 按 `pruning_fp16` / `amct_deploy` 加载输入契约
- 在 pixi 环境下消费 summary 契约与上游 `architecture_signature` 引用，不再在 ATC 阶段导入 ONNX 做 deploy 实体复核
- 构建 `atc` 子进程命令
- 收集 `.om` 与工具产物
- 生成 `atc_summary.json`

### `atc/output.py`

- ATC 输出目录命名
- 保存 `atc_summary.json`

### `atc/utils.py`

- 构建 `atc` 子进程环境
- 路径解析与文件检查
- 汇总工具链环境变量

## 当前阶段边界

- `base_model`：产出结构化基座 checkpoint
- `pruning`：消费基座 checkpoint，产出 pruning checkpoint
- `qat`：消费 pruning checkpoint，产出 QAT prepare checkpoint
- `onnx`：消费 pruning / QAT checkpoint，产出 ONNX 与评估摘要
- `amct`：消费 `qat_convert` ONNX，产出 deploy / fakequant ONNX
- `atc`：消费 `pruning_fp16` 或 `amct_deploy` ONNX，产出 `.om`
