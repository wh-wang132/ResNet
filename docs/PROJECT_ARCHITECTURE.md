# 项目架构分析

## 1. 总体架构

项目采用分阶段、模块化设计，核心分为三层：

- **入口层**：`src/base_model_main.py`、`src/pruning_main.py`
- **能力层**：`src/base_model/`、`src/pruning/`、`src/qat/`
- **文档与配置层**：`docs/`、`pyproject.toml`、`pixi.toml`

当前已完整实现的是基座训练与剪枝微调流程，QAT 目录已预留但尚未实现完整训练链路。

## 2. 训练阶段架构

### 2.1 基座模型训练（`base_model`）

基座阶段提供完整的训练、验证、测试与可视化能力：

- `dataset.py`：`.npy` 数据集加载与划分
- `resnet_lightweight.py` / `resnet_standard.py`：模型定义
- `trainer.py`：训练主循环、AMP、学习率调度、checkpoint 保存
- `tester.py`：测试评估
- `visualizer.py` / `confusionMatrix.py`：训练曲线、混淆矩阵、UMAP 可视化
- `utils.py`：模型实例化与通用工具函数

输出路径统一落在 `output/base_model/...`，并保留最佳模型与训练统计信息。

### 2.2 剪枝微调（`pruning`）

剪枝阶段基于已训练模型进行结构化剪枝并可选微调：

- `args.py`：参数定义
- `checkpoint.py`：基座 checkpoint 读取与兼容
- `topology.py`：模型拓扑与签名提取
- `pruner.py`：剪枝策略执行
- `trainer.py`：微调流程
- `evaluator.py`：评估流程

输出路径统一落在 `output/pruning/...`，包含剪枝摘要、最佳模型和日志。

## 3. 模块协作关系

1. 入口脚本解析参数并初始化运行环境  
2. 数据模块构建 DataLoader  
3. 模型模块按参数创建具体 ResNet 结构  
4. 训练/剪枝模块执行主流程并周期性评估  
5. 工具模块统一保存 checkpoint、指标和可视化结果

## 4. 当前完成度评估

- ✅ 基座训练链路：完成
- ✅ 剪枝 + 微调链路：完成
- ⏳ QAT（量化感知训练）链路：目录和接口预留，完整实现未完成

整体上，项目已形成“**基座训练 → 剪枝微调 →（预留）QAT**”的可扩展分层架构，代码组织清晰，便于后续继续补齐 QAT 阶段。
