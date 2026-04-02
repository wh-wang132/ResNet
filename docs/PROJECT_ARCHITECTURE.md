# 项目架构分析

## 总体架构

当前项目采用按阶段拆分的结构，主线为：

```text
base_model -> pruning -> （后续）qat / onnx
```

三个阶段的职责分别是：

- `base_model`：训练原始模型并保存结构化基座 checkpoint
- `pruning`：读取基座 checkpoint，执行结构化剪枝与微调，保存 pruning checkpoint
- `qat`：未来读取 pruning checkpoint，恢复剪枝后模型并继续量化感知训练

## 当前完成状态

### 1. `base_model`：已完成

当前已经具备：

- 完整的训练、验证、测试流程
- AMP + WarmupCosineAnnealingLR
- 稳定的数据集切分
- 混淆矩阵与 UMAP 可视化
- 结构化基座 checkpoint 导出

基座 checkpoint 当前不仅保存权重，还保存：

- `train_context`
- `model_structure`
- `model_kwargs`
- `input_tensor_meta`
- `architecture_signature`

这使它已经具备作为下游 pruning 输入的能力。

### 2. `pruning`：已完成主链

当前 pruning 阶段已经具备：

- 通过 `--model` 自动解析基座模型符号链接
- iterative structured pruning
- 每轮评估与可选微调
- 中间轮仅保留内存最佳权重
- 仅最终轮保存 pruning checkpoint
- `pruning_summary.json` 与最终混淆矩阵输出
- 剪枝后拓扑导出：`channel_cfg + architecture_signature`

pruning checkpoint 当前已能完整表达：

- 最终权重
- 所属模型定义信息
- 剪枝后的逐层拓扑
- 剪枝统计信息
- 微调训练上下文

### 3. `qat`：当前为占位阶段

当前仅保留：

- `src/qat/README.md`
- `src/qat/utils.py`

也就是说，QAT 的恢复与训练链路尚未实现，但上游 pruning 产物已经在为这一阶段做契约准备。

## 阶段之间的契约

### 基座 checkpoint -> pruning

pruning 当前依赖的核心字段包括：

- `model_structure.model_name`
- `model_structure.model_kwargs`
- `model_state_dict`
- `best_acc`
- `best_val_loss`
- `input_tensor_meta`

其中模型恢复入口仍然走默认模型工厂函数。

### pruning checkpoint -> 未来 QAT / ONNX

当前 pruning checkpoint 已经提供：

- `model_structure.model_name`
- `model_structure.model_kwargs`
- `model_structure.channel_cfg`
- `model_structure.architecture_signature`
- `model_state_dict`

因此后续 QAT / ONNX 阶段只需补齐“恢复入口”，即可基于这些信息完成剪枝后模型重建。

## 设计上的关键点

### 1. 模型定义与拓扑格式已对齐

`base_model/resnet_lightweight.py` 与 `base_model/resnet_standard.py` 当前都支持：

- 默认构造函数
- `*_from_cfg()` 构造函数
- 逐层 `channel_cfg` 驱动重建

这意味着 pruning 导出的拓扑格式已经能与基础模型定义紧密配合。

### 2. checkpoint 已经从“只存权重”升级成“可恢复对象”

无论是基座 checkpoint 还是 pruning checkpoint，当前都不再只是单纯的 `state_dict`，而是带有：

- 模型结构描述
- 上下文信息
- 结构签名
- 输入信息

这为后续跨阶段恢复提供了稳定基础。

### 3. 入口脚本职责清晰

- `src/base_model_main.py`：只做基座训练链路
- `src/pruning_main.py`：只做 pruning 链路
- `qat` 未来将独立拥有自己的入口，而不是挤进现有入口中

## 当前自然的下一步

当前项目主线已经推进到：

```text
基座训练完成 -> pruning 收敛并稳定产出 pruning checkpoint
```

因此下一阶段最自然的工作是：

1. 在 `qat` 中实现“读取 pruning checkpoint 并恢复模型”的正式入口
2. 基于 pruning checkpoint 开展量化感知训练
3. 再进一步衔接 ONNX / 部署导出链
