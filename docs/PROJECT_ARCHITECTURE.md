# 项目架构分析

## 总体架构

当前项目采用按阶段拆分的结构，主线为：

```text
base_model -> pruning -> qat -> （后续）onnx / deploy
```

三个阶段的职责分别是：

- `base_model`：训练原始模型并保存结构化基座 checkpoint
- `pruning`：读取基座 checkpoint，执行结构化剪枝与微调，保存 pruning checkpoint
- `qat`：读取 pruning checkpoint，恢复剪枝后模型并执行保守单路径 QAT

## 当前完成状态

### 1. `base_model`：已完成

当前已经具备：

- 完整的训练、验证、测试流程
- AMP + WarmupCosineAnnealingLR
- 稳定的数据集切分
- 数据集划分清单落盘与优先复用
- 混淆矩阵与 UMAP 可视化
- 结构化基座 checkpoint 导出

基座 checkpoint 当前不仅保存权重，还保存：

- `train_context`
- `model_structure`
- `model_kwargs`
- `input_tensor_meta`
- `architecture_signature`

这使它已经具备作为下游 pruning 输入的能力。

当前 `base_model.dataset.data_set_split()` 还会将划分结果落盘到 `output/splits/`，并在后续运行时优先读取已落盘清单。该 manifest 当前保存相对路径 `data_dir=Data`，作为训练端与后续推理端共享的数据划分真值。

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

### 3. `qat`：已完成主链

当前 QAT 阶段已经具备：

- 读取 pruning checkpoint
- 按 `model_name + channel_cfg` 调用 `*_from_cfg()` 重建剪枝结构
- 在线执行 `prepare_qat_fx`
- 保守单路径 QAT 微调
- 导出 prepare 后的 QAT checkpoint
- `qat_summary.json` 与最终混淆矩阵输出

当前 QAT 产物明确限定为 **prepare 后 graph 的权重**，暂不负责 `torch.convert` 与 ONNX 导出。

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

### pruning checkpoint -> QAT

当前 pruning checkpoint 已经提供：

- `model_structure.model_name`
- `model_structure.model_kwargs`
- `model_structure.channel_cfg`
- `model_structure.architecture_signature`
- `model_state_dict`

QAT 当前已经基于这些字段完成恢复与训练链路。

### QAT checkpoint -> 未来 ONNX / 部署

当前 QAT checkpoint 保存：

- `model_structure.model_name`
- `model_structure.model_kwargs`
- `model_structure.channel_cfg`
- `model_structure.architecture_signature`
- `quantization_meta`
- prepared model 的 `model_state_dict`

这为后续 ONNX / 部署阶段通过 QAT checkpoint 直接恢复同一条 prepare 图提供了契约基础。未来导出链应只消费 QAT checkpoint，不再回退到 pruning checkpoint。

## 设计上的关键点

### 1. 模型定义与拓扑格式已对齐

`base_model/resnet_lightweight.py` 与 `base_model/resnet_standard.py` 当前都支持：

- 默认构造函数
- `*_from_cfg()` 构造函数
- 逐层 `channel_cfg` 驱动重建

这意味着 pruning 导出的拓扑格式已经能与基础模型定义紧密配合。

### 2. checkpoint 已经从“只存权重”升级成“可恢复对象”

无论是基座 checkpoint、pruning checkpoint 还是 QAT checkpoint，当前都不再只是单纯的 `state_dict`，而是带有：

- 模型结构描述
- 上下文信息
- 结构签名
- 输入信息

这为后续跨阶段恢复提供了稳定基础。

### 3. 入口脚本职责清晰

- `src/base_model_main.py`：只做基座训练链路
- `src/pruning_main.py`：只做 pruning 链路
- `src/qat_main.py`：只做 QAT 恢复与量化感知训练链路

## 当前自然的下一步

当前项目主线已经推进到：

```text
基座训练完成 -> pruning 收敛并稳定产出 pruning checkpoint -> QAT 产出 prepare checkpoint
```

因此下一阶段最自然的工作是：

1. 在 ONNX / 部署阶段实现消费 QAT checkpoint 的恢复入口
2. 验证 Torch FX QAT 产物与后续导出链的兼容性
3. 再进一步衔接 ONNX / 部署导出链
