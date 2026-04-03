# QAT

该目录当前处于占位阶段，尚未实现完整的量化感知训练链路。

未来 QAT 阶段同样默认继承项目的标准 `pixi + uv` 运行环境：`pixi` 负责系统工具链环境，`uv` 负责 Python 依赖与运行入口，`direnv` 负责自动激活。

## 当前状态

目前 `src/qat/` 仅包含：

- `README.md`
- `utils.py`

其中 `utils.py` 主要复用 `base_model` 中与未来 QAT 兼容的稳定公共工具，例如：

- `load_model_map`
- `load_state_dict_safely`
- `get_raw_model`
- `build_architecture_signature`
- `create_optimized_dataloader`

## 未来职责

QAT 阶段的目标不是训练基座模型，也不是执行剪枝，而是：

1. 读取 pruning checkpoint
2. 恢复剪枝后的模型结构
3. 执行量化感知训练
4. 产出 QAT 训练产物与后续导出输入

## 预期上游输入

未来 QAT 将直接消费 pruning checkpoint。当前 pruning checkpoint 已经包含恢复所需的关键字段：

- `model_structure.model_name`
- `model_structure.model_kwargs`
- `model_structure.channel_cfg`
- `model_structure.architecture_signature`
- `model_state_dict`

因此，QAT 阶段未来只需补齐“恢复入口 + QAT 训练流程”即可。
