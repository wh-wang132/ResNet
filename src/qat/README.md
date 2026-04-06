# QAT

QAT 阶段已经落地，当前负责消费 pruning checkpoint、恢复剪枝结构，并执行保守的 Torch 原生 FX graph mode QAT 微调。

当前 QAT 精度策略已固定为纯 FP32：无论在 CPU 还是 GPU 上，数据集均通过 `base_model.dataset` 的 fp32 接口加载，训练/验证/测试计算也统一为 FP32。

QAT 同样默认继承项目的标准 `pixi + uv` 运行环境：`pixi` 负责系统工具链环境，`uv` 负责 Python 依赖与运行入口，`direnv` 负责自动激活。

## 当前职责

QAT 阶段当前只负责：

1. 读取 pruning checkpoint
2. 按 `model_structure.model_name + channel_cfg` 调用 `*_from_cfg()` 恢复剪枝后的浮点模型
3. 在线执行 `prepare_qat_fx`
4. 进行单路径、保守超参的 QAT 微调
5. 导出 **prepare 后** 的 QAT checkpoint
6. 提供从 QAT checkpoint 直接恢复 prepared model 的正式接口

当前阶段**不负责**：

- ONNX 导出
- `torch.convert` 后的 int8 模型落盘
- 保证 FX QAT 图已经与 CANN 8.5.0 ATC 兼容

其中 ONNX 导出已经由独立的 `src/onnx_main.py` 实现；QAT 阶段本身仍只负责产出 QAT checkpoint，不直接承担导出逻辑。

## 目录结构

当前 `src/qat/` 包含：

- `args.py`
- `checkpoint.py`
- `quantization.py`
- `trainer.py`
- `evaluator.py`
- `output.py`
- `utils.py`
- `README.md`

入口脚本位于：

- `src/qat_main.py`

## 当前恢复链

QAT 训练入口当前依赖 pruning checkpoint 中的以下字段：

- `model_structure.model_name`
- `model_structure.model_kwargs`
- `model_structure.channel_cfg`
- `model_structure.architecture_signature`
- `model_state_dict`

恢复顺序固定为：

1. 读取 pruning checkpoint
2. 用 `*_from_cfg()` 重建剪枝后的浮点模型
3. `strict=True` 加载 pruning 浮点权重
4. 执行 `prepare_qat_fx`
5. 在 prepared model 上做 QAT 微调

QAT checkpoint 则提供独立恢复接口：

1. 读取 QAT checkpoint
2. 用 `model_structure.model_name + channel_cfg` 重建剪枝后的浮点模型
3. 按当前代码内固定的 canonical QAT 方案重建同一条 `prepare_qat_fx` 图
4. `strict=True` 加载 QAT prepared 权重

当前 ONNX 导出阶段已经直接消费 `load_qat_checkpoint(...)` 这条 QAT 恢复接口，而不是重新回退到 pruning checkpoint。

当前只保证兼容新的最小 `quantization_meta` 契约；旧版 QAT checkpoint 需要重新跑一遍新的 QAT 后再进入后续导出链。

## 量化约束

当前量化方案按保守单路径固定：

- 权重：`qint8`、`per-channel symmetric`
- 激活及其余量化：`per-tensor`
- 使用显式自定义 `QConfigMapping`
- 最终不执行 `torch.convert`
- 落盘产物是 **prepare 后 graph 的权重**
- `quantization_meta` 当前只保留最小恢复信息：
  - `quantization_scheme_version=3`
  - `scheme_name="torch_fx_qat_cann_v1"`
  - `example_input_shape`

## 输出产物

```text
output/qat/<model>/from_<pruning_exp>/
├── best_qat_prepare_model.pth
├── best_qat_info.txt
├── qat_summary.json
├── Confusion_matrix.png     # 仅最终测试阶段生成
└── runs/
```

其中：

- `best_qat_prepare_model.pth`：prepare 后的 QAT checkpoint
- `best_qat_info.txt`：每次 best 刷新时追加一行
- `qat_summary.json`：记录 `baseline / quantization_meta / finetune_summary / final / final_topology`

## 阶段边界

- `pruning` 只负责产出 pruning checkpoint
- `qat` 只负责产出 QAT prepare checkpoint
- 后续 ONNX 导出将只读取 QAT checkpoint，通过 QAT 恢复接口重建对象并验证部署兼容性

`source_pruning_checkpoint_path` 仅用于实验溯源，不参与 QAT checkpoint 的恢复。
