# QAT

QAT 阶段负责消费 pruning checkpoint、恢复剪枝结构，并执行保守的 Torch 原生 FX graph mode QAT 微调。

## 环境说明

运行 QAT 阶段前，用户只需要独立手动安装：

- `git`
- `pixi`
- `uv`
- `direnv`（可选）

随后执行：

```bash
pixi install
uv sync
direnv allow
```

说明：

- Python 运行时、工具链、CUDA runtime 等由 `pixi install` 自动提供
- Python 包依赖由 `uv sync` 自动提供
- QAT 阶段只依赖公共环境层，不需要额外 `load_*_env.sh`
- README 默认从仓库根执行；若结合 `autorun/autorun_qat.sh` 使用，必须先激活 `.envrc` 以提供 `REPO_ROOT`
- `direnv` 为推荐方案；若不使用 `direnv` 自动激活，也必须手动提供与 `.envrc` 等价的环境变量

## 职责

QAT 阶段负责：

1. 读取 pruning checkpoint
2. 按 `model_structure.model_name + channel_cfg` 调用 `*_from_cfg()` 恢复剪枝后的浮点模型
3. 在线执行 `prepare_qat_fx`
4. 进行单路径、保守超参的 QAT 微调
5. 导出 prepare 后的 QAT checkpoint
6. 提供从 QAT checkpoint 直接恢复 prepared model 的正式接口

当前阶段不负责：

- ONNX 导出
- AMCT / ATC
- 额外暴露 qconfig / observer / quant scheme 的 CLI 定制接口

## 数据与量化约束

- QAT 固定纯 `fp32`
- 无论在 CPU 还是 GPU 上，训练 / 验证 / 测试统一为 `fp32`
- 量化方案采用固定 canonical 契约：
  - `quantization_scheme_version=3`
  - `scheme_name="torch_fx_qat_cann_v1"`

## 恢复链

### pruning checkpoint -> QAT 训练

QAT 训练入口依赖 pruning checkpoint 中的：

- `model_structure.model_name`
- `model_structure.model_kwargs`
- `model_structure.channel_cfg`
- `model_structure.architecture_signature`
- `model_state_dict`

恢复顺序固定为：

1. 读取 pruning checkpoint
2. 用 `*_from_cfg()` 重建剪枝后的浮点模型
3. 对 `architecture_signature` 执行强校验
4. `strict=True` 加载 pruning 浮点权重
5. 执行 `prepare_qat_fx`
6. 在 prepared model 上做 QAT 微调

说明：

- 所有直接消费 pruning / QAT checkpoint 的 `.pth` 链路，都会对 `architecture_signature` 执行强校验
- 后续 ONNX / AMCT / ATC 阶段统一通过对应 summary 读取并校验上游签名与来源信息

### QAT checkpoint -> 后续导出

QAT checkpoint 提供独立恢复接口：

1. 读取 QAT checkpoint
2. 用 `model_structure.model_name + channel_cfg` 重建剪枝后的浮点模型
3. 对 `architecture_signature` 执行强校验
4. 按当前代码内固定的 canonical QAT 方案重建同一条 `prepare_qat_fx` 图
5. `strict=True` 加载 prepared 权重

ONNX 导出阶段直接消费 `load_qat_checkpoint(...)` 恢复的 QAT checkpoint。

## 输出产物

```text
output/qat/<model>/from_<pruning_exp>/
├── best_qat_prepare_model.pth
├── best_qat_info.txt
├── qat_summary.json
├── Confusion_matrix.png
└── runs/
```

其中：

- `best_qat_prepare_model.pth`：prepare 后的 QAT checkpoint
- `best_qat_info.txt`：每次 best 刷新时追加一行
- `qat_summary.json`：记录 `baseline / quantization_meta / finetune_summary / final / final_topology`

## 阶段边界

- `pruning` 只负责产出 pruning checkpoint
- `qat` 只负责产出 QAT prepare checkpoint
- `onnx` 负责 `convert_fx`、图重写与导出
- `amct` / `atc` 继续承担 Ascend 侧下游流程
