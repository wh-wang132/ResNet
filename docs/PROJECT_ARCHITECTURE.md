# 项目架构分析

## 总体架构

当前项目采用按阶段拆分的结构，真实主线为：

```text
base_model -> pruning -> qat -> onnx -> amct -> atc -> deploy
```

各阶段职责如下：

- `base_model`
  - 训练原始 2D ResNet
  - 输出结构化基座 checkpoint
- `pruning`
  - 读取基座 checkpoint
  - 执行 iterative structured pruning + 微调
  - 输出 pruning checkpoint 与剪枝拓扑
- `qat`
  - 读取 pruning checkpoint
  - 按剪枝拓扑重建浮点模型
  - 执行 FX graph mode QAT
  - 输出 prepare 后 QAT checkpoint
- `onnx`
  - 读取 pruning 或 QAT checkpoint
  - 导出 `pruning_fp16` 或 `qat_convert` ONNX
  - 用 ONNX Runtime 做一致性评估
- `amct`
  - 读取 `qat_convert` ONNX
  - 生成 `deploy_model.onnx` 与 `fake_quant_model.onnx`
- `atc`
  - 读取 `pruning_fp16` ONNX 或 `amct_deploy` ONNX
  - 生成 `Ascend310B4` 目标 `.om`

## 架构分层

### 1. 数据层

- 数据集采用 `Data/<class>/*.npy` 目录组织
- 当前默认输入形状为 `(1, 543, 512)`
- `data_set_split()` 负责：
  - 自然排序扫描
  - 分层切分 train / val / test
  - 将切分结果落盘到 `output/splits/`
  - 后续优先复用 manifest

### 2. 模型定义层

- 轻量模型定义位于 `base_model/resnet_lightweight.py`
- 标准模型定义位于 `base_model/resnet_standard.py`
- 当前仅支持 5 个模型：
  - `resnet6_2d`
  - `resnet10_2d`
  - `resnet14_2d`
  - `resnet18_2d`
  - `resnet34_2d`
- 两类模型都支持：
  - 默认构造函数
  - `*_from_cfg()` 恢复入口
  - 基于 `channel_cfg` 的逐层重建

### 3. 阶段编排层

顶层入口当前位于 `src/` 根目录：

- `src/base_model_main.py`
- `src/pruning_main.py`
- `src/qat_main.py`
- `src/onnx_main.py`
- `src/amct_main.py`
- `src/atc_main.py`

这 6 个入口分别承担单阶段编排，不直接混写彼此逻辑。

### 4. 环境层

当前环境采用“公共层 + 阶段增量层”：

- 公共层：`.envrc`
  - `REPO_ROOT`
  - `PYTHONPATH=$REPO_ROOT/src`
  - `autorun/autorun_*.sh` 与阶段环境脚本统一直接依赖这里提供的 `REPO_ROOT`
  - 当前统一通过 `.envrc` 提供的 `REPO_ROOT` 识别仓库根，不再根据脚本位置推导
- 阶段增量层：
  - `load_base_model_env.sh`
  - `load_onnx_env.sh`
  - `load_amct_env.sh`
  - `load_atc_env.sh`

说明：

- `pixi install` 负责系统工具链、Python 运行时、CUDA runtime、cuDNN、CANN toolkit 等自动安装内容
- `uv sync` 负责 Python 包依赖
- `amct_onnx` 相关 wheel 与算子包当前不在 `uv sync` 管理范围内，是 AMCT 阶段专用的手动补充项

## 当前阶段状态

### `base_model`

已实现：

- 基座训练 / 验证 / 测试
- AMP + `torch.compile`
- Warmup + Cosine Annealing 学习率调度
- TensorBoard、混淆矩阵、UMAP
- 结构化基座 checkpoint

关键输出字段：

- `model_state_dict`
- `train_context`
- `model_structure`
- `input_tensor_meta`
- `architecture_signature`

### `pruning`

已实现：

- 通过 `--model` 自动解析 `best_model.pth` 符号链接
- iterative structured pruning
- 每轮评估与可选微调
- 仅最终轮保存 pruning checkpoint
- `pruning_summary.json`
- 最终混淆矩阵

关键输出字段：

- `model_structure.channel_cfg`
- `model_structure.architecture_signature`
- `pruning_meta`

### `qat`

已实现：

- 读取 pruning checkpoint
- 用 `*_from_cfg()` 重建剪枝后的浮点模型
- `prepare_qat_fx`
- 保守单路径 QAT 微调
- `best_qat_prepare_model.pth`
- `qat_summary.json`

当前约束：

- 数据链固定 `fp32`
- `quantization_meta` 采用最小恢复契约
- QAT 阶段只落 prepare checkpoint，不直接承担导出

### `onnx`

已实现：

- `pruning_fp16`：
  - pruning checkpoint -> FP16 ONNX
- `qat_convert`：
  - QAT checkpoint -> `convert_fx` -> quantized ONNX
- ONNX Runtime 精度评估
- 动态 batch 导出
- `rewrite + validate` 用于 CANN/AMCT/ATC 兼容约束

### `amct`

已实现：

- 只接受仓库 `qat_convert` 导出的 `model_quant.onnx`
- 自动读取同目录 `onnx_summary.json`
- 调用 `amct_onnx.convert_qat_model(...)`
- 输出：
  - `deploy_model.onnx`
  - `fake_quant_model.onnx`
  - `scale_offset_record.txt`
  - `amct_summary.json`

说明：

- AMCT 代码已接入主线
- 运行该阶段前需要额外准备仓库附带的 `amct_onnx` wheel 与算子包

### `atc`

已实现：

- 支持两条输入分支：
  - `pruning_fp16`
  - `amct_deploy`
- 调用 `atc` 编译
- 输出：
  - `.om`
  - `atc_summary.json`
  - `check_result.json` / `fusion_result.json`（若工具链生成）

当前默认：

- `soc_version=Ascend310B4`
- `input_format=NCHW`
- `input_shape` 默认从上游摘要中的输入接口派生，并将 batch 固定为 `1`
- 用户可通过 `--input_shape` 显式覆盖

## 阶段之间的契约

### 基座 checkpoint -> pruning

pruning 当前依赖：

- `model_structure.model_name`
- `model_structure.model_kwargs`
- `model_state_dict`
- `input_tensor_meta`

### pruning checkpoint -> QAT

QAT 当前依赖：

- `model_structure.model_name`
- `model_structure.model_kwargs`
- `model_structure.channel_cfg`
- `model_structure.architecture_signature`
- `model_state_dict`

说明：

- 所有 `.pth` checkpoint 消费步骤统一对 `architecture_signature` 执行强校验

### QAT checkpoint -> ONNX

ONNX 当前依赖：

- `model_structure.model_name`
- `model_structure.model_kwargs`
- `model_structure.channel_cfg`
- `model_structure.architecture_signature`
- `quantization_meta`
- prepare 后 graph 的 `model_state_dict`

说明：

- `qat_convert` 与 `pruning_fp16` 两条 ONNX 导出路径都属于 `.pth` 消费链，因此同样执行 `architecture_signature` 强校验

### ONNX -> AMCT

AMCT 当前依赖：

- ONNX 实体接口与图事实
- `onnx_summary.json.branch == "qat_convert"`
- `onnx_summary.json.onnx_path`
- `onnx_summary.json.model_name`
- `onnx_summary.json.source_checkpoint_path`
- `onnx_summary.json.source_architecture_signature`
- `onnx_summary.json.example_input_shape`
- `onnx_summary.json.opset_version`

说明：

- 由于 ONNX 当前无法可靠嵌入 `architecture_signature`，AMCT 通过 `onnx_summary.json` 消费上游签名引用作为必要补充

### ONNX / AMCT -> ATC

ATC 当前依赖：

- `pruning_fp16` 分支：
  - `model_fp16.onnx`
  - 同目录 `onnx_summary.json`
  - `onnx_summary.json.source_architecture_signature`
- `amct_deploy` 分支：
  - `deploy_model.onnx`
  - 同目录 `amct_summary.json`
  - `amct_summary.json.source_architecture_signature`
  - `amct_summary.json.source_onnx_summary_path` 指向的 `onnx_summary.json`

说明：

- ATC 仍优先校验实体 interface；当 ONNX / deploy ONNX 无法可靠承载签名时，再通过 summary 中的签名引用补充校验
- `amct_deploy -> atc` 不只检查 `amct_summary.json.source_architecture_signature` 是否存在，还会回读 `source_onnx_summary_path` 指向的 `onnx_summary.json`，并要求 `onnx_path`、`source_architecture_signature`、`interface` 三者与 `amct_summary.json` 桥接一致

## 设计上的关键点

### 1. checkpoint 从“只存权重”升级为“可恢复对象”

基座、pruning、QAT 三类 checkpoint 都不只是 `state_dict`，而是带有：

- 模型结构描述
- 输入信息
- 结构签名
- 上下文元数据

这保证了跨阶段恢复是明确契约，而不是隐式猜测。

### 2. `channel_cfg` 是剪枝与量化链的核心桥梁

- pruning 从真实模型中提取 `channel_cfg`
- QAT 用 `*_from_cfg()` 按 `channel_cfg` 重建剪枝结构
- ONNX 导出和后续部署链都建立在这条恢复链上

### 3. ONNX 阶段承担了部署兼容收敛职责

当前 ONNX 阶段不只是“导出一个文件”，还承担：

- `convert_fx`
- 量化图重写
- 结构校验
- ORT 精度对照

因此它是训练端与 Ascend 部署链之间最关键的衔接层。

## 当前重点目标

当前项目的重点目标不是继续扩展模型种类，而是收敛部署前主线：

1. 保持基座 / pruning / QAT 的恢复契约稳定
2. 让 `qat_convert` ONNX 的 rewrite / validate 规则更稳
3. 让 AMCT / ATC 在当前环境分层下可重复运行
4. 保持文档、环境脚本和代码实现三者一致
