# 项目架构分析

## 总体架构

项目采用按阶段拆分的结构，主线为：

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
- 训练期输入 shape 由数据集首个可读样本推断
- `data_set_split()` 负责：
  - 自然排序扫描
  - 推断原始样本 shape，并补齐 `CHW / NCHW`
  - 分层切分 train / val / test
  - 将切分结果落盘到 `output/splits/`
  - 后续优先复用 manifest

### 2. 模型定义层

- 轻量模型定义位于 `base_model/resnet_lightweight.py`
- 标准模型定义位于 `base_model/resnet_standard.py`
- 仅支持 5 个模型：
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

顶层入口位于 `src/` 根目录：

- `src/base_model_main.py`
- `src/pruning_main.py`
- `src/qat_main.py`
- `src/onnx_main.py`
- `src/amct_main.py`
- `src/atc_main.py`

这 6 个入口分别承担单阶段编排，不直接混写彼此逻辑。

### 4. 环境层

环境采用“公共层 + 阶段增量层”：

- 公共层：`.envrc`
  - `REPO_ROOT`
  - `PYTHONPATH=$REPO_ROOT/src`
  - `autorun/autorun_*.sh` 与阶段环境脚本统一直接依赖这里提供的 `REPO_ROOT`
  - 所有脚本统一通过 `.envrc` 提供的 `REPO_ROOT` 识别仓库根
- 阶段增量层：
  - `load_base_model_env.sh`
  - `load_onnx_env.sh`
  - `load_amct_env.sh`
  - `load_atc_env.sh`

说明：

- `pixi install` 负责系统工具链、Python 运行时、CUDA runtime、cuDNN、CANN toolkit 等自动安装内容
- `uv sync` 负责 Python 包依赖
- `amct_onnx` 相关 wheel 与算子包不在 `uv sync` 管理范围内，由 AMCT 阶段按目标环境单独准备

## 阶段状态

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

约束：

- 数据链固定 `fp32`
- `quantization_meta` 采用最小恢复契约
- QAT 阶段输出 prepare checkpoint，由后续 ONNX 导出阶段继续消费

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

默认：

- `soc_version=Ascend310B4`
- `input_format=NCHW`
- `input_shape` 默认从上游摘要中的输入接口派生，并将 batch 固定为 `1`
- 若用户显式传入 `--input_shape`，其输入名与各维度必须与自动派生结果完全一致，否则直接报错

## 阶段之间的契约

### 基座 checkpoint -> pruning

pruning 依赖：

- `model_structure.model_name`
- `model_structure.model_kwargs`
- `model_state_dict`
- `input_tensor_meta`

### pruning checkpoint -> QAT

QAT 依赖：

- `model_structure.model_name`
- `model_structure.model_kwargs`
- `model_structure.channel_cfg`
- `model_structure.architecture_signature`
- `model_state_dict`

说明：

- 所有 `.pth` checkpoint 消费步骤统一对 `architecture_signature` 执行强校验

### QAT checkpoint -> ONNX

ONNX 依赖：

- `model_structure.model_name`
- `model_structure.model_kwargs`
- `model_structure.channel_cfg`
- `model_structure.architecture_signature`
- `quantization_meta`
- prepare 后 graph 的 `model_state_dict`

说明：

- `qat_convert` 与 `pruning_fp16` 两条 ONNX 导出路径都属于 `.pth` 消费链，因此同样执行 `architecture_signature` 强校验

### ONNX -> AMCT

AMCT 依赖：

- ONNX 实体接口与图事实
- `onnx_summary.json.branch == "qat_convert"`
- `onnx_summary.json.onnx_path`
- `onnx_summary.json.model_name`
- `onnx_summary.json.source_checkpoint_path`
- `onnx_summary.json.source_architecture_signature`
- `onnx_summary.json.example_input_shape`
- `onnx_summary.json.opset_version`

说明：

- AMCT 统一通过同目录 `onnx_summary.json` 读取并校验上游签名、来源路径与输入接口信息

### ONNX / AMCT -> ATC

ATC 依赖：

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

- `pruning_fp16` 分支通过 `onnx_summary.json.source_architecture_signature` 引用并校验上游签名，同时结合 `onnx_path` 与 `interface` 完成摘要契约校验
- `amct_deploy -> atc` 在 pixi 环境下不导入 ONNX，而是基于 `amct_summary.json` 与上游 `onnx_summary.json` 做摘要契约校验
- `amct_summary.json.deploy_interface` 必须与 `source_interface` 一致，且 `source_onnx_summary_path` 指向的 `onnx_summary.json` 必须在 `onnx_path`、`source_architecture_signature`、`interface` 三者上与 `amct_summary.json` 桥接一致
- 因而当前 `amct_deploy -> atc` 形成的是摘要桥接闭环与 summary 内部一致性校验，而不再在 ATC 阶段做 deploy ONNX 实体复核

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

## 重点目标

项目的重点目标不是继续扩展模型种类，而是收敛部署前主线：

1. 保持基座 / pruning / QAT 的恢复契约稳定
2. 让 `qat_convert` ONNX 的 rewrite / validate 规则更稳
3. 让 AMCT / ATC 在当前环境分层下可重复运行
4. 保持文档、环境脚本和代码实现三者一致
