# ResNet 仓库工作约定

## 项目定位

- 本仓库是本科毕设“基于昇腾 AI 架构的高效化无人机射频信号识别”的训练端实现。
- 云端训练端仓库为 `wh-wang132/ResNet`，下游推理端仓库为 `wh-wang132/ResNet_Acl`。
- 本仓库不是单一 ResNet 训练 demo，而是面向 Ascend 部署的六阶段流水线：

```text
base_model -> pruning -> qat -> onnx_export -> amct -> atc -> ResNet_Acl
```

- 修改代码时优先保持阶段边界、产物命名、summary 字段和跨仓库接口稳定。
- 与用户沟通、仓库说明、项目级文档默认使用中文；代码标识符沿用现有英文命名。

## 顶层结构

- `src/`：训练端六阶段代码主体。
- `Data/`：按类别目录组织的 `.npy` 数据集，格式为 `Data/<class>/*.npy`。
- `output/`：训练、剪枝、QAT、ONNX、AMCT、ATC 与 split manifest 产物根目录。
- `docs/`：架构、模块、CLI、数据准备、训练、剪枝等说明文档。
- `autorun/`：六阶段批处理脚本。
- `scripts/`：公共和阶段增量环境加载脚本。
- `amct_onnx/`：AMCT 专用 wheel 与算子包，不属于 `uv sync` 自动安装依赖。
- `pixi.toml`：Pixi 环境、CANN 依赖和 `autorun-*` task。
- `pyproject.toml`：Python 依赖定义。

## 阶段边界

| 阶段 | 入口 | 主要职责 | 关键输入 | 关键输出 |
| --- | --- | --- | --- | --- |
| `base_model` | `uv run python -m base_model` | 基座训练、验证、测试、UMAP、混淆矩阵 | `Data/<class>/*.npy` | `output/base_model/<model>/<experiment>/best_model.pth` |
| `pruning` | `uv run python -m pruning` | 自动选择最佳基座，执行 iterative structured pruning 与微调 | 基座 checkpoint | `best_pruned_model.pth`、`pruning_summary.json` |
| `qat` | `uv run python -m qat` | 按剪枝拓扑恢复模型，执行 FX graph mode QAT | pruning checkpoint | `best_qat_prepare_model.pth`、`qat_summary.json` |
| `onnx_export` | `uv run python -m onnx_export` | 导出 `pruning_fp16` 或 `qat_convert` ONNX，执行 ORT 评估 | pruning/QAT checkpoint | `model_fp16.onnx` 或 `model_quant.onnx`、`onnx_summary.json` |
| `amct` | `uv run python -m amct` | 消费 `qat_convert` ONNX 并生成 Ascend 侧 deploy/fakequant ONNX | `model_quant.onnx` | `deploy_model.onnx`、`fake_quant_model.onnx`、`amct_summary.json` |
| `atc` | `pixi run python -m atc` | 编译 `pruning_fp16` 或 `amct_deploy` 分支为 `.om` | `model_fp16.onnx` 或 `deploy_model.onnx` | `.om`、`atc_summary.json` |

不要把一个阶段的核心逻辑混写到另一个阶段。共享能力优先放在已有的 `utils.py`、`checkpoint.py`、`output.py` 或明确的公共函数里，并保持调用方向清晰。

## 核心模块

- `src/base_model/dataset.py`
  - 扫描 `Data/<class>/*.npy`。
  - 使用自然排序建立 `class_names` 与 `class_to_idx`。
  - 从首个可读样本推断 `sample_shape_chw` 与 `input_shape_nchw`。
  - 固定分层切分 train/val/test = `0.6/0.2/0.2`，默认 seed 为 `42`。
  - split manifest 写入 `output/splits/`，后续阶段复用。
- `src/base_model/resnet_lightweight.py`
  - 定义 `resnet6_2d`、`resnet10_2d`、`resnet14_2d`。
  - 提供 `*_from_cfg()`，用于按剪枝后的 `channel_cfg` 恢复拓扑。
- `src/base_model/resnet_standard.py`
  - 定义 `resnet18_2d`、`resnet34_2d`。
  - 同样提供 `*_from_cfg()`。
- `src/base_model/trainer.py` / `tester.py`
  - 负责基座训练、验证、测试、结构化 checkpoint、训练曲线与混淆矩阵。
- `src/pruning/checkpoint.py`
  - 自动扫描 `output/base_model/<model>/` 下候选实验。
  - 读取 `best_val_acc_info.txt`，按验证准确率优先、验证损失次优选择最佳基座。
- `src/pruning/topology.py`
  - 从真实剪枝后模型提取 `channel_cfg` 与 `architecture_signature`。
- `src/qat/checkpoint.py`
  - 从 pruning/QAT checkpoint 恢复剪枝拓扑与 QAT prepare 图。
- `src/onnx_export/exporter.py`
  - 维护 `pruning_fp16` 与 `qat_convert` 两条导出分支。
  - 固定 ONNX opset 16，输入名 `input`，输出名 `logits`，动态 batch。
- `src/onnx_export/rewrite.py` / `validate.py`
  - 收敛 QAT ONNX 到 CANN/AMCT/ATC 可接受的图结构。
- `src/amct/converter.py`
  - 只接受仓库 `qat_convert` 分支产出的 `model_quant.onnx`。
  - 校验同目录 `onnx_summary.json`、接口、路径和结构签名。
- `src/atc/converter.py`
  - 只接受 `pruning_fp16/model_fp16.onnx` 或 AMCT 的 `deploy_model.onnx`。
  - 从 summary 派生 batch=1 的 `--input_shape` 并执行 ATC 编译。

## 数据接口规范

- 数据根目录默认为 `Data/`，可通过各阶段 CLI 的 `--data_dir` 覆盖。
- 一级子目录名就是类别名，类别映射由自然排序动态推断；不要硬编码 `num_classes` 或手写类别顺序。
- `.npy` 样本只支持：
  - 2D `(H, W)`：加载后自动补通道维，成为 `CHW=(1, H, W)`。
  - 3D `(C, H, W)`：直接沿用通道维。
- 同一数据集内所有样本 shape 必须一致。
- Dataset 返回 `(tensor, int_label)`，模型训练使用交叉熵，标签必须与 `class_to_idx` 一致。
- 当前 split manifest 位于 `output/splits/dataset_split__train0.60_val0.20_test0.20_seed42.json` 这类路径，至少包含：
  - `class_names`
  - `class_to_idx`
  - `train_files`
  - `val_files`
  - `test_files`
- 每条样本记录至少包含：
  - `path`
  - `label_name`
  - `label_idx`
- 修改切分比例、随机种子、类别扫描规则或 manifest 格式时，必须同步检查训练端后续阶段和下游 `ResNet_Acl`。

## 数据精度约定

- `base_model`：`--data_dtype` 支持 `fp16` 或 `fp32`，默认 `fp16`。
- `pruning`：`--data_dtype` 支持 `fp16` 或 `fp32`，默认 `fp16`。
- `qat`：固定 `fp32`。
- `onnx_export` 测试评估：固定 `fp32` 数据链，再由分支内部处理输入 dtype。
- `pruning_fp16` ONNX 评估输入 dtype 为 `float16`。
- `qat_convert` ONNX 评估输入 dtype 为 `float32`。

## 模型与 checkpoint 契约

- 支持模型固定为：
  - `resnet6_2d`
  - `resnet10_2d`
  - `resnet14_2d`
  - `resnet18_2d`
  - `resnet34_2d`
- `resnet*_2d` 表示网络使用 2D 卷积，不表示原始 `.npy` 样本只能是二维。
- Torch 模型输出为 logits，语义 shape 为 `[N, num_classes]`。
- 训练、剪枝、QAT checkpoint 是结构化契约，不是裸 `state_dict`。
- 直接消费 `.pth` 的阶段必须强校验 `architecture_signature.signature_hash`。
- 基座 checkpoint 关键字段包括：
  - `model_state_dict`
  - `train_context`
  - `model_structure`
  - `model_structure.model_name`
  - `model_structure.model_kwargs`
  - `model_structure.input_tensor_meta`
  - `model_structure.architecture_signature`
- pruning checkpoint 必须保留：
  - `model_structure.channel_cfg`
  - `model_structure.architecture_signature`
  - `pruning_meta`
- QAT checkpoint 必须保留：
  - `model_structure.channel_cfg`
  - `model_structure.architecture_signature`
  - `quantization_meta`
- `channel_cfg` 是 pruning -> QAT -> ONNX 恢复剪枝拓扑的核心桥梁；改动 pruning、模型定义或 `*_from_cfg()` 时必须验证恢复链路。

## ONNX / AMCT / ATC 契约

- ONNX 导出分支固定为：
  - `pruning_fp16`：`best_pruned_model.pth` -> `model_fp16.onnx`
  - `qat_convert`：`best_qat_prepare_model.pth` -> `model_quant.onnx`
- ONNX 固定 opset 16。
- ONNX 输入名固定为 `input`，输出名固定为 `logits`。
- ONNX 使用动态 batch；`example_input_shape` 中 batch=1 只是导出样例。
- `onnx_summary.json` 至少要稳定保留：
  - `summary_version`
  - `branch`
  - `model_name`
  - `labels`
  - `source_checkpoint_path`
  - `source_architecture_signature`
  - `opset_version`
  - `example_input_shape`
  - `onnx_path`
  - `interface`
- `interface` 至少包含：
  - `input_name`
  - `output_name`
  - `input_elem_type`
  - `output_elem_type`
  - `input_shape`
  - `output_shape`
  - `dynamic_batch`
- AMCT 只消费 `qat_convert` 分支同目录带 `onnx_summary.json` 的 `model_quant.onnx`。
- AMCT 输出：
  - `deploy_model.onnx`
  - `fake_quant_model.onnx`
  - `scale_offset_record.txt`
  - `amct_summary.json`
- `amct_summary.json.deploy_interface` 必须与 `source_interface` 一致。
- ATC 分支固定为：
  - `pruning_fp16`
  - `amct_deploy`
- ATC 默认：
  - `soc_version=Ascend310B4`
  - `input_format=NCHW`
  - `batch=1`
- 显式传入 ATC `--input_shape` 时，输入名和各维度必须与根据 summary 自动派生的结果完全一致。
- 不要随意改 `output/` 产物目录名、文件名、summary 名和字段名；后续阶段与下游仓库会依赖它们自动发现和校验。

## 下游 ResNet_Acl 对接契约

下游仓库 `wh-wang132/ResNet_Acl` 只消费训练端产物，不负责重新训练、重新导出 ONNX 或重新编译 ATC。跨仓库同步时，要把训练端产物整理到推理端约定目录：

```text
input/atc/<branch>/<model_name>/<experiment_name>/
├── atc_summary.json
├── fusion_result.json
└── <model_filename>
```

当前下游内置分支和 OM 文件名为：

- `pruning_fp16`：`model_fp16.om`
- `amct_deploy`：`deploy_model.om`

下游还需要同步：

- `Data/<class>/*.npy`
- split manifest，默认位置为 `input/splits/dataset_split__train0.60_val0.20_test0.20_seed42.json`

推理端对 `atc_summary.json` 的硬依赖包括：

- `stage == "atc"`
- `branch` 必须与 CLI 指定分支一致。
- `resolved_input_shape`
- `source_interface`
- `source_architecture_signature.parameter_count`

下游当前只支持：

- batch size = `1`
- 单输入、单输出 OM
- ACL elem type `1=float32`
- ACL elem type `10=float16`
- 输出为 logits，并直接对 `outputs[0]` 执行 `argmax(axis=1)`

训练端如修改以下内容，必须同步检查下游：

- 类别目录、类别自然排序、`class_names`、`class_to_idx`
- 输入 shape、通道布局、batch 语义、dtype
- 输出 shape、输出类别顺序、输出名
- branch 名、OM 文件名、artifact 目录层级
- `atc_summary.json`、`onnx_summary.json`、`amct_summary.json` 字段
- experiment 命名，例如 `from_ratio0.60_steps8_global_ft10_bs64`

下游重点文件：

- `src/common/branch_specs.py`
- `src/common/artifact_scanner.py`
- `src/common/data.py`
- `src/common/acl_runner.py`
- `src/common/model_complexity.py`
- `docs/interface_contract.md`
- `src/visualization/__main__.py`

跨仓库同步后，优先在 `ResNet_Acl` 执行：

```bash
pixi run python -m src.validate --branch <branch> --sample_limit 8
pixi run python -m src.accuracy --branch <branch> --artifact_path input/atc/<branch>/<model>/<experiment>
pixi run python -m src.efficiency --branch <branch> --artifact_path input/atc/<branch>/<model>/<experiment>
```

## 环境与运行约定

- 初次准备优先执行：

```bash
pixi install
uv sync
direnv allow
```

- `.envrc` 是公共环境入口，只提供：
  - `REPO_ROOT`
  - `PYTHONPATH=$REPO_ROOT/src`
- `scripts/load_*_env.sh` 只补充阶段增量环境。
- `scripts/load_onnx_env.sh` 负责 ONNX Runtime / TensorRT 运行时库路径和 TensorRT engine cache。
- `scripts/load_amct_env.sh` 负责 AMCT/CANN 相关增量环境。
- `scripts/load_atc_env.sh` 负责 ATC/CANN 编译环境。
- AMCT 运行前需要目标环境可导入仓库附带的 `amct_onnx` wheel，并准备对应算子包。
- ATC/AMCT 真实运行依赖 Ascend/CANN 宿主环境；无硬件或工具链时，不要把未运行误写成已验证。

## 常用命令

```bash
uv run python -m base_model --epochs 20 --model resnet6_2d
uv run python -m pruning --model resnet6_2d
uv run python -m qat --pruning_checkpoint output/pruning/.../best_pruned_model.pth
uv run python -m onnx_export --branch pruning_fp16 --checkpoint output/pruning/.../best_pruned_model.pth
uv run python -m onnx_export --branch qat_convert --checkpoint output/qat/.../best_qat_prepare_model.pth
uv run python -m amct --onnx_model output/onnx/qat_convert/.../model_quant.onnx
pixi run python -m atc --branch pruning_fp16 --onnx_model output/onnx/pruning_fp16/.../model_fp16.onnx
pixi run python -m atc --branch amct_deploy --onnx_model output/amct/.../deploy_model.onnx
```

批处理入口：

```bash
pixi run autorun-base-model
pixi run autorun-pruning
pixi run autorun-qat
pixi run autorun-onnx
pixi run autorun-amct
pixi run autorun-atc
```

## 修改代码时的检查清单

- 改数据加载：
  - 检查 `Data/<class>` 扫描、自然排序、shape 推断、manifest 复用。
  - 检查下游 `ResNet_Acl` 的 manifest 和样本读取逻辑是否仍匹配。
- 改模型结构：
  - 同步默认构造函数和 `*_from_cfg()`。
  - 检查 `channel_cfg`、`architecture_signature`、checkpoint 恢复。
  - 检查 ONNX 导出和 ATC summary 是否仍能表达输入输出接口。
- 改 pruning：
  - 确保最终模型能提取完整 `channel_cfg`。
  - 确保 QAT 能 strict 恢复 pruning checkpoint。
- 改 QAT：
  - 保护 `quantization_meta` 最小恢复契约。
  - 确保 `onnx_export --branch qat_convert` 仍能 `convert_fx`、rewrite、validate。
- 改 ONNX：
  - 保护 opset 16、输入名 `input`、输出名 `logits`、动态 batch。
  - 检查 `onnx_summary.json.interface`、`source_architecture_signature` 和路径字段。
- 改 AMCT：
  - 只接受 `model_quant.onnx` 的约束如需变化，必须同步 ATC 与下游。
  - 检查 `deploy_interface == source_interface`。
- 改 ATC：
  - 保护 batch=1、NCHW、`Ascend310B4` 默认约定。
  - 检查 `.om` 文件名是否仍符合下游 `ResNet_Acl` 的 `branch_specs.py`。
- 改输出目录或 experiment 命名：
  - 同步 autorun 脚本、summary 生成、下游可视化解析逻辑。
- 改 summary 字段：
  - 同步所有读取方，不要只改写入方。

## 验证建议

- 对纯 Python 代码改动，至少运行相关入口的 `--help` 或轻量导入检查：

```bash
uv run python -m base_model --help
uv run python -m pruning --help
uv run python -m qat --help
uv run python -m onnx_export --help
uv run python -m amct --help
pixi run python -m atc --help
```

- 对真实训练、ONNX、AMCT、ATC 改动，应按受影响阶段运行最小闭环。
- 若当前环境缺少 CUDA、TensorRT、CANN、Ascend 设备或 AMCT 组件，最终说明中必须明确哪些验证未执行。

## 协作与子任务拆分

- 做大范围结构分析或跨仓库对接时，可把任务拆为训练端本地结构、远端训练仓库、下游推理端契约等独立子任务。
- 每个子任务只负责一个明确目标，最后由主流程合并结论。
- 跨仓库 GitHub 对象查询优先使用 GitHub 专用能力；不要为了明确仓库文件读取先走通用搜索。
- 在线资料、网页抓取或第二来源核实时，遵守当前运行环境中的 MCP 路由和最多两个 MCP 规则。
