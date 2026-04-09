# 自动化脚本说明

## 概述

当前仓库在 `autorun/` 目录提供 6 份顺序执行脚本：

- `autorun/autorun_base_model.sh`
- `autorun/autorun_pruning.sh`
- `autorun/autorun_qat.sh`
- `autorun/autorun_onnx.sh`
- `autorun/autorun_amct.sh`
- `autorun/autorun_atc.sh`

这些脚本整体仍服务于顺序批处理执行；其中 `onnx` / `amct` / `atc` autorun 已包含 shell 函数、`mktemp`、`trap`、`find` 遍历与临时文件清理等基础控制逻辑，便于在服务器终端观察运行状态并安全清理中间状态。

## 运行前准备

### 需要用户手动安装的项目

- `git`
- `pixi`
- `uv`
- `direnv`（推荐）

### `pixi install` / `uv sync` 会自动安装的内容

- `pixi install`
  - Python 3.12 运行时
  - GCC / G++ / Make / CMake
  - `cuda-runtime`、`cudnn`
  - `ascend-cann-toolkit`、`ascend-cann-310b-ops`
- `uv sync`
  - `torch`
  - `onnx`
  - `onnxruntime-gpu`
  - `torch-pruning`
  - 以及 `pyproject.toml` 中声明的其余 Python 依赖

### 条件性宿主机要求

- 若要使用 CUDA 加速训练，宿主机需要可用的 NVIDIA GPU 与驱动。
- 若要执行真实的 AMCT / ATC / Ascend 部署验证，宿主机需要对应的 Ascend 设备/驱动环境。

### 推荐初始化顺序

```bash
pixi install
uv sync
direnv allow
```

其中 [`.envrc`](../.envrc) 提供仓库级公共变量：

- `REPO_ROOT`
- `PYTHONPATH=$REPO_ROOT/src`

说明：

- `direnv` 为推荐方案；若不使用 `direnv` 自动激活，也必须手动提供与 `.envrc` 等价的环境变量
- 所有脚本统一通过 `.envrc` 提供的 `REPO_ROOT` 识别仓库根

### 阶段专用手动准备

AMCT 阶段额外依赖仓库自带组件：

- `amct_onnx/amct_onnx-0.23.2-py3-none-linux_x86_64.whl`
- `amct_onnx/amct_onnx_op.tar.gz`

说明：

- 上述文件已经随仓库提供
- 它们不属于 `uv sync` / `pixi install` 的自动安装范围
- 在运行 `autorun/autorun_amct.sh` 前，需要按目标环境自行安装或部署

## 环境层次

自动化脚本默认都要求当前 shell 已激活公共环境层 `.envrc`，并通过其中的 `REPO_ROOT` 作为统一仓库根。在此基础上：

- `autorun/autorun_base_model.sh`：内部加载 `load_base_model_env.sh`
- `autorun/autorun_pruning.sh`：只依赖公共层
- `autorun/autorun_qat.sh`：只依赖公共层
- `autorun/autorun_onnx.sh`：内部加载 `load_onnx_env.sh`
- `autorun/autorun_amct.sh`：内部加载 `load_amct_env.sh`
- `autorun/autorun_atc.sh`：内部加载 `load_atc_env.sh`

## 基座模型自动训练

入口：

```bash
bash autorun/autorun_base_model.sh
```

脚本行为：
- 模型：`resnet6_2d` / `resnet10_2d` / `resnet14_2d` / `resnet18_2d` / `resnet34_2d`
- 搜索维度：模型对应的训练轮数 + `batch_size`
- 每条命令显式传入：`--full_load True`
- 内部执行：`uv run src/base_model_main.py ...`

输出目录：

```text
output/base_model/<model>/epochs<epochs>_bs<batch_size>/
```

## 剪枝自动运行

入口：

```bash
bash autorun/autorun_pruning.sh
```

脚本行为：
- 模型：全部 5 个基座模型
- 搜索维度：
  - `--model`
  - `--pruning_ratio`
  - `--pruning_steps`
- 每条命令显式传入：`--full_load True`
- 其余参数与 pruning CLI 默认值保持一致
- 内部执行：`uv run src/pruning_main.py ...`

输出目录：

```text
output/pruning/<model>/ratio<ratio>_steps<steps>_<global|local>_ft<epochs>_bs<batch_size>/
```

## QAT 自动运行

入口：

```bash
bash autorun/autorun_qat.sh
```

脚本行为：
- 输入：pruning 自动脚本产出的 `best_pruned_model.pth`
- 每条命令显式传入：`--full_load True`
- 其余参数与 QAT CLI 默认值保持一致
- 内部执行：`uv run src/qat_main.py ...`

输出目录：

```text
output/qat/<model>/from_<pruning_exp>/
```

## ONNX 自动运行

入口：

```bash
bash autorun/autorun_onnx.sh
```

脚本行为：
- 遍历：
  - `output/pruning/**/best_pruned_model.pth`
  - `output/qat/**/best_qat_prepare_model.pth`
- 内部执行：`uv run src/onnx_main.py ...`
- 默认参数与 ONNX CLI 保持一致：
  - `full_load=False`
  - `evaluate_test=True`
  - `eval_batch_size=64`

输出目录：

```text
output/onnx/pruning_fp16/<model>/from_<exp>/
output/onnx/qat_convert/<model>/from_<exp>/
```

## AMCT 自动运行

入口：

```bash
bash autorun/autorun_amct.sh
```

脚本行为：
- 只遍历 `output/onnx/qat_convert/**/model_quant.onnx`
- 内部执行：`uv run src/amct_main.py ...`
- 运行前需先完成仓库内 `amct_onnx` wheel 与算子包的手动安装或部署

输出目录：

```text
output/amct/<model>/from_<exp>/
```

## ATC 自动运行

入口：

```bash
bash autorun/autorun_atc.sh
```

脚本行为：
- 输入分支：
  - `output/onnx/pruning_fp16/**/model_fp16.onnx`
  - `output/amct/**/deploy_model.onnx`
- 内部执行：`pixi run python src/atc_main.py ...`
- 默认参数与 ATC CLI 保持一致：
  - `soc_version=Ascend310B4`
  - `input_format=NCHW`
  - `input_shape` 默认从上游摘要中的输入接口派生，并将 batch 固定为 `1`
  - 用户可通过 `--input_shape` 显式覆盖

输出目录：

```text
output/atc/pruning_fp16/<model>/from_<exp>/
output/atc/amct_deploy/<model>/from_<exp>/
```

## 使用建议

1. 先单独运行一条命令，确认环境、数据路径与驱动条件正常。
2. 先确认执行脚本前的 shell 已加载项目根目录的 `.envrc`；若不使用 `direnv` 自动激活，也必须手动提供与 `.envrc` 等价的环境变量。
3. pruning 自动脚本依赖：
   - 对应基座模型目录下已存在 `output/base_model/<model>/best_model.pth` 符号链接
4. `autorun/autorun_onnx.sh` 默认不显式传 `--eval_batch_size`；若资源不足，可在脚本中追加更小的评估 batch。
5. `autorun/autorun_amct.sh` 与 `autorun/autorun_atc.sh` 适合在对应目标环境上运行。

## 注意事项

- 脚本不做并行调度。
- 脚本不做失败重试或断点续跑。
- 若需要调整搜索网格，直接编辑脚本中的命令列表即可。
