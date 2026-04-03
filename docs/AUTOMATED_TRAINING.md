# 自动化脚本说明

## 概述

当前仓库在项目根目录提供两份顺序执行脚本：

- `autorun_base_model.sh`
- `autorun_pruning.sh`

两份脚本都采用非常直接的风格：逐行命令、顺序执行、无复杂控制流，适合在服务器终端手动监视。

## 运行前准备

当前项目默认工作流不是“只装 `uv` 就能跑”，而是：

- `pixi`：提供 GCC / Make / CMake 等系统工具链环境
- `uv`：安装 Python 依赖并执行 `uv run ...`
- `direnv`：推荐自动激活 `pixi shell-hook` 并设置 `PYTHONPATH`

推荐最小运行顺序：

```bash
pixi install
uv sync
direnv allow
```

若未使用 `direnv`，则应先手动进入 `pixi` 环境，再执行下面的脚本。

## 基座模型自动训练

入口脚本：

```bash
bash autorun_base_model.sh
```

脚本内部逐行调用：

```bash
uv run src/base_model_main.py ...
```

当前覆盖范围：

- 模型：`resnet6_2d` / `resnet10_2d` / `resnet14_2d` / `resnet18_2d` / `resnet34_2d` / `resnet50_2d`
- 搜索维度：模型对应的训练轮数 + `batch_size`
- 固定设置：每条命令显式传入 `--full_load True`

输出默认写入：

```text
output/base_model/<model>/epochs<epochs>_bs<batch_size>/
```

## 剪枝自动运行

入口脚本：

```bash
bash autorun_pruning.sh
```

脚本内部逐行调用：

```bash
uv run src/pruning_main.py ...
```

当前覆盖范围：

- 模型：全部 6 个基座模型
- 搜索维度：
  - `--model`
  - `--pruning_ratio`
  - `--pruning_steps`
- 固定设置：每条命令显式传入 `--full_load True`
- 其他 pruning 参数使用 [src/pruning/args.py](/root/ResNet/src/pruning/args.py) 的默认值，例如：
  - `batch_size=64`
  - `finetune_epochs=10`
  - `global_pruning=True`
  - `ignore_fc=True`
  - `evaluate_test=True`

输出默认写入：

```text
output/pruning/<model>/ratio<ratio>_steps<steps>_<global|local>_ft<epochs>_bs<batch_size>/
```

## 使用建议

1. 先单独运行一条命令确认环境与数据路径正常。
2. 服务器长时运行时建议直接进入项目根目录后执行脚本。
3. 若你依赖 `direnv`，先确认当前 shell 已加载项目根目录的 `.envrc`。
4. pruning 自动脚本依赖：
   - 对应基座模型目录下已存在 `output/base_model/<model>/best_model.pth` 符号链接

## 注意事项

- 当前脚本不做并行调度。
- 当前脚本不做失败重试、断点续跑或日志切分。
- 若需要修改搜索网格，直接编辑脚本中的命令列表即可。
