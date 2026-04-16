# 贡献指南

感谢您对本项目的关注。

## 开发环境设置

### 1. 克隆仓库

```bash
git clone git@github.com:wh-wang132/ResNet.git
cd ResNet
```

### 2. 手动前置项

参与开发前，用户只需要独立手动安装：

- `git`
- `pixi`
- `uv`
- `direnv`（推荐）

### 3. 安装项目环境

```bash
pixi install
uv sync
direnv allow
```

说明：

- `pixi install` 会自动安装 Python 3.12 运行时、GCC/G++、CUDA runtime、cuDNN、CANN toolkit 等工具链内容
- `uv sync` 会自动安装 `torch`、`onnx`、`torch-pruning` 等 Python 包依赖
- 文档中的手动环境依赖只记录用户需要显式准备的部分

### 4. 公共环境层

项目默认工作流为：

- `pixi`：系统工具链与运行时
- `uv`：Python 包依赖与运行入口
- `direnv`：推荐自动激活 [`.envrc`](../.envrc)

其中 [`.envrc`](../.envrc) 当前提供仓库级公共变量：

- `REPO_ROOT`
- `PYTHONPATH=$REPO_ROOT/src`

说明：

- `direnv` 为推荐方案；若不使用 `direnv` 自动激活，也必须手动提供与 `.envrc` 等价的环境变量
- 所有脚本统一通过 `.envrc` 提供的 `REPO_ROOT` 识别仓库根

阶段相关的额外环境变量统一由 `scripts/load_*_env.sh` 按需补充。

维护建议：新增或更新 autorun / 文档时，统一以 `.envrc -> REPO_ROOT` 作为仓库根约定。

### 5. 阶段专用说明

- AMCT 相关开发依赖仓库附带的 `amct_onnx` wheel 与算子包；它们不在 `uv sync` 管理范围内
- 若要做真实的 CUDA / Ascend 侧验证，宿主机需要对应硬件与驱动环境

## 代码规范

### Python 代码风格

- 遵循 [PEP 8](https://peps.python.org/pep-0008/)
- 使用 4 空格缩进
- 每行尽量不超过 120 字符
- 使用有意义的变量名、函数名和模块名

### 文档字符串

公共函数和类应带有简洁、准确的文档字符串。

### 类型注解

推荐为新增或重构代码补充类型注解。

## 提交建议

1. 修改前先确认对应阶段边界与输入输出契约
2. 修改后至少验证：
   - 入口脚本参数是否仍与文档一致
   - 输出目录与 summary 是否保持兼容
   - 文档是否同步更新

## Pull Request 建议

PR 描述中建议明确说明：

- 变更属于哪个阶段：`base_model / pruning / qat / onnx / amct / atc`
- 是否影响跨阶段契约：
  - `model_structure`
  - `channel_cfg`
  - `architecture_signature`
  - `quantization_meta`
  - `onnx_summary.json / amct_summary.json / atc_summary.json`
- 是否需要更新自动化脚本或文档

## 测试与验证

当前仓库更强调“阶段入口可运行 + 产物契约正确”，建议按变更范围做验证：

- 基座训练相关：至少跑通一次 `uv run python -m base_model`
- 剪枝相关：至少验证 `pruning_summary.json` 与 checkpoint 输出
- QAT 相关：至少验证 QAT checkpoint 可恢复
- ONNX 相关：至少验证导出与 ORT 精度评估
- AMCT / ATC 相关：至少验证输入契约检查逻辑；若环境允许，再做真实阶段运行

## 文档贡献

文档改进同样重要。更新文档时请保持以下原则：

- 以代码实现为准
- 不把 `pixi install` / `uv sync` 会自动安装的内容写成“用户手动依赖”
- 不把“已实现”误写成“已充分验证”
- 不把 `resnet*_2d` 误写成“样本维度受模型名限制”；样本支持 2D/3D，以数据集推断结果为准
- 不把类别数写成手动配置；统一描述为从 `Data/<class>/` 一级子目录动态推断
- 只保留当前有效的阶段、入口和参数描述

## 获取帮助

如果需要快速了解项目：

1. 先看 [README.md](../README.md)
2. 再看 [项目架构分析](PROJECT_ARCHITECTURE.md)
3. 最后按阶段查看：
   - [命令行参数详解](CLI_ARGUMENTS.md)
   - [自动化脚本说明](AUTOMATED_TRAINING.md)
   - [剪枝指南](PRUNING_GUIDE.md)

## 许可证

通过贡献代码，您同意您的贡献将根据项目的 [LICENSE](../LICENSE) 进行许可。
