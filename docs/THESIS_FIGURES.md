# 论文插图可视化模块

`thesis_figures` 是面向本科毕设论文插图的后处理模块。它只消费 `output/` 下已有 JSON summary 与产物路径信息，不读取 `Data/`，不加载 `.pth`、`.onnx`、`.om`，也不调用 CUDA、CANN、AMCT 或 ATC。

## 入口

```bash
uv run python -m thesis_figures --help
uv run python -m thesis_figures --output_root output --dry_run
uv run python -m thesis_figures --output_root output --formats png,svg
pixi run autorun-thesis-figures
```

`pixi run autorun-thesis-figures` 是批处理后处理入口，内部执行 `uv run python -m thesis_figures --output_root output --formats png,svg --strict`，通常在 ATC 产物准备好后运行。

默认输出目录：

```text
output/thesis_figures/
├── fig1_pruning_accuracy_complexity.png
├── fig1_pruning_accuracy_complexity.svg
├── ...
├── figures_manifest.json
└── tables/
    ├── records.csv
    ├── pruning_tradeoff.csv
    ├── stage_accuracy_summary.csv
    ├── onnx_metric_delta.csv
    └── atc_amct_interface_matrix.csv
```

重复运行会覆盖 `output/thesis_figures/` 下的同名图表、CSV 和 `figures_manifest.json`，不再创建 `figures_<timestamp>` 子目录。

## 输入范围

模块扫描以下文件：

- `output/pruning/*/*/pruning_summary.json`
- `output/qat/*/*/qat_summary.json`
- `output/onnx/*/*/*/onnx_summary.json`
- `output/amct/*/*/amct_summary.json`
- `output/atc/*/*/*/atc_summary.json`

`from_ratio...` 实验名会归一化为 `ratio...`，用于把 pruning、QAT、ONNX、AMCT、ATC 阶段对齐到同一个实验配置。

## 生成图表

- `fig1_pruning_accuracy_complexity`：剪枝率、错误率与参数保留比例关系，错误率与参数保留比例均使用对数坐标。
- `fig2_compression_by_model`：各模型最佳压缩实验的 baseline/final 参数量与 MACs 对比，参数量与 MACs 均使用对数坐标。
- `fig3_stage_accuracy_flow`：pruning、QAT、ONNX FP16、ONNX QAT 的平均测试错误率流转，使用对数坐标。
- `fig4_onnx_metric_delta`：ONNX 导出前后 error rate / loss 差异；错误率差异由 `-metric_delta_acc` 派生。
- `fig5_atc_amct_interface_matrix`：AMCT / ATC 分支接口矩阵。

错误率统一按 `1 - acc` 派生。对数坐标图会跳过缺失、为 0 或小于 0 的数据点，不使用 epsilon 伪造非零错误率。

该模块不生成或模拟推理端真实延迟、吞吐、能耗等指标；这些指标仍以 `wh-wang132/ResNet_Acl` 的 `accuracy`、`efficiency`、`visualization` 输出为准。

## 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--output_root` | `output` | 训练端产物根目录 |
| `--figure_dir` | `output/thesis_figures` | 论文插图输出根目录 |
| `--formats` | `png,svg` | 逗号分隔的图片格式 |
| `--model` | `all` | 可筛选单个模型 |
| `--experiment` | `all` | 可按实验名子串筛选 |
| `--dry_run` | `False` | 只扫描并打印摘要，不创建图表 |
| `--strict` | `False` | 遇到缺字段、坏 JSON 或无可用记录时直接失败 |

## 验证

```bash
uv run python -m thesis_figures --help
uv run python -m thesis_figures --output_root output --dry_run
uv run python -m thesis_figures --output_root output --dry_run --strict
pixi run autorun-thesis-figures
PYTHONPATH=src uv run python -m unittest discover -s tests
```
