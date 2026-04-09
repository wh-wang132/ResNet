# 数据准备指南

## 概述

本指南说明项目的数据组织方式、切分机制与不同阶段的数据精度约定。

## 数据集结构

数据集按如下目录组织：

```text
Data/
├── 0/
│   ├── sample1.npy
│   ├── sample2.npy
│   └── ...
├── 1/
│   └── ...
└── ...
```

说明：

- `Data/` 是数据根目录，可通过 `--data_dir` 指定
- 一级子目录名作为类别名
- 默认类别数为 24
- 实现使用自然排序扫描目录与文件名，以保证类别映射与切分稳定

## `.npy` 样本要求

每个样本应为 2D 数组：

- 推荐形状：`(543, 512)`
- 加载后会自动补通道维，变为 `(1, 543, 512)`
- 数值类型可为浮点型
- 数值范围建议归一化到 `[0, 1]` 或 `[-1, 1]`

## 数据加载精度约定

不同阶段的数据精度约定如下：

| 阶段 | 数据集输出精度 |
| --- | --- |
| `base_model` | `fp16` 或 `fp32`，默认 `fp16` |
| `pruning` | `fp16` 或 `fp32`，默认 `fp16` |
| `qat` | 固定 `fp32` |
| `onnx` 测试评估 | 固定 `fp32` |

说明：

- `base_model` / `pruning` 的 `--data_dtype` 只影响 DataLoader 输出 tensor 精度
- QAT 阶段训练/验证/测试统一为 `fp32`
- ONNX 导出阶段的评估数据链统一按 `fp32` 构建，再由分支内部处理输入 dtype

## 数据集切分机制

`base_model.dataset.data_set_split()` 使用固定分层切分：

- 训练集：60%
- 验证集：20%
- 测试集：20%
- 随机种子：默认 42

切分流程：

1. 扫描 `Data/` 目录
2. 建立 `class_names` 与 `class_to_idx`
3. 执行分层抽样
4. 创建 `NPYDataset`
5. 返回 train / val / test dataset

## `output/splits` manifest

切分结果会落盘到：

```text
output/splits/
```

manifest 保存：

- `data_dir`
- 划分比例
- `random_state`
- `class_names`
- `class_to_idx`
- `train_files`
- `val_files`
- `test_files`

其中样本路径以相对于 `Data/` 的相对路径落盘，而不是绝对路径。

后续运行时：

- 若 manifest 与给定配置一致，则直接复用
- 若 manifest 缺失或配置不匹配，则重新切分并覆盖落盘

## 注意事项

1. 确保所有 `.npy` 文件形状一致。
2. 数据加载阶段会校验样本可读性与 shape 合法性；若存在损坏样本，会抛 `DatasetSampleError` / `DatasetIntegrityError`，必须先修复样本再继续运行。
3. `full_load=True` 时会把该 split 的所有样本预加载到内存中；内存不足时请保持默认 `False`。
4. 若你修改了切分比例或随机种子，`output/splits` 中会生成新的 manifest 文件。

## 修改切分比例

各阶段入口都直接调用 `data_set_split(..., train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)`。

若确实需要修改，请同步调整：

- 对应入口脚本中的调用参数
- 与实验结果相关的文档说明

否则会导致不同配置实验不可直接横向比较。
