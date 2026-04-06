#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ONNX 导出阶段参数解析。"""

import argparse

from qat.utils import str2bool


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("eval_batch_size 必须大于等于 1")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export pruning/QAT checkpoints to ONNX and evaluate with ONNX Runtime"
    )
    parser.add_argument(
        "--branch",
        type=str,
        required=True,
        choices=["pruning_fp16", "qat_convert"],
        help="ONNX 导出分支",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="输入 checkpoint 路径；pruning 分支传 pruning checkpoint，QAT 分支传 QAT checkpoint",
    )
    parser.add_argument("--data_dir", type=str, default="Data", help="数据集路径")
    parser.add_argument(
        "--full_load",
        type=str2bool,
        default=False,
        help="是否全量加载数据集",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="数据加载工作线程数",
    )
    parser.add_argument(
        "--evaluate_test",
        type=str2bool,
        default=True,
        help="是否在导出后执行测试集精度评估",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=positive_int,
        default=64,
        help="导出后 Torch / ORT 精度评估使用的批次大小，不影响导出图本身 (默认 64)",
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=16,
        choices=[16],
        help="ONNX opset 版本，当前固定为 16",
    )
    return parser.parse_args()
