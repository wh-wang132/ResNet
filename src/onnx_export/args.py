#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ONNX 导出阶段参数解析。"""

import argparse

from qat.utils import str2bool


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
        "--opset_version",
        type=int,
        default=16,
        choices=[16],
        help="ONNX opset 版本，当前固定为 16",
    )
    return parser.parse_args()
