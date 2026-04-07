#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ATC 阶段参数解析。"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile repository pruning_fp16 / AMCT deploy ONNX artifacts with ATC"
    )
    parser.add_argument(
        "--branch",
        type=str,
        required=True,
        choices=["pruning_fp16", "amct_deploy"],
        help="ATC 编译输入分支",
    )
    parser.add_argument(
        "--onnx_model",
        type=str,
        required=True,
        help="输入 ONNX 路径；pruning 分支传 model_fp16.onnx，AMCT 分支传 deploy_model.onnx",
    )
    parser.add_argument(
        "--soc_version",
        type=str,
        default="Ascend310B4",
        help="目标芯片版本 (默认 Ascend310B4)",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default=None,
        help="可选显式输入形状；默认从上游摘要 interface 派生并将 batch 固定为 1",
    )
    parser.add_argument(
        "--input_format",
        type=str,
        default="NCHW",
        help="输入格式 (默认 NCHW)",
    )
    return parser.parse_args()
