#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AMCT 阶段参数解析。"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert repository qat_convert ONNX artifacts into AMCT deploy/fakequant models"
    )
    parser.add_argument(
        "--onnx_model",
        type=str,
        required=True,
        help="输入的 qat_convert ONNX 路径，固定为仓库导出的 model_quant.onnx",
    )
    return parser.parse_args()
