#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AMCT 阶段输出目录与摘要工具。"""

import json
import os


def create_output_directory(onnx_summary):
    output_root = os.path.join("output", "amct")
    model_name = onnx_summary["model_name"]
    source_exp_name = os.path.basename(os.path.dirname(onnx_summary["onnx_path"]))
    if not source_exp_name.startswith("from_"):
        source_exp_name = f"from_{source_exp_name}"
    folder_path = os.path.join(output_root, model_name, source_exp_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def save_summary(folder_path, summary):
    summary_path = os.path.join(folder_path, "amct_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary_path
