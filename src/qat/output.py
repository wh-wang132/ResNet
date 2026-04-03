#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QAT 阶段输出目录与摘要工具。"""

import json
import os


def create_output_directory(args, checkpoint_meta):
    output_root = os.path.join("output", "qat")
    model_name = checkpoint_meta["model_name"]
    pruning_exp_name = os.path.basename(os.path.dirname(checkpoint_meta["source_pruning_checkpoint_path"]))
    folder_path = os.path.join(
        output_root,
        model_name,
        f"from_{pruning_exp_name}",
    )
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def save_summary(folder_path, summary):
    summary_path = os.path.join(folder_path, "qat_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary_path
