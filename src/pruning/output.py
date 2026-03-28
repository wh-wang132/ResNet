#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""剪枝阶段输出目录与摘要工具。"""

import json
import os


def create_output_directory(args, model_name):
    output_root = os.path.join("output", "pruning")
    ratio_tag = f"ratio{args.pruning_ratio:.2f}"
    steps_tag = f"steps{args.pruning_steps}"
    scope_tag = "global" if args.global_pruning else "local"
    folder_path = os.path.join(
        output_root,
        model_name,
        f"{ratio_tag}_{steps_tag}_{scope_tag}_ft{args.finetune_epochs}_bs{args.batch_size}",
    )
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def save_summary(folder_path, summary):
    summary_path = os.path.join(folder_path, "pruning_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary_path
