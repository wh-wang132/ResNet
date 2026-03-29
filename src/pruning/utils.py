#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
剪枝阶段通用工具入口。

仅复用 base_model 中稳定、无阶段语义冲突的公共函数，
避免将基座训练专用参数解析和输出目录逻辑耦合进来。
"""

import os

from base_model.utils import (
    build_architecture_signature,
    create_optimized_dataloader,
    get_raw_model,
    load_model_map,
    load_state_dict_safely,
    release_gpu_memory,
    remove_orig_mod_prefix,
    setup_device,
    str2bool,
)

INPUT_SHAPE_NCHW = (1, 1, 543, 512)
INPUT_SIZE_CHW = (1, 543, 512)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def to_repo_relative_path(path):
    if path is None:
        return None

    normalized_path = os.path.normpath(path)
    if not os.path.isabs(normalized_path):
        return normalized_path

    return os.path.relpath(normalized_path, REPO_ROOT)


def build_compact_pruning_meta(pruning_meta, baseline_stats):
    compact_pruning_meta = {
        "step_index": pruning_meta["step_index"],
        "pruning_steps": pruning_meta["pruning_steps"],
        "step_ratio": pruning_meta["step_ratio"],
        "target_total_ratio": pruning_meta["target_total_ratio"],
        "global_pruning": pruning_meta["global_pruning"],
        "ignored_layers": pruning_meta["ignored_layers"],
        "example_input_shape": pruning_meta["example_input_shape"],
        "torch_pruning_version": pruning_meta["torch_pruning_version"],
        "params_before": baseline_stats["params"],
        "params_after": pruning_meta["params_after"],
        "macs_before": baseline_stats["macs"],
        "macs_after": pruning_meta["macs_after"],
    }
    return compact_pruning_meta

__all__ = [
    "INPUT_SHAPE_NCHW",
    "INPUT_SIZE_CHW",
    "REPO_ROOT",
    "build_compact_pruning_meta",
    "build_architecture_signature",
    "create_optimized_dataloader",
    "get_raw_model",
    "load_model_map",
    "load_state_dict_safely",
    "release_gpu_memory",
    "remove_orig_mod_prefix",
    "setup_device",
    "str2bool",
    "to_repo_relative_path",
]
