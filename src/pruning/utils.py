#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
剪枝阶段通用工具入口。

仅复用 base_model 中稳定、无阶段语义冲突的公共函数，
避免将基座训练专用参数解析和输出目录逻辑耦合进来。
"""

from base_model.utils import (
    create_optimized_dataloader,
    get_raw_model,
    load_model_map,
    load_state_dict_safely,
    release_gpu_memory,
    remove_orig_mod_prefix,
    setup_device,
)

INPUT_SHAPE_NCHW = (1, 1, 543, 512)
INPUT_SIZE_CHW = (1, 543, 512)

__all__ = [
    "INPUT_SHAPE_NCHW",
    "INPUT_SIZE_CHW",
    "create_optimized_dataloader",
    "get_raw_model",
    "load_model_map",
    "load_state_dict_safely",
    "release_gpu_memory",
    "remove_orig_mod_prefix",
    "setup_device",
]
