#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""torch-pruning 封装。"""

import math
import torch

from pruning.evaluator import count_model_stats
from pruning.utils import get_raw_model


def compute_step_pruning_ratio(target_total_ratio, pruning_steps):
    if pruning_steps <= 0:
        raise ValueError("pruning_steps 必须大于 0")
    if not 0.0 <= target_total_ratio < 1.0:
        raise ValueError("pruning_ratio 必须位于 [0, 1) 区间内")

    return 1.0 - math.pow(1.0 - target_total_ratio, 1.0 / pruning_steps)


def prune_model(
    model,
    example_inputs,
    target_total_ratio,
    global_pruning=True,
    ignore_fc=True,
    step_index=1,
    pruning_steps=1,
):
    try:
        import torch_pruning as tp
    except ImportError as exc:
        raise RuntimeError("未安装 torch-pruning，无法执行结构化剪枝") from exc

    raw_model = get_raw_model(model)
    raw_model.eval()
    step_ratio = compute_step_pruning_ratio(target_total_ratio, pruning_steps)

    pre_stats = count_model_stats(raw_model, example_inputs)

    ignored_layers = []
    ignored_layer_names = []
    if ignore_fc and hasattr(raw_model, "fc"):
        ignored_layers.append(raw_model.fc)

    for name, module in raw_model.named_modules():
        if any(module is ignored for ignored in ignored_layers):
            ignored_layer_names.append(name)

    importance = tp.importance.MagnitudeImportance(p=2)
    pruner = tp.pruner.MagnitudePruner(
        raw_model,
        example_inputs=example_inputs,
        importance=importance,
        pruning_ratio=step_ratio,
        global_pruning=global_pruning,
        ignored_layers=ignored_layers,
    )

    pruner.step()

    post_stats = count_model_stats(raw_model, example_inputs)
    pruning_meta = {
        "step_index": int(step_index),
        "pruning_steps": int(pruning_steps),
        "target_total_ratio": float(target_total_ratio),
        "step_ratio": float(step_ratio),
        "global_pruning": bool(global_pruning),
        "ignored_layers": ignored_layer_names,
        "example_input_shape": list(example_inputs.shape),
        "torch_pruning_version": getattr(tp, "__version__", "unknown"),
        "params_before": pre_stats["params"],
        "params_after": post_stats["params"],
        "macs_before": pre_stats["macs"],
        "macs_after": post_stats["macs"],
    }
    return raw_model, pruning_meta
