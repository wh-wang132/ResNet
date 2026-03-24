#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""torch-pruning 封装。"""

import torch

from pruning.evaluator import count_model_stats
from pruning.utils import get_raw_model


def prune_model(model, example_inputs, pruning_ratio, global_pruning=True, ignore_fc=True):
    try:
        import torch_pruning as tp
    except ImportError as exc:
        raise RuntimeError("未安装 torch-pruning，无法执行结构化剪枝") from exc

    raw_model = get_raw_model(model)
    raw_model.eval()

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
        pruning_ratio=pruning_ratio,
        global_pruning=global_pruning,
        ignored_layers=ignored_layers,
    )

    pruner.step()

    post_stats = count_model_stats(raw_model, example_inputs)
    pruning_meta = {
        "pruning_ratio": float(pruning_ratio),
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
