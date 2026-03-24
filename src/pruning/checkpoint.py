#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基座 checkpoint 读取与模型恢复。"""

import os
import torch

from pruning.utils import load_model_map, load_state_dict_safely


class CheckpointRestoreError(RuntimeError):
    """剪枝阶段 checkpoint 恢复错误。"""


def load_base_checkpoint(checkpoint_path, device, expected_model_name=None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到基座 checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise CheckpointRestoreError("输入 checkpoint 不包含 model_state_dict，无法作为基座模型恢复")

    model_structure = checkpoint.get("model_structure", {})
    model_name = model_structure.get("model_name")
    if expected_model_name is not None and model_name is not None and model_name != expected_model_name:
        raise CheckpointRestoreError(
            f"checkpoint 中模型名为 {model_name}，与命令行指定的 {expected_model_name} 不一致"
        )

    model_name = expected_model_name or model_name
    if model_name is None:
        raise CheckpointRestoreError("checkpoint 中缺少 model_name，且命令行未指定 --model")

    model_kwargs = dict(model_structure.get("model_kwargs", {}))
    model_kwargs.setdefault(
        "num_classes",
        checkpoint.get("train_context", {}).get("class_num", 24),
    )
    model_kwargs.setdefault("dropout_p", 0.0)

    model_map = load_model_map()
    if model_name not in model_map:
        raise CheckpointRestoreError(f"不支持的模型名: {model_name}")

    model = model_map[model_name](**model_kwargs)
    success = load_state_dict_safely(model, checkpoint["model_state_dict"], strict=True)
    if not success:
        raise CheckpointRestoreError("无法以 strict=True 加载基座 checkpoint 权重")

    model.to(device)

    checkpoint_meta = {
        "checkpoint_path": checkpoint_path,
        "model_name": model_name,
        "model_kwargs": model_kwargs,
        "train_context": checkpoint.get("train_context", {}),
        "model_structure": model_structure,
        "input_tensor_meta": model_structure.get("input_tensor_meta"),
        "best_acc": checkpoint.get("best_acc"),
        "best_val_loss": checkpoint.get("best_val_loss"),
    }
    return model, checkpoint_meta, checkpoint
