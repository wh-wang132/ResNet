#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QAT / pruning checkpoint 读取与剪枝结构恢复。"""

import copy
import os

import torch

from base_model.resnet_lightweight import (
    resnet10_2d_from_cfg,
    resnet14_2d_from_cfg,
    resnet6_2d_from_cfg,
)
from base_model.resnet_standard import (
    resnet18_2d_from_cfg,
    resnet34_2d_from_cfg,
)
from qat.quantization import prepare_model_for_qat
from qat.utils import load_state_dict_safely, to_repo_relative_path


class CheckpointRestoreError(RuntimeError):
    """QAT 阶段 checkpoint 恢复错误。"""


FROM_CFG_MODEL_MAP = {
    "resnet6_2d": resnet6_2d_from_cfg,
    "resnet10_2d": resnet10_2d_from_cfg,
    "resnet14_2d": resnet14_2d_from_cfg,
    "resnet18_2d": resnet18_2d_from_cfg,
    "resnet34_2d": resnet34_2d_from_cfg,
}


REQUIRED_MODEL_STRUCTURE_KEYS = {
    "model_name",
    "model_kwargs",
    "channel_cfg",
    "architecture_signature",
}

REQUIRED_QAT_CHECKPOINT_KEYS = {
    "model_state_dict",
    "model_structure",
    "quantization_meta",
}


def load_pruning_checkpoint(pruning_checkpoint_path, device):
    checkpoint_path = os.path.abspath(pruning_checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"找不到 pruning checkpoint: {pruning_checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise CheckpointRestoreError("输入 checkpoint 不包含 model_state_dict，无法作为 pruning checkpoint 恢复")

    model_structure = checkpoint.get("model_structure", {})
    missing_keys = sorted(REQUIRED_MODEL_STRUCTURE_KEYS - set(model_structure.keys()))
    if missing_keys:
        raise CheckpointRestoreError(
            f"pruning checkpoint 缺少关键 model_structure 字段: {', '.join(missing_keys)}"
        )

    model_name = model_structure["model_name"]
    if model_name not in FROM_CFG_MODEL_MAP:
        raise CheckpointRestoreError(f"不支持的模型名: {model_name}")

    model_kwargs = dict(model_structure.get("model_kwargs", {}))
    model_kwargs.setdefault("num_classes", checkpoint.get("train_context", {}).get("class_num", 24))
    model_kwargs.setdefault("dropout_p", model_kwargs.get("dropout_p", 0.0))
    include_top = model_structure.get("include_top", True)
    in_channels = model_structure.get("in_channels", 1)
    channel_cfg = copy.deepcopy(model_structure["channel_cfg"])

    model = FROM_CFG_MODEL_MAP[model_name](
        channel_cfg=channel_cfg,
        num_classes=model_kwargs.get("num_classes", 24),
        dropout_p=model_kwargs.get("dropout_p", 0.0),
        include_top=include_top,
        in_channels=in_channels,
    )
    success = load_state_dict_safely(model, checkpoint["model_state_dict"], strict=True)
    if not success:
        raise CheckpointRestoreError("无法以 strict=True 加载 pruning checkpoint 权重")

    model.to(device)

    checkpoint_meta = {
        "source_pruning_checkpoint_path": to_repo_relative_path(checkpoint_path),
        "checkpoint_path": checkpoint_path,
        "model_name": model_name,
        "model_kwargs": model_kwargs,
        "model_structure": copy.deepcopy(model_structure),
        "input_tensor_meta": model_structure.get("input_tensor_meta"),
        "best_acc": checkpoint.get("best_acc"),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "quantization_source": {
            "channel_cfg": copy.deepcopy(model_structure["channel_cfg"]),
            "architecture_signature": copy.deepcopy(model_structure["architecture_signature"]),
        },
    }
    return model, checkpoint_meta, checkpoint


def _restore_float_model_from_structure(model_structure, checkpoint, device):
    missing_keys = sorted(REQUIRED_MODEL_STRUCTURE_KEYS - set(model_structure.keys()))
    if missing_keys:
        raise CheckpointRestoreError(
            f"checkpoint 缺少关键 model_structure 字段: {', '.join(missing_keys)}"
        )

    model_name = model_structure["model_name"]
    if model_name not in FROM_CFG_MODEL_MAP:
        raise CheckpointRestoreError(f"不支持的模型名: {model_name}")

    model_kwargs = dict(model_structure.get("model_kwargs", {}))
    model_kwargs.setdefault("num_classes", checkpoint.get("train_context", {}).get("class_num", 24))
    model_kwargs.setdefault("dropout_p", model_kwargs.get("dropout_p", 0.0))
    include_top = model_structure.get("include_top", True)
    in_channels = model_structure.get("in_channels", 1)
    channel_cfg = copy.deepcopy(model_structure["channel_cfg"])

    model = FROM_CFG_MODEL_MAP[model_name](
        channel_cfg=channel_cfg,
        num_classes=model_kwargs.get("num_classes", 24),
        dropout_p=model_kwargs.get("dropout_p", 0.0),
        include_top=include_top,
        in_channels=in_channels,
    )
    model.to(device)
    return model, model_name, model_kwargs


def load_qat_checkpoint(qat_checkpoint_path, device):
    checkpoint_path = os.path.abspath(qat_checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"找不到 QAT checkpoint: {qat_checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    missing_top_keys = sorted(REQUIRED_QAT_CHECKPOINT_KEYS - set(checkpoint.keys()))
    if missing_top_keys:
        raise CheckpointRestoreError(
            f"输入 checkpoint 缺少关键 QAT 字段: {', '.join(missing_top_keys)}"
        )

    model_structure = checkpoint["model_structure"]
    float_model, model_name, model_kwargs = _restore_float_model_from_structure(
        model_structure=model_structure,
        checkpoint=checkpoint,
        device=device,
    )

    prepared_model, quantization_meta, example_inputs = prepare_model_for_qat(
        float_model,
        device=device,
        quantization_meta=checkpoint["quantization_meta"],
    )
    success = load_state_dict_safely(prepared_model, checkpoint["model_state_dict"], strict=True)
    if not success:
        raise CheckpointRestoreError("无法以 strict=True 加载 QAT checkpoint 的 prepared 权重")

    checkpoint_meta = {
        "checkpoint_path": checkpoint_path,
        "source_qat_checkpoint_path": to_repo_relative_path(checkpoint_path),
        "model_name": model_name,
        "model_kwargs": model_kwargs,
        "model_structure": copy.deepcopy(model_structure),
        "input_tensor_meta": model_structure.get("input_tensor_meta"),
        "quantization_meta": copy.deepcopy(quantization_meta),
        "example_input_shape": list(example_inputs[0].shape),
        "best_acc": checkpoint.get("best_acc"),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "train_context": copy.deepcopy(checkpoint.get("train_context", {})),
    }
    return prepared_model, checkpoint_meta, checkpoint
