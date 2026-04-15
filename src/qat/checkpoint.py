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
from .quantization import prepare_model_for_qat
from .utils import (
    build_architecture_signature,
    load_state_dict_safely,
    resolve_model_structure_num_classes,
    to_repo_relative_path,
)


class CheckpointRestoreError(RuntimeError):
    """QAT 阶段 checkpoint 恢复错误。"""


def _validate_architecture_signature(model, expected_signature, label):
    actual_signature = build_architecture_signature(model)
    expected_hash = expected_signature.get("signature_hash")
    actual_hash = actual_signature.get("signature_hash")
    if expected_hash is None or actual_hash is None:
        raise CheckpointRestoreError(f"{label} 缺少 architecture_signature.signature_hash，无法校验")
    if actual_hash != expected_hash:
        raise CheckpointRestoreError(
            f"{label} 的 architecture_signature 校验失败: expected={expected_hash}, actual={actual_hash}"
        )


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


def _validate_expected_num_classes(model_structure, state_dict, expected_num_classes, label):
    checkpoint_num_classes = resolve_model_structure_num_classes(
        model_structure,
        state_dict=state_dict,
    )
    if checkpoint_num_classes is None:
        raise CheckpointRestoreError(f"{label} 无法解析分类头输出维度")
    if checkpoint_num_classes != expected_num_classes:
        raise CheckpointRestoreError(
            f"{label} 的分类头输出维度与 Data 目录类别数不一致: "
            f"expected={expected_num_classes}, actual={checkpoint_num_classes}"
        )
    return checkpoint_num_classes


def load_pruning_checkpoint(pruning_checkpoint_path, device, expected_num_classes):
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

    restore_spec = _build_restore_spec(model_structure, checkpoint, expected_num_classes)
    model = _build_float_model_from_restore_spec(restore_spec)
    _validate_architecture_signature(
        model,
        model_structure["architecture_signature"],
        "pruning checkpoint",
    )
    success = load_state_dict_safely(model, checkpoint["model_state_dict"], strict=True)
    if not success:
        raise CheckpointRestoreError("无法以 strict=True 加载 pruning checkpoint 权重")

    model.to(device)

    checkpoint_meta = {
        "source_pruning_checkpoint_path": to_repo_relative_path(checkpoint_path),
        "checkpoint_path": checkpoint_path,
        "model_name": restore_spec["model_name"],
        "model_kwargs": restore_spec["model_kwargs"],
        "num_classes": restore_spec["num_classes"],
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


def _build_restore_spec(model_structure, checkpoint, expected_num_classes):
    missing_keys = sorted(REQUIRED_MODEL_STRUCTURE_KEYS - set(model_structure.keys()))
    if missing_keys:
        raise CheckpointRestoreError(
            f"checkpoint 缺少关键 model_structure 字段: {', '.join(missing_keys)}"
        )

    model_name = model_structure["model_name"]
    if model_name not in FROM_CFG_MODEL_MAP:
        raise CheckpointRestoreError(f"不支持的模型名: {model_name}")

    architecture_signature = model_structure.get("architecture_signature")
    if architecture_signature is None:
        raise CheckpointRestoreError("checkpoint 中缺少 architecture_signature，无法执行强校验")

    resolved_num_classes = _validate_expected_num_classes(
        model_structure,
        checkpoint["model_state_dict"],
        expected_num_classes,
        "checkpoint",
    )

    model_kwargs = dict(model_structure.get("model_kwargs", {}))
    model_kwargs.pop("num_classes", None)
    model_kwargs.setdefault("dropout_p", model_kwargs.get("dropout_p", 0.0))
    return {
        "model_name": model_name,
        "model_kwargs": model_kwargs,
        "num_classes": resolved_num_classes,
        "include_top": model_structure.get("include_top", True),
        "in_channels": model_structure.get("in_channels", 1),
        "channel_cfg": copy.deepcopy(model_structure["channel_cfg"]),
    }


def _build_float_model_from_restore_spec(restore_spec):
    model = FROM_CFG_MODEL_MAP[restore_spec["model_name"]](
        channel_cfg=restore_spec["channel_cfg"],
        num_classes=restore_spec["num_classes"],
        dropout_p=restore_spec["model_kwargs"].get("dropout_p", 0.0),
        include_top=restore_spec["include_top"],
        in_channels=restore_spec["in_channels"],
    )
    return model


def load_qat_checkpoint(qat_checkpoint_path, device, expected_num_classes):
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
    restore_spec = _build_restore_spec(model_structure, checkpoint, expected_num_classes)
    float_model = _build_float_model_from_restore_spec(restore_spec)
    _validate_architecture_signature(
        float_model,
        model_structure["architecture_signature"],
        "QAT checkpoint",
    )
    float_model.to(device)

    try:
        prepared_model, quantization_meta, example_inputs = prepare_model_for_qat(
            float_model,
            device=device,
            quantization_meta=checkpoint["quantization_meta"],
        )
    except Exception as exc:
        raise CheckpointRestoreError(
            f"QAT checkpoint 的 quantization_meta 与当前实现不兼容: {exc}"
        ) from exc

    success = load_state_dict_safely(prepared_model, checkpoint["model_state_dict"], strict=True)
    if not success:
        raise CheckpointRestoreError("无法以 strict=True 加载 QAT checkpoint 的 prepared 权重")

    checkpoint_meta = {
        "checkpoint_path": checkpoint_path,
        "source_qat_checkpoint_path": to_repo_relative_path(checkpoint_path),
        "model_name": restore_spec["model_name"],
        "model_kwargs": restore_spec["model_kwargs"],
        "num_classes": restore_spec["num_classes"],
        "model_structure": copy.deepcopy(model_structure),
        "input_tensor_meta": model_structure.get("input_tensor_meta"),
        "quantization_meta": copy.deepcopy(quantization_meta),
        "example_input_shape": list(example_inputs[0].shape),
        "best_acc": checkpoint.get("best_acc"),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "train_context": copy.deepcopy(checkpoint.get("train_context", {})),
    }
    return prepared_model, checkpoint_meta, checkpoint
