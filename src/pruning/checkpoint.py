#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基座 checkpoint 读取与模型恢复。"""

import os
import re

import torch

from .utils import (
    build_architecture_signature,
    load_model_map,
    load_state_dict_safely,
    to_repo_relative_path,
)


BEST_VAL_ACC_INFO_PATTERN = re.compile(
    r"^Best Validation Accuracy: (?P<val_acc>\d+(?:\.\d+)?), "
    r"Best Validation Loss: (?P<val_loss>\d+(?:\.\d+)?) at Epoch: (?P<epoch>\d+)$"
)


class CheckpointRestoreError(RuntimeError):
    """剪枝阶段 checkpoint 恢复错误。"""


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


def _parse_best_val_acc_info_line(line):
    match = BEST_VAL_ACC_INFO_PATTERN.fullmatch(line.strip())
    if match is None:
        return None
    return {
        "val_acc": float(match.group("val_acc")),
        "val_loss": float(match.group("val_loss")),
        "epoch": int(match.group("epoch")),
    }


def _read_last_valid_best_record(info_path):
    with open(info_path, "r", encoding="utf-8") as file_obj:
        for raw_line in reversed(file_obj.readlines()):
            line = raw_line.strip()
            if not line:
                continue
            parsed = _parse_best_val_acc_info_line(line)
            if parsed is not None:
                return parsed
    return None


def _collect_base_model_candidates(model_name):
    model_root = os.path.join("output", "base_model", model_name)
    if not os.path.isdir(model_root):
        raise FileNotFoundError(
            f"找不到基座模型目录: {model_root}\n"
            f"期望路径: output/base_model/{model_name}/<experiment_dir>/best_model.pth"
        )

    candidates = []
    for entry_name in sorted(os.listdir(model_root)):
        experiment_dir = os.path.join(model_root, entry_name)
        if not os.path.isdir(experiment_dir):
            continue

        info_path = os.path.join(experiment_dir, "best_val_acc_info.txt")
        checkpoint_path = os.path.join(experiment_dir, "best_model.pth")
        if not os.path.isfile(info_path) or not os.path.isfile(checkpoint_path):
            continue

        best_record = _read_last_valid_best_record(info_path)
        if best_record is None:
            continue

        candidates.append(
            {
                "experiment_name": entry_name,
                "checkpoint_path": checkpoint_path,
                **best_record,
            }
        )

    if not candidates:
        raise FileNotFoundError(
            "找不到可用的基座实验 checkpoint:\n"
            f"已扫描目录: {model_root}\n"
            "要求每个候选子目录同时包含可解析的 best_val_acc_info.txt 与 best_model.pth"
        )

    return candidates


def _select_best_base_experiment(candidates):
    return min(
        candidates,
        key=lambda item: (-item["val_acc"], item["val_loss"], item["experiment_name"]),
    )


def resolve_base_checkpoint_path(model_name):
    candidates = _collect_base_model_candidates(model_name)
    best_candidate = _select_best_base_experiment(candidates)
    selected_checkpoint_path = best_candidate["checkpoint_path"]
    return selected_checkpoint_path, selected_checkpoint_path


def load_base_checkpoint(model_name, device):
    checkpoint_link_path, resolved_checkpoint_path = resolve_base_checkpoint_path(model_name)

    checkpoint = torch.load(resolved_checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise CheckpointRestoreError("输入 checkpoint 不包含 model_state_dict，无法作为基座模型恢复")

    model_structure = checkpoint.get("model_structure", {})
    checkpoint_model_name = model_structure.get("model_name")
    if checkpoint_model_name is not None and checkpoint_model_name != model_name:
        raise CheckpointRestoreError(
            f"checkpoint 中模型名为 {checkpoint_model_name}，与命令行指定的 {model_name} 不一致"
        )

    if checkpoint_model_name is None:
        raise CheckpointRestoreError("checkpoint 中缺少 model_name，无法校验与命令行指定模型的一致性")

    architecture_signature = model_structure.get("architecture_signature")
    if architecture_signature is None:
        raise CheckpointRestoreError("checkpoint 中缺少 architecture_signature，无法执行强校验")

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
    _validate_architecture_signature(model, architecture_signature, "基座 checkpoint")
    success = load_state_dict_safely(model, checkpoint["model_state_dict"], strict=True)
    if not success:
        raise CheckpointRestoreError("无法以 strict=True 加载基座 checkpoint 权重")

    model.to(device)

    checkpoint_meta = {
        "checkpoint_link_path": to_repo_relative_path(checkpoint_link_path),
        "resolved_checkpoint_path": to_repo_relative_path(resolved_checkpoint_path),
        "checkpoint_path": resolved_checkpoint_path,
        "model_name": model_name,
        "model_kwargs": model_kwargs,
        "train_context": checkpoint.get("train_context", {}),
        "model_structure": model_structure,
        "input_tensor_meta": model_structure.get("input_tensor_meta"),
        "best_acc": checkpoint.get("best_acc"),
        "best_val_loss": checkpoint.get("best_val_loss"),
    }
    return model, checkpoint_meta, checkpoint
