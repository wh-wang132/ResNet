#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ONNX 导出编排入口。"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch.ao.quantization.quantize_fx import convert_fx

from qat.checkpoint import load_pruning_checkpoint, load_qat_checkpoint

from .output import create_output_directory
from .rewrite import rewrite_cann_qat_onnx
from .validate import validate_fp16_onnx, validate_qat_quantized_onnx


ONNX_OPSET_VERSION = 16
SUPPORTED_EXPORT_BRANCHES = ("pruning_fp16", "qat_convert")
ONNX_DYNAMIC_AXES = {
    "input": {0: "batch"},
    "logits": {0: "batch"},
}


@dataclass(frozen=True)
class BranchRuntime:
    source_device: torch.device
    dataset_dtype: str
    onnx_input_dtype: type


@dataclass
class BranchArtifacts:
    branch: str
    runtime: BranchRuntime
    model: torch.nn.Module
    checkpoint_meta: dict
    checkpoint: dict
    export_shape: list[int]
    onnx_path: str
    export_meta: dict
    folder_path: str


def inspect_branch_checkpoint(branch, checkpoint_path, device):
    if branch == "pruning_fp16":
        _, checkpoint_meta, _ = load_pruning_checkpoint(checkpoint_path, device)
        return checkpoint_meta
    if branch == "qat_convert":
        _, checkpoint_meta, _ = load_qat_checkpoint(checkpoint_path, torch.device("cpu"))
        return checkpoint_meta
    raise ValueError(f"不支持的 ONNX 导出分支: {branch}")


def normalize_export_shape(example_input_shape):
    normalized_shape = [int(dim) for dim in example_input_shape]
    if len(normalized_shape) != 4:
        raise ValueError("导出输入形状必须是 NCHW 四维")
    if any(dim <= 0 for dim in normalized_shape[1:]):
        raise ValueError("导出输入形状的 CHW 必须大于 0")
    normalized_shape[0] = 1
    return normalized_shape


def _resolve_pruning_export_shape(checkpoint_meta, checkpoint):
    model_structure = checkpoint_meta.get("model_structure", {})
    input_tensor_meta = model_structure.get("input_tensor_meta", {})
    candidate_shape = input_tensor_meta.get("batch_shape_nchw")
    if candidate_shape is None:
        candidate_shape = checkpoint.get("pruning_meta", {}).get("example_input_shape")
    if candidate_shape is None:
        raise ValueError("pruning checkpoint 缺少导出所需的输入形状")
    return normalize_export_shape(candidate_shape)


def _resolve_qat_export_shape(checkpoint_meta):
    candidate_shape = checkpoint_meta.get("example_input_shape")
    if candidate_shape is None:
        candidate_shape = checkpoint_meta.get("quantization_meta", {}).get("example_input_shape")
    if candidate_shape is None:
        raise ValueError("QAT checkpoint 缺少导出所需的输入形状")
    return normalize_export_shape(candidate_shape)


def _export_model_to_onnx(model, example_input, onnx_path, opset_version):
    torch.onnx.export(
        model,
        example_input,
        onnx_path,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=ONNX_DYNAMIC_AXES,
        dynamo=False,
    )


def _resolve_branch_runtime(branch, device):
    if branch == "pruning_fp16":
        return BranchRuntime(
            source_device=device,
            dataset_dtype="fp32",
            onnx_input_dtype=np.float16,
        )
    if branch == "qat_convert":
        return BranchRuntime(
            source_device=torch.device("cpu"),
            dataset_dtype="fp32",
            onnx_input_dtype=np.float32,
        )
    raise ValueError(f"不支持的 ONNX 导出分支: {branch}")


def resolve_branch_opset_version(branch, requested_opset_version):
    if branch not in SUPPORTED_EXPORT_BRANCHES:
        raise ValueError(f"不支持的 ONNX 导出分支: {branch}")
    if requested_opset_version is None:
        return ONNX_OPSET_VERSION
    if requested_opset_version != ONNX_OPSET_VERSION:
        raise ValueError("ONNX 导出当前仅支持 opset 16")
    return requested_opset_version


def _export_pruning_fp16_branch(checkpoint_path, runtime, folder_path, opset_version):
    model, checkpoint_meta, checkpoint = load_pruning_checkpoint(
        checkpoint_path,
        runtime.source_device,
    )
    export_shape = _resolve_pruning_export_shape(checkpoint_meta, checkpoint)

    export_model = copy.deepcopy(model).eval().to(runtime.source_device).half()
    example_input = torch.randn(*export_shape, dtype=torch.float16, device=runtime.source_device)
    onnx_path = f"{folder_path}/model_fp16.onnx"
    _export_model_to_onnx(export_model, example_input, onnx_path, opset_version)
    export_meta = validate_fp16_onnx(onnx_path, ONNX_OPSET_VERSION)
    export_meta.update(
        {
            "torch_device": str(runtime.source_device),
            "input_dtype": "float16",
            "export_shape": export_shape,
            "dynamic_batch": True,
        }
    )
    return BranchArtifacts(
        branch="pruning_fp16",
        runtime=runtime,
        model=model.eval(),
        checkpoint_meta=checkpoint_meta,
        checkpoint=checkpoint,
        export_shape=export_shape,
        onnx_path=onnx_path,
        export_meta=export_meta,
        folder_path=folder_path,
    )


def _export_qat_convert_branch(checkpoint_path, runtime, folder_path, opset_version):
    if opset_version != ONNX_OPSET_VERSION:
        raise ValueError("qat_convert 分支导出时必须使用 opset 16")

    prepared_model, checkpoint_meta, checkpoint = load_qat_checkpoint(
        checkpoint_path,
        runtime.source_device,
    )
    quantization_meta = checkpoint_meta["quantization_meta"]
    quantized_model = convert_fx(prepared_model.eval())
    export_shape = _resolve_qat_export_shape(checkpoint_meta)
    example_input = torch.randn(*export_shape, dtype=torch.float32, device=runtime.source_device)
    onnx_path = f"{folder_path}/model_quant.onnx"
    _export_model_to_onnx(quantized_model, example_input, onnx_path, opset_version)
    rewrite_cann_qat_onnx(onnx_path)
    export_meta = validate_qat_quantized_onnx(
        onnx_path,
        quantization_meta,
        ONNX_OPSET_VERSION,
    )
    export_meta.update(
        {
            "torch_device": str(runtime.source_device),
            "input_dtype": "float32",
            "export_shape": export_shape,
            "dynamic_batch": True,
        }
    )
    return BranchArtifacts(
        branch="qat_convert",
        runtime=runtime,
        model=quantized_model.eval(),
        checkpoint_meta=checkpoint_meta,
        checkpoint=checkpoint,
        export_shape=export_shape,
        onnx_path=onnx_path,
        export_meta=export_meta,
        folder_path=folder_path,
    )


def build_metric_delta(source_metrics, onnx_metrics):
    if source_metrics is None or onnx_metrics is None:
        return None
    return {
        "loss": float(onnx_metrics["loss"] - source_metrics["loss"]),
        "acc": float(onnx_metrics["acc"] - source_metrics["acc"]),
        "samples": int(onnx_metrics["samples"] - source_metrics["samples"]),
    }


def build_branch_artifacts(branch, checkpoint_path, device, opset_version):
    runtime = _resolve_branch_runtime(branch, device)
    checkpoint_meta = inspect_branch_checkpoint(branch, checkpoint_path, runtime.source_device)
    folder_path = create_output_directory(branch, checkpoint_meta)

    if branch == "pruning_fp16":
        return _export_pruning_fp16_branch(
            checkpoint_path=checkpoint_path,
            runtime=runtime,
            folder_path=folder_path,
            opset_version=opset_version,
        )
    if branch == "qat_convert":
        return _export_qat_convert_branch(
            checkpoint_path=checkpoint_path,
            runtime=runtime,
            folder_path=folder_path,
            opset_version=opset_version,
        )
    raise ValueError(f"不支持的 ONNX 导出分支: {branch}")


__all__ = [
    "BranchArtifacts",
    "BranchRuntime",
    "ONNX_OPSET_VERSION",
    "SUPPORTED_EXPORT_BRANCHES",
    "build_branch_artifacts",
    "build_metric_delta",
    "inspect_branch_checkpoint",
    "normalize_export_shape",
    "resolve_branch_opset_version",
]
