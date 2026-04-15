#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ONNX 导出编排入口。"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable

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


@dataclass(frozen=True)
class BranchSpec:
    loader: Callable[[str, BranchRuntime, int], tuple[torch.nn.Module, dict, dict]]
    runtime_factory: Callable[[torch.device], BranchRuntime]
    export_shape_resolver: Callable[[dict, dict], list[int]]
    onnx_filename: str
    input_dtype: torch.dtype
    export_input_dtype_label: str
    post_export: Callable[[str], None] = field(default=lambda _onnx_path: None)
    validate_export: Callable[[str, dict, int], dict] = field(
        default=lambda _onnx_path, _checkpoint_meta, _opset_version: {}
    )
    model_builder: Callable[[torch.nn.Module], torch.nn.Module] = field(
        default=lambda model: model.eval()
    )
    exported_model_builder: Callable[[torch.nn.Module, torch.device], torch.nn.Module] = field(
        default=lambda model, _device: copy.deepcopy(model).eval()
    )


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


def _resolve_qat_export_shape(checkpoint_meta, _checkpoint):
    candidate_shape = checkpoint_meta.get("example_input_shape")
    if candidate_shape is None:
        candidate_shape = checkpoint_meta.get("quantization_meta", {}).get(
            "example_input_shape"
        )
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


def _build_pruning_runtime(device):
    return BranchRuntime(
        source_device=device,
        dataset_dtype="fp32",
        onnx_input_dtype=np.float16,
    )


def _build_qat_runtime(_device):
    return BranchRuntime(
        source_device=torch.device("cpu"),
        dataset_dtype="fp32",
        onnx_input_dtype=np.float32,
    )


def _load_pruning_artifacts(checkpoint_path, runtime, expected_num_classes):
    return load_pruning_checkpoint(
        checkpoint_path,
        runtime.source_device,
        expected_num_classes,
    )


def _load_qat_artifacts(checkpoint_path, runtime, expected_num_classes):
    return load_qat_checkpoint(
        checkpoint_path,
        runtime.source_device,
        expected_num_classes,
    )


def _validate_pruning_export(onnx_path, _checkpoint_meta, expected_opset_version):
    return validate_fp16_onnx(onnx_path, expected_opset_version)


def _rewrite_qat_export(onnx_path):
    rewrite_cann_qat_onnx(onnx_path)


def _validate_qat_export(onnx_path, checkpoint_meta, expected_opset_version):
    return validate_qat_quantized_onnx(
        onnx_path,
        checkpoint_meta["quantization_meta"],
        expected_opset_version,
    )


def _build_pruning_exported_model(model, device):
    return copy.deepcopy(model).eval().to(device).half()


def _build_qat_quantized_model(model, _device):
    return convert_fx(model.eval())


def _resolve_branch_spec(branch):
    try:
        return BRANCH_SPECS[branch]
    except KeyError as exc:
        raise ValueError(f"不支持的 ONNX 导出分支: {branch}") from exc


def _resolve_branch_runtime(branch, device):
    return _resolve_branch_spec(branch).runtime_factory(device)


def _load_branch_artifacts(branch, checkpoint_path, runtime, expected_num_classes):
    return _resolve_branch_spec(branch).loader(
        checkpoint_path,
        runtime,
        expected_num_classes,
    )


def _build_export_meta(export_meta, runtime, export_shape, input_dtype_label):
    export_meta.update(
        {
            "torch_device": str(runtime.source_device),
            "input_dtype": input_dtype_label,
            "export_shape": export_shape,
            "dynamic_batch": True,
        }
    )
    return export_meta


def _build_branch_artifacts_from_loaded(
    branch,
    runtime,
    model,
    checkpoint_meta,
    checkpoint,
    opset_version,
):
    spec = _resolve_branch_spec(branch)
    folder_path = create_output_directory(branch, checkpoint_meta)
    export_shape = spec.export_shape_resolver(checkpoint_meta, checkpoint)
    export_model = spec.exported_model_builder(model, runtime.source_device)
    example_input = torch.randn(
        *export_shape,
        dtype=spec.input_dtype,
        device=runtime.source_device,
    )
    onnx_path = f"{folder_path}/{spec.onnx_filename}"
    _export_model_to_onnx(export_model, example_input, onnx_path, opset_version)
    spec.post_export(onnx_path)
    export_meta = spec.validate_export(onnx_path, checkpoint_meta, opset_version)
    export_meta = _build_export_meta(
        export_meta=export_meta,
        runtime=runtime,
        export_shape=export_shape,
        input_dtype_label=spec.export_input_dtype_label,
    )
    return BranchArtifacts(
        branch=branch,
        runtime=runtime,
        model=spec.model_builder(model),
        checkpoint_meta=checkpoint_meta,
        checkpoint=checkpoint,
        export_shape=export_shape,
        onnx_path=onnx_path,
        export_meta=export_meta,
        folder_path=folder_path,
    )


def inspect_branch_checkpoint(branch, checkpoint_path, device, expected_num_classes):
    runtime = _resolve_branch_runtime(branch, device)
    _, checkpoint_meta, _ = _load_branch_artifacts(
        branch,
        checkpoint_path,
        runtime,
        expected_num_classes,
    )
    return checkpoint_meta


BRANCH_SPECS = {
    "pruning_fp16": BranchSpec(
        loader=_load_pruning_artifacts,
        runtime_factory=_build_pruning_runtime,
        export_shape_resolver=_resolve_pruning_export_shape,
        onnx_filename="model_fp16.onnx",
        input_dtype=torch.float16,
        export_input_dtype_label="float16",
        validate_export=_validate_pruning_export,
        model_builder=lambda model: model.eval(),
        exported_model_builder=_build_pruning_exported_model,
    ),
    "qat_convert": BranchSpec(
        loader=_load_qat_artifacts,
        runtime_factory=_build_qat_runtime,
        export_shape_resolver=_resolve_qat_export_shape,
        onnx_filename="model_quant.onnx",
        input_dtype=torch.float32,
        export_input_dtype_label="float32",
        post_export=_rewrite_qat_export,
        validate_export=_validate_qat_export,
        model_builder=lambda model: model.eval(),
        exported_model_builder=_build_qat_quantized_model,
    ),
}


_SUPPORTED_EXPORT_BRANCH_SET = set(SUPPORTED_EXPORT_BRANCHES)


def resolve_branch_opset_version(branch, requested_opset_version):
    if branch not in _SUPPORTED_EXPORT_BRANCH_SET:
        raise ValueError(f"不支持的 ONNX 导出分支: {branch}")
    if requested_opset_version is None:
        return ONNX_OPSET_VERSION
    if requested_opset_version != ONNX_OPSET_VERSION:
        raise ValueError("ONNX 导出当前仅支持 opset 16")
    return requested_opset_version


def build_metric_delta(source_metrics, onnx_metrics):
    if source_metrics is None or onnx_metrics is None:
        return None
    return {
        "loss": float(onnx_metrics["loss"] - source_metrics["loss"]),
        "acc": float(onnx_metrics["acc"] - source_metrics["acc"]),
        "samples": int(onnx_metrics["samples"] - source_metrics["samples"]),
    }


def build_branch_artifacts(branch, checkpoint_path, device, opset_version, expected_num_classes):
    runtime = _resolve_branch_runtime(branch, device)
    model, checkpoint_meta, checkpoint = _load_branch_artifacts(
        branch=branch,
        checkpoint_path=checkpoint_path,
        runtime=runtime,
        expected_num_classes=expected_num_classes,
    )
    return _build_branch_artifacts_from_loaded(
        branch=branch,
        runtime=runtime,
        model=model,
        checkpoint_meta=checkpoint_meta,
        checkpoint=checkpoint,
        opset_version=opset_version,
    )


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
