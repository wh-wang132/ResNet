#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ONNX 导出链实现。"""

from __future__ import annotations

import copy
from collections import Counter
from typing import Any

import onnx
from onnx import TensorProto
import torch
from torch.ao.quantization.quantize_fx import convert_fx

from qat.checkpoint import load_pruning_checkpoint, load_qat_checkpoint
from qat.quantization import create_qat_qconfig_mapping_from_meta


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
        dynamic_axes=None,
        dynamo=False,
    )


def _summarize_onnx_graph(onnx_path):
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    node_domains = sorted(set(node.domain for node in model.graph.node))
    non_standard_domains = [domain for domain in node_domains if domain not in ("", "ai.onnx")]
    if non_standard_domains:
        raise ValueError(f"检测到非标准 ONNX domain: {non_standard_domains}")

    op_counter = Counter(node.op_type for node in model.graph.node)
    input_elem_type = model.graph.input[0].type.tensor_type.elem_type
    output_elem_type = model.graph.output[0].type.tensor_type.elem_type
    return {
        "node_domains": node_domains,
        "op_counts": dict(sorted(op_counter.items())),
        "input_elem_type": int(input_elem_type),
        "output_elem_type": int(output_elem_type),
    }


def _validate_fp16_onnx(onnx_path):
    summary = _summarize_onnx_graph(onnx_path)
    if summary["input_elem_type"] != TensorProto.FLOAT16:
        raise ValueError("FP16 ONNX 输入不是 FLOAT16")
    return summary


def _validate_qat_quantized_onnx(onnx_path, quantization_meta):
    summary = _summarize_onnx_graph(onnx_path)
    op_counts = summary["op_counts"]
    if "QuantizeLinear" not in op_counts or "DequantizeLinear" not in op_counts:
        raise ValueError("量化 ONNX 缺少 QuantizeLinear/DequantizeLinear")
    if quantization_meta.get("weight_qscheme") != str(torch.per_channel_symmetric):
        raise ValueError("QAT checkpoint 未使用权重 per-channel symmetric")
    if quantization_meta.get("activation_qscheme") != str(torch.per_tensor_affine):
        raise ValueError("QAT checkpoint 未使用激活 per-tensor affine")
    return summary


def export_pruning_fp16_branch(checkpoint_path, device, folder_path, opset_version):
    model, checkpoint_meta, checkpoint = load_pruning_checkpoint(checkpoint_path, device)
    export_shape = _resolve_pruning_export_shape(checkpoint_meta, checkpoint)

    export_model = copy.deepcopy(model).eval().to(device).half()
    example_input = torch.randn(*export_shape, dtype=torch.float16, device=device)
    onnx_path = f"{folder_path}/model_fp16.onnx"
    _export_model_to_onnx(export_model, example_input, onnx_path, opset_version)
    export_meta = _validate_fp16_onnx(onnx_path)
    export_meta.update(
        {
            "torch_device": str(device),
            "input_dtype": "float16",
            "export_shape": export_shape,
        }
    )
    return model.eval(), checkpoint_meta, checkpoint, export_shape, onnx_path, export_meta


def export_qat_convert_branch(checkpoint_path, folder_path, opset_version):
    prepared_model, checkpoint_meta, checkpoint = load_qat_checkpoint(
        checkpoint_path,
        torch.device("cpu"),
    )
    quantization_meta = checkpoint_meta["quantization_meta"]
    qconfig_mapping = create_qat_qconfig_mapping_from_meta(quantization_meta)
    quantized_model = convert_fx(
        prepared_model.eval(),
        qconfig_mapping=qconfig_mapping,
    )
    export_shape = _resolve_qat_export_shape(checkpoint_meta)
    example_input = torch.randn(*export_shape, dtype=torch.float32)
    onnx_path = f"{folder_path}/model_quant.onnx"
    _export_model_to_onnx(quantized_model, example_input, onnx_path, opset_version)
    export_meta = _validate_qat_quantized_onnx(onnx_path, quantization_meta)
    export_meta.update(
        {
            "torch_device": "cpu",
            "input_dtype": "float32",
            "export_shape": export_shape,
        }
    )
    return quantized_model.eval(), checkpoint_meta, checkpoint, export_shape, onnx_path, export_meta


def build_metric_delta(source_metrics, onnx_metrics):
    if source_metrics is None or onnx_metrics is None:
        return None
    return {
        "loss": float(onnx_metrics["loss"] - source_metrics["loss"]),
        "acc": float(onnx_metrics["acc"] - source_metrics["acc"]),
        "samples": int(onnx_metrics["samples"] - source_metrics["samples"]),
    }


def build_branch_artifacts(branch, checkpoint_path, device, folder_path, opset_version):
    if branch == "pruning_fp16":
        return export_pruning_fp16_branch(checkpoint_path, device, folder_path, opset_version)
    if branch == "qat_convert":
        return export_qat_convert_branch(checkpoint_path, folder_path, opset_version)
    raise ValueError(f"不支持的 ONNX 导出分支: {branch}")
