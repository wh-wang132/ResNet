#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ONNX 图校验工具。"""

from __future__ import annotations

from collections import Counter

import onnx
from onnx import TensorProto
import torch

from qat.quantization import validate_quantization_meta

from .rewrite import (
    _build_onnx_graph_maps,
    _extract_constant_tensor_value,
    _get_node_attr_int,
    _get_node_display_name,
)


QAT_EXPECTED_INPUT_PATTERNS = {
    "Conv": ("DQ", "Transpose(DQ[axis=1])", "raw"),
    "Add": ("DQ", "DQ"),
    "Gemm": ("DQ", "DQ", "raw"),
}


def summarize_onnx_graph(onnx_path):
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
        "opset_imports": {
            (item.domain or "ai.onnx"): int(item.version) for item in model.opset_import
        },
        "node_domains": node_domains,
        "op_counts": dict(sorted(op_counter.items())),
        "input_elem_type": int(input_elem_type),
        "output_elem_type": int(output_elem_type),
    }


def _validate_opset_version(summary, expected_opset_version, expected_message):
    if summary["opset_imports"].get("ai.onnx") != expected_opset_version:
        raise ValueError(expected_message)


def _describe_input_pattern(input_name, producer_map):
    producer = producer_map.get(input_name)
    if producer is None:
        return "raw"
    if producer.op_type == "Transpose" and producer.input:
        return f"Transpose({_describe_input_pattern(producer.input[0], producer_map)})"
    if producer.op_type != "DequantizeLinear":
        return producer.op_type
    axis = _get_node_attr_int(producer, "axis")
    if axis is None:
        return "DQ"
    return f"DQ[axis={axis}]"


def _get_constant_scale_value(scale_input_name, producer_map):
    scale_producer = producer_map.get(scale_input_name)
    if scale_producer is None:
        return None
    return _extract_constant_tensor_value(scale_producer)


def _validate_qat_input_patterns(model, producer_map):
    seen_ops = {key: 0 for key in QAT_EXPECTED_INPUT_PATTERNS}
    for node in model.graph.node:
        expected = QAT_EXPECTED_INPUT_PATTERNS.get(node.op_type)
        if expected is None:
            continue
        seen_ops[node.op_type] += 1
        actual = tuple(_describe_input_pattern(input_name, producer_map) for input_name in node.input)
        if actual != expected:
            raise ValueError(
                f"{node.op_type} 输入量化模式不满足 CANN 8.5 要求: 期望 {expected}, 实际 {actual}"
            )

    if seen_ops["Conv"] == 0 or seen_ops["Add"] == 0 or seen_ops["Gemm"] == 0:
        raise ValueError("QAT CANN ONNX 缺少 Conv/Add/Gemm 关键算子，无法校验量化模式")


def _validate_quantize_linear_consumers(model, producer_map, consumer_map):
    for node in model.graph.node:
        if node.op_type != "QuantizeLinear":
            continue
        consumers = consumer_map.get(node.output[0], [])
        if not consumers or any(consumer.op_type != "DequantizeLinear" for consumer in consumers):
            raise ValueError(f"{_get_node_display_name(node)} 后面必须直接连接 DequantizeLinear")

        scale_value = _get_constant_scale_value(node.input[1], producer_map)
        if scale_value is None or scale_value.size <= 1:
            continue
        axis = _get_node_attr_int(node, "axis")
        if axis != 1:
            raise ValueError(
                f"{_get_node_display_name(node)} 的 per-channel QuantizeLinear 必须使用 axis=1"
            )


def _validate_split_quantize_linear_params(model):
    split_prefix_groups = {}
    for node in model.graph.node:
        if node.op_type != "QuantizeLinear" or "_split_" not in (node.name or ""):
            continue
        base_name = node.name.rsplit("_split_", 1)[0]
        split_prefix_groups.setdefault(base_name, []).append(node)

    for base_name, split_nodes in split_prefix_groups.items():
        scale_inputs = {node.input[1] for node in split_nodes}
        zero_inputs = {node.input[2] for node in split_nodes}
        if len(scale_inputs) != len(split_nodes):
            raise ValueError(f"{base_name} 的 split QuantizeLinear 共享了 scale 节点")
        if len(zero_inputs) != len(split_nodes):
            raise ValueError(f"{base_name} 的 split QuantizeLinear 共享了 zero-point 节点")


def validate_fp16_onnx(onnx_path, expected_opset_version):
    summary = summarize_onnx_graph(onnx_path)
    _validate_opset_version(summary, expected_opset_version, "FP16 ONNX 必须使用 opset 16")
    if summary["input_elem_type"] != TensorProto.FLOAT16:
        raise ValueError("FP16 ONNX 输入不是 FLOAT16")
    return summary


def validate_qat_quantized_onnx(onnx_path, quantization_meta, expected_opset_version):
    summary = summarize_onnx_graph(onnx_path)
    _validate_opset_version(summary, expected_opset_version, "QAT CANN ONNX 必须使用 opset 16")
    op_counts = summary["op_counts"]
    if "QuantizeLinear" not in op_counts or "DequantizeLinear" not in op_counts:
        raise ValueError("量化 ONNX 缺少 QuantizeLinear/DequantizeLinear")
    if summary["input_elem_type"] != TensorProto.FLOAT:
        raise ValueError("QAT CANN ONNX 输入不是 FLOAT32")
    if summary["output_elem_type"] != TensorProto.FLOAT:
        raise ValueError("QAT CANN ONNX 输出不是 FLOAT32")
    validate_quantization_meta(quantization_meta)

    model = onnx.load(onnx_path)
    producer_map, consumer_map = _build_onnx_graph_maps(model)
    _validate_qat_input_patterns(model, producer_map)
    _validate_quantize_linear_consumers(model, producer_map, consumer_map)
    _validate_split_quantize_linear_params(model)
    return summary


__all__ = [
    "summarize_onnx_graph",
    "validate_fp16_onnx",
    "validate_qat_quantized_onnx",
]
