#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ONNX 导出链实现。"""

from __future__ import annotations

import copy
from collections import Counter

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import torch
from torch.ao.quantization.quantize_fx import convert_fx

from qat.checkpoint import load_pruning_checkpoint, load_qat_checkpoint

ONNX_OPSET_VERSION = 16


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


def resolve_branch_opset_version(branch, requested_opset_version):
    if branch not in ("pruning_fp16", "qat_convert"):
        raise ValueError(f"不支持的 ONNX 导出分支: {branch}")
    if requested_opset_version is None:
        return ONNX_OPSET_VERSION
    if requested_opset_version != ONNX_OPSET_VERSION:
        raise ValueError("ONNX 导出当前仅支持 opset 16")
    return requested_opset_version


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
        "opset_imports": {
            (item.domain or "ai.onnx"): int(item.version) for item in model.opset_import
        },
        "node_domains": node_domains,
        "op_counts": dict(sorted(op_counter.items())),
        "input_elem_type": int(input_elem_type),
        "output_elem_type": int(output_elem_type),
    }


def _build_onnx_graph_maps(model):
    producer_map = {}
    consumer_map = {}
    for node in model.graph.node:
        for output_name in node.output:
            producer_map[output_name] = node
        for input_name in node.input:
            consumer_map.setdefault(input_name, []).append(node)
    return producer_map, consumer_map


def _get_node_attr_int(node, name):
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return None


def _set_node_attr_int(node, name, value):
    kept = [attr for attr in node.attribute if attr.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.extend([helper.make_attribute(name, int(value))])


def _remove_node_attr(node, name):
    kept = [attr for attr in node.attribute if attr.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)


def _evaluate_constant_value(name, producer_map, initializer_map):
    if name in initializer_map:
        return initializer_map[name]

    node = producer_map.get(name)
    if node is None:
        raise KeyError(name)

    if node.op_type == "Constant":
        for attr in node.attribute:
            if attr.name == "value":
                return numpy_helper.to_array(attr.t)
    if node.op_type == "Identity":
        return _evaluate_constant_value(node.input[0], producer_map, initializer_map)
    if node.op_type == "Cast":
        dtype_map = {
            TensorProto.FLOAT: np.float32,
            TensorProto.UINT8: np.uint8,
            TensorProto.INT8: np.int8,
            TensorProto.INT32: np.int32,
            TensorProto.INT64: np.int64,
            TensorProto.BOOL: np.bool_,
        }
        cast_dtype = dtype_map[_get_node_attr_int(node, "to")]
        return _evaluate_constant_value(node.input[0], producer_map, initializer_map).astype(
            cast_dtype,
            copy=False,
        )
    if node.op_type == "ConstantOfShape":
        shape = _evaluate_constant_value(node.input[0], producer_map, initializer_map).astype(
            np.int64
        ).reshape(-1)
        fill_value = np.array(0, dtype=np.float32)
        for attr in node.attribute:
            if attr.name == "value":
                fill_value = numpy_helper.to_array(attr.t).reshape(-1)[0]
                break
        return np.full(
            tuple(int(item) for item in shape.tolist()),
            fill_value,
            dtype=np.array(fill_value).dtype,
        )
    raise RuntimeError(f"不支持的常量链节点: {node.op_type} ({node.name})")


def _dequantize_array(q_array, scale_array, zero_array, axis):
    qf = q_array.astype(np.float32)
    zf = zero_array.astype(np.float32)
    sf = scale_array.astype(np.float32)
    if axis is None:
        return (qf - zf) * sf

    reshape = [1] * q_array.ndim
    reshape[axis] = q_array.shape[axis]
    return (qf - zf.reshape(reshape)) * sf.reshape(reshape)


def _append_initializer(model, initializer_map, name, array):
    tensor = numpy_helper.from_array(array, name=name)
    model.graph.initializer.append(tensor)
    initializer_map[name] = array


def _build_constant_node(output_name, array, node_name):
    tensor = numpy_helper.from_array(array, name=f"{output_name}_tensor")
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[output_name],
        name=node_name,
        value=tensor,
    )


def _clone_quant_param_input(
    model,
    producer_map,
    initializer_map,
    input_name,
    cloned_output_name,
):
    producer = producer_map.get(input_name)
    if producer is None:
        if input_name not in initializer_map:
            raise KeyError(input_name)
        cloned_array = np.array(initializer_map[input_name], copy=True)
        _append_initializer(model, initializer_map, cloned_output_name, cloned_array)
        return None, cloned_output_name

    cloned_array = np.array(_evaluate_constant_value(input_name, producer_map, initializer_map), copy=True)
    cloned_node = _build_constant_node(
        output_name=cloned_output_name,
        array=cloned_array,
        node_name=f"{cloned_output_name}_const",
    )
    return cloned_node, cloned_output_name


def _replace_all_inputs(model, old_name, new_name):
    for node in model.graph.node:
        for index, input_name in enumerate(node.input):
            if input_name == old_name:
                node.input[index] = new_name
    for output in model.graph.output:
        if output.name == old_name:
            output.name = new_name


def _split_multi_output_activation_quantize_nodes(model):
    producer_map, consumer_map = _build_onnx_graph_maps(model)
    initializer_map = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    nodes_to_remove = set()
    extra_before = {}

    for node in model.graph.node:
        if node.op_type != "QuantizeLinear" or len(node.output) != 1:
            continue
        output_name = node.output[0]
        downstream = consumer_map.get(output_name, [])
        if len(downstream) <= 1:
            continue
        if any(consumer.op_type != "DequantizeLinear" for consumer in downstream):
            raise ValueError(f"{node.name or node.op_type} 存在非 DequantizeLinear consumer，无法安全拆分")

        quant_scale = _evaluate_constant_value(node.input[1], producer_map, {})
        quant_zero = _evaluate_constant_value(node.input[2], producer_map, {})
        if quant_scale.size > 1:
            raise ValueError(f"{node.name or node.op_type} 的 activation QuantizeLinear 不应为 per-channel")

        for consumer in downstream:
            dequant_scale = _evaluate_constant_value(consumer.input[1], producer_map, {})
            dequant_zero = _evaluate_constant_value(consumer.input[2], producer_map, {})
            if not (quant_scale == dequant_scale).all() or not (quant_zero == dequant_zero).all():
                raise ValueError(
                    f"{node.name or node.op_type} 的多个 consumer 使用了不一致的量化参数，拒绝重写"
                )

        nodes_to_remove.add(id(node))
        for index, consumer in enumerate(downstream):
            cloned_output = f"{node.output[0]}_split_{index}"
            cloned_name = f"{node.name}_split_{index}" if node.name else f"QuantizeLinear_split_{index}"
            scale_clone_name = f"{cloned_name}_scale"
            zero_clone_name = f"{cloned_name}_zero"
            scale_node, scale_input = _clone_quant_param_input(
                model,
                producer_map,
                initializer_map,
                node.input[1],
                scale_clone_name,
            )
            zero_node, zero_input = _clone_quant_param_input(
                model,
                producer_map,
                initializer_map,
                node.input[2],
                zero_clone_name,
            )
            cloned_q = helper.make_node(
                "QuantizeLinear",
                [node.input[0], scale_input, zero_input],
                [cloned_output],
                name=cloned_name,
            )
            cloned_nodes = []
            if scale_node is not None:
                cloned_nodes.append(scale_node)
            if zero_node is not None:
                cloned_nodes.append(zero_node)
            cloned_nodes.append(cloned_q)
            extra_before.setdefault(id(consumer), []).extend(cloned_nodes)
            for input_index, input_name in enumerate(consumer.input):
                if input_name == output_name:
                    consumer.input[input_index] = cloned_output

    if not nodes_to_remove:
        return

    kept_nodes = []
    for node in model.graph.node:
        kept_nodes.extend(extra_before.get(id(node), []))
        if id(node) not in nodes_to_remove:
            kept_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)


def _collapse_quantize_cast_pairs(model):
    producer_map, consumer_map = _build_onnx_graph_maps(model)
    nodes_to_remove = set()

    for node in model.graph.node:
        if node.op_type != "Cast" or len(node.input) != 1 or len(node.output) != 1:
            continue
        cast_input = node.input[0]
        cast_output = node.output[0]
        producer = producer_map.get(cast_input)
        consumers = consumer_map.get(cast_output, [])
        if producer is None or producer.op_type != "QuantizeLinear":
            continue
        if not consumers or any(consumer.op_type != "DequantizeLinear" for consumer in consumers):
            continue
        for consumer in consumers:
            for index, input_name in enumerate(consumer.input):
                if input_name == cast_output:
                    consumer.input[index] = cast_input
        nodes_to_remove.add(id(node))

    if not nodes_to_remove:
        return

    kept_nodes = [node for node in model.graph.node if id(node) not in nodes_to_remove]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)


def _collapse_identity_nodes(model):
    _, consumer_map = _build_onnx_graph_maps(model)
    nodes_to_remove = set()

    for node in model.graph.node:
        if node.op_type != "Identity" or len(node.input) != 1 or len(node.output) != 1:
            continue
        identity_input = node.input[0]
        identity_output = node.output[0]
        consumers = consumer_map.get(identity_output, [])
        for consumer in consumers:
            for index, input_name in enumerate(consumer.input):
                if input_name == identity_output:
                    consumer.input[index] = identity_input
        for output in model.graph.output:
            if output.name == identity_output:
                output.name = identity_input
        nodes_to_remove.add(id(node))

    if not nodes_to_remove:
        return

    kept_nodes = [node for node in model.graph.node if id(node) not in nodes_to_remove]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)


def _rewrite_quantized_weight_and_bias_paths(model):
    producer_map, _ = _build_onnx_graph_maps(model)
    initializer_map = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    nodes_to_remove = set()
    extra_before = {}
    extra_after = {}

    weight_targets = []
    bias_targets = []
    for consumer in model.graph.node:
        if consumer.op_type not in ("Conv", "Gemm"):
            continue
        if len(consumer.input) > 1:
            weight_source = producer_map.get(consumer.input[1])
            if weight_source is not None and weight_source.op_type == "DequantizeLinear":
                weight_targets.append((consumer, weight_source))
        if len(consumer.input) > 2:
            bias_source = producer_map.get(consumer.input[2])
            if bias_source is not None and bias_source.op_type == "DequantizeLinear":
                bias_targets.append((consumer, bias_source))

    for consumer, dq_node in weight_targets:
        source_node = producer_map.get(dq_node.input[0])
        if source_node is not None and source_node.op_type == "QuantizeLinear":
            continue

        axis = _get_node_attr_int(dq_node, "axis")
        q_array = _evaluate_constant_value(dq_node.input[0], producer_map, initializer_map)
        scale_array = _evaluate_constant_value(dq_node.input[1], producer_map, initializer_map)
        zero_array = _evaluate_constant_value(dq_node.input[2], producer_map, initializer_map)
        float_array = _dequantize_array(q_array, scale_array, zero_array, axis).astype(
            np.float32,
            copy=False,
        )

        q_node_name = f"{dq_node.name}_amct_quant"
        q_output_name = f"{dq_node.name}_amct_quant_out"
        dq_output_name = dq_node.output[0]

        if consumer.op_type == "Conv" and scale_array.size > 1:
            transposed_array = np.transpose(float_array, (1, 0, 2, 3)).astype(np.float32, copy=False)
            float_name = f"{dq_node.name}_amct_float"
            _append_initializer(model, initializer_map, float_name, transposed_array)
            q_node = helper.make_node(
                "QuantizeLinear",
                [float_name, dq_node.input[1], dq_node.input[2]],
                [q_output_name],
                name=q_node_name,
                axis=1,
            )
            dq_transposed_output = f"{dq_node.name}_amct_dq_transposed"
            dq_node.input[0] = q_output_name
            dq_node.output[0] = dq_transposed_output
            _set_node_attr_int(dq_node, "axis", 1)
            transpose_back = helper.make_node(
                "Transpose",
                [dq_transposed_output],
                [dq_output_name],
                name=f"{dq_node.name}_amct_transpose_back",
                perm=[1, 0, 2, 3],
            )
            extra_before.setdefault(id(dq_node), []).append(q_node)
            extra_after.setdefault(id(dq_node), []).append(transpose_back)
        else:
            float_name = f"{dq_node.name}_amct_float"
            _append_initializer(model, initializer_map, float_name, float_array)
            q_inputs = [float_name, dq_node.input[1], dq_node.input[2]]
            q_node = helper.make_node(
                "QuantizeLinear",
                q_inputs,
                [q_output_name],
                name=q_node_name,
            )
            dq_node.input[0] = q_output_name
            _remove_node_attr(dq_node, "axis")
            extra_before.setdefault(id(dq_node), []).append(q_node)

    for _, dq_node in bias_targets:
        source_node = producer_map.get(dq_node.input[0])
        if source_node is not None and source_node.op_type == "QuantizeLinear":
            continue

        axis = _get_node_attr_int(dq_node, "axis")
        q_array = _evaluate_constant_value(dq_node.input[0], producer_map, initializer_map)
        scale_array = _evaluate_constant_value(dq_node.input[1], producer_map, initializer_map)
        zero_array = _evaluate_constant_value(dq_node.input[2], producer_map, initializer_map)
        float_array = _dequantize_array(q_array, scale_array, zero_array, axis).astype(
            np.float32,
            copy=False,
        )
        bias_name = f"{dq_node.name}_amct_bias_float"
        _append_initializer(model, initializer_map, bias_name, float_array)
        _replace_all_inputs(model, dq_node.output[0], bias_name)
        nodes_to_remove.add(id(dq_node))

    if not extra_before and not extra_after and not nodes_to_remove:
        return

    kept_nodes = []
    for node in model.graph.node:
        kept_nodes.extend(extra_before.get(id(node), []))
        if id(node) not in nodes_to_remove:
            kept_nodes.append(node)
            kept_nodes.extend(extra_after.get(id(node), []))
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)


def _rewrite_cann_qat_onnx(onnx_path):
    model = onnx.load(onnx_path)
    _collapse_quantize_cast_pairs(model)
    _collapse_identity_nodes(model)
    _split_multi_output_activation_quantize_nodes(model)
    _rewrite_quantized_weight_and_bias_paths(model)
    _collapse_identity_nodes(model)
    onnx.save(model, onnx_path)


def _validate_fp16_onnx(onnx_path):
    summary = _summarize_onnx_graph(onnx_path)
    if summary["opset_imports"].get("ai.onnx") != ONNX_OPSET_VERSION:
        raise ValueError("FP16 ONNX 必须使用 opset 16")
    if summary["input_elem_type"] != TensorProto.FLOAT16:
        raise ValueError("FP16 ONNX 输入不是 FLOAT16")
    return summary


def _validate_qat_quantized_onnx(onnx_path, quantization_meta):
    summary = _summarize_onnx_graph(onnx_path)
    if summary["opset_imports"].get("ai.onnx") != ONNX_OPSET_VERSION:
        raise ValueError("QAT CANN ONNX 必须使用 opset 16")
    op_counts = summary["op_counts"]
    if "QuantizeLinear" not in op_counts or "DequantizeLinear" not in op_counts:
        raise ValueError("量化 ONNX 缺少 QuantizeLinear/DequantizeLinear")
    if summary["input_elem_type"] != TensorProto.FLOAT:
        raise ValueError("QAT CANN ONNX 输入不是 FLOAT32")
    if summary["output_elem_type"] != TensorProto.FLOAT:
        raise ValueError("QAT CANN ONNX 输出不是 FLOAT32")
    if quantization_meta.get("activation_qscheme") != str(torch.per_tensor_affine):
        raise ValueError("QAT checkpoint 未使用激活 per-tensor affine")
    if quantization_meta.get("conv_weight_qscheme") != str(torch.per_channel_symmetric):
        raise ValueError("QAT checkpoint 未使用 Conv 权重 per-channel symmetric")
    if quantization_meta.get("linear_weight_qscheme") != str(torch.per_tensor_symmetric):
        raise ValueError("QAT checkpoint 未使用 Linear 权重 per-tensor symmetric")

    model = onnx.load(onnx_path)
    producer_map = {}
    consumer_map = {}
    for node in model.graph.node:
        for output_name in node.output:
            producer_map[output_name] = node
        for input_name in node.input:
            consumer_map.setdefault(input_name, []).append(node)

    def describe_input_pattern(input_name):
        producer = producer_map.get(input_name)
        if producer is None:
            return "raw"
        if producer.op_type == "Transpose" and producer.input:
            inner_pattern = describe_input_pattern(producer.input[0])
            return f"Transpose({inner_pattern})"
        if producer.op_type != "DequantizeLinear":
            return producer.op_type
        axis = None
        for attr in producer.attribute:
            if attr.name == "axis":
                axis = int(attr.i)
                break
        if axis is None:
            return "DQ"
        return f"DQ[axis={axis}]"

    expected_patterns = {
        "Conv": ("DQ", "Transpose(DQ[axis=1])", "raw"),
        "Add": ("DQ", "DQ"),
        "Gemm": ("DQ", "DQ", "raw"),
    }
    seen_ops = {key: 0 for key in expected_patterns}
    for node in model.graph.node:
        expected = expected_patterns.get(node.op_type)
        if expected is None:
            continue
        seen_ops[node.op_type] += 1
        actual = tuple(describe_input_pattern(input_name) for input_name in node.input)
        if actual != expected:
            raise ValueError(
                f"{node.op_type} 输入量化模式不满足 CANN 8.5 要求: 期望 {expected}, 实际 {actual}"
            )

    if seen_ops["Conv"] == 0 or seen_ops["Add"] == 0 or seen_ops["Gemm"] == 0:
        raise ValueError("QAT CANN ONNX 缺少 Conv/Add/Gemm 关键算子，无法校验量化模式")
    for node in model.graph.node:
        if node.op_type != "QuantizeLinear":
            continue
        output_name = node.output[0]
        consumers = consumer_map.get(output_name, [])
        if not consumers or any(consumer.op_type != "DequantizeLinear" for consumer in consumers):
            raise ValueError(f"{node.name or node.op_type} 后面必须直接连接 DequantizeLinear")
        scale_input_name = node.input[1]
        scale_producer = producer_map.get(scale_input_name)
        if scale_producer is None:
            continue
        scale_value = None
        if scale_producer.op_type == "Constant":
            for attr in scale_producer.attribute:
                if attr.name == "value":
                    scale_value = numpy_helper.to_array(attr.t)
                    break
        if scale_value is None or scale_value.size <= 1:
            continue
        axis = _get_node_attr_int(node, "axis")
        if axis != 1:
            raise ValueError(f"{node.name or node.op_type} 的 per-channel QuantizeLinear 必须使用 axis=1")

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
    if opset_version != ONNX_OPSET_VERSION:
        raise ValueError("qat_convert 分支导出时必须使用 opset 16")
    prepared_model, checkpoint_meta, checkpoint = load_qat_checkpoint(
        checkpoint_path,
        torch.device("cpu"),
    )
    quantization_meta = checkpoint_meta["quantization_meta"]
    quantized_model = convert_fx(prepared_model.eval())
    export_shape = _resolve_qat_export_shape(checkpoint_meta)
    example_input = torch.randn(*export_shape, dtype=torch.float32)
    onnx_path = f"{folder_path}/model_quant.onnx"
    _export_model_to_onnx(quantized_model, example_input, onnx_path, opset_version)
    _rewrite_cann_qat_onnx(onnx_path)
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
