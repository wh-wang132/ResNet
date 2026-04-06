#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QAT ONNX 图重写工具。"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import helper, numpy_helper


def _build_onnx_graph_maps(model):
    producer_map = {}
    consumer_map = {}
    for node in model.graph.node:
        for output_name in node.output:
            producer_map[output_name] = node
        for input_name in node.input:
            consumer_map.setdefault(input_name, []).append(node)
    return producer_map, consumer_map


def _build_initializer_map(model):
    return {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }


def _get_node_display_name(node):
    return node.name or node.op_type


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


def _extract_constant_tensor_value(node):
    if node.op_type != "Constant":
        return None
    for attr in node.attribute:
        if attr.name == "value":
            return numpy_helper.to_array(attr.t)
    return None


def _evaluate_constant_value(name, producer_map, initializer_map):
    if name in initializer_map:
        return initializer_map[name]

    node = producer_map.get(name)
    if node is None:
        raise KeyError(name)

    constant_value = _extract_constant_tensor_value(node)
    if constant_value is not None:
        return constant_value
    if node.op_type == "Identity":
        return _evaluate_constant_value(node.input[0], producer_map, initializer_map)
    if node.op_type == "Cast":
        dtype_map = {
            onnx.TensorProto.FLOAT: np.float32,
            onnx.TensorProto.UINT8: np.uint8,
            onnx.TensorProto.INT8: np.int8,
            onnx.TensorProto.INT32: np.int32,
            onnx.TensorProto.INT64: np.int64,
            onnx.TensorProto.BOOL: np.bool_,
        }
        cast_dtype = dtype_map[_get_node_attr_int(node, "to")]
        return _evaluate_constant_value(node.input[0], producer_map, initializer_map).astype(
            cast_dtype,
            copy=False,
        )
    if node.op_type == "ConstantOfShape":
        shape = _evaluate_constant_value(
            node.input[0],
            producer_map,
            initializer_map,
        ).astype(np.int64).reshape(-1)
        fill_value = np.array(0, dtype=np.float32)
        constant_value = _extract_constant_tensor_value(node)
        if constant_value is not None:
            fill_value = constant_value.reshape(-1)[0]
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

    cloned_array = np.array(
        _evaluate_constant_value(input_name, producer_map, initializer_map),
        copy=True,
    )
    cloned_node = _build_constant_node(
        output_name=cloned_output_name,
        array=cloned_array,
        node_name=f"{cloned_output_name}_const",
    )
    return cloned_node, cloned_output_name


def _replace_node_input_name(node, old_name, new_name):
    for index, input_name in enumerate(node.input):
        if input_name == old_name:
            node.input[index] = new_name


def _replace_all_inputs(model, old_name, new_name):
    for node in model.graph.node:
        _replace_node_input_name(node, old_name, new_name)
    for output in model.graph.output:
        if output.name == old_name:
            output.name = new_name


def _stage_nodes(stage_map, anchor_node, *nodes):
    staged_nodes = [node for node in nodes if node is not None]
    if staged_nodes:
        stage_map.setdefault(id(anchor_node), []).extend(staged_nodes)


def _rebuild_graph_nodes(model, nodes_to_remove=None, extra_before=None, extra_after=None):
    nodes_to_remove = nodes_to_remove or set()
    extra_before = extra_before or {}
    extra_after = extra_after or {}
    if not nodes_to_remove and not extra_before and not extra_after:
        return

    kept_nodes = []
    for node in model.graph.node:
        kept_nodes.extend(extra_before.get(id(node), []))
        if id(node) not in nodes_to_remove:
            kept_nodes.append(node)
            kept_nodes.extend(extra_after.get(id(node), []))
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)


ORPHAN_CHAIN_REMOVABLE_NODE_TYPES = {
    "Constant",
    "Cast",
    "Identity",
    "DequantizeLinear",
}


def _remove_initializers_by_name(model, initializer_names):
    if not initializer_names:
        return

    kept_initializers = [
        initializer
        for initializer in model.graph.initializer
        if initializer.name not in initializer_names
    ]
    if len(kept_initializers) == len(model.graph.initializer):
        return

    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)


def _cleanup_orphaned_value_chains(
    model,
    detached_value_names,
    removable_node_types=ORPHAN_CHAIN_REMOVABLE_NODE_TYPES,
):
    if not detached_value_names:
        return

    graph_input_names = {item.name for item in model.graph.input}
    graph_output_names = {item.name for item in model.graph.output}
    pending_values = [value_name for value_name in detached_value_names if value_name]

    while pending_values:
        value_name = pending_values.pop()
        if not value_name or value_name in graph_output_names:
            continue

        producer_map, consumer_map = _build_onnx_graph_maps(model)
        if consumer_map.get(value_name):
            continue

        producer = producer_map.get(value_name)
        if producer is None:
            if value_name in graph_input_names:
                continue
            initializer_names = {initializer.name for initializer in model.graph.initializer}
            if value_name in initializer_names:
                _remove_initializers_by_name(model, {value_name})
            continue

        if producer.op_type not in removable_node_types:
            continue

        if any(
            output_name in graph_output_names or consumer_map.get(output_name)
            for output_name in producer.output
        ):
            continue

        upstream_inputs = [input_name for input_name in producer.input if input_name]
        _rebuild_graph_nodes(model, nodes_to_remove={id(producer)})
        pending_values.extend(upstream_inputs)


def _resolve_activation_quant_params(input_names, producer_map):
    return tuple(_evaluate_constant_value(name, producer_map, {}) for name in input_names)


def _validate_activation_quantize_split(node, downstream, producer_map):
    quant_scale, quant_zero = _resolve_activation_quant_params(node.input[1:3], producer_map)
    if quant_scale.size > 1:
        raise ValueError(
            f"{_get_node_display_name(node)} 的 activation QuantizeLinear 不应为 per-channel"
        )

    for consumer in downstream:
        dequant_scale, dequant_zero = _resolve_activation_quant_params(consumer.input[1:3], producer_map)
        if not (quant_scale == dequant_scale).all() or not (quant_zero == dequant_zero).all():
            raise ValueError(
                f"{_get_node_display_name(node)} 的多个 consumer 使用了不一致的量化参数，拒绝重写"
            )


def _build_split_activation_quantize_nodes(
    model,
    producer_map,
    initializer_map,
    source_node,
    split_index,
):
    cloned_output = f"{source_node.output[0]}_split_{split_index}"
    cloned_name = (
        f"{source_node.name}_split_{split_index}"
        if source_node.name
        else f"QuantizeLinear_split_{split_index}"
    )
    scale_node, scale_input = _clone_quant_param_input(
        model,
        producer_map,
        initializer_map,
        source_node.input[1],
        f"{cloned_name}_scale",
    )
    zero_node, zero_input = _clone_quant_param_input(
        model,
        producer_map,
        initializer_map,
        source_node.input[2],
        f"{cloned_name}_zero",
    )
    cloned_quantize = helper.make_node(
        "QuantizeLinear",
        [source_node.input[0], scale_input, zero_input],
        [cloned_output],
        name=cloned_name,
    )
    return cloned_output, scale_node, zero_node, cloned_quantize


def _split_multi_output_activation_quantize_nodes(model):
    producer_map, consumer_map = _build_onnx_graph_maps(model)
    initializer_map = _build_initializer_map(model)
    nodes_to_remove = set()
    extra_before = {}
    detached_value_names = []

    for node in model.graph.node:
        if node.op_type != "QuantizeLinear" or len(node.output) != 1:
            continue
        output_name = node.output[0]
        downstream = consumer_map.get(output_name, [])
        if len(downstream) <= 1:
            continue
        if any(consumer.op_type != "DequantizeLinear" for consumer in downstream):
            raise ValueError(
                f"{_get_node_display_name(node)} 存在非 DequantizeLinear consumer，无法安全拆分"
            )

        _validate_activation_quantize_split(node, downstream, producer_map)
        nodes_to_remove.add(id(node))
        detached_value_names.extend([node.input[1], node.input[2]])
        for split_index, consumer in enumerate(downstream):
            cloned_output, scale_node, zero_node, cloned_quantize = _build_split_activation_quantize_nodes(
                model,
                producer_map,
                initializer_map,
                node,
                split_index,
            )
            _stage_nodes(extra_before, consumer, scale_node, zero_node, cloned_quantize)
            _replace_node_input_name(consumer, output_name, cloned_output)

    _rebuild_graph_nodes(model, nodes_to_remove=nodes_to_remove, extra_before=extra_before)
    _cleanup_orphaned_value_chains(model, detached_value_names)


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
            _replace_node_input_name(consumer, cast_output, cast_input)
        nodes_to_remove.add(id(node))

    _rebuild_graph_nodes(model, nodes_to_remove=nodes_to_remove)


def _collapse_identity_nodes(model):
    _, consumer_map = _build_onnx_graph_maps(model)
    nodes_to_remove = set()

    for node in model.graph.node:
        if node.op_type != "Identity" or len(node.input) != 1 or len(node.output) != 1:
            continue
        identity_input = node.input[0]
        identity_output = node.output[0]
        for consumer in consumer_map.get(identity_output, []):
            _replace_node_input_name(consumer, identity_output, identity_input)
        for output in model.graph.output:
            if output.name == identity_output:
                output.name = identity_input
        nodes_to_remove.add(id(node))

    _rebuild_graph_nodes(model, nodes_to_remove=nodes_to_remove)


def _collect_quantized_param_targets(model, producer_map):
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
    return weight_targets, bias_targets


def _extract_dequantized_float_array(dq_node, producer_map, initializer_map):
    axis = _get_node_attr_int(dq_node, "axis")
    q_array = _evaluate_constant_value(dq_node.input[0], producer_map, initializer_map)
    scale_array = _evaluate_constant_value(dq_node.input[1], producer_map, initializer_map)
    zero_array = _evaluate_constant_value(dq_node.input[2], producer_map, initializer_map)
    float_array = _dequantize_array(q_array, scale_array, zero_array, axis).astype(
        np.float32,
        copy=False,
    )
    return axis, scale_array, float_array


def _rewrite_conv_per_channel_weight_path(
    model,
    initializer_map,
    dq_node,
    float_array,
    extra_before,
    extra_after,
):
    old_quantized_input_name = dq_node.input[0]
    q_node_name = f"{dq_node.name}_amct_quant"
    q_output_name = f"{dq_node.name}_amct_quant_out"
    dq_output_name = dq_node.output[0]
    float_name = f"{dq_node.name}_amct_float"
    transposed_array = np.transpose(float_array, (1, 0, 2, 3)).astype(np.float32, copy=False)
    _append_initializer(model, initializer_map, float_name, transposed_array)
    quantize = helper.make_node(
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
    _stage_nodes(extra_before, dq_node, quantize)
    _stage_nodes(extra_after, dq_node, transpose_back)
    return old_quantized_input_name


def _rewrite_standard_weight_path(model, initializer_map, dq_node, float_array, extra_before):
    old_quantized_input_name = dq_node.input[0]
    float_name = f"{dq_node.name}_amct_float"
    q_output_name = f"{dq_node.name}_amct_quant_out"
    _append_initializer(model, initializer_map, float_name, float_array)
    quantize = helper.make_node(
        "QuantizeLinear",
        [float_name, dq_node.input[1], dq_node.input[2]],
        [q_output_name],
        name=f"{dq_node.name}_amct_quant",
    )
    dq_node.input[0] = q_output_name
    _remove_node_attr(dq_node, "axis")
    _stage_nodes(extra_before, dq_node, quantize)
    return old_quantized_input_name


def _rewrite_bias_path(model, producer_map, initializer_map, dq_node):
    source_node = producer_map.get(dq_node.input[0])
    if source_node is not None and source_node.op_type == "QuantizeLinear":
        return None

    detached_value_names = list(dq_node.input)
    _, _, float_array = _extract_dequantized_float_array(dq_node, producer_map, initializer_map)
    bias_name = f"{dq_node.name}_amct_bias_float"
    _append_initializer(model, initializer_map, bias_name, float_array)
    _replace_all_inputs(model, dq_node.output[0], bias_name)
    return detached_value_names


def _rewrite_quantized_weight_and_bias_paths(model):
    producer_map, _ = _build_onnx_graph_maps(model)
    initializer_map = _build_initializer_map(model)
    nodes_to_remove = set()
    extra_before = {}
    extra_after = {}
    detached_value_names = []
    weight_targets, bias_targets = _collect_quantized_param_targets(model, producer_map)

    for consumer, dq_node in weight_targets:
        source_node = producer_map.get(dq_node.input[0])
        if source_node is not None and source_node.op_type == "QuantizeLinear":
            continue

        _, scale_array, float_array = _extract_dequantized_float_array(
            dq_node,
            producer_map,
            initializer_map,
        )
        if consumer.op_type == "Conv" and scale_array.size > 1:
            detached_value_names.append(
                _rewrite_conv_per_channel_weight_path(
                    model,
                    initializer_map,
                    dq_node,
                    float_array,
                    extra_before,
                    extra_after,
                )
            )
            continue
        detached_value_names.append(
            _rewrite_standard_weight_path(
                model,
                initializer_map,
                dq_node,
                float_array,
                extra_before,
            )
        )

    for _, dq_node in bias_targets:
        detached_bias_values = _rewrite_bias_path(model, producer_map, initializer_map, dq_node)
        if detached_bias_values is not None:
            nodes_to_remove.add(id(dq_node))
            detached_value_names.extend(detached_bias_values)

    _rebuild_graph_nodes(
        model,
        nodes_to_remove=nodes_to_remove,
        extra_before=extra_before,
        extra_after=extra_after,
    )
    _cleanup_orphaned_value_chains(model, detached_value_names)


def rewrite_cann_qat_onnx(onnx_path):
    model = onnx.load(onnx_path)
    _collapse_quantize_cast_pairs(model)
    _collapse_identity_nodes(model)
    _split_multi_output_activation_quantize_nodes(model)
    _rewrite_quantized_weight_and_bias_paths(model)
    _collapse_identity_nodes(model)
    onnx.save(model, onnx_path)


__all__ = [
    "_build_onnx_graph_maps",
    "_extract_constant_tensor_value",
    "_get_node_attr_int",
    "_get_node_display_name",
    "rewrite_cann_qat_onnx",
]
