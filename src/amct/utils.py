#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AMCT/ATC 阶段共享工具。"""

from __future__ import annotations

import json
import os

import onnx


SUMMARY_VERSION = 2

INTERFACE_REQUIRED_KEYS = {
    "input_name",
    "output_name",
    "input_elem_type",
    "output_elem_type",
    "input_shape",
    "output_shape",
    "dynamic_batch",
}

AMCT_DEPLOY_ALLOWED_DOMAINS = ("", "ai.onnx")
AMCT_FAKE_QUANT_ALLOWED_DOMAINS = ("", "ai.onnx", "amct.customop")
AMCT_ALLOWED_OP_TYPES = (
    "Add",
    "AscendAntiQuant",
    "AscendDequant",
    "AscendQuant",
    "Constant",
    "ConstantOfShape",
    "Conv",
    "Flatten",
    "Gemm",
    "GlobalAveragePool",
    "MaxPool",
    "Relu",
)


def get_repo_root():
    repo_root = os.environ.get("REPO_ROOT")
    if not repo_root:
        raise RuntimeError("REPO_ROOT 未设置：请先让 direnv 自动激活 .envrc")
    return os.path.abspath(repo_root)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def ensure_file_exists(path, label):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到{label}: {path}")


def resolve_repo_path(path, repo_root=None):
    repo_root = get_repo_root() if repo_root is None else repo_root
    normalized_path = os.path.normpath(path)
    if os.path.isabs(normalized_path):
        return normalized_path
    return os.path.abspath(os.path.join(repo_root, normalized_path))


def to_repo_relative_path(path, repo_root=None):
    repo_root = get_repo_root() if repo_root is None else repo_root
    if path is None:
        return None

    normalized_path = os.path.normpath(path)
    if not os.path.isabs(normalized_path):
        return normalized_path
    return os.path.relpath(normalized_path, repo_root)


def ensure_summary_keys(summary, required_keys, summary_name, error_cls=RuntimeError):
    missing_keys = sorted(required_keys - set(summary.keys()))
    if missing_keys:
        raise error_cls(f"{summary_name} 缺少关键字段: {', '.join(missing_keys)}")


def ensure_summary_version(
    summary,
    summary_name,
    expected_version=SUMMARY_VERSION,
    error_cls=RuntimeError,
):
    actual_version = summary.get("summary_version")
    if actual_version != expected_version:
        raise error_cls(
            f"{summary_name}.summary_version 必须为 {expected_version}，当前为 {actual_version}"
        )


def _extract_shape(shape_proto):
    shape = []
    for dim in shape_proto.dim:
        if dim.HasField("dim_value"):
            shape.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            shape.append(dim.dim_param)
        else:
            shape.append(None)
    return shape


def _extract_value_info_contract(value_info):
    tensor_type = value_info.type.tensor_type
    return {
        "name": value_info.name,
        "elem_type": int(tensor_type.elem_type),
        "shape": _extract_shape(tensor_type.shape),
    }


def extract_onnx_interface_from_model(model):
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    inputs = [item for item in model.graph.input if item.name not in initializer_names]
    outputs = list(model.graph.output)
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(
            f"当前契约仅支持单输入单输出模型，实际为 {len(inputs)} 输入 / {len(outputs)} 输出"
        )

    input_contract = _extract_value_info_contract(inputs[0])
    output_contract = _extract_value_info_contract(outputs[0])
    dynamic_batch = not isinstance(input_contract["shape"][0], int) or not isinstance(
        output_contract["shape"][0],
        int,
    )
    return {
        "input_name": input_contract["name"],
        "output_name": output_contract["name"],
        "input_elem_type": input_contract["elem_type"],
        "output_elem_type": output_contract["elem_type"],
        "input_shape": input_contract["shape"],
        "output_shape": output_contract["shape"],
        "dynamic_batch": bool(dynamic_batch),
    }


def extract_onnx_contract(onnx_path):
    model = onnx.load(onnx_path)
    interface = extract_onnx_interface_from_model(model)
    domains = sorted(set(node.domain or "ai.onnx" for node in model.graph.node))
    op_types = sorted(set(node.op_type for node in model.graph.node))
    return {
        "interface": interface,
        "domains": domains,
        "op_types": op_types,
    }


def validate_interface_dict(interface, label="interface", error_cls=ValueError):
    if not isinstance(interface, dict):
        raise error_cls(f"{label} 必须是对象")

    missing_keys = sorted(INTERFACE_REQUIRED_KEYS - set(interface.keys()))
    if missing_keys:
        raise error_cls(f"{label} 缺少关键字段: {', '.join(missing_keys)}")

    if not interface["input_name"] or not interface["output_name"]:
        raise error_cls(f"{label} 的输入输出名称不能为空")
    if not isinstance(interface["input_elem_type"], int) or interface["input_elem_type"] <= 0:
        raise error_cls(f"{label}.input_elem_type 非法")
    if not isinstance(interface["output_elem_type"], int) or interface["output_elem_type"] <= 0:
        raise error_cls(f"{label}.output_elem_type 非法")
    if not isinstance(interface["input_shape"], list) or len(interface["input_shape"]) != 4:
        raise error_cls(f"{label}.input_shape 必须是长度为 4 的 NCHW 形状")
    if not isinstance(interface["output_shape"], list) or len(interface["output_shape"]) != 2:
        raise error_cls(f"{label}.output_shape 必须是长度为 2 的分类输出形状")
    if not isinstance(interface["dynamic_batch"], bool):
        raise error_cls(f"{label}.dynamic_batch 必须是布尔值")


def validate_interface_equal(actual, expected, label="interface", error_cls=ValueError):
    validate_interface_dict(actual, label=f"{label}(actual)", error_cls=error_cls)
    validate_interface_dict(expected, label=f"{label}(expected)", error_cls=error_cls)
    if actual != expected:
        raise error_cls(f"{label} 与期望接口不一致: actual={actual}, expected={expected}")


def validate_allowed_domains_and_ops(
    domains,
    op_types,
    allowed_domains,
    allowed_op_types,
    label,
    error_cls=ValueError,
):
    unexpected_domains = sorted(set(domains) - set(allowed_domains))
    if unexpected_domains:
        raise error_cls(f"{label} 包含未允许的 domain: {unexpected_domains}")

    unexpected_ops = sorted(set(op_types) - set(allowed_op_types))
    if unexpected_ops:
        raise error_cls(f"{label} 包含未允许的 op types: {unexpected_ops}")


def validate_onnx_contract(
    onnx_path,
    *,
    expected_interface=None,
    allowed_domains=None,
    allowed_op_types=None,
    label="ONNX",
    error_cls=ValueError,
):
    model = onnx.load(onnx_path)
    checker_error = None
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as exc:
        checker_error = str(exc)

    contract = {
        "interface": extract_onnx_interface_from_model(model),
        "domains": sorted(set(node.domain or "ai.onnx" for node in model.graph.node)),
        "op_types": sorted(set(node.op_type for node in model.graph.node)),
        "checker_error": checker_error,
    }
    validate_interface_dict(contract["interface"], label=f"{label}.interface", error_cls=error_cls)

    if expected_interface is not None:
        validate_interface_equal(
            contract["interface"],
            expected_interface,
            label=f"{label}.interface",
            error_cls=error_cls,
        )

    allow_custom_contract = allowed_domains is not None or allowed_op_types is not None
    if allowed_domains is not None and allowed_op_types is not None:
        validate_allowed_domains_and_ops(
            contract["domains"],
            contract["op_types"],
            allowed_domains=allowed_domains,
            allowed_op_types=allowed_op_types,
            label=label,
            error_cls=error_cls,
        )

    if checker_error is not None and not allow_custom_contract:
        raise error_cls(f"{label} 未通过 onnx.checker 校验: {checker_error}")

    return contract


def parse_input_shape_argument(input_shape):
    if ":" not in input_shape:
        raise ValueError('input_shape 必须形如 "input:1,1,543,512"')
    input_name, dims_text = input_shape.split(":", 1)
    if not input_name:
        raise ValueError("input_shape 中的输入名不能为空")
    dims = [int(part) for part in dims_text.split(",") if part]
    if len(dims) != 4:
        raise ValueError("当前仅支持四维 NCHW input_shape")
    if any(dim <= 0 for dim in dims):
        raise ValueError("input_shape 的每一维都必须大于 0")
    return input_name, dims


def validate_input_shape_argument_matches_interface(
    input_shape,
    interface,
    *,
    label="ATC input_shape",
    error_cls=ValueError,
):
    validate_interface_dict(interface, label=f"{label}.interface", error_cls=error_cls)
    input_name, dims = parse_input_shape_argument(input_shape)
    if input_name != interface["input_name"]:
        raise error_cls(
            f'{label} 中的输入名必须为 "{interface["input_name"]}"，当前为 "{input_name}"'
        )

    expected_shape = interface["input_shape"]
    if interface["dynamic_batch"]:
        expected_suffix = expected_shape[1:]
        actual_suffix = dims[1:]
        if actual_suffix != expected_suffix:
            raise error_cls(
                f"{label} 的非 batch 维度必须为 {expected_suffix}，当前为 {actual_suffix}"
            )
    else:
        if dims != expected_shape:
            raise error_cls(f"{label} 必须为 {expected_shape}，当前为 {dims}")

    return {
        "input_name": input_name,
        "input_shape": dims,
    }


__all__ = [
    "AMCT_ALLOWED_OP_TYPES",
    "AMCT_DEPLOY_ALLOWED_DOMAINS",
    "AMCT_FAKE_QUANT_ALLOWED_DOMAINS",
    "INTERFACE_REQUIRED_KEYS",
    "SUMMARY_VERSION",
    "ensure_file_exists",
    "ensure_summary_keys",
    "ensure_summary_version",
    "extract_onnx_contract",
    "extract_onnx_interface_from_model",
    "get_repo_root",
    "load_json",
    "parse_input_shape_argument",
    "resolve_repo_path",
    "to_repo_relative_path",
    "validate_allowed_domains_and_ops",
    "validate_input_shape_argument_matches_interface",
    "validate_interface_dict",
    "validate_interface_equal",
    "validate_onnx_contract",
]
