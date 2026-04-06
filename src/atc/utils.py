#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ATC 阶段通用工具。"""

import json
import os


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


def build_atc_subprocess_env():
    env = os.environ.copy()
    virtual_env = env.pop("VIRTUAL_ENV", None)
    env.pop("UV", None)
    env.pop("PYTHONHOME", None)

    uv_keys = [key for key in env if key.startswith("UV_")]
    for key in uv_keys:
        env.pop(key, None)

    if virtual_env is not None:
        virtual_bin = os.path.join(virtual_env, "bin")
        path_entries = env.get("PATH", "").split(os.pathsep)
        path_entries = [
            entry
            for entry in path_entries
            if os.path.normpath(entry) != os.path.normpath(virtual_bin)
        ]
        env["PATH"] = os.pathsep.join(path_entries)

    return env


__all__ = [
    "SUMMARY_VERSION",
    "build_atc_subprocess_env",
    "ensure_file_exists",
    "ensure_summary_keys",
    "ensure_summary_version",
    "get_repo_root",
    "load_json",
    "parse_input_shape_argument",
    "resolve_repo_path",
    "to_repo_relative_path",
    "validate_input_shape_argument_matches_interface",
    "validate_interface_dict",
]
