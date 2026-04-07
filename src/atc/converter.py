#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ATC 编译与输入契约校验。"""

from __future__ import annotations

import os
import subprocess

from .output import create_output_directory
from .utils import (
    SUMMARY_VERSION,
    build_atc_subprocess_env,
    ensure_file_exists,
    ensure_summary_keys,
    ensure_summary_version,
    get_repo_root,
    load_json,
    parse_input_shape_argument,
    resolve_repo_path,
    to_repo_relative_path,
    validate_interface_dict,
)


EXPECTED_PRUNING_INPUT_MODEL_NAME = "model_fp16.onnx"
EXPECTED_AMCT_INPUT_MODEL_NAME = "deploy_model.onnx"
EXPECTED_PRUNING_SUMMARY_NAME = "onnx_summary.json"
EXPECTED_AMCT_SUMMARY_NAME = "amct_summary.json"
EXPECTED_PRUNING_BRANCH = "pruning_fp16"
EXPECTED_AMCT_STAGE = "amct"
ATC_OUTPUT_SUFFIX = ".om"
CHECK_REPORT_NAME = "check_result.json"
FUSION_RESULT_NAME = "fusion_result.json"
REQUIRED_PRUNING_SUMMARY_KEYS = {
    "summary_version",
    "branch",
    "model_name",
    "source_checkpoint_path",
    "source_architecture_signature",
    "onnx_path",
    "example_input_shape",
    "opset_version",
    "interface",
}
REQUIRED_AMCT_SUMMARY_KEYS = {
    "summary_version",
    "stage",
    "model_name",
    "source_onnx_path",
    "source_onnx_summary_path",
    "source_checkpoint_path",
    "source_architecture_signature",
    "source_branch",
    "source_interface",
    "deploy_model_path",
    "deploy_interface",
    "deploy_domains",
    "deploy_op_types",
    "fake_quant_model_path",
    "fake_quant_interface",
    "fake_quant_domains",
    "fake_quant_op_types",
    "example_input_shape",
    "opset_version",
}


class ATCCompilationError(RuntimeError):
    """ATC 阶段输入校验或编译错误。"""


def _validate_summary_architecture_signature(summary, summary_name):
    architecture_signature = summary.get("source_architecture_signature")
    if not isinstance(architecture_signature, dict):
        raise ATCCompilationError(f"{summary_name} 缺少 source_architecture_signature")
    signature_hash = architecture_signature.get("signature_hash")
    if not signature_hash:
        raise ATCCompilationError(f"{summary_name}.source_architecture_signature 缺少 signature_hash")
    return architecture_signature


def _load_summary_contract(
    *,
    onnx_model_path,
    repo_root,
    expected_input_model_name,
    expected_summary_name,
    required_summary_keys,
    expected_marker_key,
    expected_marker_value,
    path_key,
    validated_interface_key,
    additional_interface_keys=(),
):
    onnx_model_path = os.path.abspath(onnx_model_path)
    ensure_file_exists(onnx_model_path, "ATC 输入 ONNX")
    if os.path.basename(onnx_model_path) != expected_input_model_name:
        raise ATCCompilationError(
            f"当前分支只接受仓库导出的 {expected_input_model_name}"
        )

    summary_path = os.path.join(os.path.dirname(onnx_model_path), expected_summary_name)
    ensure_file_exists(summary_path, "ATC 输入摘要")
    summary = load_json(summary_path)
    ensure_summary_keys(
        summary,
        required_summary_keys,
        expected_summary_name,
        error_cls=ATCCompilationError,
    )
    ensure_summary_version(
        summary,
        expected_summary_name,
        expected_version=SUMMARY_VERSION,
        error_cls=ATCCompilationError,
    )

    if summary[expected_marker_key] != expected_marker_value:
        raise ATCCompilationError(
            f"{expected_summary_name}.{expected_marker_key} 必须为 {expected_marker_value}"
        )

    validate_interface_dict(
        summary[validated_interface_key],
        label=f"{expected_summary_name}.{validated_interface_key}",
        error_cls=ATCCompilationError,
    )
    for interface_key in additional_interface_keys:
        validate_interface_dict(
            summary[interface_key],
            label=f"{expected_summary_name}.{interface_key}",
            error_cls=ATCCompilationError,
        )

    architecture_signature = _validate_summary_architecture_signature(
        summary,
        expected_summary_name,
    )

    resolved_summary_onnx_path = resolve_repo_path(summary[path_key], repo_root=repo_root)
    if os.path.normpath(resolved_summary_onnx_path) != os.path.normpath(onnx_model_path):
        raise ATCCompilationError(
            f"{expected_summary_name} 中的 {path_key} 与当前输入文件不匹配"
        )

    return summary, summary_path, summary[validated_interface_key], architecture_signature


def _load_pruning_fp16_summary(onnx_model_path, repo_root=None):
    repo_root = get_repo_root() if repo_root is None else repo_root
    return _load_summary_contract(
        onnx_model_path=onnx_model_path,
        repo_root=repo_root,
        expected_input_model_name=EXPECTED_PRUNING_INPUT_MODEL_NAME,
        expected_summary_name=EXPECTED_PRUNING_SUMMARY_NAME,
        required_summary_keys=REQUIRED_PRUNING_SUMMARY_KEYS,
        expected_marker_key="branch",
        expected_marker_value=EXPECTED_PRUNING_BRANCH,
        path_key="onnx_path",
        validated_interface_key="interface",
    )


def _load_amct_summary(onnx_model_path, repo_root=None):
    repo_root = get_repo_root() if repo_root is None else repo_root
    return _load_summary_contract(
        onnx_model_path=onnx_model_path,
        repo_root=repo_root,
        expected_input_model_name=EXPECTED_AMCT_INPUT_MODEL_NAME,
        expected_summary_name=EXPECTED_AMCT_SUMMARY_NAME,
        required_summary_keys=REQUIRED_AMCT_SUMMARY_KEYS,
        expected_marker_key="stage",
        expected_marker_value=EXPECTED_AMCT_STAGE,
        path_key="deploy_model_path",
        validated_interface_key="deploy_interface",
        additional_interface_keys=("source_interface", "fake_quant_interface"),
    )


def load_branch_summary(branch, onnx_model_path, repo_root=None):
    repo_root = get_repo_root() if repo_root is None else repo_root
    branch_loaders = {
        "pruning_fp16": _load_pruning_fp16_summary,
        "amct_deploy": _load_amct_summary,
    }
    try:
        loader = branch_loaders[branch]
    except KeyError as exc:
        raise ValueError(f"不支持的 ATC 分支: {branch}") from exc
    return loader(onnx_model_path, repo_root=repo_root)


def _build_output_basename(source_model_path):
    return os.path.splitext(os.path.basename(source_model_path))[0]


def _collect_optional_artifacts(output_dir):
    artifacts = {}
    check_report_path = os.path.join(output_dir, CHECK_REPORT_NAME)
    fusion_result_path = os.path.join(output_dir, FUSION_RESULT_NAME)

    if os.path.exists(check_report_path):
        artifacts["check_report_path"] = check_report_path
    if os.path.exists(fusion_result_path):
        artifacts["fusion_result_path"] = fusion_result_path
    return artifacts


def _build_atc_input_shape_from_interface(interface):
    input_name = interface["input_name"]
    source_shape = interface["input_shape"]
    if len(source_shape) != 4:
        raise ATCCompilationError(
            f"当前仅支持四维 NCHW 输入，实际 interface.input_shape={source_shape}"
        )
    if any(not isinstance(dim, int) or dim <= 0 for dim in source_shape[1:]):
        raise ATCCompilationError(
            f"interface.input_shape 的非 batch 维度必须是正整数，实际为 {source_shape}"
        )

    resolved_shape = [1, *source_shape[1:]]
    input_shape = f"{input_name}:{','.join(str(dim) for dim in resolved_shape)}"
    return {
        "input_name": input_name,
        "input_shape": resolved_shape,
        "input_shape_arg": input_shape,
    }


def _resolve_effective_input_shape(interface, explicit_input_shape):
    resolved = _build_atc_input_shape_from_interface(interface)
    if explicit_input_shape is None:
        return resolved

    explicit_input_name, explicit_dims = parse_input_shape_argument(explicit_input_shape)
    if explicit_input_name != resolved["input_name"] or explicit_dims != resolved["input_shape"]:
        raise ATCCompilationError(
            "显式传入的 --input_shape 与根据上游摘要自动派生的结果不一致: "
            f"expected={resolved['input_shape_arg']}, actual={explicit_input_shape}"
        )
    return {
        **resolved,
        "input_shape_arg": explicit_input_shape,
    }


def _run_atc_compile(
    onnx_model_path,
    output_dir,
    output_basename,
    soc_version,
    input_shape,
    input_format,
):
    output_prefix = os.path.join(output_dir, output_basename)
    check_report_path = os.path.join(output_dir, CHECK_REPORT_NAME)
    command = [
        "atc",
        "--framework=5",
        f"--model={os.path.abspath(onnx_model_path)}",
        f"--output={output_prefix}",
        f"--soc_version={soc_version}",
        f"--input_format={input_format}",
        f"--input_shape={input_shape}",
        f"--check_report={check_report_path}",
    ]

    completed = subprocess.run(
        command,
        cwd=output_dir,
        env=build_atc_subprocess_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="")

    if completed.returncode != 0:
        raise ATCCompilationError(
            f"ATC 编译失败，退出码 {completed.returncode}: {os.path.abspath(onnx_model_path)}"
        )

    om_path = f"{output_prefix}{ATC_OUTPUT_SUFFIX}"
    ensure_file_exists(om_path, "ATC 输出 OM")
    return om_path


def _build_summary(
    branch,
    input_summary,
    source_architecture_signature,
    validated_interface,
    resolved_input_shape,
    onnx_model_path,
    source_summary_path,
    om_path,
    input_shape,
    input_format,
    soc_version,
    optional_artifacts,
    repo_root=None,
):
    repo_root = get_repo_root() if repo_root is None else repo_root
    source_checkpoint_path = input_summary.get("source_checkpoint_path")
    source_onnx_path = input_summary.get("source_onnx_path")
    summary = {
        "stage": "atc",
        "branch": branch,
        "model_name": input_summary["model_name"],
        "source_model_path": to_repo_relative_path(onnx_model_path, repo_root=repo_root),
        "source_summary_path": to_repo_relative_path(source_summary_path, repo_root=repo_root),
        "source_checkpoint_path": to_repo_relative_path(source_checkpoint_path, repo_root=repo_root),
        "source_architecture_signature": source_architecture_signature,
        "validated_source_summary_version": input_summary["summary_version"],
        "source_interface": validated_interface,
        "resolved_input_name": resolved_input_shape["input_name"],
        "resolved_input_shape": resolved_input_shape["input_shape"],
        "input_shape": input_shape,
        "input_format": input_format,
        "soc_version": soc_version,
        "example_input_shape": input_summary.get("example_input_shape"),
        "opset_version": input_summary.get("opset_version"),
        "om_path": to_repo_relative_path(om_path, repo_root=repo_root),
    }
    if source_onnx_path is not None:
        summary["source_onnx_path"] = to_repo_relative_path(source_onnx_path, repo_root=repo_root)
    if "source_branch" in input_summary:
        summary["source_branch"] = input_summary["source_branch"]
    if "stage" in input_summary and branch == "amct_deploy":
        summary["source_stage"] = input_summary["stage"]

    for key, path in optional_artifacts.items():
        summary[key] = to_repo_relative_path(path, repo_root=repo_root)
    return summary


def build_atc_artifacts(
    branch,
    onnx_model_path,
    soc_version,
    input_shape,
    input_format,
    repo_root=None,
):
    repo_root = get_repo_root() if repo_root is None else repo_root
    input_summary, source_summary_path, validated_interface, source_architecture_signature = load_branch_summary(
        branch,
        onnx_model_path,
        repo_root=repo_root,
    )
    resolved_input_shape = _resolve_effective_input_shape(
        validated_interface,
        input_shape,
    )
    effective_input_shape = resolved_input_shape["input_shape_arg"]

    if branch == "pruning_fp16":
        source_rel_path = input_summary["onnx_path"]
    else:
        source_rel_path = input_summary["deploy_model_path"]

    output_dir = create_output_directory(
        branch=branch,
        model_name=input_summary["model_name"],
        source_rel_path=source_rel_path,
    )
    output_dir_abs = os.path.abspath(output_dir)
    output_basename = _build_output_basename(onnx_model_path)
    om_path = _run_atc_compile(
        onnx_model_path=onnx_model_path,
        output_dir=output_dir_abs,
        output_basename=output_basename,
        soc_version=soc_version,
        input_shape=effective_input_shape,
        input_format=input_format,
    )
    optional_artifacts = _collect_optional_artifacts(output_dir_abs)
    summary = _build_summary(
        branch=branch,
        input_summary=input_summary,
        source_architecture_signature=source_architecture_signature,
        validated_interface=validated_interface,
        resolved_input_shape=resolved_input_shape,
        onnx_model_path=os.path.abspath(onnx_model_path),
        source_summary_path=os.path.abspath(source_summary_path),
        om_path=om_path,
        input_shape=effective_input_shape,
        input_format=input_format,
        soc_version=soc_version,
        optional_artifacts=optional_artifacts,
        repo_root=repo_root,
    )
    return output_dir_abs, summary
