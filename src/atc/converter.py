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
    resolve_repo_path,
    to_repo_relative_path,
    validate_input_shape_argument_matches_interface,
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


def _load_pruning_fp16_summary(onnx_model_path, repo_root=None):
    repo_root = get_repo_root() if repo_root is None else repo_root
    onnx_model_path = os.path.abspath(onnx_model_path)
    ensure_file_exists(onnx_model_path, "ATC 输入 ONNX")
    if os.path.basename(onnx_model_path) != EXPECTED_PRUNING_INPUT_MODEL_NAME:
        raise ATCCompilationError(
            "pruning_fp16 分支当前只接受仓库导出的 model_fp16.onnx"
        )

    summary_path = os.path.join(os.path.dirname(onnx_model_path), EXPECTED_PRUNING_SUMMARY_NAME)
    ensure_file_exists(summary_path, "ONNX 摘要")
    summary = load_json(summary_path)
    ensure_summary_keys(
        summary,
        REQUIRED_PRUNING_SUMMARY_KEYS,
        EXPECTED_PRUNING_SUMMARY_NAME,
        error_cls=ATCCompilationError,
    )
    ensure_summary_version(
        summary,
        EXPECTED_PRUNING_SUMMARY_NAME,
        expected_version=SUMMARY_VERSION,
        error_cls=ATCCompilationError,
    )
    validate_interface_dict(
        summary["interface"],
        label=f"{EXPECTED_PRUNING_SUMMARY_NAME}.interface",
        error_cls=ATCCompilationError,
    )

    if summary["branch"] != EXPECTED_PRUNING_BRANCH:
        raise ATCCompilationError("onnx_summary.json.branch 必须为 pruning_fp16")

    resolved_summary_onnx_path = resolve_repo_path(summary["onnx_path"], repo_root=repo_root)
    if os.path.normpath(resolved_summary_onnx_path) != os.path.normpath(onnx_model_path):
        raise ATCCompilationError("onnx_summary.json 中的 onnx_path 与当前输入文件不匹配")

    return summary, summary_path, summary["interface"]


def _load_amct_summary(onnx_model_path, repo_root=None):
    repo_root = get_repo_root() if repo_root is None else repo_root
    onnx_model_path = os.path.abspath(onnx_model_path)
    ensure_file_exists(onnx_model_path, "ATC 输入 ONNX")
    if os.path.basename(onnx_model_path) != EXPECTED_AMCT_INPUT_MODEL_NAME:
        raise ATCCompilationError(
            "amct_deploy 分支当前只接受仓库导出的 deploy_model.onnx"
        )

    summary_path = os.path.join(os.path.dirname(onnx_model_path), EXPECTED_AMCT_SUMMARY_NAME)
    ensure_file_exists(summary_path, "AMCT 摘要")
    summary = load_json(summary_path)
    ensure_summary_keys(
        summary,
        REQUIRED_AMCT_SUMMARY_KEYS,
        EXPECTED_AMCT_SUMMARY_NAME,
        error_cls=ATCCompilationError,
    )
    ensure_summary_version(
        summary,
        EXPECTED_AMCT_SUMMARY_NAME,
        expected_version=SUMMARY_VERSION,
        error_cls=ATCCompilationError,
    )
    validate_interface_dict(
        summary["source_interface"],
        label=f"{EXPECTED_AMCT_SUMMARY_NAME}.source_interface",
        error_cls=ATCCompilationError,
    )
    validate_interface_dict(
        summary["deploy_interface"],
        label=f"{EXPECTED_AMCT_SUMMARY_NAME}.deploy_interface",
        error_cls=ATCCompilationError,
    )
    validate_interface_dict(
        summary["fake_quant_interface"],
        label=f"{EXPECTED_AMCT_SUMMARY_NAME}.fake_quant_interface",
        error_cls=ATCCompilationError,
    )

    if summary["stage"] != EXPECTED_AMCT_STAGE:
        raise ATCCompilationError("amct_summary.json.stage 必须为 amct")

    resolved_summary_onnx_path = resolve_repo_path(
        summary["deploy_model_path"],
        repo_root=repo_root,
    )
    if os.path.normpath(resolved_summary_onnx_path) != os.path.normpath(onnx_model_path):
        raise ATCCompilationError("amct_summary.json 中的 deploy_model_path 与当前输入文件不匹配")

    return summary, summary_path, summary["deploy_interface"]


def load_branch_summary(branch, onnx_model_path, repo_root=None):
    repo_root = get_repo_root() if repo_root is None else repo_root
    if branch == "pruning_fp16":
        return _load_pruning_fp16_summary(onnx_model_path, repo_root=repo_root)
    if branch == "amct_deploy":
        return _load_amct_summary(onnx_model_path, repo_root=repo_root)
    raise ValueError(f"不支持的 ATC 分支: {branch}")


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
    validated_interface,
    validated_input_shape,
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
        "validated_source_summary_version": input_summary["summary_version"],
        "source_interface": validated_interface,
        "resolved_input_name": validated_input_shape["input_name"],
        "resolved_input_shape": validated_input_shape["input_shape"],
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
    input_summary, source_summary_path, validated_interface = load_branch_summary(
        branch,
        onnx_model_path,
        repo_root=repo_root,
    )
    validated_input_shape = validate_input_shape_argument_matches_interface(
        input_shape,
        validated_interface,
        label="ATC input_shape",
        error_cls=ATCCompilationError,
    )

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
        input_shape=input_shape,
        input_format=input_format,
    )
    optional_artifacts = _collect_optional_artifacts(output_dir_abs)
    summary = _build_summary(
        branch=branch,
        input_summary=input_summary,
        validated_interface=validated_interface,
        validated_input_shape=validated_input_shape,
        onnx_model_path=os.path.abspath(onnx_model_path),
        source_summary_path=os.path.abspath(source_summary_path),
        om_path=om_path,
        input_shape=input_shape,
        input_format=input_format,
        soc_version=soc_version,
        optional_artifacts=optional_artifacts,
        repo_root=repo_root,
    )
    return output_dir_abs, summary
