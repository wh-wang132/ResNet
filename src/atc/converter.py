#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ATC 编译与输入契约校验。"""

from __future__ import annotations

import json
import os
import subprocess

from atc.output import create_output_directory


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
    "branch",
    "model_name",
    "source_checkpoint_path",
    "onnx_path",
    "example_input_shape",
    "opset_version",
}
REQUIRED_AMCT_SUMMARY_KEYS = {
    "stage",
    "model_name",
    "source_onnx_path",
    "source_onnx_summary_path",
    "source_checkpoint_path",
    "deploy_model_path",
    "example_input_shape",
    "opset_version",
}


class ATCCompilationError(RuntimeError):
    """ATC 阶段输入校验或编译错误。"""


def _get_repo_root():
    repo_root = os.environ.get("REPO_ROOT")
    if not repo_root:
        raise ATCCompilationError("REPO_ROOT 未设置：请先让 direnv 自动激活 .envrc")
    return os.path.abspath(repo_root)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _ensure_file_exists(path, label):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到{label}: {path}")


def _resolve_repo_path(path, repo_root=None):
    repo_root = _get_repo_root() if repo_root is None else repo_root
    normalized_path = os.path.normpath(path)
    if os.path.isabs(normalized_path):
        return normalized_path
    return os.path.abspath(os.path.join(repo_root, normalized_path))


def _to_repo_relative_path(path, repo_root=None):
    repo_root = _get_repo_root() if repo_root is None else repo_root
    if path is None:
        return None

    normalized_path = os.path.normpath(path)
    if not os.path.isabs(normalized_path):
        return normalized_path
    return os.path.relpath(normalized_path, repo_root)


def _ensure_summary_keys(summary, required_keys, summary_name):
    missing_keys = sorted(required_keys - set(summary.keys()))
    if missing_keys:
        raise ATCCompilationError(
            f"{summary_name} 缺少关键字段: {', '.join(missing_keys)}"
        )


def _load_pruning_fp16_summary(onnx_model_path, repo_root=None):
    repo_root = _get_repo_root() if repo_root is None else repo_root
    onnx_model_path = os.path.abspath(onnx_model_path)
    _ensure_file_exists(onnx_model_path, "ATC 输入 ONNX")
    if os.path.basename(onnx_model_path) != EXPECTED_PRUNING_INPUT_MODEL_NAME:
        raise ATCCompilationError(
            "pruning_fp16 分支当前只接受仓库导出的 model_fp16.onnx"
        )

    summary_path = os.path.join(
        os.path.dirname(onnx_model_path),
        EXPECTED_PRUNING_SUMMARY_NAME,
    )
    _ensure_file_exists(summary_path, "ONNX 摘要")
    summary = _load_json(summary_path)
    _ensure_summary_keys(summary, REQUIRED_PRUNING_SUMMARY_KEYS, EXPECTED_PRUNING_SUMMARY_NAME)

    if summary["branch"] != EXPECTED_PRUNING_BRANCH:
        raise ATCCompilationError("onnx_summary.json.branch 必须为 pruning_fp16")

    resolved_summary_onnx_path = _resolve_repo_path(summary["onnx_path"], repo_root=repo_root)
    if os.path.normpath(resolved_summary_onnx_path) != os.path.normpath(onnx_model_path):
        raise ATCCompilationError("onnx_summary.json 中的 onnx_path 与当前输入文件不匹配")

    return summary, summary_path


def _load_amct_summary(onnx_model_path, repo_root=None):
    repo_root = _get_repo_root() if repo_root is None else repo_root
    onnx_model_path = os.path.abspath(onnx_model_path)
    _ensure_file_exists(onnx_model_path, "ATC 输入 ONNX")
    if os.path.basename(onnx_model_path) != EXPECTED_AMCT_INPUT_MODEL_NAME:
        raise ATCCompilationError(
            "amct_deploy 分支当前只接受仓库导出的 deploy_model.onnx"
        )

    summary_path = os.path.join(
        os.path.dirname(onnx_model_path),
        EXPECTED_AMCT_SUMMARY_NAME,
    )
    _ensure_file_exists(summary_path, "AMCT 摘要")
    summary = _load_json(summary_path)
    _ensure_summary_keys(summary, REQUIRED_AMCT_SUMMARY_KEYS, EXPECTED_AMCT_SUMMARY_NAME)

    if summary["stage"] != EXPECTED_AMCT_STAGE:
        raise ATCCompilationError("amct_summary.json.stage 必须为 amct")

    resolved_summary_onnx_path = _resolve_repo_path(
        summary["deploy_model_path"],
        repo_root=repo_root,
    )
    if os.path.normpath(resolved_summary_onnx_path) != os.path.normpath(onnx_model_path):
        raise ATCCompilationError("amct_summary.json 中的 deploy_model_path 与当前输入文件不匹配")

    return summary, summary_path


def load_branch_summary(branch, onnx_model_path, repo_root=None):
    repo_root = _get_repo_root() if repo_root is None else repo_root
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


def _build_atc_subprocess_env():
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
        env=_build_atc_subprocess_env(),
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
    _ensure_file_exists(om_path, "ATC 输出 OM")
    return om_path


def _build_summary(
    branch,
    input_summary,
    onnx_model_path,
    source_summary_path,
    om_path,
    input_shape,
    input_format,
    soc_version,
    optional_artifacts,
    repo_root=None,
):
    repo_root = _get_repo_root() if repo_root is None else repo_root
    source_checkpoint_path = input_summary.get("source_checkpoint_path")
    source_onnx_path = input_summary.get("source_onnx_path")
    summary = {
        "stage": "atc",
        "branch": branch,
        "model_name": input_summary["model_name"],
        "source_model_path": _to_repo_relative_path(onnx_model_path, repo_root=repo_root),
        "source_summary_path": _to_repo_relative_path(source_summary_path, repo_root=repo_root),
        "source_checkpoint_path": _to_repo_relative_path(source_checkpoint_path, repo_root=repo_root),
        "input_shape": input_shape,
        "input_format": input_format,
        "soc_version": soc_version,
        "example_input_shape": input_summary.get("example_input_shape"),
        "opset_version": input_summary.get("opset_version"),
        "om_path": _to_repo_relative_path(om_path, repo_root=repo_root),
    }
    if source_onnx_path is not None:
        summary["source_onnx_path"] = _to_repo_relative_path(source_onnx_path, repo_root=repo_root)
    if "source_branch" in input_summary:
        summary["source_branch"] = input_summary["source_branch"]
    if "stage" in input_summary and branch == "amct_deploy":
        summary["source_stage"] = input_summary["stage"]

    for key, path in optional_artifacts.items():
        summary[key] = _to_repo_relative_path(path, repo_root=repo_root)
    return summary


def build_atc_artifacts(
    branch,
    onnx_model_path,
    soc_version,
    input_shape,
    input_format,
    repo_root=None,
):
    repo_root = _get_repo_root() if repo_root is None else repo_root
    input_summary, source_summary_path = load_branch_summary(
        branch,
        onnx_model_path,
        repo_root=repo_root,
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
