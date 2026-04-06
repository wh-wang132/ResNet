#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AMCT 转换与产物校验。"""

import importlib
import json
import os

import onnx

from amct.output import create_output_directory
from qat.utils import REPO_ROOT


EXPECTED_INPUT_MODEL_NAME = "model_quant.onnx"
EXPECTED_ONNX_SUMMARY_NAME = "onnx_summary.json"
EXPECTED_ONNX_BRANCH = "qat_convert"
EXPECTED_DEPLOY_MODEL_NAME = "deploy_model.onnx"
EXPECTED_FAKE_QUANT_MODEL_NAME = "fake_quant_model.onnx"
EXPECTED_RECORD_FILE_NAME = "scale_offset_record.txt"
REQUIRED_ONNX_SUMMARY_KEYS = {
    "branch",
    "model_name",
    "source_checkpoint_path",
    "onnx_path",
    "example_input_shape",
    "opset_version",
}


class AMCTConversionError(RuntimeError):
    """AMCT 阶段输入校验或转换错误。"""


def _load_json(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _resolve_repo_path(path, repo_root=REPO_ROOT):
    normalized_path = os.path.normpath(path)
    if os.path.isabs(normalized_path):
        return normalized_path
    return os.path.abspath(os.path.join(repo_root, normalized_path))


def _to_repo_relative_path(path, repo_root=REPO_ROOT):
    if path is None:
        return None

    normalized_path = os.path.normpath(path)
    if not os.path.isabs(normalized_path):
        return normalized_path
    return os.path.relpath(normalized_path, repo_root)


def _ensure_file_exists(path, label):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到{label}: {path}")


def _ensure_summary_keys(summary):
    missing_keys = sorted(REQUIRED_ONNX_SUMMARY_KEYS - set(summary.keys()))
    if missing_keys:
        raise AMCTConversionError(
            f"onnx_summary.json 缺少关键字段: {', '.join(missing_keys)}"
        )


def _validate_input_onnx_model(onnx_model_path):
    if os.path.splitext(onnx_model_path)[1] != ".onnx":
        raise AMCTConversionError("AMCT 输入文件必须是 .onnx")
    if os.path.basename(onnx_model_path) != EXPECTED_INPUT_MODEL_NAME:
        raise AMCTConversionError("AMCT 当前只接受仓库 qat_convert 产出的 model_quant.onnx")


def load_qat_onnx_summary(onnx_model_path, repo_root=REPO_ROOT):
    onnx_model_path = os.path.abspath(onnx_model_path)
    _ensure_file_exists(onnx_model_path, "AMCT 输入 ONNX")
    _validate_input_onnx_model(onnx_model_path)

    summary_path = os.path.join(os.path.dirname(onnx_model_path), EXPECTED_ONNX_SUMMARY_NAME)
    _ensure_file_exists(summary_path, "ONNX 摘要")
    onnx_summary = _load_json(summary_path)
    _ensure_summary_keys(onnx_summary)

    if onnx_summary["branch"] != EXPECTED_ONNX_BRANCH:
        raise AMCTConversionError("AMCT 当前只接受 branch=qat_convert 的 ONNX 摘要")

    resolved_summary_onnx_path = _resolve_repo_path(onnx_summary["onnx_path"], repo_root=repo_root)
    if os.path.normpath(resolved_summary_onnx_path) != os.path.normpath(onnx_model_path):
        raise AMCTConversionError("onnx_summary.json 中的 onnx_path 与当前输入文件不匹配")

    return onnx_summary, summary_path


def _import_amct_module():
    try:
        return importlib.import_module("amct_onnx")
    except Exception as exc:  # pragma: no cover - 真实环境依赖错误
        raise AMCTConversionError(f"无法导入 amct_onnx: {exc}") from exc


def _validate_onnx_model(path):
    model = onnx.load(path)
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError:
        has_custom_ascend_ops = any(node.op_type.startswith("Ascend") for node in model.graph.node)
        has_non_standard_domains = any(node.domain not in ("", "ai.onnx") for node in model.graph.node)
        if not has_custom_ascend_ops and not has_non_standard_domains:
            raise


def _collect_external_data_files(folder_path):
    external_files = []
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith(".external"):
            external_files.append(os.path.join(folder_path, file_name))
    return external_files


def _build_summary(
    onnx_summary,
    onnx_model_path,
    onnx_summary_path,
    output_dir,
    external_files,
    repo_root=REPO_ROOT,
):
    deploy_model_path = os.path.join(output_dir, EXPECTED_DEPLOY_MODEL_NAME)
    fake_quant_model_path = os.path.join(output_dir, EXPECTED_FAKE_QUANT_MODEL_NAME)
    record_file_path = os.path.join(output_dir, EXPECTED_RECORD_FILE_NAME)
    amct_log_path = os.path.join(repo_root, "amct_log", "amct_onnx.log")
    summary = {
        "stage": "amct",
        "model_name": onnx_summary["model_name"],
        "source_onnx_path": _to_repo_relative_path(onnx_model_path, repo_root=repo_root),
        "source_onnx_summary_path": _to_repo_relative_path(onnx_summary_path, repo_root=repo_root),
        "source_checkpoint_path": _to_repo_relative_path(
            onnx_summary["source_checkpoint_path"],
            repo_root=repo_root,
        ),
        "source_branch": EXPECTED_ONNX_BRANCH,
        "example_input_shape": onnx_summary["example_input_shape"],
        "opset_version": onnx_summary["opset_version"],
        "deploy_model_path": _to_repo_relative_path(deploy_model_path, repo_root=repo_root),
        "fake_quant_model_path": _to_repo_relative_path(fake_quant_model_path, repo_root=repo_root),
        "record_file_path": _to_repo_relative_path(record_file_path, repo_root=repo_root),
        "amct_log_path": _to_repo_relative_path(amct_log_path, repo_root=repo_root)
        if os.path.exists(amct_log_path)
        else None,
    }
    if external_files:
        summary["external_data_files"] = [
            _to_repo_relative_path(path, repo_root=repo_root) for path in external_files
        ]
    return summary


def build_amct_artifacts(onnx_model_path, repo_root=REPO_ROOT):
    onnx_summary, onnx_summary_path = load_qat_onnx_summary(onnx_model_path, repo_root=repo_root)
    output_dir = create_output_directory(onnx_summary)
    output_dir_abs = os.path.abspath(output_dir)
    save_path = os.path.join(output_dir_abs, "")
    record_file = os.path.join(output_dir_abs, EXPECTED_RECORD_FILE_NAME)

    amct_onnx = _import_amct_module()
    amct_onnx.convert_qat_model(
        model_file=os.path.abspath(onnx_model_path),
        save_path=save_path,
        record_file=record_file,
    )

    deploy_model_path = os.path.join(output_dir_abs, EXPECTED_DEPLOY_MODEL_NAME)
    fake_quant_model_path = os.path.join(output_dir_abs, EXPECTED_FAKE_QUANT_MODEL_NAME)
    _ensure_file_exists(deploy_model_path, "AMCT deploy ONNX")
    _ensure_file_exists(fake_quant_model_path, "AMCT fakequant ONNX")
    _ensure_file_exists(record_file, "AMCT record 文件")
    _validate_onnx_model(deploy_model_path)
    _validate_onnx_model(fake_quant_model_path)

    external_files = _collect_external_data_files(output_dir_abs)
    summary = _build_summary(
        onnx_summary=onnx_summary,
        onnx_model_path=os.path.abspath(onnx_model_path),
        onnx_summary_path=os.path.abspath(onnx_summary_path),
        output_dir=output_dir_abs,
        external_files=external_files,
        repo_root=repo_root,
    )
    return output_dir_abs, summary
