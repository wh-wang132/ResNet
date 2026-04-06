#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AMCT 转换与产物校验。"""

import importlib
import os

from .output import create_output_directory
from .utils import (
    AMCT_ALLOWED_OP_TYPES,
    AMCT_DEPLOY_ALLOWED_DOMAINS,
    AMCT_FAKE_QUANT_ALLOWED_DOMAINS,
    SUMMARY_VERSION,
    ensure_file_exists,
    ensure_summary_keys,
    ensure_summary_version,
    get_repo_root,
    load_json,
    resolve_repo_path,
    to_repo_relative_path,
    validate_interface_dict,
    validate_onnx_contract,
)


EXPECTED_INPUT_MODEL_NAME = "model_quant.onnx"
EXPECTED_ONNX_SUMMARY_NAME = "onnx_summary.json"
EXPECTED_ONNX_BRANCH = "qat_convert"
EXPECTED_DEPLOY_MODEL_NAME = "deploy_model.onnx"
EXPECTED_FAKE_QUANT_MODEL_NAME = "fake_quant_model.onnx"
EXPECTED_RECORD_FILE_NAME = "scale_offset_record.txt"
REQUIRED_ONNX_SUMMARY_KEYS = {
    "summary_version",
    "branch",
    "model_name",
    "source_checkpoint_path",
    "onnx_path",
    "example_input_shape",
    "opset_version",
    "interface",
}


class AMCTConversionError(RuntimeError):
    """AMCT 阶段输入校验或转换错误。"""


def _validate_input_onnx_model(onnx_model_path):
    if os.path.splitext(onnx_model_path)[1] != ".onnx":
        raise AMCTConversionError("AMCT 输入文件必须是 .onnx")
    if os.path.basename(onnx_model_path) != EXPECTED_INPUT_MODEL_NAME:
        raise AMCTConversionError("AMCT 当前只接受仓库 qat_convert 产出的 model_quant.onnx")


def load_and_validate_qat_source(onnx_model_path, repo_root=None):
    repo_root = get_repo_root() if repo_root is None else repo_root
    onnx_model_path = os.path.abspath(onnx_model_path)
    ensure_file_exists(onnx_model_path, "AMCT 输入 ONNX")
    _validate_input_onnx_model(onnx_model_path)

    summary_path = os.path.join(os.path.dirname(onnx_model_path), EXPECTED_ONNX_SUMMARY_NAME)
    ensure_file_exists(summary_path, "ONNX 摘要")
    onnx_summary = load_json(summary_path)
    ensure_summary_keys(
        onnx_summary,
        REQUIRED_ONNX_SUMMARY_KEYS,
        EXPECTED_ONNX_SUMMARY_NAME,
        error_cls=AMCTConversionError,
    )
    ensure_summary_version(
        onnx_summary,
        EXPECTED_ONNX_SUMMARY_NAME,
        expected_version=SUMMARY_VERSION,
        error_cls=AMCTConversionError,
    )
    validate_interface_dict(
        onnx_summary["interface"],
        label=f"{EXPECTED_ONNX_SUMMARY_NAME}.interface",
        error_cls=AMCTConversionError,
    )

    if onnx_summary["branch"] != EXPECTED_ONNX_BRANCH:
        raise AMCTConversionError("AMCT 当前只接受 branch=qat_convert 的 ONNX 摘要")

    resolved_summary_onnx_path = resolve_repo_path(onnx_summary["onnx_path"], repo_root=repo_root)
    if os.path.normpath(resolved_summary_onnx_path) != os.path.normpath(onnx_model_path):
        raise AMCTConversionError("onnx_summary.json 中的 onnx_path 与当前输入文件不匹配")

    source_contract = validate_onnx_contract(
        onnx_model_path,
        expected_interface=onnx_summary["interface"],
        label="AMCT source ONNX",
        error_cls=AMCTConversionError,
    )
    return {
        "onnx_summary": onnx_summary,
        "onnx_summary_path": os.path.abspath(summary_path),
        "source_onnx_path": onnx_model_path,
        "source_interface": source_contract["interface"],
    }


def _import_amct_module():
    try:
        return importlib.import_module("amct_onnx")
    except Exception as exc:  # pragma: no cover - 真实环境依赖错误
        raise AMCTConversionError(f"无法导入 amct_onnx: {exc}") from exc


def run_amct_convert(source_onnx_path, output_dir_abs):
    save_path = os.path.join(output_dir_abs, "")
    record_file = os.path.join(output_dir_abs, EXPECTED_RECORD_FILE_NAME)

    amct_onnx = _import_amct_module()
    amct_onnx.convert_qat_model(
        model_file=os.path.abspath(source_onnx_path),
        save_path=save_path,
        record_file=record_file,
    )
    return {
        "deploy_model_path": os.path.join(output_dir_abs, EXPECTED_DEPLOY_MODEL_NAME),
        "fake_quant_model_path": os.path.join(output_dir_abs, EXPECTED_FAKE_QUANT_MODEL_NAME),
        "record_file_path": record_file,
    }


def _collect_external_data_files(folder_path):
    external_files = []
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith(".external"):
            external_files.append(os.path.join(folder_path, file_name))
    return external_files


def validate_amct_outputs_and_build_summary(
    *,
    source_info,
    output_paths,
    output_dir_abs,
    repo_root=None,
):
    repo_root = get_repo_root() if repo_root is None else repo_root
    deploy_model_path = output_paths["deploy_model_path"]
    fake_quant_model_path = output_paths["fake_quant_model_path"]
    record_file_path = output_paths["record_file_path"]

    ensure_file_exists(deploy_model_path, "AMCT deploy ONNX")
    ensure_file_exists(fake_quant_model_path, "AMCT fakequant ONNX")
    ensure_file_exists(record_file_path, "AMCT record 文件")

    deploy_contract = validate_onnx_contract(
        deploy_model_path,
        expected_interface=source_info["source_interface"],
        allowed_domains=AMCT_DEPLOY_ALLOWED_DOMAINS,
        allowed_op_types=AMCT_ALLOWED_OP_TYPES,
        label="AMCT deploy ONNX",
        error_cls=AMCTConversionError,
    )
    fake_quant_contract = validate_onnx_contract(
        fake_quant_model_path,
        expected_interface=source_info["source_interface"],
        allowed_domains=AMCT_FAKE_QUANT_ALLOWED_DOMAINS,
        allowed_op_types=AMCT_ALLOWED_OP_TYPES,
        label="AMCT fakequant ONNX",
        error_cls=AMCTConversionError,
    )
    external_files = _collect_external_data_files(output_dir_abs)
    onnx_summary = source_info["onnx_summary"]
    amct_log_path = os.path.join(repo_root, "amct_log", "amct_onnx.log")

    summary = {
        "summary_version": SUMMARY_VERSION,
        "stage": "amct",
        "model_name": onnx_summary["model_name"],
        "source_onnx_path": to_repo_relative_path(source_info["source_onnx_path"], repo_root=repo_root),
        "source_onnx_summary_path": to_repo_relative_path(
            source_info["onnx_summary_path"],
            repo_root=repo_root,
        ),
        "source_checkpoint_path": to_repo_relative_path(
            onnx_summary["source_checkpoint_path"],
            repo_root=repo_root,
        ),
        "source_branch": EXPECTED_ONNX_BRANCH,
        "source_interface": source_info["source_interface"],
        "example_input_shape": onnx_summary["example_input_shape"],
        "opset_version": onnx_summary["opset_version"],
        "deploy_model_path": to_repo_relative_path(deploy_model_path, repo_root=repo_root),
        "deploy_interface": deploy_contract["interface"],
        "deploy_domains": deploy_contract["domains"],
        "deploy_op_types": deploy_contract["op_types"],
        "fake_quant_model_path": to_repo_relative_path(fake_quant_model_path, repo_root=repo_root),
        "fake_quant_interface": fake_quant_contract["interface"],
        "fake_quant_domains": fake_quant_contract["domains"],
        "fake_quant_op_types": fake_quant_contract["op_types"],
        "record_file_path": to_repo_relative_path(record_file_path, repo_root=repo_root),
        "amct_log_path": to_repo_relative_path(amct_log_path, repo_root=repo_root)
        if os.path.exists(amct_log_path)
        else None,
    }
    if external_files:
        summary["external_data_files"] = [
            to_repo_relative_path(path, repo_root=repo_root) for path in external_files
        ]
    return summary


def build_amct_artifacts(onnx_model_path, repo_root=None):
    repo_root = get_repo_root() if repo_root is None else repo_root
    source_info = load_and_validate_qat_source(onnx_model_path, repo_root=repo_root)
    output_dir = create_output_directory(source_info["onnx_summary"])
    output_dir_abs = os.path.abspath(output_dir)
    output_paths = run_amct_convert(source_info["source_onnx_path"], output_dir_abs)
    summary = validate_amct_outputs_and_build_summary(
        source_info=source_info,
        output_paths=output_paths,
        output_dir_abs=output_dir_abs,
        repo_root=repo_root,
    )
    return output_dir_abs, summary
