#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""output summary 到论文图表记录的只读契约适配。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any


SUMMARY_FILENAMES = {
    "pruning_summary.json",
    "qat_summary.json",
    "onnx_summary.json",
    "amct_summary.json",
    "atc_summary.json",
}

ELEM_TYPE_LABELS = {
    1: "float32",
    10: "float16",
}


@dataclass(frozen=True)
class FigureRecord:
    stage: str
    model_name: str
    experiment_name: str
    summary_path: str
    branch: str | None = None
    raw_experiment_name: str | None = None
    pruning_ratio: float | None = None
    pruning_steps: int | None = None
    val_acc: float | None = None
    val_loss: float | None = None
    test_acc: float | None = None
    test_loss: float | None = None
    source_test_acc: float | None = None
    source_test_loss: float | None = None
    exported_test_acc: float | None = None
    exported_test_loss: float | None = None
    metric_delta_acc: float | None = None
    metric_delta_loss: float | None = None
    params: int | None = None
    macs: int | None = None
    baseline_params: int | None = None
    baseline_macs: int | None = None
    signature_hash: str | None = None
    input_elem_type: int | None = None
    output_elem_type: int | None = None
    input_dtype: str | None = None
    output_dtype: str | None = None
    input_shape: str | None = None
    output_shape: str | None = None
    soc_version: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    if not isinstance(data, dict):
        raise ValueError(f"{path} 顶层必须是 JSON object")
    return data


def normalize_experiment_name(name: str | None) -> str:
    if not name:
        return "unknown"
    return name[5:] if name.startswith("from_") else name


def parse_experiment_config(experiment_name: str | None) -> dict[str, Any]:
    normalized = normalize_experiment_name(experiment_name)
    pattern = re.compile(
        r"ratio(?P<ratio>[0-9.]+)_steps(?P<steps>\d+)_(?P<mode>[^_]+)_ft(?P<ft>\d+)_bs(?P<bs>\d+)"
    )
    match = pattern.search(normalized)
    if match is None:
        return {}

    return {
        "pruning_ratio": _to_float(match.group("ratio")),
        "pruning_steps": _to_int(match.group("steps")),
        "pruning_mode": match.group("mode"),
        "finetune_epochs": _to_int(match.group("ft")),
        "batch_size": _to_int(match.group("bs")),
    }


def record_from_summary(summary_path: Path, output_root: Path) -> FigureRecord:
    summary = load_json(summary_path)
    filename = summary_path.name
    if filename == "pruning_summary.json":
        return _pruning_record(summary, summary_path, output_root)
    if filename == "qat_summary.json":
        return _qat_record(summary, summary_path, output_root)
    if filename == "onnx_summary.json":
        return _onnx_record(summary, summary_path, output_root)
    if filename == "amct_summary.json":
        return _amct_record(summary, summary_path, output_root)
    if filename == "atc_summary.json":
        return _atc_record(summary, summary_path, output_root)
    raise ValueError(f"不支持的 summary 文件: {summary_path}")


def natural_key(value: str) -> list[Any]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]


def _pruning_record(summary: dict[str, Any], path: Path, root: Path) -> FigureRecord:
    raw_experiment = path.parent.name
    config = parse_experiment_config(raw_experiment)
    final = _dict(summary.get("final"))
    baseline = _dict(summary.get("baseline"))
    final_stats = _dict(final.get("stats"))
    baseline_stats = _dict(baseline.get("stats"))
    final_val = _dict(final.get("val"))
    final_test = _dict(final.get("test"))
    pruning_meta = _dict(summary.get("pruning_meta"))
    topology = _dict(summary.get("final_topology"))

    warnings = []
    _warn_missing(warnings, summary, ["model_name", "final"], path)

    return FigureRecord(
        stage="pruning",
        model_name=str(summary.get("model_name") or path.parent.parent.name),
        experiment_name=normalize_experiment_name(raw_experiment),
        raw_experiment_name=raw_experiment,
        summary_path=_rel(path, root),
        pruning_ratio=_first_float(
            pruning_meta.get("target_total_ratio"),
            config.get("pruning_ratio"),
        ),
        pruning_steps=_first_int(
            summary.get("pruning_steps"),
            pruning_meta.get("pruning_steps"),
            config.get("pruning_steps"),
        ),
        val_acc=_to_float(final_val.get("acc")),
        val_loss=_to_float(final_val.get("loss")),
        test_acc=_to_float(final_test.get("acc")),
        test_loss=_to_float(final_test.get("loss")),
        params=_first_int(final_stats.get("params"), pruning_meta.get("params_after")),
        macs=_first_int(final_stats.get("macs"), pruning_meta.get("macs_after")),
        baseline_params=_first_int(
            baseline_stats.get("params"),
            pruning_meta.get("params_before"),
        ),
        baseline_macs=_first_int(
            baseline_stats.get("macs"),
            pruning_meta.get("macs_before"),
        ),
        signature_hash=_signature_hash(topology.get("architecture_signature")),
        warnings=tuple(warnings),
    )


def _qat_record(summary: dict[str, Any], path: Path, root: Path) -> FigureRecord:
    raw_experiment = path.parent.name
    config = parse_experiment_config(raw_experiment)
    final = _dict(summary.get("final"))
    baseline = _dict(summary.get("baseline"))
    final_stats = _dict(final.get("stats"))
    baseline_stats = _dict(baseline.get("stats"))
    final_val = _dict(final.get("val"))
    final_test = _dict(final.get("test"))
    topology = _dict(summary.get("final_topology"))

    warnings = []
    _warn_missing(warnings, summary, ["model_name", "final"], path)

    return FigureRecord(
        stage="qat",
        model_name=str(summary.get("model_name") or path.parent.parent.name),
        experiment_name=normalize_experiment_name(raw_experiment),
        raw_experiment_name=raw_experiment,
        summary_path=_rel(path, root),
        pruning_ratio=config.get("pruning_ratio"),
        pruning_steps=config.get("pruning_steps"),
        val_acc=_to_float(final_val.get("acc")),
        val_loss=_to_float(final_val.get("loss")),
        test_acc=_to_float(final_test.get("acc")),
        test_loss=_to_float(final_test.get("loss")),
        params=_first_int(final_stats.get("params"), baseline_stats.get("params")),
        macs=_first_int(final_stats.get("macs"), baseline_stats.get("macs")),
        baseline_params=_to_int(baseline_stats.get("params")),
        baseline_macs=_to_int(baseline_stats.get("macs")),
        signature_hash=_signature_hash(topology.get("architecture_signature")),
        warnings=tuple(warnings),
    )


def _onnx_record(summary: dict[str, Any], path: Path, root: Path) -> FigureRecord:
    raw_experiment = path.parent.name
    config = parse_experiment_config(raw_experiment)
    source_metrics = _dict(summary.get("source_test_metrics"))
    onnx_metrics = _dict(summary.get("onnx_test_metrics"))
    metric_delta = _dict(summary.get("metric_delta"))
    interface = _dict(summary.get("interface"))
    signature = _dict(summary.get("source_architecture_signature"))

    warnings = []
    _warn_missing(
        warnings,
        summary,
        ["summary_version", "branch", "model_name", "interface"],
        path,
    )

    input_elem_type = _to_int(interface.get("input_elem_type"))
    output_elem_type = _to_int(interface.get("output_elem_type"))

    return FigureRecord(
        stage="onnx",
        branch=str(summary.get("branch") or path.parent.parent.parent.name),
        model_name=str(summary.get("model_name") or path.parent.parent.name),
        experiment_name=normalize_experiment_name(raw_experiment),
        raw_experiment_name=raw_experiment,
        summary_path=_rel(path, root),
        pruning_ratio=config.get("pruning_ratio"),
        pruning_steps=config.get("pruning_steps"),
        test_acc=_to_float(onnx_metrics.get("acc")),
        test_loss=_to_float(onnx_metrics.get("loss")),
        source_test_acc=_to_float(source_metrics.get("acc")),
        source_test_loss=_to_float(source_metrics.get("loss")),
        exported_test_acc=_to_float(onnx_metrics.get("acc")),
        exported_test_loss=_to_float(onnx_metrics.get("loss")),
        metric_delta_acc=_to_float(metric_delta.get("acc")),
        metric_delta_loss=_to_float(metric_delta.get("loss")),
        params=_to_int(signature.get("parameter_count")),
        signature_hash=_signature_hash(signature),
        input_elem_type=input_elem_type,
        output_elem_type=output_elem_type,
        input_dtype=ELEM_TYPE_LABELS.get(input_elem_type),
        output_dtype=ELEM_TYPE_LABELS.get(output_elem_type),
        input_shape=_shape_label(interface.get("input_shape")),
        output_shape=_shape_label(interface.get("output_shape")),
        warnings=tuple(warnings),
    )


def _amct_record(summary: dict[str, Any], path: Path, root: Path) -> FigureRecord:
    raw_experiment = path.parent.name
    config = parse_experiment_config(raw_experiment)
    interface = _dict(summary.get("deploy_interface") or summary.get("source_interface"))
    signature = _dict(summary.get("source_architecture_signature"))

    warnings = []
    _warn_missing(warnings, summary, ["summary_version", "stage", "model_name"], path)

    input_elem_type = _to_int(interface.get("input_elem_type"))
    output_elem_type = _to_int(interface.get("output_elem_type"))

    return FigureRecord(
        stage="amct",
        branch="amct_deploy",
        model_name=str(summary.get("model_name") or path.parent.parent.name),
        experiment_name=normalize_experiment_name(raw_experiment),
        raw_experiment_name=raw_experiment,
        summary_path=_rel(path, root),
        pruning_ratio=config.get("pruning_ratio"),
        pruning_steps=config.get("pruning_steps"),
        params=_to_int(signature.get("parameter_count")),
        signature_hash=_signature_hash(signature),
        input_elem_type=input_elem_type,
        output_elem_type=output_elem_type,
        input_dtype=ELEM_TYPE_LABELS.get(input_elem_type),
        output_dtype=ELEM_TYPE_LABELS.get(output_elem_type),
        input_shape=_shape_label(interface.get("input_shape")),
        output_shape=_shape_label(interface.get("output_shape")),
        warnings=tuple(warnings),
    )


def _atc_record(summary: dict[str, Any], path: Path, root: Path) -> FigureRecord:
    raw_experiment = path.parent.name
    config = parse_experiment_config(raw_experiment)
    interface = _dict(summary.get("source_interface"))
    signature = _dict(summary.get("source_architecture_signature"))

    warnings = []
    _warn_missing(warnings, summary, ["stage", "branch", "model_name"], path)

    input_elem_type = _to_int(interface.get("input_elem_type"))
    output_elem_type = _to_int(interface.get("output_elem_type"))

    return FigureRecord(
        stage="atc",
        branch=str(summary.get("branch") or path.parent.parent.parent.name),
        model_name=str(summary.get("model_name") or path.parent.parent.name),
        experiment_name=normalize_experiment_name(raw_experiment),
        raw_experiment_name=raw_experiment,
        summary_path=_rel(path, root),
        pruning_ratio=config.get("pruning_ratio"),
        pruning_steps=config.get("pruning_steps"),
        params=_to_int(signature.get("parameter_count")),
        signature_hash=_signature_hash(signature),
        input_elem_type=input_elem_type,
        output_elem_type=output_elem_type,
        input_dtype=ELEM_TYPE_LABELS.get(input_elem_type),
        output_dtype=ELEM_TYPE_LABELS.get(output_elem_type),
        input_shape=_shape_label(summary.get("resolved_input_shape") or interface.get("input_shape")),
        output_shape=_shape_label(interface.get("output_shape")),
        soc_version=_none_or_str(summary.get("soc_version")),
        warnings=tuple(warnings),
    )


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root.parent))
    except ValueError:
        return str(path)


def _warn_missing(warnings: list[str], data: dict[str, Any], keys: list[str], path: Path) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        warnings.append(f"{path} 缺少字段: {', '.join(missing)}")


def _signature_hash(value: Any) -> str | None:
    signature = _dict(value)
    value = signature.get("signature_hash")
    return str(value) if value else None


def _shape_label(value: Any) -> str | None:
    if isinstance(value, (list, tuple)):
        return "x".join(str(item) for item in value)
    if value is None:
        return None
    return str(value)


def _none_or_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _to_float(value)
        if parsed is not None:
            return parsed
    return None


def _first_int(*values: Any) -> int | None:
    for value in values:
        parsed = _to_int(value)
        if parsed is not None:
            return parsed
    return None
