#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扫描 output 产物并构建论文图表记录。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .contracts import (
    FigureRecord,
    SUMMARY_FILENAMES,
    natural_key,
    record_from_summary,
)


@dataclass(frozen=True)
class ScanResult:
    records: tuple[FigureRecord, ...]
    warnings: tuple[str, ...]

    @property
    def counts_by_stage(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record.stage] = counts.get(record.stage, 0) + 1
        return counts


def scan_output(
    output_root,
    model="all",
    experiment="all",
    strict=False,
) -> ScanResult:
    output_root = Path(output_root)
    if not output_root.exists():
        message = f"output_root 不存在: {output_root}"
        if strict:
            raise FileNotFoundError(message)
        return ScanResult(records=(), warnings=(message,))

    warnings: list[str] = []
    records: list[FigureRecord] = []
    for summary_path in _iter_summary_paths(output_root):
        try:
            record = record_from_summary(summary_path, output_root)
        except Exception as exc:
            message = f"跳过 {summary_path}: {exc}"
            if strict:
                raise RuntimeError(message) from exc
            warnings.append(message)
            continue

        if model != "all" and record.model_name != model:
            continue
        if experiment != "all" and experiment not in record.experiment_name:
            continue

        if strict and record.warnings:
            raise RuntimeError("; ".join(record.warnings))

        records.append(record)
        warnings.extend(record.warnings)

    records.sort(
        key=lambda item: (
            natural_key(item.model_name),
            natural_key(item.experiment_name),
            item.stage,
            item.branch or "",
        )
    )

    if strict and not records:
        raise RuntimeError("没有发现可用于论文插图的 output summary")
    if not records:
        warnings.append("没有发现可用于论文插图的 output summary")

    return ScanResult(records=tuple(records), warnings=tuple(warnings))


def _iter_summary_paths(output_root: Path):
    paths = []
    for path in output_root.rglob("*_summary.json"):
        if "thesis_figures" in path.parts:
            continue
        if path.name in SUMMARY_FILENAMES:
            paths.append(path)
    return sorted(paths, key=lambda item: natural_key(str(item)))
