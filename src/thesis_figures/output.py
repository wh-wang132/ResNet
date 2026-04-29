#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""论文插图输出目录和表格工具。"""

from __future__ import annotations

from datetime import datetime
import csv
import json
from pathlib import Path
from typing import Any, Iterable


def create_figure_directory(figure_dir) -> Path:
    root = Path(figure_dir)
    timestamp = datetime.now().strftime("figures_%Y%m%d_%H%M%S")
    candidate = root / timestamp
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = root / f"{timestamp}_{suffix}"
    candidate.mkdir(parents=True, exist_ok=False)
    (candidate / "tables").mkdir(parents=True, exist_ok=True)
    return candidate


def write_csv(path, rows: Iterable[dict[str, Any]]) -> Path:
    path = Path(path)
    rows = list(rows)
    fieldnames = _fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _serialize(row.get(field)) for field in fieldnames})
    return path


def write_json(path, data: Any) -> Path:
    path = Path(path)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, indent=2, ensure_ascii=False)
    return path


def build_manifest(output_root, records, warnings, figures, tables) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(output_root),
        "record_count": len(records),
        "warnings": list(warnings),
        "figures": [str(path) for path in figures],
        "tables": [str(path) for path in tables],
    }


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def _serialize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10g}"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)
