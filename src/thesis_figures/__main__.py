#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""论文插图生成入口。"""

from __future__ import annotations

from pathlib import Path
import sys

from .args import parse_args
from .output import build_manifest, create_figure_directory, write_json
from .plots import write_all_outputs
from .scanner import scan_output


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    result = scan_output(
        output_root=output_root,
        model=args.model,
        experiment=args.experiment,
        strict=args.strict,
    )

    _print_scan_summary(result)
    if args.dry_run:
        print("\nDry run 完成：未创建论文插图目录。")
        return 0

    if not result.records:
        print("没有可用记录，未生成图表。", file=sys.stderr)
        return 1

    figure_dir = create_figure_directory(args.figure_dir)
    figures, tables = write_all_outputs(result.records, figure_dir, args.formats)
    manifest = build_manifest(
        output_root=output_root,
        records=result.records,
        warnings=result.warnings,
        figures=figures,
        tables=tables,
    )
    manifest_path = write_json(figure_dir / "figures_manifest.json", manifest)

    print(f"\n论文插图输出目录: {figure_dir}")
    print(f"图像数量: {len(figures)}")
    print(f"表格数量: {len(tables)}")
    print(f"Manifest: {manifest_path}")
    return 0


def _print_scan_summary(result):
    print("\n扫描 output 产物完成")
    print(f"记录数: {len(result.records)}")
    for stage, count in sorted(result.counts_by_stage.items()):
        print(f"  {stage}: {count}")
    if result.warnings:
        print(f"警告数: {len(result.warnings)}")
        for warning in result.warnings[:8]:
            print(f"  - {warning}")
        if len(result.warnings) > 8:
            print(f"  ... 另有 {len(result.warnings) - 8} 条警告")


if __name__ == "__main__":
    raise SystemExit(main())
