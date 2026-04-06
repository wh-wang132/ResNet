#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AMCT 转换统一入口。"""

import json

from amct.args import parse_args
from amct.converter import build_amct_artifacts
from amct.output import save_summary


def main():
    args = parse_args()
    print(args)

    folder_path, summary = build_amct_artifacts(args.onnx_model)
    summary_path = save_summary(folder_path, summary)

    print("\nAMCT 转换流程完成")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n摘要已保存至: {summary_path}")


if __name__ == "__main__":
    main()
