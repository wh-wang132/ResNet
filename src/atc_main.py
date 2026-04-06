#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ATC 编译统一入口。"""

import json

from atc.args import parse_args
from atc.converter import build_atc_artifacts
from atc.output import save_summary


def main():
    args = parse_args()
    print(args)

    folder_path, summary = build_atc_artifacts(
        branch=args.branch,
        onnx_model_path=args.onnx_model,
        soc_version=args.soc_version,
        input_shape=args.input_shape,
        input_format=args.input_format,
    )
    summary_path = save_summary(folder_path, summary)

    print("\nATC 编译流程完成")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n摘要已保存至: {summary_path}")


if __name__ == "__main__":
    main()
