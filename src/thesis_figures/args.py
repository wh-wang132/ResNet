#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""论文插图 CLI 参数。"""

import argparse


SUPPORTED_FORMATS = ("png", "svg")
SUPPORTED_MODELS = (
    "all",
    "resnet6_2d",
    "resnet10_2d",
    "resnet14_2d",
    "resnet18_2d",
    "resnet34_2d",
)


def _parse_formats(value):
    formats = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not formats:
        raise argparse.ArgumentTypeError("至少需要指定一种输出格式")

    unsupported = sorted(set(formats) - set(SUPPORTED_FORMATS))
    if unsupported:
        raise argparse.ArgumentTypeError(
            f"不支持的输出格式: {', '.join(unsupported)}；支持: {', '.join(SUPPORTED_FORMATS)}"
        )
    return tuple(dict.fromkeys(formats))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate thesis figures from existing output artifacts only"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="output",
        help="训练端产物根目录 (默认 output)",
    )
    parser.add_argument(
        "--figure_dir",
        type=str,
        default="output/thesis_figures",
        help="论文插图输出根目录 (默认 output/thesis_figures)",
    )
    parser.add_argument(
        "--formats",
        type=_parse_formats,
        default=("png", "svg"),
        help="逗号分隔的图片格式，支持 png,svg (默认 png,svg)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=SUPPORTED_MODELS,
        default="all",
        help="筛选模型名 (默认 all)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        help="筛选实验名子串；from_ratio... 会归一化为 ratio... 后再匹配 (默认 all)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="只扫描并打印摘要，不创建图表目录",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="遇到缺字段、坏 JSON 或无可用记录时直接失败",
    )
    return parser.parse_args()
