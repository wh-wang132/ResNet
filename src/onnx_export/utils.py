#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ONNX 阶段通用工具入口。"""

import os

from base_model.utils import (
    create_optimized_dataloader,
    release_gpu_memory,
    setup_device,
    str2bool,
)


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def to_repo_relative_path(path):
    if path is None:
        return None

    normalized_path = os.path.normpath(path)
    if not os.path.isabs(normalized_path):
        return normalized_path

    return os.path.relpath(normalized_path, REPO_ROOT)


__all__ = [
    "REPO_ROOT",
    "create_optimized_dataloader",
    "release_gpu_memory",
    "setup_device",
    "str2bool",
    "to_repo_relative_path",
]
