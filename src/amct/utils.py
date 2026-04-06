#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AMCT 阶段通用工具入口。"""

import json
import os


def get_repo_root():
    repo_root = os.environ.get("REPO_ROOT")
    if not repo_root:
        raise RuntimeError("REPO_ROOT 未设置：请先让 direnv 自动激活 .envrc")
    return os.path.abspath(repo_root)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def ensure_file_exists(path, label):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到{label}: {path}")


def resolve_repo_path(path, repo_root=None):
    repo_root = get_repo_root() if repo_root is None else repo_root
    normalized_path = os.path.normpath(path)
    if os.path.isabs(normalized_path):
        return normalized_path
    return os.path.abspath(os.path.join(repo_root, normalized_path))


def to_repo_relative_path(path, repo_root=None):
    repo_root = get_repo_root() if repo_root is None else repo_root
    if path is None:
        return None

    normalized_path = os.path.normpath(path)
    if not os.path.isabs(normalized_path):
        return normalized_path
    return os.path.relpath(normalized_path, repo_root)


__all__ = [
    "ensure_file_exists",
    "get_repo_root",
    "load_json",
    "resolve_repo_path",
    "to_repo_relative_path",
]
