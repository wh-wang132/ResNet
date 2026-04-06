#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ATC 阶段通用工具入口。"""

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


def build_atc_subprocess_env():
    env = os.environ.copy()
    virtual_env = env.pop("VIRTUAL_ENV", None)
    env.pop("UV", None)
    env.pop("PYTHONHOME", None)

    uv_keys = [key for key in env if key.startswith("UV_")]
    for key in uv_keys:
        env.pop(key, None)

    if virtual_env is not None:
        virtual_bin = os.path.join(virtual_env, "bin")
        path_entries = env.get("PATH", "").split(os.pathsep)
        path_entries = [
            entry
            for entry in path_entries
            if os.path.normpath(entry) != os.path.normpath(virtual_bin)
        ]
        env["PATH"] = os.pathsep.join(path_entries)

    return env


__all__ = [
    "build_atc_subprocess_env",
    "ensure_file_exists",
    "get_repo_root",
    "load_json",
    "resolve_repo_path",
    "to_repo_relative_path",
]
