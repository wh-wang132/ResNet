#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.npy 数据集加载模块
用于加载 Data 目录下的 24 类.npy 格式数据集
支持多线程预加载和性能监控
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, TypeAlias, cast

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

Sample: TypeAlias = tuple[torch.Tensor, int]
SplitEntries: TypeAlias = list[dict[str, object]]

DTYPE_MAP = {
    "fp16": (np.float16, torch.float16),
    "fp32": (np.float32, torch.float32),
}

SPLIT_MANIFEST_VERSION = 1
SPLIT_OUTPUT_DIR = os.path.join("output", "splits")


class DatasetSampleError(RuntimeError):
    """单个样本读取或校验失败。"""


class DatasetIntegrityError(RuntimeError):
    """数据集完整性校验失败。"""

    def __init__(self, message, sample_errors=None):
        super().__init__(message)
        self.sample_errors = [] if sample_errors is None else list(sample_errors)


def _format_sample_context(path, *, idx=None, split=None, stage=None):
    parts = [f"path={path}"]
    if idx is not None:
        parts.append(f"idx={idx}")
    if split is not None:
        parts.append(f"split={split}")
    if stage is not None:
        parts.append(f"stage={stage}")
    return ", ".join(parts)


def _build_sample_error(
    path,
    *,
    idx=None,
    split=None,
    stage=None,
    reason=None,
    expected_shape=None,
    actual_shape=None,
):
    context = _format_sample_context(path, idx=idx, split=split, stage=stage)
    if expected_shape is not None:
        return DatasetSampleError(
            f"样本形状不合法: {context}, expected_shape={expected_shape}, actual_shape={actual_shape}"
        )
    return DatasetSampleError(f"样本读取失败: {context}, reason={reason}")


def _build_integrity_error(sample_errors, *, split=None, stage=None, total_samples=None):
    summary = [
        "数据集完整性校验失败",
        f"invalid_samples={len(sample_errors)}",
    ]
    if total_samples is not None:
        summary.append(f"total_samples={total_samples}")
    if split is not None:
        summary.append(f"split={split}")
    if stage is not None:
        summary.append(f"stage={stage}")

    preview_lines = [f"  - {error}" for error in sample_errors[:5]]
    if len(sample_errors) > 5:
        preview_lines.append(f"  ... 还有 {len(sample_errors) - 5} 个错误")

    message = ", ".join(summary)
    if preview_lines:
        message = f"{message}\n" + "\n".join(preview_lines)
    return DatasetIntegrityError(message, sample_errors=sample_errors)


def _load_and_validate_numpy_sample(
    file_path,
    numpy_dtype,
    *,
    expected_shape=None,
    idx=None,
    split=None,
    stage=None,
):
    try:
        data = np.load(file_path)
    except Exception as exc:
        raise _build_sample_error(
            file_path,
            idx=idx,
            split=split,
            stage=stage,
            reason=str(exc),
        ) from exc

    if not isinstance(data, np.ndarray):
        raise _build_sample_error(
            file_path,
            idx=idx,
            split=split,
            stage=stage,
            reason=f"np.load 返回类型必须为 numpy.ndarray，当前为 {type(data).__name__}",
        )

    actual_shape = tuple(data.shape)
    if expected_shape is not None and actual_shape != tuple(expected_shape):
        raise _build_sample_error(
            file_path,
            idx=idx,
            split=split,
            stage=stage,
            expected_shape=tuple(expected_shape),
            actual_shape=actual_shape,
        )

    if data.dtype != numpy_dtype:
        data = data.astype(numpy_dtype)

    return data


def _numpy_sample_to_tensor(data, tensor_dtype):
    return torch.from_numpy(data).to(tensor_dtype).unsqueeze(0)


def _normalize_inferred_sample_shape(raw_shape):
    shape = tuple(int(dim) for dim in raw_shape)
    if len(shape) == 2:
        sample_shape_chw = (1, *shape)
        input_shape_nchw = (1, 1, *shape)
        return shape, sample_shape_chw, input_shape_nchw
    if len(shape) == 3:
        sample_shape_chw = shape
        input_shape_nchw = (1, *shape)
        return shape, sample_shape_chw, input_shape_nchw
    raise ValueError(
        f"数据集样本维度必须为 2 或 3，当前 shape={shape}"
    )


def _normalize_dataset_shape_tuple(shape, expected_dims, label):
    if shape is None:
        raise ValueError(f"{label} 不能为空；请通过 data_set_split() 构造 NPYDataset")
    normalized_shape = tuple(int(dim) for dim in shape)
    if len(normalized_shape) != expected_dims:
        raise ValueError(
            f"{label} 必须是 {expected_dims} 维形状，当前为 {normalized_shape}"
        )
    if any(dim <= 0 for dim in normalized_shape):
        raise ValueError(f"{label} 的每一维都必须大于 0，当前为 {normalized_shape}")
    return normalized_shape


def infer_dataset_sample_shapes(file_paths):
    if not file_paths:
        raise DatasetIntegrityError("数据集为空，无法推断样本形状")

    sample_errors = []
    for idx, file_path in enumerate(file_paths):
        try:
            data = _load_and_validate_numpy_sample(
                file_path,
                np.float32,
                idx=idx,
                stage="shape_inference",
            )
        except DatasetSampleError as exc:
            sample_errors.append(exc)
            continue

        actual_shape = tuple(data.shape)
        try:
            return _normalize_inferred_sample_shape(actual_shape)
        except ValueError as exc:
            raise DatasetIntegrityError(
                "数据集样本形状推断失败: "
                f"path={file_path}, idx={idx}, actual_shape={actual_shape}, reason={exc}"
            ) from exc

    raise _build_integrity_error(
        sample_errors,
        stage="shape_inference",
        total_samples=len(file_paths),
    )


def _validate_split_file_paths(split_name, file_paths, numpy_dtype, expected_shape):
    sample_errors = []
    for idx, file_path in enumerate(file_paths):
        try:
            _load_and_validate_numpy_sample(
                file_path,
                numpy_dtype,
                expected_shape=expected_shape,
                idx=idx,
                split=split_name,
                stage="split_validation",
            )
        except DatasetSampleError as exc:
            sample_errors.append(exc)

    if sample_errors:
        raise _build_integrity_error(
            sample_errors,
            split=split_name,
            stage="split_validation",
            total_samples=len(file_paths),
        )


class NPYDataset(Dataset):
    """自定义 Dataset 类用于加载 .npy 文件，支持可配置精度和多线程预加载"""

    def __init__(
        self,
        file_paths,
        labels,
        transform=None,
        full_load=False,
        num_workers=None,
        data_dtype="fp16",
        split_name=None,
        expected_sample_shape=None,
        sample_shape_chw=None,
        input_shape_nchw=None,
    ):
        """
        Args:
            file_paths: .npy 文件路径列表
            labels: 对应的标签列表
            transform: 数据变换
            full_load: 是否全量加载到内存
            num_workers: 预加载使用的线程数（None表示自动检测）
            data_dtype: 数据加载后的 tensor 精度（fp16 或 fp32）
            split_name: 数据集 split 名称（train / val / test）
            expected_sample_shape: 推断得到的原始样本形状
            sample_shape_chw: 推断得到的 CHW 形状
            input_shape_nchw: 推断得到的 NCHW 形状
        """
        if data_dtype not in DTYPE_MAP:
            raise ValueError(f"不支持的数据精度: {data_dtype}")

        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.full_load = full_load
        self.data_dtype = data_dtype
        self.numpy_dtype, self.tensor_dtype = DTYPE_MAP[data_dtype]
        self.data_cache: Optional[list[Optional[Sample]]] = None
        self.split_name: Optional[str] = split_name
        if expected_sample_shape is None:
            raise ValueError(
                "expected_sample_shape 不能为空；请通过 data_set_split() 构造 NPYDataset"
            )
        normalized_expected_sample_shape = tuple(int(dim) for dim in expected_sample_shape)
        self._expected_sample_shape = _normalize_dataset_shape_tuple(
            normalized_expected_sample_shape,
            expected_dims=(
                2 if len(normalized_expected_sample_shape) == 2 else 3
            ),
            label="expected_sample_shape",
        )
        self._sample_shape_chw = _normalize_dataset_shape_tuple(
            sample_shape_chw,
            expected_dims=3,
            label="sample_shape_chw",
        )
        self._input_shape_nchw = _normalize_dataset_shape_tuple(
            input_shape_nchw,
            expected_dims=4,
            label="input_shape_nchw",
        )
        cpu_count = os.cpu_count() or 1
        self.num_workers = num_workers if num_workers is not None else max(1, cpu_count)

        self.load_count = 0
        self.load_time_total = 0.0

        if self.full_load:
            self._preload_data_multithreaded()

    def _load_tensor_sample(self, idx, *, stage):
        data = _load_and_validate_numpy_sample(
            self.file_paths[idx],
            self.numpy_dtype,
            expected_shape=self._expected_sample_shape,
            idx=idx,
            split=self.split_name,
            stage=stage,
        )
        return _numpy_sample_to_tensor(data, self.tensor_dtype)

    def _load_single_sample(self, idx):
        """加载单个样本（线程安全）"""
        start_time = time.time()
        try:
            data = self._load_tensor_sample(idx, stage="full_load_preload")
            label = int(self.labels[idx])
            load_time = time.time() - start_time
            return idx, (data, label), load_time, None
        except DatasetSampleError as exc:
            return idx, None, 0.0, exc

    def _preload_data_multithreaded(self):
        """多线程预加载所有数据到内存"""
        print(f"\n{'='*80}")
        print(f"开始多线程预加载 {len(self.file_paths)} 个样本到内存...")
        print(f"使用 {self.num_workers} 个工作线程")
        print(f"{'='*80}")

        start_total = time.time()
        cache: list[Optional[Sample]] = cast(
            list[Optional[Sample]], [None] * len(self.file_paths)
        )
        self.data_cache = cache
        sample_errors = []
        total_load_time = 0.0

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(self._load_single_sample, idx): idx
                for idx in range(len(self.file_paths))
            }

            completed = 0
            for future in as_completed(futures):
                idx, data, load_time, error = future.result()
                if data is not None:
                    cache[idx] = data
                    total_load_time += load_time
                if error is not None:
                    sample_errors.append(error)
                completed += 1

                if completed % 1000 == 0:
                    elapsed = time.time() - start_total
                    print(
                        f"预加载进度: {completed}/{len(self.file_paths)} "
                        f"({completed/len(self.file_paths)*100:.1f}%), "
                        f"已用时间: {elapsed:.1f}s"
                    )

        total_time = time.time() - start_total
        avg_load_time = total_load_time / len(self.file_paths) * 1000 if self.file_paths else 0.0

        assert self.data_cache is not None
        if sample_errors:
            raise _build_integrity_error(
                sample_errors,
                split=self.split_name,
                stage="full_load_preload",
                total_samples=len(self.file_paths),
            )

        print(f"\n{'='*80}")
        print(f"✓ 预加载完成")
        print(f"  总样本数: {len(self.data_cache)}")
        print(f"  总耗时: {total_time:.2f}s")
        print(f"  平均每个样本: {avg_load_time:.2f}ms")
        print(f"  吞吐量: {len(self.file_paths)/total_time:.1f} 样本/秒")
        print(f"{'='*80}\n")

    def __len__(self):
        return len(self.file_paths)

    @property
    def expected_sample_shape(self):
        return self._expected_sample_shape

    @property
    def sample_shape_chw(self):
        return self._sample_shape_chw

    @property
    def input_shape_nchw(self):
        return self._input_shape_nchw

    def __getitem__(self, idx):
        """加载单个样本（可配置精度，带性能监控）"""
        start_time = time.time()

        if self.full_load and self.data_cache is not None:
            cache = self.data_cache
            cached_sample = cache[idx]
            if cached_sample is None:
                raise DatasetIntegrityError(
                    "预加载缓存缺少样本: "
                    + _format_sample_context(
                        self.file_paths[idx],
                        idx=idx,
                        split=self.split_name,
                        stage="cache_lookup",
                    )
                )
            data, label = cached_sample
            if self.transform:
                data = self.transform(data)
            load_time = (time.time() - start_time) * 1000
            self._record_load_time(load_time)
            return data, label

        try:
            data = self._load_tensor_sample(idx, stage="lazy_getitem")
            label = int(self.labels[idx])

            if self.transform:
                data = self.transform(data)

            load_time = (time.time() - start_time) * 1000
            self._record_load_time(load_time)
            return data, label
        except DatasetSampleError:
            load_time = (time.time() - start_time) * 1000
            self._record_load_time(load_time)
            raise

    def _record_load_time(self, load_time_ms):
        """记录加载时间用于性能监控"""
        self.load_count += 1
        self.load_time_total += load_time_ms

    def get_load_stats(self):
        """获取加载统计信息"""
        if self.load_count == 0:
            return {"count": 0, "avg_time_ms": 0.0}
        return {
            "count": self.load_count,
            "total_time_ms": self.load_time_total,
            "avg_time_ms": self.load_time_total / self.load_count,
        }


def data_set_split(
    data_dir,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    random_state=42,
    full_load=False,
    num_workers=None,
    data_dtype="fp16",
):
    """
    划分数据集（支持可配置精度和多线程预加载）

    Args:
        data_dir: 数据根目录（如./Data）
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_state: 随机种子
        full_load: 是否全量加载到内存
        num_workers: 预加载使用的线程数（None表示自动检测）
        data_dtype: 数据加载后的 tensor 精度（fp16 或 fp32）

    Returns:
        train_dataset, validate_dataset, test_dataset, labels__
    """
    if data_dtype not in DTYPE_MAP:
        raise ValueError(f"不支持的数据精度: {data_dtype}")

    normalized_data_dir = os.path.normpath(data_dir)
    numpy_dtype, _ = DTYPE_MAP[data_dtype]

    def natural_sort_key(text):
        """自然排序键：支持 0,1,2,...,10 的数字顺序，同时兼容非数字字符串。"""
        return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]

    def build_manifest_path():
        file_name = (
            "dataset_split__"
            f"train{train_ratio:.2f}_"
            f"val{val_ratio:.2f}_"
            f"test{test_ratio:.2f}_"
            f"seed{random_state}.json"
        )
        return os.path.join(SPLIT_OUTPUT_DIR, file_name)

    def scan_dataset():
        file_paths: list[str] = []
        labels: list[str] = []
        labels__: list[str] = []
        label_map: dict[str, int] = {}
        label_index = 0

        for label_folder in sorted(os.listdir(data_dir), key=natural_sort_key):
            label_folder_path = os.path.join(data_dir, label_folder)
            if os.path.isdir(label_folder_path):
                labels__.append(label_folder)
                for file_name in sorted(
                    os.listdir(label_folder_path), key=natural_sort_key
                ):
                    if file_name.endswith(".npy"):
                        file_path = os.path.join(label_folder_path, file_name)
                        file_paths.append(file_path)
                        labels.append(label_folder)
                label_map[label_folder] = label_index
                label_index += 1

        indexed_labels = [label_map[label] for label in labels]
        return file_paths, indexed_labels, labels__, label_map

    def to_manifest_entries(paths, indexed_labels, labels__):
        entries: SplitEntries = []
        for path, label_idx in zip(paths, indexed_labels):
            entries.append(
                {
                    "path": os.path.relpath(path, data_dir),
                    "label_name": labels__[label_idx],
                    "label_idx": int(label_idx),
                }
            )
        return entries

    def restore_from_entries(entries: SplitEntries):
        restored_paths: list[str] = []
        restored_labels: list[int] = []
        for entry in entries:
            rel_path = cast(str, entry["path"])
            restored_paths.append(os.path.join(data_dir, rel_path))
            restored_labels.append(int(entry["label_idx"]))
        return restored_paths, restored_labels

    def write_split_manifest(manifest_path, labels__, label_map, split_entries):
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        manifest = {
            "version": SPLIT_MANIFEST_VERSION,
            "data_dir": normalized_data_dir,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "random_state": random_state,
            "class_names": labels__,
            "class_to_idx": label_map,
            "train_files": split_entries["train_files"],
            "val_files": split_entries["val_files"],
            "test_files": split_entries["test_files"],
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    def try_load_split_manifest(manifest_path, labels__):
        if not os.path.exists(manifest_path):
            return None

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        manifest_matches = (
            manifest.get("version") == SPLIT_MANIFEST_VERSION
            and manifest.get("data_dir") == normalized_data_dir
            and manifest.get("train_ratio") == train_ratio
            and manifest.get("val_ratio") == val_ratio
            and manifest.get("test_ratio") == test_ratio
            and manifest.get("random_state") == random_state
            and manifest.get("class_names") == labels__
        )

        if not manifest_matches:
            print(f"检测到划分清单不匹配，重新划分并覆盖: {manifest_path}")
            return None

        train_paths, train_labels = restore_from_entries(
            cast(SplitEntries, manifest["train_files"])
        )
        val_paths, val_labels = restore_from_entries(
            cast(SplitEntries, manifest["val_files"])
        )
        test_paths, test_labels = restore_from_entries(
            cast(SplitEntries, manifest["test_files"])
        )
        print(f"检测到已落盘划分清单，直接复用: {manifest_path}")
        return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels

    file_paths, labels, labels__, label_map = scan_dataset()
    manifest_path = build_manifest_path()
    expected_sample_shape, sample_shape_chw, input_shape_nchw = infer_dataset_sample_shapes(
        file_paths
    )

    print(f"类别标签映射：{labels__}")
    print(f"总样本数：{len(file_paths)}")
    print(
        "推断得到数据集样本形状: "
        f"expected_sample_shape={expected_sample_shape}, "
        f"sample_shape_chw={sample_shape_chw}, "
        f"input_shape_nchw={input_shape_nchw}"
    )

    manifest_split = try_load_split_manifest(manifest_path, labels__)
    if manifest_split is None:
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            file_paths,
            labels,
            test_size=(1 - train_ratio),
            random_state=random_state,
            stratify=labels,
        )

        val_test_ratio = test_ratio / (test_ratio + val_ratio)
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths,
            temp_labels,
            test_size=val_test_ratio,
            random_state=random_state,
            stratify=temp_labels,
        )

        split_entries = {
            "train_files": to_manifest_entries(train_paths, train_labels, labels__),
            "val_files": to_manifest_entries(val_paths, val_labels, labels__),
            "test_files": to_manifest_entries(test_paths, test_labels, labels__),
        }
        write_split_manifest(manifest_path, labels__, label_map, split_entries)
        print(f"划分结果已落盘: {manifest_path}")
    else:
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = (
            manifest_split
        )

    if not full_load:
        _validate_split_file_paths("train", train_paths, numpy_dtype, expected_sample_shape)
        _validate_split_file_paths("val", val_paths, numpy_dtype, expected_sample_shape)
        _validate_split_file_paths("test", test_paths, numpy_dtype, expected_sample_shape)

    print(f"训练集：{len(train_paths)} 样本")
    print(f"验证集：{len(val_paths)} 样本")
    print(f"测试集：{len(test_paths)} 样本")

    train_dataset = NPYDataset(
        train_paths,
        train_labels,
        full_load=full_load,
        num_workers=num_workers,
        data_dtype=data_dtype,
        split_name="train",
        expected_sample_shape=expected_sample_shape,
        sample_shape_chw=sample_shape_chw,
        input_shape_nchw=input_shape_nchw,
    )
    validate_dataset = NPYDataset(
        val_paths,
        val_labels,
        full_load=full_load,
        num_workers=num_workers,
        data_dtype=data_dtype,
        split_name="val",
        expected_sample_shape=expected_sample_shape,
        sample_shape_chw=sample_shape_chw,
        input_shape_nchw=input_shape_nchw,
    )
    test_dataset = NPYDataset(
        test_paths,
        test_labels,
        full_load=full_load,
        num_workers=num_workers,
        data_dtype=data_dtype,
        split_name="test",
        expected_sample_shape=expected_sample_shape,
        sample_shape_chw=sample_shape_chw,
        input_shape_nchw=input_shape_nchw,
    )

    return train_dataset, validate_dataset, test_dataset, labels__
