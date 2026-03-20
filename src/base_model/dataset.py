#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.npy 数据集加载模块
用于加载 Data 目录下的 24 类.npy 格式数据集
支持多线程预加载和性能监控
"""

import os
import time
import numpy as np
import torch
import re
from typing import Optional, TypeAlias, cast
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed

Sample: TypeAlias = tuple[torch.Tensor, int]


class NPYDataset(Dataset):
    """自定义 Dataset 类用于加载.npy 文件（FP16 版本，支持多线程预加载）"""

    def __init__(
        self, file_paths, labels, transform=None, full_load=False, num_workers=None
    ):
        """
        Args:
            file_paths: .npy 文件路径列表
            labels: 对应的标签列表
            transform: 数据变换
            full_load: 是否全量加载到内存
            num_workers: 预加载使用的线程数（None表示自动检测）
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.full_load = full_load
        self.data_cache: Optional[list[Optional[Sample]]] = None
        cpu_count = os.cpu_count() or 1
        self.num_workers = num_workers if num_workers is not None else max(1, cpu_count)

        self.load_count = 0
        self.load_time_total = 0.0

        if self.full_load:
            self._preload_data_multithreaded()

    def _load_single_sample(self, idx):
        """加载单个样本（线程安全）"""
        try:
            start_time = time.time()
            data = np.load(self.file_paths[idx])
            label = self.labels[idx]

            if data.dtype != np.float16:
                data = data.astype(np.float16)

            data = torch.from_numpy(data).to(torch.float16)
            data = data.unsqueeze(0)

            load_time = time.time() - start_time
            return idx, (data, int(label)), load_time, None
        except Exception as e:
            return idx, (torch.zeros(1, 543, 512, dtype=torch.float16), 0), 0.0, str(e)

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
        errors = []
        total_load_time = 0.0

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(self._load_single_sample, idx): idx
                for idx in range(len(self.file_paths))
            }

            completed = 0
            for future in as_completed(futures):
                idx, data, load_time, error = future.result()
                cache[idx] = data
                total_load_time += load_time
                completed += 1

                if error:
                    errors.append((self.file_paths[idx], error))

                if completed % 1000 == 0:
                    elapsed = time.time() - start_total
                    print(
                        f"预加载进度: {completed}/{len(self.file_paths)} "
                        f"({completed/len(self.file_paths)*100:.1f}%), "
                        f"已用时间: {elapsed:.1f}s"
                    )

        total_time = time.time() - start_total
        avg_load_time = total_load_time / len(self.file_paths) * 1000

        assert self.data_cache is not None
        print(f"\n{'='*80}")
        print(f"✓ 预加载完成")
        print(f"  总样本数: {len(self.data_cache)}")
        print(f"  总耗时: {total_time:.2f}s")
        print(f"  平均每个样本: {avg_load_time:.2f}ms")
        print(f"  吞吐量: {len(self.file_paths)/total_time:.1f} 样本/秒")

        if errors:
            print(f"\n⚠️  警告：{len(errors)} 个样本加载失败")
            for file_path, error in errors[:5]:
                print(f"  - {file_path}: {error}")
            if len(errors) > 5:
                print(f"  ... 还有 {len(errors)-5} 个错误")
        print(f"{'='*80}\n")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """加载单个样本（FP16，带性能监控）"""
        start_time = time.time()

        if self.full_load and self.data_cache is not None:
            cache = self.data_cache
            cached_sample = cache[idx]
            if cached_sample is None:
                # 理论上不应发生；兜底保证类型安全和运行稳定性
                return torch.zeros(1, 543, 512, dtype=torch.float16), 0
            data, label = cached_sample
            if self.transform:
                data = self.transform(data)
            load_time = (time.time() - start_time) * 1000
            self._record_load_time(load_time)
            return data, label

        try:
            # 加载.npy 文件
            data = np.load(self.file_paths[idx])
            label = self.labels[idx]

            # 确保数据类型为 float16
            if data.dtype != np.float16:
                data = data.astype(np.float16)

            # 转换为 torch tensor (保持 float16)
            data = torch.from_numpy(data).to(torch.float16)

            # 添加通道维度 (H, W) -> (1, H, W)
            data = data.unsqueeze(0)

            # 应用变换
            if self.transform:
                data = self.transform(data)

            load_time = (time.time() - start_time) * 1000
            self._record_load_time(load_time)
            return data, label

        except Exception as e:
            print(f"Error loading {self.file_paths[idx]}: {e}")
            load_time = (time.time() - start_time) * 1000
            self._record_load_time(load_time)
            # 返回空样本（为了训练继续）
            return torch.zeros(1, 543, 512, dtype=torch.float16), 0

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
):
    """
    划分数据集（FP16 版本，支持多线程预加载）

    Args:
        data_dir: 数据根目录（如./Data）
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_state: 随机种子
        full_load: 是否全量加载到内存
        num_workers: 预加载使用的线程数（None表示自动检测）

    Returns:
        train_dataset, validate_dataset, test_dataset, labels__
    """
    file_paths = []
    labels = []
    labels__ = []
    label_map = {}
    label_index = 0

    def natural_sort_key(text):
        """自然排序键：支持 0,1,2,...,10 的数字顺序，同时兼容非数字字符串。"""
        return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]

    # 读取所有文件路径和标签
    for label_folder in sorted(os.listdir(data_dir), key=natural_sort_key):
        label_folder_path = os.path.join(data_dir, label_folder)
        if os.path.isdir(label_folder_path):
            labels__.append(label_folder)
            for file_name in os.listdir(label_folder_path):
                if file_name.endswith(".npy"):
                    file_path = os.path.join(label_folder_path, file_name)
                    file_paths.append(file_path)
                    labels.append(label_folder)
            label_map[label_folder] = label_index
            label_index += 1

    # 标签转换为索引
    labels = [label_map[label] for label in labels]

    print(f"类别标签映射：{labels__}")
    print(f"总样本数：{len(file_paths)}")

    # 两步划分法
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        file_paths,
        labels,
        test_size=(1 - train_ratio),
        random_state=random_state,
        stratify=labels,  # 保持类别比例
    )

    val_test_ratio = test_ratio / (test_ratio + val_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=val_test_ratio,
        random_state=random_state,
        stratify=temp_labels,
    )

    print(f"训练集：{len(train_paths)} 样本")
    print(f"验证集：{len(val_paths)} 样本")
    print(f"测试集：{len(test_paths)} 样本")

    # 创建数据集（FP16 版本）
    train_dataset = NPYDataset(
        train_paths, train_labels, full_load=full_load, num_workers=num_workers
    )
    validate_dataset = NPYDataset(
        val_paths, val_labels, full_load=full_load, num_workers=num_workers
    )
    test_dataset = NPYDataset(
        test_paths, test_labels, full_load=full_load, num_workers=num_workers
    )

    return train_dataset, validate_dataset, test_dataset, labels__
