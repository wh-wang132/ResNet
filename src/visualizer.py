#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UMAP 可视化模块
包含特征提取和 UMAP 降维可视化功能
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from torch.amp import autocast
from tqdm import tqdm
import umap
from sklearn.decomposition import PCA
import psutil


def visualize_umap(model, device, test_loader, args, folder_path, labels__):
    """
    UMAP 可视化（内存优化版 + 16线程配置）

    Args:
        model: 要使用的模型
        device: 计算设备
        test_loader: 测试数据加载器
        args: 命令行参数
        folder_path: 输出目录路径
        labels__: 类别标签列表
    """
    print(f"\n{'='*80}")
    print("UMAP 可视化（内存优化版 + 16线程配置）")
    print(f"{'='*80}")

    print("配置 UMAP 使用 16 个逻辑线程...")
    print(f"系统 CPU 核心数: {psutil.cpu_count(logical=False)}")
    print(f"系统逻辑线程数: {psutil.cpu_count(logical=True)}")

    all_features = []
    all_labels = []

    model.eval()
    with torch.no_grad(), autocast("cuda", enabled=torch.cuda.is_available()):
        for val_data in tqdm(test_loader):
            val_images, val_labels = val_data
            val_images = val_images.to(device)

            features_dict = model.get_features(val_images, layer=["layer3"])
            features = features_dict["layer3"]
            features = features.cpu().numpy()
            features = features.reshape(features.shape[0], -1)
            all_features.append(features)
            all_labels.extend(val_labels.numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.array(all_labels)

    print("内存优化：首先使用 PCA 降低特征维度...")
    pca = PCA(n_components=50, random_state=42)
    all_features_pca = pca.fit_transform(all_features)
    print(
        f"原始维度: {all_features.shape[1]}, PCA降维后维度: {all_features_pca.shape[1]}"
    )

    print("执行 UMAP 降维（使用 16 个逻辑线程）...")
    umap_embedding = umap.UMAP(
        n_components=2,
        random_state=None,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        low_memory=True,
        n_jobs=16,
        verbose=True,
    )

    print(f"UMAP 线程配置: n_jobs={umap_embedding.n_jobs}")
    X_umap = umap_embedding.fit_transform(all_features_pca)

    cmaps = "viridis"
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        X_umap[:, 0], X_umap[:, 1], c=all_labels, cmap=cmaps, alpha=0.6
    )
    plt.colorbar(scatter)
    plt.title("UMAP Visualization (Memory-Optimized, 16 Threads)")
    plt.xlabel("UMAP Feature 1")
    plt.ylabel("UMAP Feature 2")

    cmap = cm.get_cmap(cmaps, len(labels__))
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=label,
            markerfacecolor=cmap(i),
            markersize=10,
        )
        for i, label in enumerate(labels__)
    ]
    plt.legend(handles=legend_elements, ncol=3)

    plt.savefig(os.path.join(folder_path, "umap_plot.png"))
    plt.close()
    print(f"✓ UMAP可视化图已保存至: {os.path.join(folder_path, 'umap_plot.png')}")

