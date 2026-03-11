#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试 UMAP 可视化模块
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm

from dataset_npy import data_set_split
from resnet_lightweight import resnet6_2d


def main():
    print("=" * 80)
    print("快速测试 UMAP 可视化模块")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    data_dir = "Data"

    print("\n开始加载数据...")
    train_dataset, validate_dataset, test_dataset, labels__ = data_set_split(
        data_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
    )

    batch_size = 16
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    print(f"测试集大小: {len(test_dataset)}, 批次大小: {batch_size}")

    model = resnet6_2d(num_classes=24, dropout_p=0.3)
    model.to(device)

    model_path = "/root/ResNet/output/resnet6_2d_Data/epochs1_bs64_lr0.0003_drop0.3/best_model.pth"
    if os.path.exists(model_path):
        print(f"\n加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"\n警告: 找不到模型文件 {model_path}, 将使用未训练的模型")

    all_features = []
    all_labels = []
    num_samples = 100
    count = 0

    print("\n提取特征（仅采样前 100 个样本用于快速测试）...")
    model.eval()
    with torch.no_grad():
        for val_data in tqdm(test_loader):
            if count >= num_samples:
                break

            val_images, val_labels_batch = val_data
            val_images = val_images.to(device)

            features_dict = model.get_features(val_images, layer=["layer3"])
            features = features_dict["layer3"]
            features = features.cpu().numpy()
            features = features.reshape(features.shape[0], -1)
            all_features.append(features)
            all_labels.extend(val_labels_batch.numpy())
            count += len(val_labels_batch)

    all_features = np.concatenate(all_features, axis=0)[:num_samples]
    all_labels = np.array(all_labels)[:num_samples]
    print(f"特征矩阵形状: {all_features.shape}")

    print("\n使用 PCA 降低特征维度...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=20, random_state=42)
    all_features_pca = pca.fit_transform(all_features)
    print(f"PCA 降维后形状: {all_features_pca.shape}")

    print("\n执行 UMAP 降维...")
    import umap
    umap_embedding = umap.UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=10,
        min_dist=0.1,
        metric='euclidean',
        low_memory=True
    )
    X_umap = umap_embedding.fit_transform(all_features_pca)

    print("\n生成可视化图...")
    cmaps = "viridis"
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        X_umap[:, 0], X_umap[:, 1], c=all_labels, cmap=cmaps, alpha=0.6
    )
    plt.colorbar(scatter)
    plt.title("UMAP Visualization (Quick Test)")
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

    output_path = "/root/ResNet/umap_test_plot.png"
    plt.savefig(output_path)
    plt.close()
    print(f"\n✓ UMAP 测试图已保存至: {output_path}")
    print("\n" + "=" * 80)
    print("UMAP 测试完成")
    print("=" * 80)


if __name__ == "__main__":
    main()

