#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.npy 数据集 2D ResNet 训练脚本（FP16 全流程版本）
使用轻量级架构，GPU 加速，内存监控，过拟合缓解，自动混合精度 (AMP)
"""

import os
import sys
import gc
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import umap
from sklearn.decomposition import PCA

plt.rcParams["font.sans-serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False

# 导入项目模块
from dataset_npy import data_set_split
from resnet_lightweight import resnet6_2d, resnet10_2d, resnet14_2d
from confusionMatrix import ConfusionMatrix
from lr_scheduler import WarmupCosineAnnealingLR, plot_lr_schedule


def release_gpu_memory():
    """释放 GPU 内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_gpu_memory_info():
    """获取 GPU 内存使用信息"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        cached = torch.cuda.memory_reserved() / (1024**3)
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        return {"allocated_gb": allocated, "cached_gb": cached, "peak_gb": peak}
    return {"allocated_gb": 0, "cached_gb": 0, "peak_gb": 0}


def print_training_summary(table_title, train_loss, val_loss, val_acc, gpu_info, epoch):
    """打印训练摘要"""
    print(f"\n{'='*80}")
    print(f"{table_title} (Epoch {epoch+1})")
    print(f"{'='*80}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Acc: {val_acc:.4f}")
    print(
        f"GPU Memory: Allocated {gpu_info['allocated_gb']:.2f} GB, Cached {gpu_info['cached_gb']:.2f} GB"
    )
    print(f"Peak Memory: {gpu_info['peak_gb']:.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="Train a 2D Lightweight ResNet for .npy Dataset"
    )

    # 训练参数
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数 (默认 60)")
    parser.add_argument("--lr", type=float, default=0.0003, help="学习率 (默认 0.001)")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小 (默认 32)")
    parser.add_argument(
        "--model_path", type=str, default="best_model.pth", help="模型保存路径"
    )
    parser.add_argument("--class_num", type=int, default=24, help="分类数 (默认 24)")

    # 模型选择
    parser.add_argument(
        "--model",
        type=str,
        default="resnet6_2d",
        choices=["resnet6_2d", "resnet10_2d", "resnet14_2d"],
        help="选择轻量级模型 (默认 resnet10_2d",
    )

    # 数据路径
    parser.add_argument("--data_dir", type=str, default="Data", help="数据集路径")

    # 功能开关
    parser.add_argument("--Train", type=bool, default=True, help="是否训练 (默认 True)")
    parser.add_argument("--Test", type=bool, default=True, help="是否测试 (默认 True)")
    parser.add_argument(
        "--TSNE", type=bool, default=True, help="t-SNE 可视化 (默认 False)"
    )

    # 正则化参数
    parser.add_argument(
        "--dropout_p", type=float, default=0.3, help="Dropout 概率 (默认 0.3)"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="权重衰减 (默认 1e-4)"
    )

    # 学习率调度器参数
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="Warmup 占总步数的比例 (默认 0.05, 即 5%%)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Warmup 步数 (如果为 0，则使用 warmup_ratio, 默认 0)",
    )
    parser.add_argument(
        "--min_lr", type=float, default=1e-6, help="最小学习率 (默认 1e-6)"
    )
    parser.add_argument(
        "--plot_lr_schedule",
        type=bool,
        default=True,
        help="是否绘制学习率调度曲线 (默认 True)",
    )

    args = parser.parse_args()
    print(args)

    # 释放 GPU 内存
    release_gpu_memory()

    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
        )

    # 创建输出目录
    folder_name = (
        os.path.basename(args.data_dir)
        if os.path.basename(args.data_dir)
        else "npy_dataset"
    )
    output_path = "./output"
    folder_path = os.path.join(
        output_path,
        f"{args.model}_{folder_name}",
        f"epochs{args.epochs}_bs{args.batch_size}_lr{args.lr}_drop{args.dropout_p}",
    )
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    print(f"\n输出目录: {folder_path}")

    # 数据加载（FP16 版本）
    print("\n开始加载数据 (FP16)...")
    train_dataset, validate_dataset, test_dataset, labels__ = data_set_split(
        args.data_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
    )

    # 数据加载器配置
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    print(f"\n使用 {nw} 个数据加载工作线程")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=nw
    )
    validate_loader = torch.utils.data.DataLoader(
        validate_dataset, batch_size=args.batch_size, shuffle=False, num_workers=nw
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=nw
    )

    train_num = len(train_dataset)
    val_num = len(validate_dataset)
    test_num = len(test_dataset)

    print(f"训练样本数: {train_num}, 验证样本数: {val_num}, 测试样本数: {test_num}")

    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # 模型初始化
    print(f"\n初始化模型: {args.model}")
    model_map = {
        "resnet6_2d": resnet6_2d,
        "resnet10_2d": resnet10_2d,
        "resnet14_2d": resnet14_2d,
    }
    model = model_map[args.model](num_classes=args.class_num, dropout_p=args.dropout_p)
    model.to(device)

    # 打印模型信息
    try:
        from torchsummary import summary

        print("\n模型结构:")
        summary(model, input_size=(1, 543, 512), device=str(device).split(":")[0])
    except Exception as e:
        print(f"无法使用 torchsummary: {e}")
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练循环（FP16 + AMP 版本）
    if args.Train:
        print(f"\n{'='*80}")
        print(f"开始训练 (FP16 + AMP)")
        print(f"{'='*80}")
        print(
            f"训练轮数: {args.epochs}, 批次大小: {args.batch_size}, 学习率: {args.lr}"
        )

        writer = SummaryWriter(folder_path + "/runs")

        # 损失函数和优化器
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        # 初始化 GradScaler 用于自动混合精度
        scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

        # 计算总训练步数
        train_steps_per_epoch = len(train_loader)
        total_train_steps = args.epochs * train_steps_per_epoch

        # 初始化 Warmup + Cosine Annealing 学习率调度器
        scheduler = WarmupCosineAnnealingLR(
            optimizer,
            total_steps=total_train_steps,
            warmup_steps=args.warmup_steps,
            warmup_ratio=args.warmup_ratio,
            min_lr=args.min_lr,
        )

        # 绘制学习率调度曲线（可选）
        if args.plot_lr_schedule:
            lr_plot_path = os.path.join(folder_path, "lr_schedule.png")
            plot_lr_schedule(
                scheduler,
                total_steps=total_train_steps,
                save_path=lr_plot_path,
                title=f"Learning Rate Schedule (Warmup: {scheduler.warmup_steps} steps)",
            )

        best_acc = 0.0
        best_train_losses = []
        best_val_losses = []
        best_val_accs = []
        lr_history = []  # 记录学习率历史

        save_path = os.path.join(folder_path, args.model_path)
        global_step = 0  # 全局步数计数器

        for epoch in range(args.epochs):
            # 训练阶段
            model.train()
            running_loss = 0.0
            train_bar = tqdm(train_loader, file=sys.stdout)
            nan_occurred = False

            for step, data in enumerate(train_bar):
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                # 前向传播：使用 autocast
                with autocast("cuda", enabled=torch.cuda.is_available()):
                    outputs = model(images)
                    loss = loss_function(outputs, labels)

                # 反向传播和优化：使用 scaler
                if torch.cuda.is_available():
                    scaler.scale(loss).backward()

                    # 梯度裁剪（数值稳定性措施）
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                # 更新学习率（每步更新）
                scheduler.step()
                current_lr = optimizer.param_groups[0]["lr"]
                lr_history.append(current_lr)

                # 记录学习率到 TensorBoard
                writer.add_scalar("Learning_Rate", current_lr, global_step)

                running_loss += loss.item()

                # 检查 NaN
                if torch.isnan(loss):
                    nan_occurred = True
                    print(f"\n⚠️  NaN loss detected at epoch {epoch+1}, step {step}")

                train_bar.desc = f"Train epoch[{epoch+1}/{args.epochs}] loss: {loss:.3f} lr: {current_lr:.2e}"

                global_step += 1

            # 验证阶段
            model.eval()
            acc = 0.0
            val_loss = 0.0

            with torch.no_grad():
                val_bar = tqdm(validate_loader, file=sys.stdout)
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    val_images, val_labels = val_images.to(device), val_labels.to(
                        device
                    )

                    with autocast("cuda", enabled=torch.cuda.is_available()):
                        outputs = model(val_images)
                        val_loss_batch = loss_function(outputs, val_labels)

                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels).sum().item()
                    val_loss += val_loss_batch.item()

                    val_bar.desc = f"Valid epoch[{epoch+1}/{args.epochs}]"

            train_loss_epoch = running_loss / train_steps_per_epoch
            val_loss_epoch = val_loss / len(validate_loader)
            val_accurate = acc / val_num

            best_train_losses.append(train_loss_epoch)
            best_val_losses.append(val_loss_epoch)
            best_val_accs.append(val_accurate)

            # 获取 GPU 信息
            gpu_info = get_gpu_memory_info()

            # 打印摘要
            print_training_summary(
                "训练摘要 (FP16)",
                train_loss_epoch,
                val_loss_epoch,
                val_accurate,
                gpu_info,
                epoch,
            )

            if nan_occurred:
                print("⚠️  注意：本轮训练中检测到 NaN")

            # TensorBoard 记录
            writer.add_scalar("Loss/train", train_loss_epoch, epoch)
            writer.add_scalar("Loss/val", val_loss_epoch, epoch)
            writer.add_scalar("Acc/val", val_accurate, epoch)
            writer.add_scalar("GPU/allocated_gb", gpu_info["allocated_gb"], epoch)
            writer.add_scalar("GPU/peak_gb", gpu_info["peak_gb"], epoch)

            # 保存最佳模型（同时保存 scaler 状态用于恢复训练）
            if val_accurate > best_acc:
                best_acc = val_accurate
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "scaler_state_dict": (
                        scaler.state_dict() if torch.cuda.is_available() else None
                    ),
                    "epoch": epoch,
                    "best_acc": best_acc,
                }
                torch.save(checkpoint, save_path)
                print(f"\n✓ 保存最佳模型 (Acc: {best_acc:.4f}, 包含 AMP scaler 状态)")

        print(f"\n{'='*80}")
        print(f"训练完成")
        print(f"最佳验证准确率: {best_acc:.4f}")
        print(f"{'='*80}")
        writer.close()

        # 加载最佳模型（包含 AMP scaler 状态）
        checkpoint = torch.load(save_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        # 绘制训练曲线（包含学习率）
        plt.figure(figsize=(16, 4))

        # 损失曲线
        plt.subplot(1, 3, 1)
        plt.plot(best_train_losses, label="Train Loss")
        plt.plot(best_val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training Loss Curves")

        # 准确率曲线
        plt.subplot(1, 3, 2)
        plt.plot(best_val_accs, label="Val Acc", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Validation Accuracy")

        # 学习率曲线
        plt.subplot(1, 3, 3)
        if lr_history:
            steps = list(range(len(lr_history)))
            plt.plot(
                steps, lr_history, label="Learning Rate", color="green", linewidth=2
            )
            # 标记 warmup 结束点
            if scheduler.warmup_steps > 0 and scheduler.warmup_steps < len(lr_history):
                plt.axvline(
                    x=scheduler.warmup_steps,
                    color="r",
                    linestyle="--",
                    label=f"Warmup End (Step {scheduler.warmup_steps})",
                )
            plt.xlabel("Step")
            plt.ylabel("Learning Rate")
            plt.yscale("log")
            plt.legend()
            plt.title("Learning Rate Schedule")

        plt.tight_layout()
        plt.savefig(
            os.path.join(folder_path, "training_curves.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # 测试阶段（FP16 版本）
    if args.Test:
        print(f"\n{'='*80}")
        print("开始测试 (FP16)")
        print(f"{'='*80}")

        model_path = os.path.join(folder_path, args.model_path)
        assert os.path.exists(model_path), f"找不到模型文件: {model_path}"

        # 加载模型（同时兼容旧格式和新 checkpoint 格式）
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)

        confusion = ConfusionMatrix(num_classes=args.class_num, labels=labels__)
        model.eval()

        with torch.no_grad(), autocast("cuda", enabled=torch.cuda.is_available()):
            for val_data in tqdm(test_loader):
                val_images, val_labels = val_data
                val_images = val_images.to(device)
                outputs = model(val_images)
                outputs = torch.softmax(outputs, dim=1)
                outputs = torch.argmax(outputs, dim=1)
                confusion.update(outputs.to("cpu").numpy(), val_labels.numpy())

        confusion.plot(folder_path)
        confusion.summary()

    # UMAP 可视化（内存优化版）
    if args.TSNE:
        print(f"\n{'='*80}")
        print("UMAP 可视化（内存优化版）")
        print(f"{'='*80}")

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

        print("执行 UMAP 降维...")
        umap_embedding = umap.UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
            low_memory=True,
        )
        X_umap = umap_embedding.fit_transform(all_features_pca)

        cmaps = "viridis"
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            X_umap[:, 0], X_umap[:, 1], c=all_labels, cmap=cmaps, alpha=0.6
        )
        plt.colorbar(scatter)
        plt.title("UMAP Visualization (Memory-Optimized)")
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

    print(f"\n{'='*80}")
    print("所有任务完成")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
