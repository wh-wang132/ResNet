#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练模块
包含完整的训练和验证流程，支持 FP16 和 AMP 混合精度
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil

from lr_scheduler import WarmupCosineAnnealingLR, plot_lr_schedule
from utils import (
    get_gpu_memory_info,
    print_training_summary,
    configure_cudnn,
    compile_model,
)


def train_model(
    model,
    device,
    train_loader,
    validate_loader,
    args,
    folder_path,
    val_num,
):
    """
    训练模型的主函数

    Args:
        model: 要训练的模型
        device: 计算设备
        train_loader: 训练数据加载器
        validate_loader: 验证数据加载器
        args: 命令行参数
        folder_path: 输出目录路径
        val_num: 验证样本数

    Returns:
        model: 训练后的模型（加载最佳权重）
    """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)
    print(f"\n{'='*80}")
    print(f"开始训练 (FP16 + AMP)")
    print(f"{'='*80}")
    print(f"训练轮数: {args.epochs}, 批次大小: {args.batch_size}, 学习率: {args.lr}")
    writer = SummaryWriter(folder_path + "/runs")

    # 损失函数和优化器
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # 配置 cuDNN 性能优化
    configure_cudnn(args)

    # 模型编译（训练前完成）
    model = compile_model(
        model=model,
        args=args,
        device=device,
        loss_function=loss_function,
        optimizer=optimizer,
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
    best_epoch = 0
    best_train_losses = []
    best_val_losses = []
    best_val_accs = []
    lr_history = []

    save_path = os.path.join(folder_path, args.model_path)
    best_acc_info_path = os.path.join(folder_path, "best_val_acc_info.txt")
    global_step = 0
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
                val_images, val_labels = val_images.to(device), val_labels.to(device)

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
            best_epoch = epoch + 1
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "scaler_state_dict": (
                    scaler.state_dict() if torch.cuda.is_available() else None
                ),
                "epoch": epoch,
                "best_acc": best_acc,
            }
            torch.save(checkpoint, save_path)
            print(
                f"\n✓ 保存最佳模型 (Acc: {best_acc:.4f} at Epoch {best_epoch}, 包含 AMP scaler 状态)"
            )

            # 记录最优验证准确率信息到文本文件
            with open(best_acc_info_path, "a", encoding="utf-8") as f:
                f.write(
                    f"Best Validation Accuracy: {best_acc:.4f} at Epoch: {best_epoch}\n"
                )

    print(f"\n{'='*80}")
    print(f"训练完成")
    print(f"最佳验证准确率: {best_acc:.4f} (Epoch: {best_epoch})")
    print(f"{'='*80}")
    writer.close()

    # 加载最佳模型
    checkpoint = torch.load(save_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 绘制训练曲线（包含学习率）
    plot_training_curves(
        best_train_losses,
        best_val_losses,
        best_val_accs,
        lr_history,
        scheduler,
        folder_path,
    )

    return model


def plot_training_curves(
    train_losses,
    val_losses,
    val_accs,
    lr_history,
    scheduler,
    folder_path,
):
    """
    绘制训练曲线（2×2布局，包含Val错误率曲线）

    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        val_accs: 验证准确率列表
        lr_history: 学习率历史列表
        scheduler: 学习率调度器
        folder_path: 保存路径
    """
    plt.figure(figsize=(14, 12))
    epochs = list(range(1, len(train_losses) + 1))

    # 计算验证错误率 (1 - 准确率)，并添加微小正值防止log(0)
    epsilon = 1e-8
    val_errors = [max(1.0 - acc, epsilon) for acc in val_accs]

    # 统一样式设置
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    # 子图1: 训练损失和验证损失
    plt.subplot(2, 2, 1)
    plt.plot(
        epochs,
        train_losses,
        label="Train Loss",
        color="blue",
        linewidth=2,
        marker="o",
        markersize=4,
        alpha=0.8,
    )
    plt.plot(
        epochs,
        val_losses,
        label="Val Loss",
        color="red",
        linewidth=2,
        marker="s",
        markersize=4,
        alpha=0.8,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss")
    plt.grid(True, alpha=0.3, linestyle="--")

    # 子图2: 验证准确率
    plt.subplot(2, 2, 2)
    plt.plot(
        epochs,
        val_accs,
        label="Val Accuracy",
        color="orange",
        linewidth=2,
        marker="^",
        markersize=4,
        alpha=0.8,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Validation Accuracy")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.ylim([0, 1.05])

    # 子图3: 验证错误率（对数坐标轴）
    plt.subplot(2, 2, 3)
    plt.plot(
        epochs,
        val_errors,
        label="Val Error Rate",
        color="purple",
        linewidth=2,
        marker="D",
        markersize=4,
        alpha=0.8,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Error Rate (Log Scale)")
    plt.yscale("log")
    plt.legend()
    plt.title("Validation Error Rate (Logarithmic Scale)")
    plt.grid(True, alpha=0.3, linestyle="--", which="both")

    # 子图4: 学习率曲线
    plt.subplot(2, 2, 4)
    if lr_history:
        steps = list(range(len(lr_history)))
        plt.plot(
            steps,
            lr_history,
            label="Learning Rate",
            color="green",
            linewidth=2,
            alpha=0.8,
        )
        # 标记 warmup 结束点
        if scheduler.warmup_steps > 0 and scheduler.warmup_steps < len(lr_history):
            plt.axvline(
                x=scheduler.warmup_steps,
                color="r",
                linestyle="--",
                linewidth=1.5,
                label=f"Warmup End (Step {scheduler.warmup_steps})",
            )
        plt.xlabel("Step")
        plt.ylabel("Learning Rate (Log Scale)")
        plt.yscale("log")
        plt.legend()
        plt.title("Learning Rate Schedule")
        plt.grid(True, alpha=0.3, linestyle="--", which="both")

    plt.tight_layout(pad=3.0)
    plt.savefig(
        os.path.join(folder_path, "training_curves.png"),
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
