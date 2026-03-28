#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""剪枝后微调训练。"""

import copy
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from base_model.lr_scheduler import WarmupCosineAnnealingLR
from base_model.utils import configure_cudnn
from pruning.topology import build_topology_metadata
from pruning.utils import get_raw_model, load_state_dict_safely, to_repo_relative_path


def _build_pruning_checkpoint(
    model,
    epoch,
    best_acc,
    best_val_loss,
    train_context,
    checkpoint_meta,
    pruning_meta,
    input_tensor_meta,
):
    raw_model = get_raw_model(model)
    topology_meta = build_topology_metadata(raw_model)
    model_kwargs = dict(checkpoint_meta.get("model_kwargs", {}))

    checkpoint = {
        "model_state_dict": raw_model.state_dict(),
        "epoch": int(epoch),
        "best_acc": float(best_acc),
        "best_val_loss": float(best_val_loss),
        "train_context": train_context,
        "model_structure": {
            "model_structure_version": 1,
            "model_name": checkpoint_meta["model_name"],
            "model_class": raw_model.__class__.__name__,
            "model_kwargs": model_kwargs,
            "include_top": bool(getattr(raw_model, "include_top", True)),
            "in_channels": int(raw_model.conv1.in_channels) if hasattr(raw_model, "conv1") else None,
            "init_channels": int(raw_model.conv1.out_channels) if hasattr(raw_model, "conv1") else None,
            "input_tensor_meta": input_tensor_meta,
            "channel_cfg": topology_meta["channel_cfg"],
            "architecture_signature": topology_meta["architecture_signature"],
        },
        "pruning_meta": pruning_meta,
    }
    return checkpoint


def append_round_best_info(best_info_path, round_index, summary):
    with open(best_info_path, "a", encoding="utf-8") as f:
        f.write(
            f"Round: {round_index}, "
            f"Best Validation Accuracy: {summary['best_acc']:.4f}, "
            f"Best Validation Loss: {summary['best_val_loss']:.4f}, "
            f"Best Epoch: {summary['best_epoch']}\n"
        )


def finetune_model(
    model,
    device,
    train_loader,
    validate_loader,
    val_num,
    args,
    folder_path,
    checkpoint_meta,
    pruning_meta,
    initial_val_metrics,
    round_index,
    save_checkpoint=False,
):
    os.makedirs(folder_path, exist_ok=True)
    writer = SummaryWriter(os.path.join(folder_path, "runs"))

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    configure_cudnn(args)
    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

    train_steps_per_epoch = max(len(train_loader), 1)
    total_train_steps = max(args.finetune_epochs * train_steps_per_epoch, 1)
    scheduler = WarmupCosineAnnealingLR(
        optimizer,
        total_steps=total_train_steps,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        min_lr=args.min_lr,
    )

    train_context = {
        "stage": "pruning_finetune",
        "source_checkpoint": checkpoint_meta["resolved_checkpoint_path"],
        "checkpoint_link_path": checkpoint_meta["checkpoint_link_path"],
        "resolved_checkpoint_path": checkpoint_meta["resolved_checkpoint_path"],
        "model_name": checkpoint_meta["model_name"],
        "round_index": int(round_index),
        "class_num": checkpoint_meta["model_kwargs"].get("num_classes", 24),
        "finetune_epochs": args.finetune_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps": args.warmup_steps,
        "min_lr": args.min_lr,
        "data_dtype": args.data_dtype,
        "full_load": args.full_load,
    }

    save_path = os.path.join(folder_path, args.model_path)
    best_acc = float(initial_val_metrics["acc"])
    best_val_loss = float(initial_val_metrics["loss"])
    best_epoch = 0
    input_tensor_meta = checkpoint_meta.get("input_tensor_meta")
    best_state_dict = copy.deepcopy(get_raw_model(model).state_dict())

    global_step = 0
    for epoch in range(args.finetune_epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)

        for images, labels in train_bar:
            images = images.to(device)
            labels = labels.to(device)

            if input_tensor_meta is None:
                input_tensor_meta = {
                    "batch_shape_nchw": list(images.shape),
                    "sample_shape_chw": list(images.shape[1:]),
                    "dtype": str(images.dtype),
                    "device_type": str(images.device.type),
                }

            optimizer.zero_grad()
            with autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = loss_function(outputs, labels)

            if torch.cuda.is_available():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("Learning_Rate", current_lr, global_step)
            running_loss += loss.item()
            train_bar.desc = (
                f"Prune FT epoch[{epoch+1}/{args.finetune_epochs}] "
                f"loss: {loss:.3f} lr: {current_lr:.2e}"
            )
            global_step += 1

        model.eval()
        val_loss = 0.0
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for images, labels in val_bar:
                images = images.to(device)
                labels = labels.to(device)
                with autocast("cuda", enabled=torch.cuda.is_available()):
                    outputs = model(images)
                    loss = loss_function(outputs, labels)
                predictions = torch.argmax(outputs, dim=1)
                acc += torch.eq(predictions, labels).sum().item()
                val_loss += loss.item()
                val_bar.desc = f"Prune Valid epoch[{epoch+1}/{args.finetune_epochs}]"

        train_loss_epoch = running_loss / train_steps_per_epoch
        val_loss_epoch = val_loss / max(len(validate_loader), 1)
        val_acc_epoch = acc / max(val_num, 1)

        writer.add_scalar("Loss/train", train_loss_epoch, epoch)
        writer.add_scalar("Loss/val", val_loss_epoch, epoch)
        writer.add_scalar("Acc/val", val_acc_epoch, epoch)

        if val_acc_epoch > best_acc or (
            val_acc_epoch == best_acc and val_loss_epoch < best_val_loss
        ):
            best_acc = val_acc_epoch
            best_val_loss = val_loss_epoch
            best_epoch = epoch + 1
            best_state_dict = copy.deepcopy(get_raw_model(model).state_dict())

    writer.close()

    raw_model = get_raw_model(model)
    success = load_state_dict_safely(raw_model, best_state_dict, strict=True)
    if not success:
        raise RuntimeError("无法重新加载最佳剪枝模型权重")

    checkpoint_path = None
    if save_checkpoint:
        checkpoint = _build_pruning_checkpoint(
            model=raw_model,
            epoch=best_epoch - 1 if best_epoch > 0 else -1,
            best_acc=best_acc,
            best_val_loss=best_val_loss,
            train_context=train_context,
            checkpoint_meta=checkpoint_meta,
            pruning_meta=pruning_meta,
            input_tensor_meta=input_tensor_meta,
        )
        torch.save(checkpoint, save_path)
        checkpoint_path = to_repo_relative_path(save_path)

    return model, {
        "round_index": int(round_index),
        "best_acc": best_acc,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "checkpoint_path": checkpoint_path,
    }


def save_pruned_checkpoint_without_finetune(
    model,
    device,
    folder_path,
    args,
    checkpoint_meta,
    pruning_meta,
    metrics,
):
    os.makedirs(folder_path, exist_ok=True)
    save_path = os.path.join(folder_path, args.model_path)
    checkpoint = _build_pruning_checkpoint(
        model=model,
        epoch=-1,
        best_acc=metrics["acc"],
        best_val_loss=metrics["loss"],
        train_context={
            "stage": "pruning_only",
            "source_checkpoint": checkpoint_meta["resolved_checkpoint_path"],
            "checkpoint_link_path": checkpoint_meta["checkpoint_link_path"],
            "resolved_checkpoint_path": checkpoint_meta["resolved_checkpoint_path"],
            "model_name": checkpoint_meta["model_name"],
            "batch_size": args.batch_size,
            "data_dtype": args.data_dtype,
            "full_load": args.full_load,
        },
        checkpoint_meta=checkpoint_meta,
        pruning_meta=pruning_meta,
        input_tensor_meta=checkpoint_meta.get("input_tensor_meta"),
    )
    torch.save(checkpoint, save_path)
    return to_repo_relative_path(save_path)
