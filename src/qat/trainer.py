#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QAT 微调训练。"""

import copy
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from base_model.lr_scheduler import WarmupCosineAnnealingLR
from base_model.utils import configure_cudnn
from qat.evaluator import evaluate_model
from qat.quantization import maybe_apply_qat_freeze_policy
from qat.utils import get_raw_model, load_state_dict_safely


def _build_qat_checkpoint(
    model,
    epoch,
    best_acc,
    best_val_loss,
    train_context,
    checkpoint_meta,
    quantization_meta,
):
    raw_model = get_raw_model(model)
    model_structure = copy.deepcopy(checkpoint_meta["model_structure"])
    model_structure["model_structure_version"] = model_structure.get("model_structure_version", 1)

    checkpoint = {
        "model_state_dict": raw_model.state_dict(),
        "epoch": int(epoch),
        "best_acc": float(best_acc),
        "best_val_loss": float(best_val_loss),
        "train_context": train_context,
        "model_structure": model_structure,
        "quantization_meta": quantization_meta,
    }
    return checkpoint


def write_best_qat_info(best_info_path, best_acc, best_val_loss, best_epoch):
    with open(best_info_path, "a", encoding="utf-8") as f:
        f.write(
            "Best Validation Accuracy: "
            f"{best_acc:.4f}, Best Validation Loss: {best_val_loss:.4f} "
            f"at Epoch: {best_epoch}\n"
        )


def finetune_qat_model(
    model,
    device,
    train_loader,
    validate_loader,
    val_num,
    args,
    folder_path,
    checkpoint_meta,
    quantization_meta,
    initial_val_metrics,
):
    os.makedirs(folder_path, exist_ok=True)
    writer = SummaryWriter(os.path.join(folder_path, "runs"))

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    configure_cudnn(args)

    train_steps_per_epoch = max(len(train_loader), 1)
    total_train_steps = max(args.qat_epochs * train_steps_per_epoch, 1)
    scheduler = WarmupCosineAnnealingLR(
        optimizer,
        total_steps=total_train_steps,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        min_lr=args.min_lr,
    )

    train_context = {
        "stage": "qat_finetune",
        "source_pruning_checkpoint_path": checkpoint_meta["source_pruning_checkpoint_path"],
        "model_name": checkpoint_meta["model_name"],
        "class_num": checkpoint_meta["model_kwargs"].get("num_classes", 24),
        "qat_epochs": args.qat_epochs,
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
    best_info_path = os.path.join(folder_path, "best_qat_info.txt")
    best_acc = float(initial_val_metrics["acc"])
    best_val_loss = float(initial_val_metrics["loss"])
    best_epoch = 0
    best_state_dict = copy.deepcopy(get_raw_model(model).state_dict())
    freeze_state = {"bn_frozen": False, "observer_frozen": False}

    global_step = 0
    for epoch in range(args.qat_epochs):
        model.train()
        freeze_state = maybe_apply_qat_freeze_policy(model, epoch, args.qat_epochs, freeze_state)
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)

        for images, labels in train_bar:
            images = images.to(device=device, dtype=torch.float32)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("Learning_Rate", current_lr, global_step)
            running_loss += loss.item()
            train_bar.desc = (
                f"QAT epoch[{epoch+1}/{args.qat_epochs}] "
                f"loss: {loss:.3f} lr: {current_lr:.2e}"
            )
            global_step += 1

        train_loss_epoch = running_loss / train_steps_per_epoch
        val_metrics = evaluate_model(
            model=model,
            device=device,
            dataloader=validate_loader,
            num_samples=val_num,
        )

        writer.add_scalar("Loss/train", train_loss_epoch, epoch)
        writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("Acc/val", val_metrics["acc"], epoch)

        if val_metrics["acc"] > best_acc or (
            val_metrics["acc"] == best_acc and val_metrics["loss"] < best_val_loss
        ):
            best_acc = val_metrics["acc"]
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1
            best_state_dict = copy.deepcopy(get_raw_model(model).state_dict())
            write_best_qat_info(
                best_info_path=best_info_path,
                best_acc=best_acc,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
            )

    writer.close()

    raw_model = get_raw_model(model)
    success = load_state_dict_safely(raw_model, best_state_dict, strict=True)
    if not success:
        raise RuntimeError("无法重新加载最佳 QAT 模型权重")

    checkpoint = _build_qat_checkpoint(
        model=raw_model,
        epoch=best_epoch - 1 if best_epoch > 0 else -1,
        best_acc=best_acc,
        best_val_loss=best_val_loss,
        train_context=train_context,
        checkpoint_meta=checkpoint_meta,
        quantization_meta=quantization_meta,
    )
    torch.save(checkpoint, save_path)

    return model, {
        "best_acc": best_acc,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "checkpoint_path": save_path,
    }


def save_prepared_qat_checkpoint_without_finetune(
    model,
    folder_path,
    args,
    checkpoint_meta,
    quantization_meta,
    metrics,
):
    os.makedirs(folder_path, exist_ok=True)
    save_path = os.path.join(folder_path, args.model_path)
    checkpoint = _build_qat_checkpoint(
        model=model,
        epoch=-1,
        best_acc=metrics["acc"],
        best_val_loss=metrics["loss"],
        train_context={
            "stage": "qat_prepare_only",
            "source_pruning_checkpoint_path": checkpoint_meta["source_pruning_checkpoint_path"],
            "model_name": checkpoint_meta["model_name"],
            "batch_size": args.batch_size,
            "data_dtype": args.data_dtype,
            "full_load": args.full_load,
        },
        checkpoint_meta=checkpoint_meta,
        quantization_meta=quantization_meta,
    )
    torch.save(checkpoint, save_path)
    return save_path
