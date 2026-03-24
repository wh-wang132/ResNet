#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""剪枝阶段评估工具。"""

import torch
from torch import nn
from torch.amp import autocast


def count_model_stats(model, example_inputs):
    params = int(sum(param.numel() for param in model.parameters()))
    macs = None
    try:
        import torch_pruning as tp

        macs, params_from_tp = tp.utils.count_ops_and_params(model, example_inputs)
        params = int(params_from_tp)
        macs = int(macs)
    except Exception:
        macs = None
    return {"params": params, "macs": macs}


@torch.no_grad()
def evaluate_model(model, device, dataloader, num_samples, use_amp=True):
    model.eval()
    loss_function = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        with autocast("cuda", enabled=use_amp and torch.cuda.is_available()):
            logits = model(images)
            loss = loss_function(logits, labels)

        predictions = torch.argmax(logits, dim=1)
        total_loss += loss.item()
        total_correct += torch.eq(predictions, labels).sum().item()
        total_seen += labels.size(0)

    avg_loss = total_loss / max(len(dataloader), 1)
    accuracy = total_correct / max(num_samples, total_seen, 1)
    return {
        "loss": float(avg_loss),
        "acc": float(accuracy),
        "samples": int(total_seen),
    }
