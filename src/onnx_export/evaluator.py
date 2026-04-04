#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ONNX 阶段评估工具。"""

from __future__ import annotations

import sys

import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from tqdm import tqdm

from base_model.confusionMatrix import ConfusionMatrix


def create_onnx_session(onnx_path):
    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        try:
            session = ort.InferenceSession(
                onnx_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            return session, session.get_providers()
        except Exception:
            pass

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return session, session.get_providers()


@torch.no_grad()
def evaluate_torch_model(
    model,
    device,
    dataloader,
    num_samples,
    input_dtype=torch.float32,
    progress_desc=None,
):
    model.eval()
    loss_function = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    eval_loader = dataloader
    if progress_desc is not None:
        eval_loader = tqdm(dataloader, file=sys.stdout, desc=progress_desc)

    for images, labels in eval_loader:
        images = images.to(device=device, dtype=input_dtype)
        labels = labels.to(device)
        logits = model(images)
        loss = loss_function(logits.float(), labels)
        predictions = torch.argmax(logits, dim=1)
        total_loss += loss.item()
        total_correct += torch.eq(predictions, labels).sum().item()
        total_seen += labels.size(0)

    return {
        "loss": float(total_loss / max(len(dataloader), 1)),
        "acc": float(total_correct / max(num_samples, total_seen, 1)),
        "samples": int(total_seen),
    }


def _evaluate_onnx_model_core(
    session,
    dataloader,
    num_samples,
    input_dtype=np.float32,
    batch_callback=None,
    progress_desc=None,
):
    loss_function = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    input_name = session.get_inputs()[0].name

    eval_loader = dataloader
    if progress_desc is not None:
        eval_loader = tqdm(dataloader, file=sys.stdout, desc=progress_desc)

    for images, labels in eval_loader:
        ort_inputs = {input_name: images.cpu().numpy().astype(input_dtype, copy=False)}
        logits = session.run(None, ort_inputs)[0]
        logits_tensor = torch.from_numpy(logits).to(torch.float32)
        labels_tensor = labels.to(torch.long)

        loss = loss_function(logits_tensor, labels_tensor)
        predictions = torch.argmax(logits_tensor, dim=1)
        if batch_callback is not None:
            batch_callback(predictions, labels_tensor)
        total_loss += loss.item()
        total_correct += torch.eq(predictions, labels_tensor).sum().item()
        total_seen += labels_tensor.size(0)

    return {
        "loss": float(total_loss / max(len(dataloader), 1)),
        "acc": float(total_correct / max(num_samples, total_seen, 1)),
        "samples": int(total_seen),
    }


def evaluate_onnx_model(
    session,
    dataloader,
    num_samples,
    input_dtype=np.float32,
    progress_desc=None,
):
    return _evaluate_onnx_model_core(
        session=session,
        dataloader=dataloader,
        num_samples=num_samples,
        input_dtype=input_dtype,
        progress_desc=progress_desc,
    )


def evaluate_onnx_model_with_confusion_matrix(
    session,
    dataloader,
    num_samples,
    input_dtype,
    labels,
    folder_path,
    progress_desc=None,
):
    confusion = ConfusionMatrix(num_classes=len(labels), labels=labels)

    def update_confusion(predictions, batch_labels):
        confusion.update(
            predictions.to("cpu").numpy(),
            batch_labels.to("cpu").numpy(),
        )

    metrics = _evaluate_onnx_model_core(
        session=session,
        dataloader=dataloader,
        num_samples=num_samples,
        input_dtype=input_dtype,
        batch_callback=update_confusion,
        progress_desc=progress_desc,
    )
    confusion.plot(folder_path)
    return metrics
