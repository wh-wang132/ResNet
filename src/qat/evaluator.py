#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QAT 阶段评估工具。"""

from pruning.evaluator import (
    count_model_stats,
    evaluate_model as _evaluate_model,
    evaluate_model_with_confusion_matrix as _evaluate_model_with_confusion_matrix,
)


def evaluate_model(model, device, dataloader, num_samples, progress_desc=None):
    return _evaluate_model(
        model=model,
        device=device,
        dataloader=dataloader,
        num_samples=num_samples,
        use_amp=False,
        progress_desc=progress_desc,
    )


def evaluate_model_with_confusion_matrix(
    model,
    device,
    dataloader,
    num_samples,
    labels,
    folder_path,
    progress_desc=None,
):
    return _evaluate_model_with_confusion_matrix(
        model=model,
        device=device,
        dataloader=dataloader,
        num_samples=num_samples,
        labels=labels,
        folder_path=folder_path,
        use_amp=False,
        progress_desc=progress_desc,
    )


__all__ = [
    "count_model_stats",
    "evaluate_model",
    "evaluate_model_with_confusion_matrix",
]
