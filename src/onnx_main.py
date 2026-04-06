#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ONNX 导出统一入口。"""

import json

import numpy as np
import torch

from base_model.dataset import data_set_split
from base_model.plotting import configure_matplotlib
from base_model.utils import create_optimized_dataloader
from onnx_export.args import parse_args
from onnx_export.evaluator import (
    create_onnx_session,
    evaluate_onnx_model_with_confusion_matrix,
    evaluate_torch_model,
)
from onnx_export.exporter import (
    build_branch_artifacts,
    build_metric_delta,
    inspect_branch_checkpoint,
    resolve_branch_opset_version,
)
from onnx_export.output import create_output_directory, save_summary
from qat.utils import release_gpu_memory, setup_device, to_repo_relative_path

configure_matplotlib()


def main():
    args = parse_args()
    print(args)

    release_gpu_memory()
    device = setup_device()

    if args.branch == "pruning_fp16":
        source_device = device
        dataset_dtype = "fp32"
        onnx_input_dtype = np.float16
    else:
        source_device = torch.device("cpu")
        dataset_dtype = "fp32"
        onnx_input_dtype = np.float32
    actual_opset_version = resolve_branch_opset_version(args.branch, args.opset_version)

    checkpoint_meta = inspect_branch_checkpoint(
        branch=args.branch,
        checkpoint_path=args.checkpoint,
        device=source_device,
    )
    folder_path = create_output_directory(args.branch, checkpoint_meta)
    model, checkpoint_meta, checkpoint, export_shape, onnx_path, export_meta = build_branch_artifacts(
        branch=args.branch,
        checkpoint_path=args.checkpoint,
        device=source_device,
        folder_path=folder_path,
        opset_version=actual_opset_version,
    )

    train_dataset, validate_dataset, test_dataset, labels__ = data_set_split(
        args.data_dir,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        full_load=args.full_load,
        num_workers=args.num_workers,
        data_dtype=dataset_dtype,
    )

    source_test_metrics = None
    onnx_test_metrics = None
    ort_providers = None
    if args.evaluate_test:
        test_loader, _ = create_optimized_dataloader(
            test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=False,
            drop_last=False,
            loader_name="ONNX 测试集 DataLoader",
        )
        source_test_metrics = evaluate_torch_model(
            model=model,
            device=source_device,
            dataloader=test_loader,
            num_samples=len(test_dataset),
            input_dtype=torch.float32,
            progress_desc="ONNX source test",
        )
        ort_session, ort_providers = create_onnx_session(onnx_path)
        onnx_test_metrics = evaluate_onnx_model_with_confusion_matrix(
            session=ort_session,
            dataloader=test_loader,
            num_samples=len(test_dataset),
            input_dtype=onnx_input_dtype,
            labels=labels__,
            folder_path=folder_path,
            progress_desc="ONNX exported test",
        )

    summary = {
        "branch": args.branch,
        "model_name": checkpoint_meta["model_name"],
        "labels": labels__,
        "source_checkpoint_path": to_repo_relative_path(args.checkpoint),
        "opset_version": actual_opset_version,
        "example_input_shape": export_shape,
        "onnx_path": to_repo_relative_path(onnx_path),
        "source_test_metrics": source_test_metrics,
        "onnx_test_metrics": onnx_test_metrics,
        "metric_delta": build_metric_delta(source_test_metrics, onnx_test_metrics),
        "export_meta": {
            **export_meta,
            "ort_providers": ort_providers,
        },
    }

    summary_path = save_summary(folder_path, summary)
    print("\nONNX 导出流程完成")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n摘要已保存至: {summary_path}")


if __name__ == "__main__":
    main()
