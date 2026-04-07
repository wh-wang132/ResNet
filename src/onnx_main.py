#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ONNX 导出统一入口。"""

import json

import torch

from amct.utils import SUMMARY_VERSION, extract_onnx_contract
from base_model.dataset import data_set_split
from base_model.plotting import configure_matplotlib
from onnx_export.args import parse_args
from onnx_export.evaluator import (
    create_onnx_session,
    evaluate_onnx_model_with_confusion_matrix,
    evaluate_torch_model,
)
from onnx_export.exporter import (
    build_branch_artifacts,
    build_metric_delta,
    resolve_branch_opset_version,
)
from onnx_export.output import save_summary
from onnx_export.utils import (
    create_optimized_dataloader,
    release_gpu_memory,
    setup_device,
    to_repo_relative_path,
)

configure_matplotlib()


def _resolve_source_test_model_and_device(artifacts, default_device):
    source_test_model = artifacts.model
    source_test_device = artifacts.runtime.source_device
    if artifacts.branch != "qat_convert" or default_device.type != "cuda":
        return source_test_model, source_test_device
    source_test_model = artifacts.model.to(default_device).eval()
    source_test_device = default_device
    return source_test_model, source_test_device


def main():
    args = parse_args()
    print(args)

    release_gpu_memory()
    device = setup_device()

    actual_opset_version = resolve_branch_opset_version(args.branch, args.opset_version)
    artifacts = build_branch_artifacts(
        branch=args.branch,
        checkpoint_path=args.checkpoint,
        device=device,
        opset_version=actual_opset_version,
    )

    _, _, test_dataset, labels__ = data_set_split(
        args.data_dir,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        full_load=args.full_load,
        num_workers=args.num_workers,
        data_dtype=artifacts.runtime.dataset_dtype,
    )

    source_test_metrics = None
    onnx_test_metrics = None
    ort_provider_meta = None
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
        source_test_model, source_test_device = _resolve_source_test_model_and_device(
            artifacts=artifacts,
            default_device=device,
        )
        source_test_metrics = evaluate_torch_model(
            model=source_test_model,
            device=source_test_device,
            dataloader=test_loader,
            num_samples=len(test_dataset),
            input_dtype=torch.float32,
            progress_desc="ONNX source test",
        )
        ort_session, ort_provider_meta = create_onnx_session(
            artifacts.onnx_path,
            artifacts.branch,
        )
        onnx_test_metrics = evaluate_onnx_model_with_confusion_matrix(
            session=ort_session,
            dataloader=test_loader,
            num_samples=len(test_dataset),
            input_dtype=artifacts.runtime.onnx_input_dtype,
            labels=labels__,
            folder_path=artifacts.folder_path,
            progress_desc="ONNX exported test",
        )

    interface = extract_onnx_contract(artifacts.onnx_path)["interface"]
    export_meta = {
        **artifacts.export_meta,
        **({} if ort_provider_meta is None else ort_provider_meta),
    }
    summary = {
        "summary_version": SUMMARY_VERSION,
        "branch": artifacts.branch,
        "model_name": artifacts.checkpoint_meta["model_name"],
        "labels": labels__,
        "source_checkpoint_path": to_repo_relative_path(args.checkpoint),
        "opset_version": actual_opset_version,
        "example_input_shape": artifacts.export_shape,
        "onnx_path": to_repo_relative_path(artifacts.onnx_path),
        "interface": interface,
        "source_test_metrics": source_test_metrics,
        "onnx_test_metrics": onnx_test_metrics,
        "metric_delta": build_metric_delta(source_test_metrics, onnx_test_metrics),
        "export_meta": export_meta,
    }

    summary_path = save_summary(artifacts.folder_path, summary)
    print("\nONNX 导出流程完成")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n摘要已保存至: {summary_path}")


if __name__ == "__main__":
    main()
