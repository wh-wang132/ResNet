#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标准 ResNet 模型单元测试
验证 ResNet-18, ResNet-34, ResNet-50 的前向传播和参数量
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn


def count_parameters(model):
    """计算模型的可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_resnet18():
    """测试 ResNet-18 模型"""
    print("=" * 80)
    print("测试 ResNet-18 模型")
    print("=" * 80)
    
    from resnet_standard import resnet18_2d
    
    model = resnet18_2d(num_classes=24)
    print(f"✓ 模型创建成功")
    
    param_count = count_parameters(model)
    print(f"✓ 参数量: {param_count:,}")
    
    # 标准 ResNet-18 的预期参数量（约 11M）
    expected_min = 10_000_000
    expected_max = 12_000_000
    assert expected_min < param_count < expected_max, \
        f"ResNet-18 参数量异常: {param_count}，期望在 {expected_min} 和 {expected_max} 之间"
    print("✓ 参数量在预期范围内")
    
    # 测试前向传播
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # 创建测试输入（单通道，543x512）
    test_input = torch.randn(1, 1, 543, 512).to(device)
    
    with torch.no_grad():
        output = model(test_input)
    
    assert output.shape == (1, 24), f"输出形状异常: {output.shape}，期望 (1, 24)"
    print(f"✓ 前向传播成功，输出形状: {output.shape}")
    
    # 测试 get_features 方法
    features = model.get_features(test_input, layer=["layer1", "layer2", "layer3", "layer4"])
    assert "layer1" in features
    assert "layer2" in features
    assert "layer3" in features
    assert "layer4" in features
    print("✓ get_features 方法工作正常")
    
    print("✓ ResNet-18 测试通过\n")


def test_resnet34():
    """测试 ResNet-34 模型"""
    print("=" * 80)
    print("测试 ResNet-34 模型")
    print("=" * 80)
    
    from resnet_standard import resnet34_2d
    
    model = resnet34_2d(num_classes=24)
    print(f"✓ 模型创建成功")
    
    param_count = count_parameters(model)
    print(f"✓ 参数量: {param_count:,}")
    
    # 标准 ResNet-34 的预期参数量（约 21M）
    expected_min = 20_000_000
    expected_max = 23_000_000
    assert expected_min < param_count < expected_max, \
        f"ResNet-34 参数量异常: {param_count}，期望在 {expected_min} 和 {expected_max} 之间"
    print("✓ 参数量在预期范围内")
    
    # 测试前向传播
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # 创建测试输入（单通道，543x512）
    test_input = torch.randn(1, 1, 543, 512).to(device)
    
    with torch.no_grad():
        output = model(test_input)
    
    assert output.shape == (1, 24), f"输出形状异常: {output.shape}，期望 (1, 24)"
    print(f"✓ 前向传播成功，输出形状: {output.shape}")
    
    # 测试 get_features 方法
    features = model.get_features(test_input, layer=["layer1", "layer2", "layer3", "layer4"])
    assert "layer1" in features
    assert "layer2" in features
    assert "layer3" in features
    assert "layer4" in features
    print("✓ get_features 方法工作正常")
    
    print("✓ ResNet-34 测试通过\n")


def test_resnet50():
    """测试 ResNet-50 模型"""
    print("=" * 80)
    print("测试 ResNet-50 模型")
    print("=" * 80)
    
    from resnet_standard import resnet50_2d
    
    model = resnet50_2d(num_classes=24)
    print(f"✓ 模型创建成功")
    
    param_count = count_parameters(model)
    print(f"✓ 参数量: {param_count:,}")
    
    # 标准 ResNet-50 的预期参数量（约 25M）
    expected_min = 23_000_000
    expected_max = 27_000_000
    assert expected_min < param_count < expected_max, \
        f"ResNet-50 参数量异常: {param_count}，期望在 {expected_min} 和 {expected_max} 之间"
    print("✓ 参数量在预期范围内")
    
    # 测试前向传播
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # 创建测试输入（单通道，543x512）
    test_input = torch.randn(1, 1, 543, 512).to(device)
    
    with torch.no_grad():
        output = model(test_input)
    
    assert output.shape == (1, 24), f"输出形状异常: {output.shape}，期望 (1, 24)"
    print(f"✓ 前向传播成功，输出形状: {output.shape}")
    
    # 测试 get_features 方法
    features = model.get_features(test_input, layer=["layer1", "layer2", "layer3", "layer4"])
    assert "layer1" in features
    assert "layer2" in features
    assert "layer3" in features
    assert "layer4" in features
    print("✓ get_features 方法工作正常")
    
    print("✓ ResNet-50 测试通过\n")


def compare_models():
    """比较所有模型的参数量"""
    print("=" * 80)
    print("模型参数量对比")
    print("=" * 80)
    
    from resnet_standard import resnet18_2d, resnet34_2d, resnet50_2d
    from resnet_lightweight import resnet6_2d, resnet10_2d, resnet14_2d
    
    models = {
        "ResNet-6 (轻量)": resnet6_2d,
        "ResNet-10 (轻量)": resnet10_2d,
        "ResNet-14 (轻量)": resnet14_2d,
        "ResNet-18 (标准)": resnet18_2d,
        "ResNet-34 (标准)": resnet34_2d,
        "ResNet-50 (标准)": resnet50_2d,
    }
    
    print(f"{'模型':<20} {'参数量':>15}")
    print("-" * 37)
    
    for name, model_func in models.items():
        model = model_func(num_classes=24)
        param_count = count_parameters(model)
        print(f"{name:<20} {param_count:>15,}")
    
    print()


def main():
    print("\n" + "=" * 80)
    print("标准 ResNet 模型单元测试")
    print("=" * 80 + "\n")
    
    try:
        test_resnet18()
        test_resnet34()
        test_resnet50()
        compare_models()
        
        print("=" * 80)
        print("所有测试通过 ✓")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

