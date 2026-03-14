#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复已编译模型的权重文件
移除 "_orig_mod." 前缀，使其可被未编译的模型加载
"""

import os
import sys
import shutil
import torch
import argparse
from datetime import datetime


def backup_file(file_path):
    """
    创建文件备份
    
    Args:
        file_path: 原始文件路径
        
    Returns:
        str: 备份文件路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.backup_{timestamp}"
    shutil.copy2(file_path, backup_path)
    print(f"✓ 已创建备份文件: {backup_path}")
    return backup_path


def fix_weights_file(file_path, dry_run=False):
    """
    修复权重文件，移除 "_orig_mod." 前缀
    
    Args:
        file_path: 权重文件路径
        dry_run: 是否只预览不实际修改
        
    Returns:
        bool: 是否成功修复
    """
    print(f"\n{'='*80}")
    print(f"处理文件: {file_path}")
    print(f"{'='*80}")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"✗ 错误：文件不存在: {file_path}")
        return False
    
    # 加载权重文件
    try:
        print("\n正在加载权重文件...")
        checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        print("✓ 权重文件加载成功")
    except Exception as e:
        print(f"✗ 加载权重文件失败: {e}")
        return False
    
    # 确定要处理的 state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        is_checkpoint_format = True
        print("✓ 检测到 checkpoint 格式（包含 model_state_dict）")
    else:
        state_dict = checkpoint
        is_checkpoint_format = False
        print("✓ 检测到纯 state_dict 格式")
    
    # 分析键名
    print("\n正在分析键名...")
    keys_with_prefix = []
    keys_without_prefix = []
    
    for key in state_dict.keys():
        if key.startswith("_orig_mod."):
            keys_with_prefix.append(key)
        else:
            keys_without_prefix.append(key)
    
    print(f"  总键数: {len(state_dict)}")
    print(f"  带 '_orig_mod.' 前缀的键数: {len(keys_with_prefix)}")
    print(f"  不带前缀的键数: {len(keys_without_prefix)}")
    
    if len(keys_with_prefix) == 0:
        print("\n✓ 没有需要修复的键名，文件无需修改")
        return True
    
    # 显示前几个需要修复的键名
    print(f"\n需要修复的键名示例（前10个）:")
    for i, key in enumerate(keys_with_prefix[:10]):
        new_key = key[len("_orig_mod."):]
        print(f"  {i+1}. {key} -> {new_key}")
    
    if len(keys_with_prefix) > 10:
        print(f"  ... 还有 {len(keys_with_prefix) - 10} 个键")
    
    # 如果是 dry_run，到此为止
    if dry_run:
        print("\n⚠️  这是 dry run 模式，不进行实际修改")
        print(f"{'='*80}\n")
        return True
    
    # 创建备份
    print("\n正在创建备份...")
    backup_path = backup_file(file_path)
    
    # 创建新的 state_dict
    print("\n正在创建修复后的 state_dict...")
    new_state_dict = {}
    
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            new_key = key[len("_orig_mod."):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    # 更新 checkpoint
    if is_checkpoint_format:
        checkpoint["model_state_dict"] = new_state_dict
        output_data = checkpoint
    else:
        output_data = new_state_dict
    
    # 保存修复后的文件
    print("\n正在保存修复后的权重文件...")
    try:
        torch.save(output_data, file_path)
        print(f"✓ 修复后的权重文件已保存: {file_path}")
    except Exception as e:
        print(f"✗ 保存文件失败: {e}")
        print(f"⚠️  正在从备份恢复原文件...")
        shutil.copy2(backup_path, file_path)
        print(f"✓ 已从备份恢复原文件")
        return False
    
    # 验证修复结果
    print("\n正在验证修复结果...")
    try:
        # 重新加载验证
        verify_checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        
        if isinstance(verify_checkpoint, dict) and "model_state_dict" in verify_checkpoint:
            verify_state_dict = verify_checkpoint["model_state_dict"]
        else:
            verify_state_dict = verify_checkpoint
        
        # 检查是否还有带前缀的键
        remaining_prefix_keys = [k for k in verify_state_dict.keys() if k.startswith("_orig_mod.")]
        
        if len(remaining_prefix_keys) == 0:
            print("✓ 验证通过：所有 '_orig_mod.' 前缀已被移除")
            print(f"✓ 修复后的键数: {len(verify_state_dict)}")
        else:
            print(f"✗ 验证失败：仍有 {len(remaining_prefix_keys)} 个键包含 '_orig_mod.' 前缀")
            print(f"⚠️  正在从备份恢复原文件...")
            shutil.copy2(backup_path, file_path)
            print(f"✓ 已从备份恢复原文件")
            return False
        
    except Exception as e:
        print(f"✗ 验证过程出错: {e}")
        print(f"⚠️  正在从备份恢复原文件...")
        shutil.copy2(backup_path, file_path)
        print(f"✓ 已从备份恢复原文件")
        return False
    
    print(f"\n{'='*80}")
    print("✓ 权重文件修复完成！")
    print(f"{'='*80}\n")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="修复已编译模型的权重文件，移除 '_orig_mod.' 前缀"
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="权重文件路径（.pth 或 .pt 文件）"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅预览修复内容，不实际修改文件"
    )
    
    args = parser.parse_args()
    
    success = fix_weights_file(args.file_path, dry_run=args.dry_run)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
