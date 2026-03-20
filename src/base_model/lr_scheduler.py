#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学习率调度器模块
包含 Warmup + Cosine Annealing 调度策略
与 AMP 完全兼容
"""

import math
from typing import Optional
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineAnnealingLR(_LRScheduler):
    """
    Warmup + Cosine Annealing 学习率调度器
    
    阶段 1: Warmup (线性增长)
    阶段 2: Cosine Annealing (余弦衰减)
    """
    
    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        warmup_ratio: float = 0.05,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer: 优化器
            total_steps: 总训练步数
            warmup_steps: Warmup 步数（如果为 0，则使用 warmup_ratio）
            warmup_ratio: Warmup 占总步数的比例（默认 5%）
            min_lr: 最小学习率（默认 1e-6）
            last_epoch: 最后一个 epoch（用于恢复训练）
        """
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps if warmup_steps > 0 else int(total_steps * warmup_ratio)
        self.min_lr = min_lr
        
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """计算当前步的学习率"""
        step = self.last_epoch + 1
        
        if step <= self.warmup_steps:
            # Warmup 阶段：线性增长
            warmup_factor = step / max(self.warmup_steps, 1)
            return [
                base_lr * warmup_factor
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine Annealing 阶段：余弦衰减
            progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]


def plot_lr_schedule(
    scheduler,
    total_steps: int,
    save_path: Optional[str] = None,
    title: str = "Learning Rate Schedule",
):
    """
    绘制学习率调度曲线
    
    Args:
        scheduler: WarmupCosineAnnealingLR 实例
        total_steps: 总步数
        save_path: 保存路径（可选）
        title: 图表标题
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    lrs = []
    steps = list(range(total_steps))
    
    # 复制优化器以避免影响原优化器
    import copy
    test_optimizer = copy.deepcopy(scheduler.optimizer)
    test_scheduler = WarmupCosineAnnealingLR(
        test_optimizer,
        total_steps=total_steps,
        warmup_steps=scheduler.warmup_steps,
        min_lr=scheduler.min_lr,
    )
    
    for _ in steps:
        lrs.append(test_optimizer.param_groups[0]['lr'])
        # 模拟 optimizer.step()（虽然我们没有实际梯度，但为了避免警告）
        test_optimizer.step()
        test_scheduler.step()
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, lrs, label='Learning Rate', linewidth=2)
    
    # 标记 warmup 结束点
    if scheduler.warmup_steps > 0 and scheduler.warmup_steps < total_steps:
        plt.axvline(
            x=scheduler.warmup_steps,
            color='r',
            linestyle='--',
            label=f'Warmup End (Step {scheduler.warmup_steps})'
        )
    
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 学习率曲线已保存至: {save_path}")
    
    plt.close()
    
    return lrs
