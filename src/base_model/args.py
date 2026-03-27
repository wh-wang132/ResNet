import argparse

from .utils import str2bool

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Train a 2D Lightweight ResNet for .npy Dataset"
    )

    # 训练参数
    parser.add_argument("--epochs", type=int, default=60, help="训练轮数 (默认 60)")
    parser.add_argument("--lr", type=float, default=0.0003, help="学习率 (默认 0.003)")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小 (默认 64)")
    parser.add_argument(
        "--model_path", type=str, default="best_model.pth", help="模型保存路径"
    )
    parser.add_argument("--class_num", type=int, default=24, help="分类数 (默认 24)")

    # 模型选择
    parser.add_argument(
        "--model",
        type=str,
        default="resnet6_2d",
        choices=[
            "resnet6_2d",
            "resnet10_2d",
            "resnet14_2d",
            "resnet18_2d",
            "resnet34_2d",
            "resnet50_2d",
        ],
        help="选择模型 (默认 resnet6_2d)",
    )

    # 数据路径
    parser.add_argument("--data_dir", type=str, default="Data", help="数据集路径")
    parser.add_argument(
        "--data_dtype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32"],
        help="数据加载后的 tensor 精度，仅影响数据集输出 (默认 fp16)",
    )

    # 数据加载选项
    parser.add_argument(
        "--full_load",
        type=str2bool,
        default=False,
        help="全量加载数据集到内存 (默认 False)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="数据加载工作线程数 (None=自动检测CPU核心数)",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="DataLoader预取因子 (默认 2)",
    )
    parser.add_argument(
        "--persistent_workers",
        type=str2bool,
        default=True,
        help="保持DataLoader工作线程活跃 (默认 True)",
    )
    parser.add_argument(
        "--pin_memory",
        type=str2bool,
        default=True,
        help="启用CUDA内存钉住 (默认 True, GPU训练时启用)",
    )

    # cuDNN和性能优化选项
    parser.add_argument(
        "--cudnn_benchmark",
        type=str2bool,
        default=True,
        help="启用cuDNN自动调优 (默认 True, 优化卷积性能)",
    )
    parser.add_argument(
        "--cudnn_deterministic",
        type=str2bool,
        default=False,
        help="启用cuDNN确定性算法 (默认 False, 禁用以提高速度)",
    )
    parser.add_argument(
        "--compile_model",
        type=str2bool,
        default=True,
        help="启用模型编译 (默认 True, 使用torch.compile优化)",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="default",
        choices=[
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
        help="模型编译模式 (默认 default)",
    )

    # 功能开关
    parser.add_argument(
        "--Train",
        type=str2bool,
        default=True,
        help="是否训练 (默认 True)",
    )
    parser.add_argument(
        "--Test",
        type=str2bool,
        default=True,
        help="是否测试 (默认 True)",
    )
    parser.add_argument(
        "--UMAP",
        type=str2bool,
        default=False,
        help="UMAP 可视化 (默认 False)",
    )

    # 正则化参数
    parser.add_argument(
        "--dropout_p", type=float, default=0.3, help="Dropout 概率 (默认 0.3)"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="权重衰减 (默认 1e-4)"
    )

    # 学习率调度器参数
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="Warmup 占总步数的比例 (默认 0.05, 即 5%)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Warmup 步数 (如果为 0，则使用 warmup_ratio, 默认 0)",
    )
    parser.add_argument(
        "--min_lr", type=float, default=1e-6, help="最小学习率 (默认 1e-6)"
    )
    parser.add_argument(
        "--plot_lr_schedule",
        type=str2bool,
        default=True,
        help="是否绘制学习率调度曲线 (默认 True)",
    )

    return parser.parse_args()