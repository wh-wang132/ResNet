#!/bin/bash
# 自动化训练脚本包装器
# 用于系统地验证不同网络结构、批次大小和训练轮数对模型性能的影响
# 所有命令均为Linux命令

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "ResNet 自动化训练实验"
echo "=========================================="
echo ""

# 检查Python脚本是否存在（Linux命令）
if [ ! -f "automated_training.py" ]; then
    echo "错误: 找不到 automated_training.py 文件"
    exit 1
fi

# 检查是否有uv（Linux命令）
if ! command -v uv &> /dev/null; then
    echo "错误: 找不到 uv 命令，请先安装 uv"
    exit 1
fi

# 创建必要的目录（Linux命令）
mkdir -p logs
mkdir -p experiments

echo "开始自动化训练实验..."
echo ""

# 运行Python自动化训练脚本（Linux命令）
uv run python automated_training.py

echo ""
echo "=========================================="
echo "Shell脚本执行完成"
echo "=========================================="
echo ""
echo "注意：自动关机功能已在Python脚本中实现"
echo "实验结果保存在 experiments/ 目录下"
echo "日志文件保存在 logs/ 目录下"

sudo shutdown -h now
