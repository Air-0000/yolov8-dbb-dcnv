#!/bin/bash
# ========================================
# 血细胞检测系统 - 一键环境配置脚本
# Windows (Git Bash) / Linux / macOS 通用
# ========================================

set -e

echo "========================================"
echo "  血细胞检测系统 - 环境配置"
echo "========================================"

# 1. 检查 Python
echo ""
echo "[1/4] 检查 Python..."
if command -v python &> /dev/null; then
    PYTHON=python
elif command -v python3 &> /dev/null; then
    PYTHON=python3
else
    echo "❌ 没找到 Python，请先安装 Python 3.9+"
    exit 1
fi

PY_VERSION=$($PYTHON --version 2>&1)
echo "  找到 $PY_VERSION"

# 2. 检查 pip
echo ""
echo "[2/4] 检查 pip..."
if ! command -v pip &> /dev/null && ! $PYTHON -m pip --version &> /dev/null; then
    echo "❌ 没找到 pip"
    exit 1
fi
echo "  pip 正常"

# 3. 创建虚拟环境（可选）
echo ""
echo "[3/4] 安装依赖..."
echo "  安装 PyTorch（CPU版，有NVIDIA显卡请装CUDA版）..."
echo "  CPU用户用这个："
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q
echo "  NVIDIA显卡用户请装：pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
echo "  安装 Ultralytics + 其他依赖..."
pip install ultralytics opencv-python pyyaml -q
echo "  ✅ 依赖安装完成"

# 4. 下载数据集
echo ""
echo "[4/4] 准备数据集..."
if [ -d "data/images" ]; then
    echo "  数据集已存在，跳过"
else
    $PYTHON download_bccd.py
fi

echo ""
echo "========================================"
echo "  ✅ 环境配置完成！"
echo "  运行训练：python train_dcnv4.py"
echo "  运行评估：python evaluate.py"
echo "  运行推理：python infer.py"
echo "========================================"
