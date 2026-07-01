#!/bin/bash
# ========================================
# 血细胞检测系统 - 一键运行脚本
# 用法: bash setup.sh
# ========================================

set -e

echo "========================================"
echo "  血细胞检测系统"
echo "========================================"

# 环境变量
PY="/d/Environment/anaconda3/python.exe"
PIP="python -m pip"
export PYTHONPATH=""
export KMP_DUPLICATE_LIB_OK=TRUE

# ---------- 第1步: Python ----------
echo ""
echo "━━━ 第1步/4: Python ━━━━━━━━━━━━━━━━━━━"
echo "  路径: $PY"
echo "  版本: $($PY --version 2>&1)"
echo "  → 已就绪，跳过"

# ---------- 第2步: 依赖 ----------
echo ""
echo "━━━ 第2步/4: 依赖安装 ━━━━━━━━━━━━━━━━━━"
if $PY -c "import ultralytics" 2>/dev/null; then
    echo "  ultralytics: $($PY -c "import ultralytics; print(ultralytics.__version__)")"
    echo "  → 已安装，跳过"
else
    echo "  安装 ultralytics opencv-python pyyaml..."
    $PIP install ultralytics opencv-python pyyaml -q
    echo "  → 完成"
fi

if $PY -c "import torch" 2>/dev/null; then
    echo "  PyTorch: $($PY -c "import torch; print(torch.__version__)")"
    echo "  → 已安装，跳过"
else
    echo "  安装 PyTorch..."
    $PIP install torch torchvision --index-url https://download.pytorch.org/whl/cu124 -q
    echo "  → 完成"
fi

# ---------- 第3步: 模型权重 ----------
echo ""
echo "━━━ 第3步/4: 模型权重 ━━━━━━━━━━━━━━━━━━"
CKPT="runs/medical/manual_epoch50.pt"
if [ -f "$CKPT" ]; then
    echo "  checkpoint: $CKPT"
    echo "  → 已存在，跳过训练"
else
    echo "  未找到 checkpoint，开始训练..."
    $PY train_dcnv4.py
    echo "  → 训练完成"
fi

# ---------- 第4步: 推理 ----------
echo ""
echo "━━━ 第4步/4: 推理 ━━━━━━━━━━━━━━━━━━━━━━"
echo "  运行推理..."
$PY infer.py

# ---------- 完成 ----------
echo ""
echo "========================================"
echo "  ✅ 全部完成！"
echo "  结果图片: runs/infer_results/predict/"
echo "========================================"
