#!/bin/bash
# ========================================
# 血细胞检测系统 - 一键运行脚本
# Windows (Git Bash) 运行: bash setup.sh
# ========================================

set -e

echo "========================================"
echo "  血细胞检测系统"
echo "========================================"

# 用 Anaconda Python（base 环境）
PY="/d/Environment/anaconda3/python.exe"
PIP="python -m pip"

# 清理 PYTHONPATH 防止冲突
export PYTHONPATH=""
export KMP_DUPLICATE_LIB_OK=TRUE

echo ""
echo "[检查依赖]..."
$PY -c "import ultralytics" 2>/dev/null || $PIP install ultralytics opencv-python pyyaml -q
$PY -c "import torch" 2>/dev/null || $PIP install torch torchvision --index-url https://download.pytorch.org/whl/cu124 -q

# 检查 checkpoint，没有就训练
CKPT="runs/medical/manual_epoch50.pt"
if [ ! -f "$CKPT" ]; then
    echo "  checkpoint 不存在，开始训练..."
    $PY train_dcnv4.py
fi

# 推理
echo ""
echo "  推理中..."
$PY infer.py

echo ""
echo "========================================"
echo "  完成！结果: runs/infer_results/predict/"
echo "========================================"
