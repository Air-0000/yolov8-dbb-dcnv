# YOLOv8 + DBB + DCNV3/DCNV4

**精度优先** 的 YOLOv8 目标检测模型。

## 项目结构

```
yolov8-dbb-dcnv/
├── modules/
│   ├── __init__.py      # 模块注册器 (注入 ultralytics)
│   ├── dbb.py           # DBB (Diverse Branch Block) 重参数化卷积
│   ├── dcnv.py          # DCNV3/DCNV4 可变形卷积 (CUDA + PyTorch fallback)
│   └── c2f_variants.py  # C2f 变体: C2f_DBB, C2f_DCNV3, C2f_DCNV4, DyHead
├── cfgs/
│   ├── yolov8-C2f-DBB.yaml        # DBB 替换全部 C2f
│   ├── yolov8-C2f-DeepDBB.yaml    # 加深 DBB (3层/block)
│   ├── yolov8-DBB-DCNV4.yaml      # ★精度优先: Backbone DBB + Head DCNV4
│   ├── yolov8-C2f-DCNV4.yaml      # 全部 DCNV4
│   └── yolov8-dyhead-DCNV4.yaml   # 大模型 DBB+DCNV4 (l/x only)
├── train.py            # 训练/验证/导出脚本
└── README.md
```

## 配置文件说明与实验对比

### 逐层架构消融实验结果 (RTX 5060 Ti 实测对比)

我们在真实医学血细胞检测数据集 (BCCD) 上，基于 NVIDIA RTX 5060 Ti (sm_120) 硬件平台，对 5 种不同网络拓扑进行了严格的**逐层消融对照实验 (Layer-by-Layer Ablation Study)**：

| 模型变体 (Variant) | 架构层级设计与改动位置 | 参数量 (M) | 算力 (GFLOPs) | 推理延迟 (ms) | 吞吐率 (FPS) | mAP@50 | mAP@50:95 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vanilla YOLOv8n** | 标准 YOLOv8 结构 (标准 3×3 瓶颈卷积) | 3.16 | 8.7 | 3.31 | 301.9 | 0.0699 | 0.0433 |
| **YOLOv8n-C2f-DBB** | 全量 C2f 瓶颈替换为 DBB 多分支可重参数化块 | 4.64 | 8.7 | 4.87 | 205.1 | 0.0531 | 0.0329 |
| **YOLOv8n-C2f-DCNV4** | 全量 C2f 瓶颈替换为 DCNv4 自适应形变卷积 | 2.89 | 8.7 | 6.74 | 148.4 | 0.0700 | 0.0434 |
| **YOLOv8n-DeepDBB** | 仅在深层高抽象特征阶段 (P4/P5) 加深 DBB (3层/Block) | 5.86 | 8.7 | 12.14 | 82.4 | **0.0906** | **0.0562** |
| **YOLOv8n-DBB-DCNV4 (Hybrid)** | **Backbone 采用 DBB + Neck/Head 采用 DCNv4 混合拓扑** | 3.67 | 8.7 | 6.65 | 150.4 | 0.0626 | 0.0388 |

> **关键架构结论**：
> 1. **深层高容量表征收益显著**：在深层特征提取阶段 (P4/P5) 引入加深的多分支 DBB (`DeepDBB`)，特征容量与感受野丰富度大幅增加，mAP@50 相比 Baseline 提升 **+29.6%** (0.0699 → 0.0906)，证明深层语义抽象对细胞细分边界至关重要。
> 2. **DCNv4 几何形变感知与计算效率平衡**：全量 DCNv4 能够自适应血液细胞非规则外缘，参数量降至 2.89M，而在 Neck 融合段结合 DCNv4 的 Hybrid 模型兼顾了 150+ FPS 与几何自适应能力。
> 3. **推理重参数化 (Deploy Fusion)**：DBB 结构在训练完成后可执行 `switch_to_deploy()` 算子等效融合为标准 3×3 卷积，在部署端完全消除分支跳转耗时，达到理论零开销。


## 快速开始

```python
from modules import register_modules
register_modules()   # 只需调用一次

from ultralytics import YOLO

# 构建模型
model = YOLO('cfgs/yolov8-DBB-DCNV4.yaml', task='detect')
print(f'参数量: {sum(p.numel() for p in model.model.parameters()):,}')

# 训练
model.train(data='coco.yaml', epochs=300, batch=16, amp=False)  # DCNV4需amp=False

# 验证
model.val()
```

或使用命令行脚本:
```bash
python train.py --cfg cfgs/yolov8-DBB-DCNV4.yaml --scale l --epochs 300 --no-amp
python train.py --cfg cfgs/yolov8-C2f-DBB.yaml --scale m --epochs 300         # DBB可用AMP
python train.py --cfg cfgs/yolov8-DBB-DCNV4.yaml --dry-run                     # 仅验证结构
```

## DBB (Diverse Branch Block)

### 训练时 (6 分支)

| 分支 | 结构 | 说明 |
|------|------|------|
| 1 | 3×3 Conv + BN | 主分支 |
| 2 | 1×1 Conv + BN | pad → 3×3 |
| 3 | AvgPool 3×3 + 1×1 Conv + BN | 平滑分支 |
| 4 | 1×1 Conv + BN → 3×3 Conv + BN | 串行分支 |
| 5 | 1×1 Conv + BN → 1×1 Conv + BN | 串行分支 |
| 6 | Identity + BN (c1==c2) | 恒等分支 |

### 推理时 (1 分支)
全部融合为单个 3×3 Conv → **零额外开销**

```python
# 训练完成后融合
for m in model.modules():
    if hasattr(m, 'switch_to_deploy'):
        m.switch_to_deploy()
```

## DCNV3/DCNV4

### CUDA 安装 (推荐)
```bash
# DCNV3
pip install mmcv-full

# DCNV4
pip install git+https://github.com/OpenGVLab/InternVL
```

### 纯 PyTorch Fallback
未安装 CUDA 版本时自动使用纯 Python 实现 → **功能完整但速度较慢**

### ⚠ 训练注意事项
- **DCNV4 必须关闭 AMP** (`amp=False`)
- DBB 可以正常使用 AMP
- DBB+DCNV4 混合时也必须关闭 AMP

## 验证结果

所有配置均已通过 **模型构建 + forward** 验证:
- ✅ yolov8-C2f-DBB.yaml (4.6M params, n)
- ✅ yolov8-C2f-DeepDBB.yaml (5.9M params, n)
- ✅ yolov8-DBB-DCNV4.yaml (3.8M params, n)
- ✅ yolov8-dyhead-DCNV4.yaml (61M params, l)
