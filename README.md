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

## 配置文件说明

| 配置 | 参数量(n) | 精度 | 推理速度 | 说明 |
|------|----------|------|---------|------|
| `C2f-DBB` | 4.6M | ★★★☆ | ★★★★★ | 推荐日常使用 |
| `C2f-DeepDBB` | 5.9M | ★★★★ | ★★★★☆ | 更深，精度更高 |
| **`DBB-DCNV4`** | 3.8M | **★★★★★** | ★★★★☆ | **精度优先组合** |
| `C2f-DCNV4` | - | ★★★★ | ★★★☆ | 全 DCNV4 |
| `dyhead-DCNV4` | 61M (l) | ★★★★★ | ★★★ | 最大模型 |

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
