"""
C2f 变体模块 - 将 YOLOv8 的 C2f 内部 Bottleneck 替换为 DBB / DCNV 版本

C2f = CSP bottleneck with 2 convolutions
  - cv1: 1x1 Conv (通道拆分)
  - cv2: 1x1 Conv (拼接后融合)
  - m:   N 个 Bottleneck 变体

使用方法:
  在 YAML 配置中将 "C2f" 替换为 "C2f_DBB" / "C2f_DCNV3" / "C2f_DCNV4"
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv

from .dbb import DBB, DBBottleNeck, DeepDBBottleNeck, WDBBottleNeck
from .dcnv import DCNv3_pytorch, DCNv4_pytorch, get_dcnv3, get_dcnv4


class C2f_DBB(nn.Module):
    """
    C2f with DBB Bottleneck
    将标准 Bottleneck 替换为 DBB (Diverse Branch Block) 版本
    训练时多分支 → 推理时融合为单分支
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(DBBottleNeck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2f_DeepDBB(nn.Module):
    """
    C2f with Deep DBB Bottleneck (3 layers deep per block)
    每个 block 比 C2f_DBB 多一层 DBB → 精度更高
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(DeepDBBottleNeck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2f_WDBB(nn.Module):
    """
    C2f with Wide DBB Bottleneck (e=1.0)
    每个 block 的隐藏通道 = 输出通道 → 特征表达能力最强
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=1.0):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(WDBBottleNeck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2f_DCNV3(nn.Module):
    """
    C2f with DCNV3 Bottleneck
    使用可变形卷积 V3 替换标准卷积
    更好的几何适应能力，精度更高
    
    ⚠ DCNV3 默认使用纯 PyTorch 实现
      如需 CUDA 加速，请安装 mmcv-full 或 InternImage
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        # DCNV3 Bottleneck: 1x1 Conv + DCNV3
        self.m = nn.ModuleList(self._make_dcnv3_block(self.c, g) for _ in range(n))

    def _make_dcnv3_block(self, c, g):
        """创建 DCNV3 瓶颈块"""
        return nn.Sequential(
            Conv(c, c, 1, 1),  # 1x1 降维/保持
            get_dcnv3(c, kernel_size=3, stride=1, groups=g, offset_scale=1.0),
            Conv(c, c, 1, 1),
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2f_DCNV4(nn.Module):
    """
    C2f with DCNV4 Bottleneck
    使用可变形卷积 V4 替换标准卷积
    
    ⚠ DCNV4 训练时需关闭 AMP (混合精度)
      使用 trainer.amp = False
    
    ⚠ 默认使用纯 PyTorch 实现
      如需 CUDA 加速，请安装 InternVL
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(self._make_dcnv4_block(self.c, g) for _ in range(n))

    def _make_dcnv4_block(self, c, g):
        """创建 DCNV4 瓶颈块"""
        return nn.Sequential(
            Conv(c, c, 1, 1),
            get_dcnv4(c, kernel_size=3, stride=1, groups=g),
            Conv(c, c, 1, 1),
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class DCNV4_Detect(nn.Module):
    """
    在 Detection Head 中使用 DCNV4 替代标准卷积
    提升检测头对形变目标的适应能力
    
    注意: 仅在 head 的最后几层使用 DCNV4，不替换整个 backbone
    """
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = get_dcnv4(c1, kernel_size=3, stride=1, groups=1)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
