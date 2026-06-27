"""
DBB (Diverse Branch Block) - 重参数化卷积模块
Paper: Diverse Branch Block: Building a Convolution as an Inception-like Unit
       (Ding et al., CVPR 2021)

训练时: 多分支结构 (3x3, 1x1, avg pool, identity) 丰富特征表达
推理时: 合并为单分支 3x3 Conv，零额外开销

适用于: YOLOv8 的 Backbone/Head 替换标准 Conv
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


class DBB(nn.Module):
    """
    Diverse Branch Block - 多分支重参数化卷积
    
    训练分支 (6路):
      1. 3x3 Conv + BN
      2. 1x1 Conv + BN (padding成3x3)
      3. Avg Pool 3x3 + 1x1 Conv + BN
      4. 1x1 Conv + BN -> 3x3 Conv + BN
      5. 1x1 Conv + BN -> 1x1 Conv + BN (padding成3x3)
      6. Identity (仅c1==c2且s==1)
    
    推理: 全部融合为单个 3x3 Conv
    """
    def __init__(self, c1, c2, k=3, s=1, padding=None, dilation=1, groups=1):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.s = s
        self.groups = groups
        padding = autopad(k, padding, dilation)

        # 分支1: 标准 3x3 Conv + BN
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, s, padding, groups=groups, bias=False),
            nn.BatchNorm2d(c2),
        )

        # 分支2: 1x1 Conv + BN
        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(c1, c2, 1, s, 0, groups=groups, bias=False),
            nn.BatchNorm2d(c2),
        )

        # 分支3: Avg Pool 3x3 + 1x1 Conv + BN
        if s != 1:
            self.branch_avg = nn.Sequential(
                nn.AvgPool2d(3, s, padding=1),
                nn.Conv2d(c1, c2, 1, 1, 0, groups=groups, bias=False),
                nn.BatchNorm2d(c2),
            )
        else:
            self.branch_avg = nn.Sequential(
                nn.AvgPool2d(3, 1, padding=1),
                nn.Conv2d(c1, c2, 1, 1, 0, groups=groups, bias=False),
                nn.BatchNorm2d(c2),
            )

        # 分支4: 1x1 + 3x3 Conv 序列
        self.branch_1x1_3x3 = nn.Sequential(
            nn.Conv2d(c1, c2, 1, 1, 0, groups=groups, bias=False),
            nn.BatchNorm2d(c2),
            nn.Conv2d(c2, c2, 3, s, padding, groups=groups, bias=False),
            nn.BatchNorm2d(c2),
        )

        # 分支5: 1x1 + 1x1 Conv 序列
        self.branch_1x1_1x1 = nn.Sequential(
            nn.Conv2d(c1, c2, 1, 1, 0, groups=groups, bias=False),
            nn.BatchNorm2d(c2),
            nn.Conv2d(c2, c2, 1, s, 0, groups=groups, bias=False),
            nn.BatchNorm2d(c2),
        )

        # 分支6: Identity (仅当 c1==c2 and s==1)
        self.branch_identity = nn.BatchNorm2d(c2) if c1 == c2 and s == 1 else None

    def forward(self, x):
        out = self.branch_3x3(x) + self.branch_1x1(x) + \
              self.branch_avg(x) + self.branch_1x1_3x3(x) + \
              self.branch_1x1_1x1(x)
        if self.branch_identity is not None:
            out += self.branch_identity(x)
        return out

    def _fuse_bn(self, conv, bn):
        w = conv.weight
        mean = bn.running_mean
        var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return w * t, beta - mean * gamma / std

    def _pad_1x1_to_3x3(self, kernel):
        return F.pad(kernel, [1, 1, 1, 1])

    def _avg_to_conv(self):
        conv_w, conv_b = self._fuse_bn(self.branch_avg[1], self.branch_avg[2])
        avg_kernel = torch.zeros(self.c1, 1, 3, 3, device=conv_w.device, dtype=conv_w.dtype)
        avg_kernel[:, :, :, :] = 1.0 / 9.0
        # 将 avg pool (1/9均值滤波) 与 1x1 conv 组合
        k_eff = F.conv2d(
            avg_kernel.expand(-1, self.c1 // self.groups, -1, -1).transpose(0, 1),
            conv_w,
            padding=1,
            groups=self.groups,
        )
        return k_eff, conv_b

    def _fuse_1x1_seq(self, seq):
        """融合 1x1->Conv 序列为单个等效卷积"""
        conv1, bn1, conv2, bn2 = seq
        w1, b1 = self._fuse_bn(conv1, bn1)
        w2, b2 = self._fuse_bn(conv2, bn2)
        
        if conv1.kernel_size == (1, 1) and conv2.kernel_size == (1, 1):
            w = torch.matmul(w2.view(w2.size(0), -1), w1.view(w1.size(0), -1)).view(w2.size(0), -1, 1, 1)
        else:
            # 1x1 -> 3x3 序列
            w = F.conv2d(w1.permute(1, 0, 2, 3), w2, padding=conv2.padding).permute(1, 0, 2, 3)
        b = w2.view(w2.size(0), -1) @ b1 + b2
        return w.to(dtype=w1.dtype), b.to(dtype=b1.dtype)

    def switch_to_deploy(self):
        """融合所有分支为单个 3x3 Conv (推理模式)"""
        device = self.branch_3x3[0].weight.device
        dtype = self.branch_3x3[0].weight.dtype
        K = torch.zeros(self.c2, self.c1, 3, 3, device=device, dtype=dtype)
        B = torch.zeros(self.c2, device=device, dtype=dtype)

        # 1. 3x3 分支
        k, b = self._fuse_bn(self.branch_3x3[0], self.branch_3x3[1])
        K += k
        B += b

        # 2. 1x1 分支 -> pad 成 3x3
        k, b = self._fuse_bn(self.branch_1x1[0], self.branch_1x1[1])
        K += self._pad_1x1_to_3x3(k)
        B += b

        # 3. Avg Pool 分支
        k, b = self._avg_to_conv()
        K += k
        B += b

        # 4. 1x1 -> 3x3 序列
        k, b = self._fuse_1x1_seq(self.branch_1x1_3x3)
        K += k
        B += b

        # 5. 1x1 -> 1x1 序列
        k, b = self._fuse_1x1_seq(self.branch_1x1_1x1)
        K += self._pad_1x1_to_3x3(k)
        B += b

        # 6. Identity 分支
        if self.branch_identity is not None:
            id_k = torch.zeros(self.c2, self.c1, 3, 3, device=device, dtype=dtype)
            for i in range(min(self.c1, self.c2)):
                id_k[i, i % self.c1, 1, 1] = 1.0
            id_b = self.branch_identity.bias.detach() - \
                   self.branch_identity.running_mean.detach() * \
                   self.branch_identity.weight.detach() / \
                   (self.branch_identity.running_var.detach() + self.branch_identity.eps).sqrt()
            K += id_k
            B += id_b

        conv = nn.Conv2d(self.c1, self.c2, self.k, self.s,
                        padding=autopad(self.k), dilation=1,
                        groups=self.groups, bias=True)
        conv.weight.data = K
        conv.bias.data = B

        # 清理训练分支
        for attr in ['branch_3x3', 'branch_1x1', 'branch_avg',
                     'branch_1x1_3x3', 'branch_1x1_1x1', 'branch_identity']:
            if hasattr(self, attr):
                delattr(self, attr)

        self.conv = conv
        self.forward = self._forward_deploy

    def _forward_deploy(self, x):
        return self.conv(x)


class DBBottleNeck(nn.Module):
    """
    使用 DBB 的标准 Bottleneck
    用于替换 C2f 中的标准 Bottleneck
    精度优于标准 Bottleneck，参数量略增
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = DBB(c1, c_, k[0], 1)
        self.cv2 = DBB(c_, c2, k[1], 1, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class DeepDBBottleNeck(nn.Module):
    """
    加深的 DBB Bottleneck: 3 个 DBB 堆叠
    比标准 DBBottleNeck 更深 → 精度更高，参数量更大
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = DBB(c1, c_, k[0], 1)
        self.cv2 = DBB(c_, c_, k[1], 1, groups=g)
        self.cv3 = DBB(c_, c2, k[1], 1, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))


class WDBBottleNeck(nn.Module):
    """
    宽 DBB Bottleneck: e=1.0，隐藏通道数 = 输出通道数
    精度最高，参数量显著增加
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=1.0):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = DBB(c1, c_, k[0], 1)
        self.cv2 = DBB(c_, c2, k[1], 1, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
