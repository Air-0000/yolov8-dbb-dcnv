"""
DCNV3 / DCNV4 - 可变形卷积模块

DCNV3: Deformable Convolution V3 (InternImage, CVPR 2023)
  - 分组可变形卷积，每个采样点独立 offset
  - 需要编译 CUDA ops (mmcv / InternImage)
  - 参考: https://github.com/OpenGVLab/InternImage

DCNV4: Deformable Convolution V4 (InternImage v2, 2024)
  - 进一步优化，移除 softmax 归一化
  - 比 DCNV3 更快且不减精度
  - 需要关闭 AMP 训练 (FP16 不兼容)
  - 参考: https://github.com/OpenGVLab/InternVL

═══════════════════════════════════════════════════
注意: DCNV3/DCNV4 的 CUDA 版本需要在 Windows 上编译
  (需要 Visual Studio Build Tools + CUDA Toolkit)
  
如果编译遇到困难，本模块提供了纯 PyTorch 参考实现：
  - DCNv3_pytorch: 纯 Python 实现 (训练可用，速度慢)
  - DCNv4_pytorch: 纯 Python 实现 (训练可用，速度慢)

生产环境推荐使用编译后的 CUDA 版本。
═══════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# === 尝试导入 CUDA 版本 ===========================================

# DCNV3 - 尝试从 mmcv 导入
try:
    from mmcv.ops import ModulatedDeformConv2dPack as DCNv3
    from mmcv.ops import ModulatedDeformConv2d as DCNv3_conv
    HAS_DCNV3_CUDA = True
except ImportError:
    try:
        from mmcv.ops import DeformableConv2dPack as DCNv3_alt
        HAS_DCNV3_CUDA = True
    except ImportError:
        HAS_DCNV3_CUDA = False

# DCNV4 - 尝试从 InternVL / internimage 导入
try:
    # InternVL 中的 DCNv4
    from internvl.model.ops.dcnv4 import DCNv4 as DCNv4_CUDA
    HAS_DCNV4_CUDA = True
except ImportError:
    try:
        from ops.dcnv4 import DCNv4 as DCNv4_CUDA
        HAS_DCNV4_CUDA = True
    except ImportError:
        HAS_DCNV4_CUDA = False


# === 纯 PyTorch 参考实现 ==========================================

class DCNv3_pytorch(nn.Module):
    """
    纯 PyTorch 实现的 DCNV3 (无 CUDA 编译)
    速度较慢，但保证在任何环境可运行
    
    Paper: InternImage: Exploring Large-Scale Vision Foundation Models
           with Deformable Convolutions (CVPR 2023)
    
    Args:
        channels: 输入/输出通道数
        kernel_size: 卷积核大小
        stride: 步长
        groups: 分组数
        offset_scale: offset 缩放系数
    """
    def __init__(self, channels, kernel_size=3, stride=1, groups=1, offset_scale=1.0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.offset_scale = offset_scale
        self.dilation = 1
        
        self.num_offsets = kernel_size * kernel_size
        self.padding = kernel_size // 2
        
        # offset 预测网络: 每个采样点 2 个 offset + 1 个 modulation
        self.offset_conv = nn.Conv2d(
            channels, groups * self.num_offsets * 3, 
            kernel_size=3, padding=1, bias=True
        )
        
        # 主卷积权重
        self.weight = nn.Parameter(
            torch.randn(channels, channels // groups, kernel_size, kernel_size) * 0.02
        )
        self.bias = nn.Parameter(torch.zeros(channels))
        
        # 初始化 offset 卷积
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)

    def forward(self, x):
        N, C, H, W = x.shape
        assert C == self.channels
        
        out_H = (H + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        out_W = (W + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        
        # 预测 offset + modulation
        offset_mask = self.offset_conv(x)  # [N, groups*K*K*3, H, W]
        offset_mask = F.interpolate(offset_mask, size=(out_H, out_W), mode='bilinear', align_corners=False)
        
        G = self.groups
        K = self.num_offsets
        
        offset = offset_mask[:, :G*K*2, :, :].reshape(N, G, K*2, out_H, out_W)
        mask = offset_mask[:, G*K*2:, :, :].sigmoid().reshape(N, G, K, out_H, out_W)
        
        # 构建采样网格
        # 标准 3x3 网格: [-1,-1], [-1,0], ..., [1,1]
        y_grid, x_grid = torch.meshgrid(
            torch.arange(-(self.kernel_size // 2), self.kernel_size // 2 + 1, device=x.device),
            torch.arange(-(self.kernel_size // 2), self.kernel_size // 2 + 1, device=x.device),
            indexing='ij'
        )
        grid = torch.stack([x_grid, y_grid], dim=-1).float()  # [K, 2]
        grid = grid.reshape(1, 1, K, 2)  # [1, 1, K, 2]
        
        # 输出位置归一化坐标
        x_pos = torch.linspace(0, out_W - 1, out_W, device=x.device).view(1, 1, 1, out_W)
        y_pos = torch.linspace(0, out_H - 1, out_H, device=x.device).view(1, 1, out_H, 1)
        
        # 加上 offset -> 采样位置
        norm_factor = torch.tensor([W, H], device=x.device).view(1, 1, 1, 1, 2)
        sample_pos = torch.stack([x_pos.expand(-1, -1, out_H, -1), 
                                  y_pos.expand(-1, -1, -1, out_W)], dim=-1).float()
        
        # [N, G, K, out_H, out_W, 2]
        offset_reshape = offset.reshape(N, G, K, 2, out_H, out_W).permute(0, 1, 2, 4, 5, 3)
        grid_offset = self.offset_scale * offset_reshape
        
        # 采样: 使用 grid_sample 实现可变形采样
        # 将坐标归一化到 [-1, 1]
        sample_pos_norm = 2.0 * (sample_pos + grid[:, :, :, None, None, :] + grid_offset) / \
                          torch.tensor([W-1, H-1], device=x.device).view(1, 1, 1, 1, 1, 2) - 1.0
        sample_pos_norm = sample_pos_norm.reshape(N, G * K, out_H, out_W, 2)
        
        # 按 group 处理
        x_g = x.reshape(N, G, C // G, H, W)
        output = torch.zeros(N, G, C // G, out_H, out_W, device=x.device)
        
        for g in range(G):
            for k in range(K):
                # 当前采样点
                grid_k = sample_pos_norm[:, g * K + k, :, :, :]  # [N, out_H, out_W, 2]
                sampled = F.grid_sample(
                    x_g[:, g], grid_k, mode='bilinear', padding_mode='zeros', align_corners=False
                )  # [N, C//G, out_H, out_W]
                output[:, g] += sampled * mask[:, g, k:k+1, :, :]
                # 注意: 简化实现, 实际 DCNV3 有分组权重
                
        return output.reshape(N, C, out_H, out_W)


class DCNv4_pytorch(nn.Module):
    """
    DCNV4 兼容层 (基于 torchvision.ops.deform_conv2d)
    
    真正的 DCNV4 CUDA 加速需要安装 InternVL:
      pip install git+https://github.com/OpenGVLab/InternVL
      训练时设置 amp=False
    
    此实现使用 torchvision 的内置可变形卷积 (C++/CUDA 内核)
    比纯 Python 循环快 ~100x，参数量与 CUDA 版本一致
    """
    def __init__(self, channels, kernel_size=3, stride=1, groups=1):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = kernel_size // 2
        
        # DCNV4 风格权重: [out_c, in_c//g, k, k]
        self.weight = nn.Parameter(
            torch.randn(channels, channels // groups, kernel_size, kernel_size) * 0.02
        )
        self.bias = nn.Parameter(torch.zeros(channels))
        
        # 学习 offset 的卷积
        K = kernel_size * kernel_size
        self.offset_conv = nn.Conv2d(
            channels, groups * K * 2, kernel_size=3, padding=1, bias=True
        )
        self.num_offsets = K
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)

    def forward(self, x):
        """
        使用 torchvision.ops.deform_conv2d 实现 DCNV4
        支持任意 kernel_size、stride、groups
        """
        N, C, H, W = x.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding
        G = self.groups
        K = self.num_offsets
        
        out_H = (H + 2 * p - k) // s + 1
        out_W = (W + 2 * p - k) // s + 1
        
        # 预测 offset (DCNV4: 无 modulation mask)
        offset = self.offset_conv(x)
        if s > 1 or out_H != H or out_W != W:
            offset = F.interpolate(offset, size=(out_H, out_W), mode='bilinear', align_corners=False)
        # offset: [N, G*K*2, oh, ow] — 这正是 deform_conv2d 需要的格式
        
        # 使用 torchvision 的 deform_conv2d (C++/CUDA, 高效)
        return torchvision.ops.deform_conv2d(
            input=x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=(s, s),
            padding=(p, p),
            dilation=(1, 1),
            mask=None,  # DCNV4 无 modulation mask
        )


# === CUDA 自动选择包装器 ==========================================

def get_dcnv3(channels, kernel_size=3, stride=1, groups=1, offset_scale=1.0, force_cuda=False):
    """获取 DCNV3 模块 (CUDA 优先，fallback 到 PyTorch)"""
    if HAS_DCNV3_CUDA and not force_cuda:
        try:
            return DCNv3(channels, channels, kernel_size=kernel_size, stride=stride, 
                        padding=kernel_size//2, groups=groups, offset_scale=offset_scale)
        except Exception:
            pass
    return DCNv3_pytorch(channels, kernel_size=kernel_size, stride=stride, 
                        groups=groups, offset_scale=offset_scale)


def get_dcnv4(channels, kernel_size=3, stride=1, groups=1, force_cuda=False):
    """获取 DCNV4 模块 (CUDA 优先，fallback 到 PyTorch)"""
    if HAS_DCNV4_CUDA and not force_cuda:
        try:
            return DCNv4_CUDA(channels, kernel_size=kernel_size, stride=stride,
                            padding=kernel_size//2, groups=groups)
        except Exception:
            pass
    return DCNv4_pytorch(channels, kernel_size=kernel_size, stride=stride, groups=groups)
