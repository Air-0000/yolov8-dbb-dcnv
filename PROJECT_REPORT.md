# 基于YOLOv8-DBB-DCNV4的血细胞智能检测系统

---

## 一、项目概述

### 1.1 项目背景

血细胞计数是临床检验的基础项目，常规血常规检查需人工在显微镜下识别红细胞（RBC）、白细胞（WBC）和血小板（Platelets）的数量与形态。传统方案存在以下痛点：

| 痛点 | 说明 |
|------|------|
| **效率低下** | 人工镜检每张涂片需5-10分钟，难以满足大规模筛查需求 |
| **主观差异** | 不同检验人员的判读标准不一致，复检率高达15-30% |
| **疲劳误判** | 长时间工作导致漏检率上升，尤其对小目标（血小板）易遗漏 |
| **自动化程度低** | 传统图像处理方法对细胞重叠、染色差异的鲁棒性差 |

### 1.2 项目目标

基于深度学习目标检测技术，构建血细胞自动识别系统，设定量化指标：

| 指标 | 目标值 | 考核方式 |
|------|--------|---------|
| 平均精度 mAP@0.5 | ≥80% | YOLO官方val评估 |
| 精确率 Precision | ≥75% | 混淆矩阵计算 |
| 召回率 Recall | ≥85% | 混淆矩阵计算 |
| 单图推理耗时 | ≤10ms | 批量测试平均 |
| 系统稳定性 | 连续运行100+次无崩溃 | 压力测试 |

### 1.3 团队分工

| 成员 | 职责 |
|------|------|
| 成员A | 方案设计、模型架构选型（DBB + DCNV4）、技术文档撰写 |
| 成员B | 算法建模、训练调参、性能优化、Git版本管理 |
| 成员C | 数据预处理、可视化界面搭建、报告撰写、视频录制 |

---

## 二、总体方案设计

### 2.1 总体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    数据模块                                  │
│  BCCD公开数据集 → YOLO格式标注 → 图像增强 → 训练/验证划分    │
│       (RGB血细胞图像)  (184+32)    (Mosaic等)   (85%/15%)   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    模型模块                                  │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │ Backbone    │ →  │ Neck/Head    │ →  │ 检测输出     │    │
│  │ DBB卷积     │    │ DCNV4        │    │ 框+类别+置信度 │   │
│  │ (重参数化)  │    │ (可变形卷积) │    │              │    │
│  └─────────────┘    └──────────────┘    └──────────────┘    │
│         ↓                                                      │
│  训练配置: batch=8, AdamW, epochs=50, amp=False               │
│  Blackwell GPU (RTX 5060 Ti) 加速训练                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               界面/交互模块                                   │
│  YOLO.predict() → 结果可视化 → 检测图像输出 → 统计报表       │
│  (带标注框+类别标签+置信度的效果图)                          │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 技术选型说明

| 技术 | 选型原因 | 优势 |
|------|---------|------|
| **Python 3.13** | 生态成熟，深度学习框架支持完善 | NumPy/PyTorch/OpenCV 原生支持 |
| **PyTorch 2.12** | 动态图计算，调试方便 | CUDA 13.3 加速，Blackwell GPU 支持 |
| **Ultralytics YOLOv8** | 工业级目标检测框架，开箱即用 | 完善的 DataLoader、Loss、评估体系 |
| **DBB (重参数化卷积)** | 提升特征提取能力 | 训练时分叉丰富梯度，推理时融合为单路，零额外开销 |
| **DCNV4 (可变形卷积V4)** | 适应细胞形态变化 | 可变形感受野，对圆形/椭圆细胞更鲁棒 |
| **CUDA 13.3** | RTX 5060 Ti Blackwell 架构 | 16GB显存支撑 batch=8 训练 |

### 2.3 模块划分

| 模块 | 功能 | 技术栈 | 负责人 |
|------|------|--------|--------|
| 数据模块 | 数据集加载、增强、缓存 | YOLODataset, Albumentations | 成员C |
| 模型模块 | DBB + DCNV4 网络定义 | PyTorch, Ultralytics | 成员B |
| 训练模块 | 手动训练循环、损失计算、参数更新 | PyTorch, AdamW | 成员B |
| 评估模块 | mAP/Precision/Recall 计算 | Ultralytics val | 成员A |
| 推理模块 | 实时检测、结果可视化 | YOLO.predict(), OpenCV | 成员C |
| 版本管理 | 代码管理、实验追踪 | Git, GitHub | 成员A |

---

## 三、详细实现过程

### 3.1 数据模块

**数据集来源：** BCCD (Blood Cell Count Dataset) 公开数据集，含 364 张血细胞显微图像，3 类标注。

**数据清洗与预处理：**

```python
# YOLODataset 自动完成：图像归一化、尺寸调整、Mosaic增强
dataset = YOLODataset(
    img_path='data/images/train',
    imgsz=640,           # 统一输入尺寸
    augment=True,        # Albumentations 自动增强
    data=data_cfg,
    stride=32,
)
```

**数据增强策略（Albumentations）：**
- Blur（模糊，p=0.01）
- MedianBlur（中值模糊，p=0.01）
- ToGray（灰度化，p=0.01）
- CLAHE（直方图均衡，p=0.01）

**数据集划分：**

| 子集 | 图像数 | 标注数 | 占比 |
|------|--------|--------|------|
| 训练集 | 184 | 2,518 | 85% |
| 验证集 | 32 | 459 | 15% |

**类别分布：**

| 类别 | 训练集 | 验证集 | 平均每图 |
|------|--------|--------|---------|
| RBC (红细胞) | 2,170 | 399 | ~13 |
| WBC (白细胞) | 174 | 32 | ~1 |
| Platelets (血小板) | 174 | 28 | ~1 |

### 3.2 模型模块

#### 3.2.1 模型基础原理

**YOLOv8 架构** 采用 Anchor-Free 检测范式，核心包括：
1. **Backbone**：CSPDarknet 结构提取多尺度特征
2. **Neck**：FPN+PAN 特征金字塔融合多尺度信息
3. **Head**：解耦头分别预测分类和回归分支

**DBB (Diverse Branch Block) — 重参数化卷积：**

训练阶段使用多分支结构（1×1卷积、3×3卷积、平均池化等并行分支），通过不同感受野丰富梯度流。推理阶段将所有分支融合为单一3×3卷积，保持原始推理速度。

**DCNV4 (Deformable Convolution V4) — 可变形卷积：**

在特征图每个采样点引入可学习的偏移量，使卷积核能自适应形变，特别适合血细胞这类形态各异的圆形目标。

#### 3.2.2 网络配置

```yaml
# cfgs/yolov8-DBB-DCNV4.yaml 核心配置
nc: 3  # 3类：RBC, WBC, Platelets
scales:
  n: [0.33, 0.25, 1024]  # Nano 规模，~3.7M参数

# Backbone: DBB 提升特征提取
backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 3, C2f_DBB, [128, True]]    # ← DBB
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 6, C2f_DBB, [256, True]]    # ← DBB
  # ... (深层同理)

# Head: DCNV4 增强几何适应
head:
  - [-1, 1, nn.Upsample, [2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 3, C2f_DCNV4, [512]]       # ← DCNV4
  # ... (P4/P5层同理)
```

#### 3.2.3 训练参数设置

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | 8 | 充分利用16GB显存 |
| epochs | 50 | 验证集收敛后停止 |
| imgsz | 640 | 输入图像尺寸 |
| optimizer | AdamW | 权重衰减正则化 |
| lr0 | 0.001 | 初始学习率 |
| weight_decay | 0.0005 | L2正则化系数 |
| warmup_epochs | 3 | 学习率预热 |
| amp | False | DCNV4需关闭混合精度 |
| deterministic | True | Blackwell GPU稳定性 |

#### 3.2.4 核心代码（手动训练循环）

```python
# 手动训练循环（绕过YOLO.train()在Blackwell GPU的segfault）
for epoch in range(EPOCHS):
    for batch_idx, batch in enumerate(dataloader):
        # 数据搬至GPU
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        batch['img'] = batch['img'].float() / 255.0

        # 前向 + 损失 + 反向
        optimizer.zero_grad()
        pred = model(batch['img'])
        loss_tuple = model.loss(batch, pred)
        loss = loss_tuple[0].sum()
        loss.backward()
        optimizer.step()
```

#### 3.2.5 训练过程

```
E01: loss=41.2  [box=12.5 cls=14.8 dfl=13.9]
E10: loss=35.1  [box=10.2 cls=12.4 dfl=12.5]
E20: loss=34.5  [box=10.1 cls=11.8 dfl=12.6]
E30: loss=33.5  [box=9.8  cls=11.2 dfl=12.5]
E40: loss=32.7  [box=9.6  cls=10.8 dfl=12.3]
E50: loss=30.5  [box=9.0  cls=9.8  dfl=11.7]
```

三个分支（box回归 + 分类 + DFL）同步下降，训练充分。

#### 3.2.6 性能指标

| 类别 | 精确率(P) | 召回率(R) | mAP@0.5 | mAP@0.5:0.95 |
|------|:---------:|:---------:|:-------:|:------------:|
| **整体** | **0.819** | **0.905** | **0.840** | **0.513** |
| RBC 红细胞 | 0.790 | 0.752 | 0.760 | 0.497 |
| WBC 白细胞 | **0.970** | **1.000** | **0.973** | 0.683 |
| Platelets 血小板 | 0.698 | 0.964 | 0.786 | 0.359 |

**全部达到甚至超越目标值：**
- mAP@0.5 = 84.0% > 目标80% ✅
- 召回率 = 90.5% > 目标85% ✅
- 精确率 = 81.9% > 目标75% ✅
- 单图推理 = 2.6ms < 目标10ms ✅

### 3.3 大模型模块

（本阶段未集成大模型，留作后续拓展方向）

**规划应用场景：**
- **报告生成**：根据检测结果自动生成血常规分析报告
- **异常预警**：当特定细胞计数超出正常范围时，大模型解读可能的临床意义
- **智能问答**：对检测结果提供自然语言解释

### 3.4 界面/交互模块

基于 YOLO 内置 `predict()` 接口实现推理可视化：

```python
results = model.predict(
    source=val_images,
    conf=0.25,       # 置信度阈值
    iou=0.45,        # NMS阈值
    imgsz=640,
    device='cuda:0',
    save=True,       # 自动保存标注结果
)
```

**效果图输出：**
- 每张图像自动绘制检测框 + 类别标签 + 置信度
- 框颜色：RBC=绿色, WBC=蓝色, Platelets=紫色

---

## 四、测试与结果分析

### 4.1 功能测试

| 测试用例 | 预期效果 | 实际结果 | 状态 |
|----------|---------|---------|:----:|
| 单张血细胞图像检测 | 正确框出所有细胞 | 22/22 个目标检出 | ✅ |
| 多类别识别（RBC/WBC/血小板） | 区分3类细胞 | 精确区分，WBC=0.97 P/R | ✅ |
| 低置信度过滤 | conf<0.25的忽略 | 过滤有效，精度81.9% | ✅ |
| 重叠目标处理 | NMS去除重复框 | NMS正常，无重复框 | ✅ |
| 32张验证集批量推理 | 全部完成 | 32/32 全部成功 | ✅ |
| 异常输入（空白图像） | 返回0检测结果 | 无报错，返回空结果 | ✅ |

### 4.2 性能测试

| 指标 | 测试值 | 环境 |
|------|--------|------|
| 模型参数量 | 3.67M | - |
| 计算量 | 10.3 GFLOPs | - |
| 单图推理耗时 | **2.6ms** | RTX 5060 Ti @ 480×640 |
| 预处理耗时 | 0.9ms | Resize + Normalize |
| 后处理耗时 | 1.0ms | NMS + 框解码 |
| 50 epochs训练总耗时 | **~3min** | batch=8 |
| GPU峰值显存 | ~4.2GB | batch=8, imgsz=640 |

### 4.3 稳定性测试

| 测试场景 | 运行表现 | 容错处理 |
|----------|---------|---------|
| 空图像输入 | 返回空检测结果 | OpenCV imread 返回 None 时跳过 |
| 非图像文件输入 | 自动跳过并告警 | Ultralytics 内置异常处理 |
| 连续推理100次 | 无显存泄漏 | torch.no_grad() 避免计算图累积 |
| GPU显存不足 | OOM自动恢复 | workers=0 避免DataLoader预加载溢出 |
| Blackwell (sm_120) | 需关闭cuDNN autotune | deterministic=True + benchmark=False |

---

## 五、总结与展望

### 5.1 项目完成情况

| 初始目标 | 完成情况 | 完成度 |
|----------|---------|:------:|
| 血细胞多类检测 | RBC/WBC/Platelets全检 | 100% |
| mAP@0.5 ≥ 80% | **84.0%** | 超额完成 |
| 精确率 ≥ 75% | **81.9%** | 超额完成 |
| 召回率 ≥ 85% | **90.5%** | 超额完成 |
| 推理速度 ≤ 10ms | **2.6ms** | 超额完成 |

**解决的实际工程问题：**
1. 血细胞自动计数替代人工镜检，提升效率
2. 统一检测标准，消除主观差异
3. 小目标（血小板）的高召回率检测

### 5.2 关键技术问题与解决方案

| 问题 | 分析过程 | 解决方案 |
|------|---------|---------|
| Blackwell GPU segfault | YOLO.train()在cuDNN autotune初始化时崩溃 | 手动训练循环绕过DetectionTrainer |
| 分类分支不学习 | 发现box_loss=dfl_loss=0，assigner未分配到正样本 | `classes=`参数误传为过滤条件，去掉后修复 |
| YOLO.yaml nc=80 vs 实际3类 | 模型输出144通道，分类头80类，数据仅3类正样本 | 改yaml nc: 3，匹配数据类别数 |
| 训练输出无缓冲 | 后台进程输出无法实时查看 | 添加PYTHONUNBUFFERED=1 + tee日志 |
| Git push超时 | 516MB对象含checkpoint和runs/目录 | 添加.gitignore排除大文件后清理回收 |

### 5.3 不足与改进方向

| 不足 | 原因 | 改进方向 |
|------|------|---------|
| 血小板ap@0.5:0.95仅0.359 | 目标占像素极少，IoU计算对偏差敏感 | 多尺度训练+大分辨率输入 |
| 训练loss收敛到30停滞 | 模型容量有限（3.7M参数） | 升级到m/l规模或加入注意力机制 |
| 无大模型交互 | 本阶段聚焦基础检测 | 集成LangChain+本地LLM实现报告解读 |
| 无Web界面 | 仅CLI运行 | Gradio/Streamlit快速搭建演示界面 |
| 数据集量小（184张） | BCCD为小型公开数据集 | 数据增强增强、MixUp/CutMix、半监督 |

**拓展方向：**
1. 集成大模型（如DeepSeek）自动生成血常规分析报告
2. 加入时序检测，实现视频流实时血细胞追踪
3. 部署为Web服务（Flask/FastAPI），支持在线API调用

---

## 附录

### A. 项目仓库

GitHub: https://github.com/Air-0000/yolov8-dbb-dcnv

### B. 核心技术代码（完整版）

**train_dcnv4.py** — 手动训练循环核心实现
```python
import sys, os, torch, yaml, time
from types import SimpleNamespace
from ultralytics import YOLO
from ultralytics.data import YOLODataset
from torch.utils.data import DataLoader

# 初始化
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda:0'
model = YOLO('cfgs/yolov8-DBB-DCNV4.yaml').model.to(device)
model.nc = 3; model.names = ['RBC', 'WBC', 'Platelets']
model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, ...)
model.train()

# 数据集
dataset = YOLODataset(img_path='data/images/train', imgsz=640,
                      augment=True, data=data_cfg, stride=32)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True,
                        num_workers=0)

# 训练循环
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
for epoch in range(50):
    for batch in dataloader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        batch['img'] = batch['img'].float() / 255.0
        optimizer.zero_grad()
        pred = model(batch['img'])
        loss = model.loss(batch, pred)[0].sum()
        loss.backward()
        optimizer.step()
```

### C. 数据集说明

BCCD (Blood Cell Count Dataset)
- 来源：https://github.com/Shenggan/BCCD_Dataset
- 图像数：364张（JPEG，~20KB/张）
- 标注格式：YOLO（`cls cx cy w h`）
- 类别：RBC(0), WBC(1), Platelets(2)
- 分辨率：640×480（原始）

### D. 项目效果截图

| 检测效果示例 | 混淆矩阵 |
|:-----------:|:--------:|
| （见文件 runs/infer_results/val_00.jpg） | （见 runs/detect/val/confusion_matrix.png） |

### E. 推理性能

```
每张图像 ~2.6ms 推理（RTX 5060 Ti）
├── 预处理: 0.9ms (Resize 640×640 + Normalize)
├── 模型推理: 2.6ms (3.7M参数, 10.3 GFLOPs)
└── 后处理: 1.0ms (NMS + 框解码)
```
