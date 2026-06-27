"""
YOLOv8 + DBB + DCNV3/DCNV4 训练/验证脚本

用法:
  # 训练 DBB+DCNV4 精度优先模型
  python train.py --cfg cfgs/yolov8-DBB-DCNV4.yaml --scale l --epochs 300
  
  # 训练标准 DBB 模型
  python train.py --cfg cfgs/yolov8-C2f-DBB.yaml --scale m --epochs 300
  
  # 只验证模型结构
  python train.py --cfg cfgs/yolov8-DBB-DCNV4.yaml --dry-run

  # 导出为 ONNX/TensorRT
  python train.py --cfg cfgs/yolov8-DBB-DCNV4.yaml --export
"""

import argparse
import sys
import os

# 确保项目在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 注册自定义模块 (必须在导入 ultralytics 之前或之后立即调用)
from modules import register_modules
register_modules()

from ultralytics import YOLO
from ultralytics import checks


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 + DBB + DCNV4 训练')
    parser.add_argument('--cfg', type=str, default='cfgs/yolov8-DBB-DCNV4.yaml',
                        help='模型配置文件')
    parser.add_argument('--scale', type=str, default='m', choices=['n', 's', 'm', 'l', 'x'],
                        help='模型尺度 (n/s/m/l/x)')
    parser.add_argument('--data', type=str, default='coco.yaml',
                        help='数据集配置')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图像尺寸')
    parser.add_argument('--device', type=str, default='0',
                        help='设备 (0=GPU, cpu)')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        help='优化器 (SGD, Adam, AdamW)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='初始学习率')
    parser.add_argument('--weights', type=str, default=None,
                        help='预训练权重路径')
    parser.add_argument('--project', type=str, default='runs/detect',
                        help='项目保存目录')
    parser.add_argument('--name', type=str, default=None,
                        help='实验名称')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='启用混合精度训练 (DCNV4 需关闭)')
    parser.add_argument('--no-amp', action='store_false', dest='amp',
                        help='禁用混合精度 (DCNV4 推荐关闭)')
    parser.add_argument('--dry-run', action='store_true',
                        help='仅验证模型结构，不训练')
    parser.add_argument('--export', action='store_true',
                        help='导出模型为 ONNX/TensorRT')
    parser.add_argument('--resume', action='store_true',
                        help='从断点恢复训练')
    parser.add_argument('--val', action='store_true',
                        help='仅验证')
    
    args = parser.parse_args()
    
    # 检查 DCNV4 配置是否需要关闭 AMP
    if 'DCNV4' in args.cfg and args.amp:
        print("⚠ 检测到 DCNV4 配置，建议关闭 AMP (使用 --no-amp)")
        print("   DCNV4 在 FP16 模式下可能不兼容\n")
    
    # === DRY RUN: 仅验证模型结构 ===
    if args.dry_run:
        print(f"\n{'═'*60}")
        print(f"  模型结构验证: {args.cfg} (scale={args.scale})")
        print(f"{'═'*60}\n")
        
        model = YOLO(args.cfg)
        
        print(f"\n{'─'*60}")
        print(f"  模型: {model.model.model_name if hasattr(model.model, 'model_name') else 'YOLOv8 Custom'}")
        print(f"  参数量: {sum(p.numel() for p in model.model.parameters()):,}")
        print(f"  可训练参数: {sum(p.numel() for p in model.model.parameters() if p.requires_grad):,}")
        print(f"{'─'*60}\n")
        
        # 测试 forward
        import torch
        dummy = torch.randn(1, 3, args.imgsz, args.imgsz)
        
        # CPU 前向测试
        print("  测试 forward pass (CPU)...")
        out = model.model(dummy)
        if isinstance(out, list):
            for i, o in enumerate(out):
                print(f"    输出 {i}: {o.shape}")
        else:
            print(f"    输出: {out.shape if hasattr(out, 'shape') else type(out)}")
        
        # 打印层信息
        print(f"\n{'─'*60}")
        print(f"  层列表 (前10层):")
        for i, (name, layer) in enumerate(list(model.model.named_modules())[:10]):
            params = sum(p.numel() for p in layer.parameters())
            print(f"    {i:3d}. {name:40s} params={params:>8,}")
        print(f"  ... (共 {len(list(model.model.named_modules())):,} 层)")
        
        print(f"\n{'═'*60}")
        print(f"  ✅ 模型结构验证通过!")
        print(f"  总参数量: {sum(p.numel() for p in model.model.parameters()):,}")
        print(f"{'═'*60}\n")
        return
    
    # === EXPORT ===
    if args.export:
        print(f"\n{'═'*60}")
        print(f"  导出模型: {args.cfg}")
        print(f"{'═'*60}\n")
        
        weights = args.weights or f'runs/detect/{args.name or "train"}/weights/best.pt'
        model = YOLO(weights)
        
        print("  导出 ONNX...")
        model.export(format='onnx', imgsz=args.imgsz)
        
        print("  导出 TensorRT...")
        model.export(format='engine', imgsz=args.imgsz, half=args.amp)
        
        print(f"\n✅ 导出完成!\n")
        return
    
    # === VALIDATE ===
    if args.val:
        weights = args.weights or f'runs/detect/{args.name or "train"}/weights/best.pt'
        model = YOLO(weights)
        results = model.val(data=args.data, device=args.device, batch=args.batch, imgsz=args.imgsz)
        return
    
    # === TRAIN ===
    print(f"\n{'═'*60}")
    print(f"  YOLOv8 + DBB/DCNV 训练开始")
    print(f"{'═'*60}")
    print(f"  配置: {args.cfg}")
    print(f"  尺度: {args.scale}")
    print(f"  数据: {args.data}")
    print(f"  批次: {args.batch} | 轮数: {args.epochs}")
    print(f"  设备: {args.device} | AMP: {args.amp}")
    print(f"  优化器: {args.optimizer} | LR: {args.lr}")
    print(f"{'─'*60}\n")
    
    # 构建模型
    if args.resume:
        model = YOLO(args.weights or f'{args.project}/{args.name or "train"}/weights/last.pt')
    else:
        model = YOLO(args.cfg)
        if args.weights:
            print(f"  加载预训练权重: {args.weights}")
            model = YOLO(args.weights)
    
    # 开始训练
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        optimizer=args.optimizer,
        lr0=args.lr,
        project=args.project,
        name=args.name,
        amp=args.amp,
        resume=args.resume,
        scale=args.scale,
        close_mosaic=10,          # 最后10轮关闭 mosaic
        deterministic=False,       # 提高速度
        workers=4,
        cache='ram',              # 缓存到内存加速
        warmup_epochs=3,
        cos_lr=True,              # 余弦学习率衰减
        val=True,
        save=True,
        save_period=10,           # 每10轮保存一次
    )
    
    print(f"\n{'═'*60}")
    print(f"  训练完成!")
    print(f"{'═'*60}\n")


if __name__ == '__main__':
    main()
