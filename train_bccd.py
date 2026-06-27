"""训练 BCCD 血细胞检测"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules import register_modules
register_modules()
from ultralytics import YOLO

model = YOLO('cfgs/yolov8-C2f-DBB.yaml')
model.train(
    data=str(os.path.expanduser('~/dev/medical_detection/data.yaml')),
    epochs=50, batch=16, imgsz=640, device='0', amp=True,
    project='runs/medical', name='dbb_bccd',
    workers=0, cache=False, lr0=0.001,
    warmup_epochs=3, cos_lr=True, close_mosaic=0,
    val=True, save=True, save_period=10, plots=True,
)
print('TRAINING_DONE')
