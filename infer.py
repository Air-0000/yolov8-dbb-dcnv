"""Inference: use YOLO.predict() with trained weights"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules import register_modules
register_modules()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from ultralytics import YOLO
from ultralytics.utils.torch_utils import init_seeds
init_seeds()

device = 'cuda:0'
print(f'Device: {device}')

# Build model (must match training architecture)
model = YOLO('cfgs/yolov8-DBB-DCNV4.yaml', task='detect')
# Override nc for detection head
with open('data/data.yaml') as f:
    import yaml
    data_cfg = yaml.safe_load(f)

# Load checkpoint weights into raw model
ckpt = torch.load('runs/medical/manual_epoch50.pt', map_location=device, weights_only=True)
model.model.load_state_dict(ckpt['model_state_dict'])
model.model.nc = data_cfg['nc']
model.model.names = data_cfg['names']
model.predictor = None  # force fresh predictor
print(f'Loaded epoch {ckpt["epoch"]} | nc={data_cfg["nc"]} names={data_cfg["names"]}')

# Run predict on all val images
val_dir = 'data/images/val'
val_imgs = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.jpg')]
val_imgs.sort()
print(f'Running predict on {len(val_imgs)} val images...')

results = model.predict(
    source=val_imgs,
    conf=0.25,
    iou=0.45,
    imgsz=640,
    device=device,
    save=True,
    project='runs/infer_results',
    name='predict',
    exist_ok=True,
    amp=False,
)

print(f'\nDone! Results saved to runs/infer_results/predict/')

# Print stats
total_det = 0
for r in results:
    total_det += len(r.boxes)
print(f'Total detections: {total_det} across {len(results)} images')
print(f'Avg detections per image: {total_det/max(len(results),1):.1f}')
