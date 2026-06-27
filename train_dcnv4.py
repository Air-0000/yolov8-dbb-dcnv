"""Training script for DBB-DCNV4 on BCCD

NOTE: Uses manual training loop instead of YOLO.train() to avoid
segfault on Blackwell (RTX 5060 Ti) GPUs. YOLO.train() crashes during
internal setup_model() due to cuDNN graph compilation issues on sm_120.
"""
import sys, os, torch, yaml, time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules import register_modules
register_modules()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from ultralytics import YOLO
from ultralytics.data import YOLODataset
from ultralytics.utils.torch_utils import init_seeds
from torch.utils.data import DataLoader

init_seeds()
device = 'cuda:0'
print(f'Device: {device}')

# --- Build model ---
model = YOLO('cfgs/yolov8-DBB-DCNV4.yaml', task='detect').model.to(device)

with open(os.path.expanduser('~/dev/medical_detection/data.yaml')) as f:
    data_cfg = yaml.safe_load(f)
model.nc = data_cfg['nc']
model.names = data_cfg['names']
model.args = SimpleNamespace(
    box=7.5, cls=0.5, cls_pw=1.0, obj=0.7, dfl=1.5, fl_gamma=0.0,
    lr0=0.001, lrf=0.01, momentum=0.937, weight_decay=0.0005,
    warmup_epochs=3, warmup_momentum=0.8, warmup_bias_lr=0.1,
)
model.train()
print(f'Model ready | nc={model.nc} | params={sum(p.numel() for p in model.parameters())/1e6:.1f}M')

# --- Dataset ---
train_dir = os.path.join(data_cfg['path'], data_cfg['train'])
dataset = YOLODataset(
    img_path=train_dir, imgsz=640, augment=True,
    classes=data_cfg['nc'], data=data_cfg, stride=32,
)
dataloader = DataLoader(
    dataset, batch_size=8, shuffle=True, num_workers=0,
    collate_fn=getattr(dataset, 'collate_fn', None),
)
print(f'Dataset: {len(dataset)} images, {len(dataloader)} batches')

# --- Optimizer ---
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)

# --- Training ---
os.makedirs('runs/medical', exist_ok=True)
EPOCHS = 50
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    n_batches = 0
    t0 = time.time()

    for batch_idx, batch in enumerate(dataloader):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        batch['img'] = batch['img'].float() / 255.0

        optimizer.zero_grad()
        pred = model(batch['img'])
        loss = model.loss(batch, pred)[0].sum()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        epoch_loss += loss_val
        n_batches += 1

        if (batch_idx + 1) % 15 == 0:
            print(f'E{epoch+1:02d}/{EPOCHS} B{batch_idx+1}/{len(dataloader)} loss={loss_val:.4f}')

    avg_loss = epoch_loss / max(n_batches, 1)
    elapsed = time.time() - t0
    print(f'E{epoch+1:02d}/{EPOCHS} avg_loss={avg_loss:.4f} time={elapsed:.1f}s')

    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f'runs/medical/manual_epoch{epoch+1:02d}.pt')
        print(f'Checkpoint saved: epoch {epoch+1}')

print('TRAINING DONE')
