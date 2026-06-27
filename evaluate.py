"""Evaluate DBB-DCNV4 on val set: compute precision, recall, mAP"""
import sys, os, torch, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules import register_modules
register_modules()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from ultralytics import YOLO
from ultralytics.utils.torch_utils import init_seeds
import cv2
import numpy as np

init_seeds()
device = 'cuda:0'

# --- Load model ---
model = YOLO('cfgs/yolov8-DBB-DCNV4.yaml', task='detect')
ckpt = torch.load('runs/medical/manual_epoch50.pt', map_location=device, weights_only=True)
model.model.load_state_dict(ckpt['model_state_dict'])
with open('data/data.yaml') as f:
    data_cfg = yaml.safe_load(f)
model.model.nc = 3
model.model.names = data_cfg['names']

names = data_cfg['names']
NUM_CLASSES = 3

# --- Load val images and GT ---
val_dir = 'data/images/val'
val_imgs = sorted([os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.jpg')])

def load_gt(img_path):
    """Load YOLO format GT labels, return list of [cls, cx, cy, w, h]"""
    basename = os.path.splitext(os.path.basename(img_path))[0]
    objs = []
    for lbl_dir in ['data/labels/val', 'data/labels']:
        lbl_path = os.path.join(lbl_dir, basename + '.txt')
        if os.path.isfile(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        objs.append([float(p) for p in parts[:5]])
            break
    return objs

def yolo_to_xyxy(bbox, img_w, img_h):
    """YOLO cx,cy,w,h -> x1,y1,x2,y2"""
    cx, cy, w, h = bbox
    x1 = (cx - w/2) * img_w
    y1 = (cy - h/2) * img_h
    x2 = (cx + w/2) * img_w
    y2 = (cy + h/2) * img_h
    return [x1, y1, x2, y2]

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

# --- Evaluate ---
all_preds = []  # (img_id, cls, conf, box_xyxy)
all_gts = []    # (img_id, cls, box_xyxy)

print(f'Evaluating {len(val_imgs)} val images...')
for idx, img_path in enumerate(val_imgs):
    # Load GT
    gt_objs = load_gt(img_path)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    for gt in gt_objs:
        cls_id = int(gt[0])
        box = yolo_to_xyxy(gt[1:], w, h)
        all_gts.append((idx, cls_id, box))

    if (idx + 1) % 10 == 0:
        print(f'  processed {idx+1}/{len(val_imgs)}')

# Run predict on all images at once
results = model.predict(
    source=val_imgs,
    conf=0.001, iou=0.45, imgsz=640, device=device, amp=False,
    verbose=False, max_det=300,
)

print(f'Processing {len(results)} prediction results...')
for idx, r in enumerate(results):
    if r.boxes is not None and len(r.boxes) > 0:
        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        for cls_id, conf, box in zip(cls_ids, confs, boxes_xyxy):
            all_preds.append((idx, cls_id, conf, box.tolist()))

print(f'\nTotal GT: {len(all_gts)}, Total predictions: {len(all_preds)}')

# --- Compute AP per class ---
IOU_THRESH = 0.5

for cls_id in range(NUM_CLASSES):
    # Filter GT for this class
    gt_cls = [(gid, box) for gid, c, box in all_gts if c == cls_id]
    # Filter predictions for this class, sorted by confidence descending
    pred_cls = [(pid, conf, box) for pid, c, conf, box in all_preds if c == cls_id]
    pred_cls.sort(key=lambda x: -x[1])  # high conf first

    # Track matched GT
    gt_matched = set()  # (img_id, gt_idx)
    
    tp = 0
    fp = 0
    total_gt = len(gt_cls)

    precision_vals = []
    recall_vals = []

    for pid, conf, pred_box in pred_cls:
        # Find best matching GT in same image
        best_iou = 0
        best_gt_idx = -1
        for gidx, (gid, gt_box) in enumerate(gt_cls):
            if gid == pid and (pid, gidx) not in gt_matched:
                i = iou(pred_box, gt_box)
                if i > best_iou:
                    best_iou = i
                    best_gt_idx = gidx

        if best_iou >= IOU_THRESH and best_gt_idx >= 0:
            tp += 1
            gt_matched.add((pid, best_gt_idx))
        else:
            fp += 1

        # At each prediction, compute interpolated precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / total_gt if total_gt > 0 else 0
        precision_vals.append(precision)
        recall_vals.append(recall)

    # Compute AP (11-point interpolation)
    if total_gt == 0:
        ap = 0.0
    else:
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            # Find max precision for recall >= t
            prec_at_recall = 0
            for p, r in zip(precision_vals, recall_vals):
                if r >= t:
                    prec_at_recall = max(prec_at_recall, p)
            ap += prec_at_recall / 11

    final_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    final_recall = tp / total_gt if total_gt > 0 else 0

    print(f'\n--- {names[cls_id]} (class {cls_id}) ---')
    print(f'  GT: {total_gt}, TP: {tp}, FP: {fp}, FN: {total_gt - tp}')
    print(f'  Precision: {final_precision:.3f} ({tp}/{tp+fp})')
    print(f'  Recall:    {final_recall:.3f} ({tp}/{total_gt})')
    print(f'  AP@0.5:    {ap:.3f}')

# Overall mAP
total_tp = sum(1 for pid, c, conf, box in all_preds if any(
    iou(box, gt_box) >= IOU_THRESH and c == gt_c and pid == gt_pid
    for gt_pid, gt_c, gt_box in all_gts
))
# This is approximate - let me use the per-class totals
print(f'\n{"="*40}')
print(f'OVERALL @ IoU={IOU_THRESH}')
print(f'Total GT: {len(all_gts)}, Total Pred: {len(all_preds)}')
