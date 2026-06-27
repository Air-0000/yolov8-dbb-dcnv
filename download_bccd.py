"""
BCCD 血细胞检测数据集下载 & YOLO 格式转换

BCCD (Blood Cell Count and Detection) Dataset
- 3 classes: RBC (红血球), WBC (白血球), Platelets (血小板)
- 364 张血细胞显微图像
- Pascal VOC XML 格式 → 转换为 YOLO txt 格式

用法:
    python download_bccd.py
"""

import os
import requests
import xml.etree.ElementTree as ET
import random
from pathlib import Path

# ===== 配置 =====
DATA_DIR = Path(os.path.expanduser('~/dev/medical_detection'))
IMG_DIR = DATA_DIR / 'images'
LABEL_DIR = DATA_DIR / 'labels'
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

# BCCD 类别
CLASSES = ['RBC', 'WBC', 'Platelets']
CLASS_MAP = {name: idx for idx, name in enumerate(CLASSES)}

# GitHub raw 前缀
RAW_BASE = 'https://raw.githubusercontent.com/Shenggan/BCCD_Dataset/master/BCCD'

# 图片总数
NUM_IMAGES = 364


def download_all():
    """下载所有图片和标注"""
    downloaded = 0
    for idx in range(NUM_IMAGES):
        fname = f'BloodImage_{idx:05d}'
        
        # 下载图片
        img_url = f'{RAW_BASE}/JPEGImages/{fname}.jpg'
        img_path = IMG_DIR / f'{fname}.jpg'
        
        if not img_path.exists():
            try:
                r = requests.get(img_url, timeout=10)
                if r.status_code == 200:
                    with open(img_path, 'wb') as f:
                        f.write(r.content)
                else:
                    continue
            except:
                continue
        
        # 下载 XML 标注
        xml_url = f'{RAW_BASE}/Annotations/{fname}.xml'
        xml_path = IMG_DIR / f'{fname}.xml'
        
        if not xml_path.exists():
            try:
                r = requests.get(xml_url, timeout=10)
                if r.status_code == 200:
                    with open(xml_path, 'wb') as f:
                        f.write(r.content)
            except:
                continue
        
        downloaded += 1
        if downloaded % 50 == 0:
            print(f'  已下载: {downloaded}/{NUM_IMAGES}')
    
    print(f'✅ 下载完成: {downloaded} 张图片')


def convert_xml_to_yolo(xml_path, img_width, img_height):
    """将 Pascal VOC XML 转换为 YOLO 格式的标注"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    yolo_annots = []
    for obj in root.findall('object'):
        cls_name = obj.find('name').text
        if cls_name not in CLASS_MAP:
            continue
        cls_id = CLASS_MAP[cls_name]
        
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        # YOLO 格式: class x_center y_center width height (归一化到 [0,1])
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        w = (xmax - xmin) / img_width
        h = (ymax - ymin) / img_height
        
        yolo_annots.append(f'{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}')
    
    return yolo_annots


def convert_all():
    """将所有 XML 标注转换为 YOLO 格式"""
    converted = 0
    for xml_path in sorted(IMG_DIR.glob('*.xml')):
        fname = xml_path.stem
        img_path = IMG_DIR / f'{fname}.jpg'
        
        if not img_path.exists():
            continue
        
        # 获取图片尺寸
        from PIL import Image
        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except:
            continue
        
        # 转换
        yolo_lines = convert_xml_to_yolo(xml_path, w, h)
        if not yolo_lines:
            continue
        
        # 写入 YOLO txt
        txt_path = LABEL_DIR / f'{fname}.txt'
        with open(txt_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
        
        converted += 1
        
        # 删除 XML 释放空间
        xml_path.unlink()
    
    print(f'✅ 转换完成: {converted} 张标注')


def split_dataset(val_ratio=0.15):
    """划分训练集和验证集"""
    images = sorted(IMG_DIR.glob('*.jpg'))
    random.seed(42)
    random.shuffle(images)
    
    n_val = max(1, int(len(images) * val_ratio))
    val_imgs = set(images[:n_val])
    
    # 创建训练/验证列表
    train_lines = []
    val_lines = []
    
    for img_path in images:
        fname = img_path.stem
        txt_path = LABEL_DIR / f'{fname}.txt'
        if not txt_path.exists():
            continue
        
        if img_path in val_imgs:
            val_lines.append(f'data/images/{fname}.jpg')
        else:
            train_lines.append(f'data/images/{fname}.jpg')
    
    # 写入列表文件
    with open(DATA_DIR / 'train.txt', 'w') as f:
        f.write('\n'.join(train_lines))
    with open(DATA_DIR / 'val.txt', 'w') as f:
        f.write('\n'.join(val_lines))
    
    print(f'✅ 划分完成: 训练 {len(train_lines)} 张, 验证 {len(val_lines)} 张')


def create_data_yaml():
    """创建数据集 YAML 配置文件"""
    yaml_content = f"""# BCCD 血细胞检测数据集
train: {DATA_DIR.as_posix()}/train.txt
val: {DATA_DIR.as_posix()}/val.txt

nc: {len(CLASSES)}
names: {CLASSES}
"""
    with open(DATA_DIR / 'data.yaml', 'w') as f:
        f.write(yaml_content)
    print(f'✅ data.yaml 已创建')


def main():
    print('=' * 50)
    print('BCCD 血细胞检测数据集 下载 & 转换')
    print('=' * 50)
    
    print('\n1/4 下载数据...')
    download_all()
    
    print('\n2/4 转换标注格式...')
    convert_all()
    
    print('\n3/4 划分训练/验证集...')
    split_dataset()
    
    print('\n4/4 创建配置文件...')
    create_data_yaml()
    
    print('\n' + '=' * 50)
    print(f'✅ 全部完成! 数据保存在: {DATA_DIR}')
    print('=' * 50)


if __name__ == '__main__':
    main()
