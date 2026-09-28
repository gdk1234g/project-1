from pathlib import Path
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import IMAGE_SIZE

VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def read_image(path: Path, flags: int) -> np.ndarray:
    """兼容 Windows 中文路径的 OpenCV 读图函数。"""
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    if image is None:
        raise ValueError(f"无法读取图像：{path}")
    return image


def read_mask(path: Path) -> np.ndarray:
    """读取调色板 PNG 的原始类别索引，不能用 OpenCV 灰度转换。

    ATLDSD 的标签为 P 模式 PNG：像素值 0~5 是语义类别索引。若以
    IMREAD_GRAYSCALE 读取，OpenCV 会读取调色板颜色而不是索引，导致类别
    变成 38、75 等灰度值。
    """
    with Image.open(path) as label:
        return np.asarray(label, dtype=np.uint8).copy()


def collect_pairs(dataset_root: Path):
    """收集 image 与同名 label PNG 的配对，并检查数据完整性。"""
    pairs = []
    for class_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        image_dir, label_dir = class_dir / "image", class_dir / "label"
        if not image_dir.is_dir() or not label_dir.is_dir():
            continue
        for image_path in image_dir.iterdir():
            if image_path.suffix.lower() not in VALID_SUFFIXES:
                continue
            label_path = label_dir / f"{image_path.stem}.png"
            if label_path.exists():
                pairs.append((image_path, label_path))
            else:
                print(f"跳过：未找到标签 {label_path}")
    if not pairs:
        raise FileNotFoundError(f"没有在 {dataset_root} 找到有效 image/label 配对")
    return pairs


def split_pairs(pairs, train_ratio=0.7, val_ratio=0.15, seed=42):
    """固定随机种子划分训练、验证、测试集。"""
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    train_end = int(len(pairs) * train_ratio)
    val_end = train_end + int(len(pairs) * val_ratio)
    return pairs[:train_end], pairs[train_end:val_end], pairs[val_end:]


class AppleLeafDataset(Dataset):
    def __init__(self, pairs, training=False, image_size=IMAGE_SIZE):
        self.pairs = pairs
        self.training = training
        self.image_size = image_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, label_path = self.pairs[index]
        image = read_image(image_path, cv2.IMREAD_COLOR)
        mask = read_mask(label_path)

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        if self.training and random.random() < 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        if self.training and random.random() < 0.2:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        if self.training and random.random() < 0.5:
            rotation = random.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE])
            image = cv2.rotate(image, rotation)
            mask = cv2.rotate(mask, rotation)
        if self.training and random.random() < 0.5:
            # 模拟田间拍摄中的明暗差异，保持掩膜不变。
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-12, 12)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask.astype(np.int64))
        return image, mask, str(image_path)
