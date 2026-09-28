"""可配置的数据增强数据集，专用于可复现实验。"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import read_image, read_mask


class BaselineDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path]], image_size: int, training: bool = False, augmentation: str = "none"):
        self.pairs, self.image_size, self.training, self.augmentation = pairs, image_size, training, augmentation

    def __len__(self) -> int:
        return len(self.pairs)

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.augmentation == "none":
            return image, mask
        if random.random() < 0.5:
            image, mask = cv2.flip(image, 1), cv2.flip(mask, 1)
        if random.random() < 0.2:
            image, mask = cv2.flip(image, 0), cv2.flip(mask, 0)
        if random.random() < 0.6:
            height, width = image.shape[:2]
            angle, scale = random.uniform(-25, 25), random.uniform(0.9, 1.1)
            transform = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
            image = cv2.warpAffine(image, transform, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, transform, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if self.augmentation == "strong":
            if random.random() < 0.7:
                image = cv2.convertScaleAbs(image, alpha=random.uniform(0.75, 1.25), beta=random.uniform(-24, 24))
            if random.random() < 0.35:
                gamma = random.uniform(0.75, 1.35)
                image = np.clip(((image / 255.0) ** gamma) * 255, 0, 255).astype(np.uint8)
            if random.random() < 0.2:
                image = cv2.GaussianBlur(image, (3, 3), 0)
            if random.random() < 0.2:
                noise = np.random.normal(0, 7, image.shape).astype(np.int16)
                image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return image, mask

    def __getitem__(self, index: int):
        image_path, label_path = self.pairs[index]
        image = read_image(image_path, cv2.IMREAD_COLOR)
        mask = read_mask(label_path)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        if self.training:
            image, mask = self._augment(image, mask)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        return torch.from_numpy(image.transpose(2, 0, 1)).float(), torch.from_numpy(mask.astype(np.int64)), str(image_path)
