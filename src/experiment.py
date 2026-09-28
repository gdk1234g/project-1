"""共享实验配置、随机种子和固定数据划分。"""
from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from .config import DATASET_ROOT, NUM_CLASSES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Generic runnable defaults, not measured values or archived experiment settings.
DEFAULT_CONFIG = {
    "seed": 0,
    "image_size": 256,
    "num_classes": NUM_CLASSES,
    "split": {"train_ratio": 0.7, "val_ratio": 0.15, "directory": "splits"},
    "training": {
        "epochs": 10, "batch_size": 4, "learning_rate": 0.001, "workers": 0,
        "early_stopping_patience": 5, "weight_decay": 0.0001,
    },
    "evaluation": {"batch_size": 4},
    "experiment": {"augmentation": "none", "loss": "ce_dice"},
}


def load_config(path: Path | None = None) -> dict:
    """Load user-owned JSON configuration or independent source-code defaults."""
    config_path = path.expanduser().resolve() if path is not None else None
    if config_path is None:
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["dataset_root"] = str(DATASET_ROOT)
    else:
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    config["dataset_root"] = Path(
        os.environ.get("ATLDSD_ROOT", config["dataset_root"])
    ).expanduser()
    if not config["dataset_root"].is_absolute():
        config["dataset_root"] = PROJECT_ROOT / config["dataset_root"]
    config["dataset_root"] = config["dataset_root"].resolve()
    config["config_path"] = config_path
    if config["num_classes"] != NUM_CLASSES:
        raise ValueError("本项目训练入口要求六类标签索引，与网页类别定义一致。")
    image_size = config["image_size"]
    if not isinstance(image_size, int) or image_size < 32 or image_size % 16:
        raise ValueError("image_size 必须至少为 32，且为 16 的整数倍。")
    train_ratio, val_ratio = config["split"]["train_ratio"], config["split"]["val_ratio"]
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("训练、验证和测试划分比例必须均为正数。")
    training = config["training"]
    for name in ("epochs", "batch_size", "early_stopping_patience"):
        if not isinstance(training[name], int) or training[name] < 1:
            raise ValueError(f"training.{name} 必须为正整数。")
    if training["workers"] < 0 or training["learning_rate"] <= 0 or training["weight_decay"] < 0:
        raise ValueError("workers/weight_decay 不能为负数，learning_rate 必须为正数。")
    return config


def seed_everything(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 的随机性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_directory(config: dict) -> Path:
    directory = Path(config["split"]["directory"])
    return directory if directory.is_absolute() else PROJECT_ROOT / directory


def _key(image_path: Path, dataset_root: Path) -> str:
    return image_path.relative_to(dataset_root).as_posix()


def _write_manifest(path: Path, pairs: list[tuple[Path, Path]], dataset_root: Path) -> None:
    path.write_text(json.dumps([_key(image, dataset_root) for image, _ in pairs], ensure_ascii=False, indent=2), encoding="utf-8")


def fixed_splits(pairs: list[tuple[Path, Path]], dataset_root: Path, config: dict, regenerate: bool = False) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """读取或首次生成固定清单，保证训练、验证、测试互不变化。"""
    manifests = {name: split_directory(config) / f"{name}.json" for name in ("train", "val", "test")}
    missing = [str(path) for path in manifests.values() if not path.is_file()]
    if missing and not regenerate:
        raise FileNotFoundError("划分清单缺失，请明确使用 --regenerate-splits 初始化：" + ", ".join(missing))
    if regenerate:
        shuffled = list(pairs)
        random.Random(config["seed"]).shuffle(shuffled)
        train_end = int(len(shuffled) * config["split"]["train_ratio"])
        val_end = train_end + int(len(shuffled) * config["split"]["val_ratio"])
        if train_end < 1 or val_end <= train_end or val_end >= len(shuffled):
            raise ValueError("数据量过少，无法建立非空的训练、验证和测试集。")
        split_directory(config).mkdir(parents=True, exist_ok=True)
        for name, subset in {"train": shuffled[:train_end], "val": shuffled[train_end:val_end], "test": shuffled[val_end:]}.items():
            _write_manifest(manifests[name], subset, dataset_root)
    pair_by_key = {_key(image, dataset_root): (image, label) for image, label in pairs}
    if len(pair_by_key) != len(pairs):
        raise ValueError("输入图像配对中存在重复样本。")
    result, used = [], set()
    for name in ("train", "val", "test"):
        keys = json.loads(manifests[name].read_text(encoding="utf-8"))
        if len(keys) != len(set(keys)):
            raise ValueError(f"{name} 划分内部存在重复样本。")
        missing, overlap = [key for key in keys if key not in pair_by_key], used.intersection(keys)
        if missing:
            raise ValueError(f"{name} 划分中有 {len(missing)} 个文件不在当前数据集，例如：{missing[0]}")
        if overlap:
            raise ValueError(f"固定划分出现重复样本，例如：{next(iter(overlap))}")
        used.update(keys)
        result.append([pair_by_key[key] for key in keys])
    if used != set(pair_by_key):
        raise ValueError("固定划分未覆盖当前全部样本；请使用 --regenerate-splits 明确重建。")
    return tuple(result)  # type: ignore[return-value]
