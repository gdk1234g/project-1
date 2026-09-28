"""审计 ATLDSD 数据、生成固定划分并保存可复现实验报告。"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.config import CLASS_NAMES
from src.dataset import collect_pairs, read_image, read_mask
from src.experiment import fixed_splits, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="审计 ATLDSD 并固化数据划分")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset_audit.json"))
    parser.add_argument("--regenerate-splits", action="store_true", help="明确重建 train/val/test 清单")
    args = parser.parse_args()
    config = load_config(args.config)
    pairs = collect_pairs(config["dataset_root"])
    splits = fixed_splits(pairs, config["dataset_root"], config, args.regenerate_splits)
    pixels = np.zeros(config["num_classes"], dtype=np.int64)
    dimensions, invalid_masks, unreadable_images = Counter(), [], []
    severity_percentages = []
    for image_path, mask_path in pairs:
        try:
            image = read_image(image_path, flags=1)
            dimensions[f"{image.shape[1]}x{image.shape[0]}"] += 1
        except ValueError as error:
            unreadable_images.append(str(error))
        mask = read_mask(mask_path)
        values, counts = np.unique(mask, return_counts=True)
        if values.min() < 0 or values.max() >= config["num_classes"]:
            invalid_masks.append({"path": str(mask_path), "values": values.tolist()})
            continue
        pixels += np.bincount(mask.ravel(), minlength=config["num_classes"])
        leaf_area = int(np.count_nonzero(mask > 0))
        lesion_area = int(np.count_nonzero(mask >= 2))
        if leaf_area:
            severity_percentages.append(100 * lesion_area / leaf_area)
    report = {
        "dataset_root": str(config["dataset_root"]), "total_pairs": len(pairs),
        "split_sizes": {name: len(samples) for name, samples in zip(("train", "val", "test"), splits)},
        "class_pixel_counts": {CLASS_NAMES[index]: int(value) for index, value in enumerate(pixels)},
        "class_pixel_percentages": {CLASS_NAMES[index]: round(float(value / pixels.sum() * 100), 6) if pixels.sum() else 0.0 for index, value in enumerate(pixels)},
        "image_dimensions": dict(dimensions), "invalid_masks": invalid_masks,
        "unreadable_images": unreadable_images,
        "lesion_severity_percent": {"count": len(severity_percentages), "mean": round(float(np.mean(severity_percentages)), 4) if severity_percentages else 0.0, "median": round(float(np.median(severity_percentages)), 4) if severity_percentages else 0.0},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已审计 {len(pairs)} 对样本，固定划分：" + "/".join(str(len(samples)) for samples in splits))
    print(f"报告：{args.output}")
    if invalid_masks or unreadable_images:
        raise SystemExit("数据审计发现异常，请查看报告。")


if __name__ == "__main__":
    main()
