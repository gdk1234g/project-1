"""只在固定验证集上计算分割指标，不访问测试集。"""
from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.baseline_dataset import BaselineDataset
from src.baseline_metrics import confusion_matrix, metrics_from_confusion
from src.config import CLASS_NAMES
from src.dataset import collect_pairs
from src.experiment import PROJECT_ROOT, fixed_splits, load_config, seed_everything, split_directory
from src.model import UNet
from src.reporting import save_evaluation_report


def main() -> None:
    parser = argparse.ArgumentParser(description="仅使用固定验证集评估权重并导出逐类别指标")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None, help="需要时覆盖配置中的数据根目录")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_root is not None:
        config["dataset_root"] = args.data_root
    seed_everything(config["seed"])

    manifests = {name: split_directory(config) / f"{name}.json" for name in ("train", "val", "test")}
    missing = [str(path) for path in manifests.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "固定划分清单缺失；验证评估不会自动生成新划分：" + ", ".join(missing)
        )

    weights_path = args.weights if args.weights.is_absolute() else PROJECT_ROOT / args.weights
    if not weights_path.is_file():
        raise FileNotFoundError(f"找不到权重：{weights_path}")
    output_dir = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise FileExistsError(f"输出路径已被占用，为避免覆盖旧报告请换一个新路径：{args.output}")

    pairs = collect_pairs(config["dataset_root"])
    _, validation_pairs, _ = fixed_splits(pairs, config["dataset_root"], config)
    checkpoint = torch.load(io.BytesIO(weights_path.read_bytes()), map_location="cpu", weights_only=True)
    model = UNet(num_classes=config["num_classes"])
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    batch_size = args.batch_size or config.get("evaluation", {}).get("batch_size", 8)
    loader = DataLoader(
        BaselineDataset(validation_pairs, image_size=config["image_size"]),
        batch_size=batch_size,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    matrix = np.zeros((config["num_classes"], config["num_classes"]), dtype=np.int64)
    with torch.no_grad():
        for images, masks, _ in loader:
            predictions = model(images.to(device)).argmax(1).cpu().numpy()
            matrix += confusion_matrix(predictions, masks.numpy(), config["num_classes"])
    metrics = metrics_from_confusion(matrix)
    split_hash = hashlib.sha256(manifests["val"].read_bytes()).hexdigest()
    metadata = {
        "evaluation_split": "validation",
        "weights": str(weights_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_miou": checkpoint.get("val_miou"),
        "validation_samples": len(validation_pairs),
        "validation_manifest_sha256": split_hash,
        "dataset_root": str(config["dataset_root"]),
        "config": str(config["config_path"] or "built_in_defaults"),
        "device": str(device),
    }
    save_evaluation_report(output_dir, CLASS_NAMES, matrix, metrics, metadata)
    print(
        f"验证集 mIoU={metrics['miou']:.4f}，mDice={metrics['mdice']:.4f}，"
        f"样本数={len(validation_pairs)}"
    )
    print(f"逐类别报告：{output_dir / 'per_class_metrics.csv'}")


if __name__ == "__main__":
    main()
