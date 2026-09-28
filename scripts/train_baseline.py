"""固定划分、动态类别权重的 U-Net 训练入口。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.baseline_dataset import BaselineDataset
from src.baseline_losses import make_loss
from src.baseline_metrics import confusion_matrix, metrics_from_confusion
from src.dataset import collect_pairs, read_mask
from src.experiment import PROJECT_ROOT, fixed_splits, load_config, seed_everything, split_directory
from src.model import UNet


def dynamic_class_weights(pairs: list[tuple[Path, Path]], num_classes: int) -> torch.Tensor:
    """根据训练集真实像素频率计算平方根逆频率权重。"""
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, mask_path in pairs:
        mask = read_mask(mask_path)
        if mask.min() < 0 or mask.max() >= num_classes:
            raise ValueError(f"掩膜类别索引超出范围：{mask_path}")
        counts += np.bincount(mask.ravel(), minlength=num_classes)[:num_classes]
    frequency = counts / counts.sum()
    weights = 1 / np.sqrt(np.maximum(frequency, 1e-8))
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def validate(model, loader, device, criterion, num_classes: int) -> tuple[float, dict]:
    model.eval()
    losses, matrix = [], np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for images, masks, _ in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            losses.append(criterion(logits, masks).item())
            matrix += confusion_matrix(logits.argmax(1).cpu().numpy(), masks.cpu().numpy(), num_classes)
    return float(np.mean(losses)), metrics_from_confusion(matrix)


def save_history(path: Path, history: list[dict]) -> None:
    """每轮原子写入，异常中断时仍保留已完成轮次。"""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
        file.flush()
    temporary_path.replace(path)


def save_run_snapshot(run_dir: Path, config: dict, manifest_paths: dict[str, Path]) -> dict[str, str]:
    """从实际运行所用配置和 split 生成快照，避免手工复制占位文件。"""
    (run_dir / "config_snapshot.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    hashes = {}
    for name, source in manifest_paths.items():
        shutil.copy2(source, run_dir / source.name)
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    (run_dir / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description="训练可复现 U-Net 基线")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--name", required=True, help="本次实验的唯一名称")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None, choices=(256, 384, 512))
    parser.add_argument("--augmentation", choices=("none", "strong"), default=None)
    parser.add_argument("--loss", choices=("ce_dice", "focal", "tversky", "focal_tversky", "ce_dice_tversky"), default=None)
    parser.add_argument("--regenerate-splits", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="只检查配置、固定划分和动态权重")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config["seed"])
    training = config["training"]
    experiment_settings = config.get("experiment", {})
    loss_parameters = experiment_settings.get("loss_parameters", {})
    if not isinstance(loss_parameters, dict):
        raise ValueError("experiment.loss_parameters 必须是 JSON 对象")
    epochs = args.epochs if args.epochs is not None else training["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else training["batch_size"]
    image_size = args.image_size or config["image_size"]
    augmentation = args.augmentation or experiment_settings.get("augmentation", "none")
    loss_name = args.loss or experiment_settings.get("loss", "ce_dice")
    if epochs < 1 or batch_size < 1:
        parser.error("epochs 和 batch-size 必须为正整数。")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", args.name):
        parser.error("name 只能包含字母、数字、下划线、连字符和点，且不能以点开头。")
    config["image_size"] = image_size
    training["epochs"], training["batch_size"] = epochs, batch_size
    experiment_settings.update({"augmentation": augmentation, "loss": loss_name})
    config["experiment"] = experiment_settings

    pairs = collect_pairs(config["dataset_root"])
    split_dir = split_directory(config)
    manifest_paths = {name: split_dir / f"{name}.json" for name in ("train", "val", "test")}
    missing_manifests = [str(path) for path in manifest_paths.values() if not path.is_file()]
    if missing_manifests and not args.regenerate_splits:
        raise FileNotFoundError(
            "固定划分清单缺失；为防止静默生成新划分，训练已停止。缺少："
            + ", ".join(missing_manifests)
            + "。只有明确要建立新实验划分时才使用 --regenerate-splits。"
        )
    train_pairs, val_pairs, _ = fixed_splits(pairs, config["dataset_root"], config, args.regenerate_splits)
    weights = dynamic_class_weights(train_pairs, config["num_classes"])
    run_dir = PROJECT_ROOT / "runs" / args.name
    experiment = {"name": args.name, "seed": config["seed"], "dataset_root": str(config["dataset_root"]), "image_size": image_size, "augmentation": augmentation, "loss": loss_name, "loss_parameters": loss_parameters, "epochs": epochs, "batch_size": batch_size, "learning_rate": training["learning_rate"], "weight_decay": training["weight_decay"], "early_stopping_patience": training["early_stopping_patience"], "workers": training["workers"], "dynamic_class_weights": weights.tolist(), "train_samples": len(train_pairs), "val_samples": len(val_pairs), "config": str(config["config_path"] or "built_in_defaults")}
    print("动态类别权重：", np.round(weights.numpy(), 4).tolist())
    if args.dry_run:
        print(f"检查通过：train={len(train_pairs)}，val={len(val_pairs)}；未创建训练目录。")
        return
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "experiment.json").write_text(json.dumps(experiment, ensure_ascii=False, indent=2), encoding="utf-8")
    split_hashes = save_run_snapshot(run_dir, config, manifest_paths)
    experiment["split_sha256"] = split_hashes
    (run_dir / "experiment.json").write_text(json.dumps(experiment, ensure_ascii=False, indent=2), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, int(os.environ.get("ATLDSD_CPU_THREADS", "4"))))

    train_loader = DataLoader(BaselineDataset(train_pairs, image_size, training=True, augmentation=augmentation), batch_size=batch_size, shuffle=True, num_workers=training["workers"], pin_memory=device.type == "cuda")
    val_loader = DataLoader(BaselineDataset(val_pairs, image_size), batch_size=batch_size, num_workers=training["workers"], pin_memory=device.type == "cuda")
    model = UNet(num_classes=config["num_classes"]).to(device)
    criterion = make_loss(loss_name, config["num_classes"], weights.to(device), **loss_parameters)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    best_iou, stale_epochs, history = -1.0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        val_loss, metrics = validate(model, val_loader, device, criterion, config["num_classes"])
        scheduler.step(metrics["miou"])
        record = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss, "val_miou": metrics["miou"], "val_mdice": metrics["mdice"], "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(record)
        save_history(run_dir / "history.csv", history)
        print(f"Epoch {epoch}/{epochs}: mIoU={metrics['miou']:.4f}, mDice={metrics['mdice']:.4f}")
        if metrics["miou"] > best_iou:
            best_iou, stale_epochs = metrics["miou"], 0
            buffer = io.BytesIO()
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_miou": best_iou, "experiment": experiment}, buffer)
            (run_dir / "best_unet.pth").write_bytes(buffer.getvalue())
        else:
            stale_epochs += 1
            if stale_epochs >= training["early_stopping_patience"]:
                print("触发 Early Stopping")
                break
    best_record = max(history, key=lambda row: row["val_miou"])
    print(f"训练完成：best_epoch={best_record['epoch']}, best_val_mIoU={best_record['val_miou']:.6f}, best_val_mDice={best_record['val_mdice']:.6f}")


if __name__ == "__main__":
    main()

