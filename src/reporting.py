"""将评估结果导出为机器可读报告和可视化图。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_evaluation_report(output_dir: Path, class_names: dict[int, str], matrix: np.ndarray, metrics: dict, metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = {**metadata, "confusion_matrix": matrix.tolist(), "pixel_accuracy": metrics["pixel_accuracy"], "miou": metrics["miou"], "mdice": metrics["mdice"], "per_class": []}
    for class_id in sorted(class_names):
        serializable["per_class"].append({"class_id": class_id, "class_name": class_names[class_id], "iou": float(metrics["iou"][class_id]), "dice": float(metrics["dice"][class_id]), "precision": float(metrics["precision"][class_id]), "recall": float(metrics["recall"][class_id]), "f1": float(metrics["f1"][class_id]), "support_pixels": int(metrics["support"][class_id])})
    (output_dir / "metrics.json").write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=serializable["per_class"][0].keys())
        writer.writeheader()
        writer.writerows(serializable["per_class"])
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(matrix, cmap="Blues")
    labels = [class_names[i] for i in sorted(class_names)]
    axis.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)), xticklabels=labels, yticklabels=labels, xlabel="预测类别", ylabel="真实类别", title="像素级混淆矩阵")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(figure)
