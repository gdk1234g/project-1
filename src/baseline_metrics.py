"""可信基线的完整像素级分割指标。"""
from __future__ import annotations

import numpy as np


def confusion_matrix(prediction: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    encoded = target.reshape(-1) * num_classes + prediction.reshape(-1)
    return np.bincount(encoded, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def metrics_from_confusion(matrix: np.ndarray) -> dict:
    matrix = matrix.astype(np.float64)
    true_positive = np.diag(matrix)
    actual, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    union = actual + predicted - true_positive
    precision = np.divide(true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0)
    recall = np.divide(true_positive, actual, out=np.zeros_like(true_positive), where=actual > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    dice = np.divide(2 * true_positive, actual + predicted, out=np.zeros_like(true_positive), where=(actual + predicted) > 0)
    iou = np.divide(true_positive, union, out=np.zeros_like(true_positive), where=union > 0)
    valid = union > 0
    return {"pixel_accuracy": float(true_positive.sum() / matrix.sum()) if matrix.sum() else 0.0,
            "miou": float(iou[valid].mean()) if valid.any() else 0.0,
            "mdice": float(dice[valid].mean()) if valid.any() else 0.0,
            "iou": iou, "dice": dice, "precision": precision, "recall": recall,
            "f1": f1, "support": actual.astype(np.int64)}
