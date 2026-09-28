"""用于消融实验的多类别分割损失函数。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, include_background: bool = False):
        super().__init__()
        self.num_classes, self.include_background = num_classes, include_background

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probability = torch.softmax(logits, dim=1)
        one_hot = functional.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        intersection = (probability * one_hot).sum(dim=(0, 2, 3))
        denominator = probability.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
        dice = (2 * intersection + 1.0) / (denominator + 1.0)
        return 1 - (dice if self.include_background else dice[1:]).mean()


class FocalLoss(nn.Module):
    def __init__(self, weights: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("weights", weights)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cross_entropy = functional.cross_entropy(logits, target, weight=self.weights, reduction="none")
        return ((1 - torch.exp(-cross_entropy)) ** self.gamma * cross_entropy).mean()


class TverskyLoss(nn.Module):
    def __init__(self, num_classes: int, alpha: float = 0.3, beta: float = 0.7, focal: bool = False):
        super().__init__()
        self.num_classes, self.alpha, self.beta, self.focal = num_classes, alpha, beta, focal

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probability = torch.softmax(logits, dim=1)
        one_hot = functional.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        tp = (probability * one_hot).sum(dim=(0, 2, 3))
        fp = (probability * (1 - one_hot)).sum(dim=(0, 2, 3))
        fn = ((1 - probability) * one_hot).sum(dim=(0, 2, 3))
        score = (tp + 1.0) / (tp + self.alpha * fp + self.beta * fn + 1.0)
        loss = 1 - score[1:]
        return (loss ** 1.33).mean() if self.focal else loss.mean()


class WeightedCEDiceTverskyLoss(nn.Module):
    """保留基线稳定性，同时额外惩罚少数类病斑漏检的混合损失。"""

    def __init__(
        self,
        num_classes: int,
        weights: torch.Tensor,
        tversky_weight: float = 0.5,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
    ):
        super().__init__()
        if tversky_weight < 0:
            raise ValueError("tversky_weight 必须大于或等于 0")
        self.cross_entropy = nn.CrossEntropyLoss(weight=weights)
        self.dice = DiceLoss(num_classes)
        self.tversky = TverskyLoss(num_classes, alpha=tversky_alpha, beta=tversky_beta)
        self.tversky_weight = tversky_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.cross_entropy(logits, target) + self.dice(logits, target) + self.tversky_weight * self.tversky(logits, target)


def make_loss(
    name: str,
    num_classes: int,
    weights: torch.Tensor,
    tversky_weight: float = 0.5,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
) -> nn.Module:
    if name == "ce_dice":
        ce, dice = nn.CrossEntropyLoss(weight=weights), DiceLoss(num_classes)
        return _CombinedLoss(ce, dice)
    if name == "focal":
        return FocalLoss(weights)
    if name == "tversky":
        return TverskyLoss(num_classes)
    if name == "focal_tversky":
        return TverskyLoss(num_classes, focal=True)
    if name == "ce_dice_tversky":
        return WeightedCEDiceTverskyLoss(
            num_classes,
            weights,
            tversky_weight=tversky_weight,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
        )
    raise ValueError(f"未知损失函数：{name}")


class _CombinedLoss(nn.Module):
    def __init__(self, first: nn.Module, second: nn.Module):
        super().__init__()
        self.first, self.second = first, second

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.first(logits, target) + self.second(logits, target)

