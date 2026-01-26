from __future__ import annotations

import torch


def dice_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
    valid_mask: torch.Tensor | None = None,
) -> float:
    pred = pred.float()
    target = target.float()
    if valid_mask is not None:
        valid = valid_mask.float()
        pred = pred * valid
        target = target * valid
    intersection = (pred * target).sum()
    total = pred.sum() + target.sum()
    return float((2 * intersection + eps) / (total + eps))


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
    valid_mask: torch.Tensor | None = None,
) -> float:
    pred = pred.float()
    target = target.float()
    if valid_mask is not None:
        valid = valid_mask.float()
        pred = pred * valid
        target = target * valid
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return float((intersection + eps) / (union + eps))


def precision_recall(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
    valid_mask: torch.Tensor | None = None,
) -> tuple[float, float]:
    pred = pred.bool()
    target = target.bool()
    if valid_mask is not None:
        valid = valid_mask.bool()
        pred = pred & valid
        target = target & valid
    tp = (pred & target).sum().float()
    fp = (pred & ~target).sum().float()
    fn = (~pred & target).sum().float()
    precision = float((tp + eps) / (tp + fp + eps))
    recall = float((tp + eps) / (tp + fn + eps))
    return precision, recall
