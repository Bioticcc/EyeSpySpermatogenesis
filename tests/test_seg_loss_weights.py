import torch

from src.training.seg_train import build_loss_weight_map, masked_bce_with_logits


def test_neg_tubule_weighting_increases_loss() -> None:
    logits = torch.zeros((1, 1, 3, 3))
    targets = torch.zeros_like(logits)
    valid_mask = torch.ones_like(targets, dtype=torch.bool)
    neg_mask = torch.zeros_like(targets, dtype=torch.bool)
    neg_mask[0, 0, 1, 1] = True

    weight_map = build_loss_weight_map(targets, neg_mask, bg_weight=1.0, neg_tubule_weight=2.0)
    loss_base = masked_bce_with_logits(logits, targets, valid_mask=valid_mask, weight_map=weight_map)

    logits_neg = logits.clone()
    logits_neg[0, 0, 1, 1] = 5.0
    loss_neg = masked_bce_with_logits(logits_neg, targets, valid_mask=valid_mask, weight_map=weight_map)

    logits_bg = logits.clone()
    logits_bg[0, 0, 0, 0] = 5.0
    loss_bg = masked_bce_with_logits(logits_bg, targets, valid_mask=valid_mask, weight_map=weight_map)

    assert (loss_neg - loss_base) > (loss_bg - loss_base)
