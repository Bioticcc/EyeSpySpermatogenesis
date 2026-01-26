import torch

from src.training.seg_train import dice_loss, masked_bce_with_logits, split_ignore_targets


def test_ignore_pixels_do_not_change_loss() -> None:
    logits = torch.zeros((1, 1, 3, 3))
    targets = torch.zeros((1, 1, 3, 3))
    targets[0, 0, 1, 1] = 1.0
    targets[0, 0, 0, 0] = -1.0

    targets_bin, valid = split_ignore_targets(targets)
    loss1 = masked_bce_with_logits(logits, targets_bin, valid_mask=valid) + dice_loss(
        logits, targets_bin, valid_mask=valid
    )

    logits2 = logits.clone()
    logits2[0, 0, 0, 0] = 10.0
    loss2 = masked_bce_with_logits(logits2, targets_bin, valid_mask=valid) + dice_loss(
        logits2, targets_bin, valid_mask=valid
    )

    torch.testing.assert_close(loss1, loss2)
