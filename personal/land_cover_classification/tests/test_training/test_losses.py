"""Tests for training losses."""

import torch

from land_cover_segmentation.training.losses import DiceCELoss


def test_dice_ce_loss_forward_and_backward():
    loss_fn = DiceCELoss(ignore_index=255)
    logits = torch.randn(2, 3, 8, 8, requires_grad=True)
    target = torch.randint(0, 3, (2, 8, 8))
    loss = loss_fn(logits, target)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_dice_ce_loss_ignores_index():
    loss_fn = DiceCELoss(ignore_index=255)
    logits = torch.zeros(1, 3, 4, 4)
    logits[:, 1, :, :] = 10.0
    logits.requires_grad_(True)
    target = torch.full((1, 4, 4), 255, dtype=torch.long)
    loss = loss_fn(logits, target)
    assert torch.isnan(loss)
    target[0, 0, 0] = 1
    loss = loss_fn(logits, target)
    assert torch.isfinite(loss)
    loss.backward()


def test_dice_ce_loss_accepts_class_weights():
    weights = torch.tensor([1.0, 2.0, 0.5])
    loss_fn = DiceCELoss(ignore_index=255, class_weights=weights)
    logits = torch.randn(1, 3, 4, 4)
    target = torch.randint(0, 3, (1, 4, 4))
    loss = loss_fn(logits, target)
    assert torch.isfinite(loss)
