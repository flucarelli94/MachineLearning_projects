"""Tests for training losses and streaming metrics."""

import math

import pytest
import torch

from land_cover_segmentation.engine.losses import DiceCELoss
from land_cover_segmentation.engine.metrics import StreamingConfusionMatrix


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


def test_confusion_matrix_perfect_prediction():
    cm = StreamingConfusionMatrix(num_classes=3, ignore_index=255)
    targets = torch.tensor([[0, 1], [2, 1]])
    preds = targets.clone()
    cm.update(preds, targets)
    out = cm.compute()
    assert out["miou"].item() == pytest.approx(1.0)
    assert out["pixel_acc"].item() == pytest.approx(1.0)
    assert torch.allclose(out["per_class_iou"], torch.ones(3))


def test_confusion_matrix_excludes_ignore_index():
    cm = StreamingConfusionMatrix(num_classes=2, ignore_index=255)
    targets = torch.tensor([[0, 255], [1, 255]])
    preds = torch.tensor([[1, 0], [1, 0]])
    cm.update(preds, targets)
    out = cm.compute()
    assert out["confusion_matrix"].sum().item() == 2
    assert out["confusion_matrix"][0, 1].item() == 1
    assert out["confusion_matrix"][1, 1].item() == 1


def test_confusion_matrix_reset():
    cm = StreamingConfusionMatrix(num_classes=2, ignore_index=255)
    cm.update(preds=torch.ones(2, 2), targets=torch.ones(2, 2))
    assert cm.confusion_matrix.sum().item() == 4
    cm.reset()
    assert cm.confusion_matrix.sum().item() == 0
    out = cm.compute()
    assert math.isnan(out["miou"].item())
