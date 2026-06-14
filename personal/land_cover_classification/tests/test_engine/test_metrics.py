"""Tests for streaming evaluation metrics."""

import math

import pytest
import torch

from land_cover_segmentation.engine.metrics import StreamingConfusionMatrix


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
