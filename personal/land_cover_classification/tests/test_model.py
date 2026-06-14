"""Tests for model construction and forward pass."""

import pytest
import torch

from land_cover_segmentation.config import Config, DataConfig, ModelConfig
from land_cover_segmentation.models.custom_model import build_model as build_custom_model
from land_cover_segmentation.models.factory import build_model


def test_custom_unet_forward_shape():
    cfg = Config(
        data=DataConfig(classes=["a", "b", "c"]),
        model=ModelConfig(source="custom", in_channels=3),
    )
    model = build_custom_model(cfg)
    x = torch.randn(2, 3, 64, 64)
    logits = model(x)
    assert logits.shape == (2, 3, 64, 64)
    assert logits.dtype == torch.float32


def test_custom_unet_backward():
    cfg = Config(model=ModelConfig(source="custom", in_channels=3))
    model = build_custom_model(cfg)
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    loss = model(x).sum()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_factory_custom_matches_direct_build():
    cfg = Config(
        data=DataConfig(classes=["a", "b", "c"]),
        model=ModelConfig(source="custom", in_channels=3),
    )
    x = torch.randn(1, 3, 64, 64)
    assert build_model(cfg)(x).shape == build_custom_model(cfg)(x).shape


def test_factory_smp_forward_shape():
    cfg = Config(
        data=DataConfig(classes=["a", "b", "c"]),
        model=ModelConfig(
            source="smp",
            encoder="resnet18",
            encoder_weights=None,
            in_channels=3,
        ),
    )
    model = build_model(cfg)
    x = torch.randn(2, 3, 64, 64)
    assert model(x).shape == (2, 3, 64, 64)


def test_factory_rejects_unknown_source():
    cfg = Config(model=ModelConfig(source="unknown"))
    with pytest.raises(ValueError, match="Unknown model.source"):
        build_model(cfg)
