"""Build a segmentation model from configuration."""

from __future__ import annotations

import torch.nn as nn
import segmentation_models_pytorch as smp

from land_cover_segmentation.config import Config
from land_cover_segmentation.models import custom_model


def build_model(cfg: Config) -> nn.Module:
    """Return a segmentation network for training or inference.

    Both sources expose the same contract: raw logits `(B, num_classes, H, W)`
    with `num_classes = cfg.data.num_classes` and no final activation.

    Parameters
    ----------
    cfg : Config
        Resolved project configuration.

    Returns
    -------
    nn.Module
        Segmentation model.

    Raises
    ------
    ValueError
        If `cfg.model.source` is not `"smp"` or `"custom"`.
    """
    if cfg.model.source == "smp":
        return smp.Unet(
            encoder_name=cfg.model.encoder,
            encoder_weights=cfg.model.encoder_weights,
            in_channels=cfg.model.in_channels,
            classes=cfg.data.num_classes,
            activation=None,
        )
    if cfg.model.source == "custom":
        return custom_model.build_model(cfg)
    raise ValueError(
        f"Unknown model.source {cfg.model.source!r}; expected 'smp' or 'custom'."
    )


__all__ = ["build_model"]
