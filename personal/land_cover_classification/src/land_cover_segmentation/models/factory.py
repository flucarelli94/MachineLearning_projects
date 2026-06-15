"""Build a segmentation model from configuration."""

from __future__ import annotations

import torch.nn as nn
import segmentation_models_pytorch as smp

from land_cover_segmentation.config import Config
from land_cover_segmentation.models import custom_model
from land_cover_segmentation.utils import configure_logging

logger = configure_logging(__name__)


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
    """
    if cfg.model.source == "smp":
        logger.info(
            "Building segmentation model: smp.Unet("
            "encoder=%r, encoder_weights=%r, in_channels=%d, classes=%d)",
            cfg.model.encoder,
            cfg.model.encoder_weights,
            cfg.model.in_channels,
            cfg.data.num_classes,
        )
        return smp.Unet(
            encoder_name=cfg.model.encoder,
            encoder_weights=cfg.model.encoder_weights,
            in_channels=cfg.model.in_channels,
            classes=cfg.data.num_classes,
            activation=None,
        )

    logger.info(
        "Building segmentation model: custom _UNet("
        "in_channels=%d, classes=%d)",
        cfg.model.in_channels,
        cfg.data.num_classes,
    )
    return custom_model.build_model(cfg)


__all__ = ["build_model"]
