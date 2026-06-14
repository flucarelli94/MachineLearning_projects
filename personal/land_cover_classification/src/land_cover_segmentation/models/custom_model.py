"""User-defined segmentation model.

When `ModelConfig.source` is `"custom"`, the model factory imports this
module and calls `build_model` — the only supported entry point.

Edit this file to supply your own architecture. The returned module must emit
raw logits of shape `(batch, num_classes, height, width)` with
`num_classes = cfg.data.num_classes`.
"""

from __future__ import annotations

import torch.nn as nn

from land_cover_segmentation.config import Config


def build_model(cfg: Config) -> nn.Module:
    """Build a custom segmentation model from the resolved config.

    Parameters
    ----------
    cfg : Config
        Full project configuration. Read `cfg.data.num_classes`,
        `cfg.model.in_channels`, etc.

    Returns
    -------
    nn.Module
        Network producing multiclass logits `(B, C, H, W)` with `C ==
        cfg.data.num_classes` (no softmax).

    Raises
    ------
    NotImplementedError
        Until you replace this stub with a real implementation.
    """
    raise NotImplementedError(
        "Implement build_model(cfg) in "
        "land_cover_segmentation.models.custom_model for model.source='custom'."
    )
