"""User-defined segmentation model.

When `ModelConfig.source` is `"custom"`, `models.factory.build_model`
delegates here and calls `build_model` — the only supported entry point.

The default implementation is a plain PyTorch U-Net with skip connections. It
uses the same pipeline contract as `model.source: smp` (logits of shape
`(B, num_classes, H, W)` with no activation), but the architecture lives
in this file so you can edit it without touching the training or inference
code. `ModelConfig.encoder` / `encoder_weights` are ignored for the custom
path.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from land_cover_segmentation.config import Config


class _ConvBlock(nn.Module):
    """Two 3x3 convolutions, batch norm, and ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UNet(nn.Module):
    """Compact U-Net with encoder levels and skip connections."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        features: tuple[int, ...] = (32, 64, 128, 256),
    ) -> None:
        super().__init__()
        f0, f1, f2, f3 = features

        self.enc1 = _ConvBlock(in_channels, f0)
        self.enc2 = _ConvBlock(f0, f1)
        self.enc3 = _ConvBlock(f1, f2)
        self.enc4 = _ConvBlock(f2, f3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = _ConvBlock(f3, f3 * 2)

        self.up4 = nn.ConvTranspose2d(f3 * 2, f3, kernel_size=2, stride=2)
        self.dec4 = _ConvBlock(f3 * 2, f3)
        self.up3 = nn.ConvTranspose2d(f3, f2, kernel_size=2, stride=2)
        self.dec3 = _ConvBlock(f2 * 2, f2)
        self.up2 = nn.ConvTranspose2d(f2, f1, kernel_size=2, stride=2)
        self.dec2 = _ConvBlock(f1 * 2, f1)
        self.up1 = nn.ConvTranspose2d(f1, f0, kernel_size=2, stride=2)
        self.dec1 = _ConvBlock(f0 * 2, f0)

        self.head = nn.Conv2d(f0, num_classes, kernel_size=1)

    @staticmethod
    def _align(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Center-crop `x` to `ref` spatial size for skip concatenation."""
        dh = x.shape[2] - ref.shape[2]
        dw = x.shape[3] - ref.shape[3]
        if dh == 0 and dw == 0:
            return x
        top, left = dh // 2, dw // 2
        bottom, right = x.shape[2] - (dh - top), x.shape[3] - (dw - left)
        return x[:, :, top:bottom, left:right]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        b = self.bottleneck(self.pool(s4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([self._align(d4, s4), s4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([self._align(d3, s3), s3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([self._align(d2, s2), s2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([self._align(d1, s1), s1], dim=1))
        return self.head(d1)


def build_model(cfg: Config) -> nn.Module:
    """Build the custom U-Net from the resolved config.

    Parameters
    ----------
    cfg : Config
        Full project configuration. Reads `cfg.model.in_channels` and
        `cfg.data.num_classes`.

    Returns
    -------
    nn.Module
        U-Net producing multiclass logits `(B, C, H, W)` with `C ==
        cfg.data.num_classes` (no softmax).
    """
    return _UNet(
        in_channels=cfg.model.in_channels,
        num_classes=cfg.data.num_classes,
    )
