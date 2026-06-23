"""User-defined segmentation model.

When `ModelConfig.source` is `"custom"`, `models.factory.build_model`
delegates here and calls `build_model` — the only supported entry point.

The default implementation is a plain PyTorch U-Net with skip connections. It
uses the same pipeline contract as `model.source: smp` (logits of shape
`(B, num_classes, H, W)` with no activation), but the architecture lives
in this file so you can edit it without touching the training or inference
code. `ModelConfig.encoder` / `encoder_weights` are ignored for the custom
path; set `ModelConfig.unet_features` in YAML to change width and depth.
"""

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
        unet_features: tuple[int, ...],
    ) -> None:
        super().__init__()
        features = (in_channels, *unet_features)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoders = nn.ModuleList(
            _ConvBlock(in_ch, out_ch)
            for in_ch, out_ch in zip(features[:-1], features[1:], strict=True)
        )

        deepest = features[-1]
        self.bottleneck = _ConvBlock(deepest, deepest * 2)

        decode_widths = (deepest * 2, *reversed(features[1:]))
        self.upsamplers = nn.ModuleList(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            for in_ch, out_ch in zip(decode_widths[:-1], decode_widths[1:], strict=True)
        )
        self.decoders = nn.ModuleList(
            _ConvBlock(out_ch * 2, out_ch) for out_ch in reversed(features[1:])
        )

        self.head = nn.Conv2d(features[1], num_classes, kernel_size=1)

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
        skips: list[torch.Tensor] = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for upsampler, decoder, skip in zip(
            self.upsamplers, self.decoders, reversed(skips), strict=True
        ):
            x = upsampler(x)
            x = decoder(torch.cat([self._align(x, skip), skip], dim=1))

        return self.head(x)

def build_model(cfg: Config) -> nn.Module:
    """Build the custom U-Net from the resolved config.

    Parameters
    ----------
    cfg : Config
        Full project configuration. Reads `cfg.model.in_channels`,
        `cfg.model.unet_features`, and `cfg.data.num_classes`.

    Returns
    -------
    nn.Module
        U-Net producing multiclass logits `(B, C, H, W)` with `C ==
        cfg.data.num_classes` (no softmax).
    """
    return _UNet(
        in_channels=cfg.model.in_channels,
        num_classes=cfg.data.num_classes,
        unet_features=cfg.model.unet_features,
    )

__all__ = ["build_model"]
