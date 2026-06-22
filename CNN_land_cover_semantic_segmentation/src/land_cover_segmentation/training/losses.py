"""Combined cross-entropy and Dice loss for multiclass segmentation."""

from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional
from segmentation_models_pytorch.losses import DiceLoss

class DiceCELoss(nn.Module):
    """0.5 * cross-entropy + 0.5 * multiclass Dice on raw logits.

    Parameters
    ----------
    ignore_index : int
        Target label value to skip in both CE and Dice.
    class_weights : torch.Tensor or None, optional
        Per-class CE weights of shape `(C,)` matching the logit channel
        count. When `None`, CE uses uniform weights.
    """

    def __init__(
        self,
        ignore_index: int,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is None else class_weights.float(),
        )
        self.ce_loss = partial(functional.cross_entropy,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
        )
        self.dice_loss = DiceLoss(
            mode="multiclass",
            ignore_index=self.ignore_index,
            from_logits=True,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the combined loss.

        Parameters
        ----------
        logits : torch.Tensor
            Raw predictions of shape `(B, C, H, W)`.
        target : torch.Tensor
            Ground-truth class indices of shape `(B, H, W)`.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        ce = self.ce_loss(logits, target)
        dice = self.dice_loss(logits, target)
        return 0.5 * ce + 0.5 * dice

__all__ = ["DiceCELoss"]
