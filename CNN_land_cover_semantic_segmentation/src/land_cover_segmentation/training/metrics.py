"""Streaming evaluation metrics for semantic segmentation."""

from __future__ import annotations

from typing import Any

import torch


class StreamingConfusionMatrix:
    """Accumulate a confusion matrix across batches on CPU.

    Parameters
    ----------
    num_classes : int
        Number of classes (matrix is `num_classes x num_classes`).
    ignore_index : int
        Target pixels with this label are excluded from updates.
    """

    def __init__(self, num_classes: int, ignore_index: int) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self._cm = torch.zeros(
            (num_classes, num_classes), dtype=torch.int64, device="cpu"
        )

    @property
    def confusion_matrix(self) -> torch.Tensor:
        """Return the accumulated confusion matrix `(C, C)`."""
        return self._cm

    def reset(self) -> None:
        """Clear accumulated counts."""
        self._cm.zero_()

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Add one batch of integer predictions and targets.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted class indices `(B, H, W)` or `(N,)`.
        targets : torch.Tensor
            Ground-truth class indices, same shape as `preds`.
        """
        preds = preds.detach().reshape(-1).to(torch.int64, copy=False)
        targets = targets.detach().reshape(-1).to(torch.int64, copy=False)
        valid = targets != self.ignore_index
        if valid.any():
            preds = preds[valid].cpu()
            targets = targets[valid].cpu()
            indices = targets * self.num_classes + preds
            counts = torch.bincount(indices, minlength=self.num_classes**2)
            self._cm += counts.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, Any]:
        """Derive segmentation metrics from the accumulated matrix.

        Returns
        -------
        dict
            Keys: `per_class_iou`, `miou`, `pixel_acc`, `per_class_f1`,
            `confusion_matrix`. IoU / F1 entries are NaN where a class is
            absent from both predictions and targets.
        """
        cm = self._cm.float()
        tp = torch.diag(cm)
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp

        iou_denom = tp + fp + fn
        per_class_iou = torch.where(iou_denom > 0, tp / iou_denom, torch.nan)

        f1_denom = 2 * tp + fp + fn
        per_class_f1 = torch.where(f1_denom > 0, 2 * tp / f1_denom, torch.nan)

        total = cm.sum()
        pixel_acc = tp.sum() / total if total > 0 else torch.tensor(float("nan"))

        return {
            "per_class_iou": per_class_iou,
            "miou": torch.nanmean(per_class_iou),
            "pixel_acc": pixel_acc,
            "per_class_f1": per_class_f1,
            "confusion_matrix": self._cm.clone(),
        }


__all__ = ["StreamingConfusionMatrix"]
