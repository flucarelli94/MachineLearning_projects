"""Evaluation loop for a trained segmentation run."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from land_cover_segmentation.config import Config
from land_cover_segmentation.dataset.loveda import LoveDADataModule
from land_cover_segmentation.training.checkpoint import CheckpointIO
from land_cover_segmentation.training.losses import DiceCELoss
from land_cover_segmentation.training.metrics import StreamingConfusionMatrix
from land_cover_segmentation.models.factory import build_model
from land_cover_segmentation.utils import resolve_device, seed_everything

Split = Literal["val", "test"]


def metrics_from_confusion(
    cm: StreamingConfusionMatrix,
    loss: float,
) -> dict[str, Any]:
    """Build a JSON-friendly metric dict from a confusion matrix and mean loss."""
    metrics = cm.compute()
    per_class_iou = metrics["per_class_iou"].tolist()
    per_class_f1 = metrics["per_class_f1"].tolist()
    return {
        "loss": loss,
        "miou": metrics["miou"].item(),
        "pixel_acc": metrics["pixel_acc"].item(),
        "per_class_iou": [
            None if isinstance(v, float) and math.isnan(v) else v for v in per_class_iou
        ],
        "per_class_f1": [
            None if isinstance(v, float) and math.isnan(v) else v for v in per_class_f1
        ],
    }


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: DiceCELoss,
    cfg: Config,
    device: torch.device,
    *,
    desc: str = "eval",
) -> dict[str, Any]:
    """Run inference on a dataloader and aggregate segmentation metrics.

    Parameters
    ----------
    model : nn.Module
        Segmentation network.
    loader : DataLoader
        Batches of `(image, mask)` tuples.
    loss_fn : DiceCELoss
        Loss used to compute the reported `loss` metric.
    cfg : Config
        Project configuration (`num_classes`, `ignore_index`).
    device : torch.device
        Compute device.
    desc : str, optional
        Progress bar label.

    Returns
    -------
    dict
        Keys: `loss`, `miou`, `pixel_acc`, `per_class_iou`, `per_class_f1`.
    """
    model.eval()
    cm = StreamingConfusionMatrix(cfg.data.num_classes, cfg.data.ignore_index)
    loss_sum = 0.0
    num_batches = 0
    autocast_device = device.type

    for images, masks in tqdm(loader, desc=desc):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type=autocast_device,
            enabled=autocast_device == "cuda",
        ):
            logits = model(images)
            loss = loss_fn(logits, masks)

        loss_sum += loss.item()
        num_batches += 1
        cm.update(logits.argmax(dim=1), masks)

    return metrics_from_confusion(cm, loss_sum / max(num_batches, 1))


def evaluate_run(run_dir: Path, split: Split = "val") -> dict[str, Any]:
    """Evaluate `best.pth` from a training run and write `metrics.json`.

    Parameters
    ----------
    run_dir : pathlib.Path
        Directory containing `config.yaml` and `best.pth`.
    split : {"val", "test"}, optional
        Dataset split to evaluate.

    Returns
    -------
    dict
        Metrics including a `split` key, also written to `run_dir/metrics.json`.
    """
    run_dir = Path(run_dir)
    cfg = CheckpointIO.load_run_config(run_dir)
    seed_everything(cfg.run.seed, deterministic=cfg.run.deterministic)

    device = resolve_device(cfg.run.device)
    model = build_model(cfg)
    checkpoint_path = CheckpointIO.default_checkpoint_path(run_dir)
    CheckpointIO.load(model, checkpoint_path, device=device)
    model.to(device)

    datamodule = LoveDADataModule(cfg.data)
    datamodule.setup()

    class_weights = _load_class_weights(run_dir) if cfg.loss.use_class_weights else None
    loss_fn = DiceCELoss(
        ignore_index=cfg.data.ignore_index,
        class_weights=class_weights,
    ).to(device)

    loader = (
        datamodule.val_dataloader()
        if split == "val"
        else datamodule.test_dataloader()
    )
    metrics = evaluate_loader(model, loader, loss_fn, cfg, device, desc=split)
    output = {"split": split, **metrics}

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(output, indent=2))
    return output


def _load_class_weights(run_dir: Path) -> torch.Tensor | None:
    path = run_dir / "class_weights.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    return torch.tensor(raw["weights"], dtype=torch.float32)


__all__ = [
    "evaluate_loader",
    "evaluate_run",
    "metrics_from_confusion",
]
