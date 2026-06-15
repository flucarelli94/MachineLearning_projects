"""Evaluation loop for a trained segmentation run."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
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
from land_cover_segmentation.visualization import (
    denormalize_image,
    save_prediction_grid,
)

Split = Literal["val", "test"]
DEFAULT_VIZ_SAMPLES = 4


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


@torch.no_grad()
def save_png_predictions(
    model: nn.Module,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
    output_path: Path,
    *,
    mean: list[float],
    std: list[float],
    num_samples: int = DEFAULT_VIZ_SAMPLES,
) -> Path:
    """Run inference on a few batches and write a qualitative PNG grid.

    Parameters
    ----------
    model : nn.Module
        Segmentation network in eval mode.
    loader : DataLoader
        Batches of `(image, mask)` tuples.
    cfg : Config
        Project configuration (palette, classes, `ignore_index`).
    device : torch.device
        Compute device.
    output_path : pathlib.Path
        Destination PNG path (e.g. `run_dir/predictions.png`).
    mean, std : list[float]
        Per-channel normalization used to denormalize inputs for display.
    num_samples : int, optional
        Number of rows in the output grid.

    Returns
    -------
    pathlib.Path
        Path to the written PNG.
    """
    model.eval()
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    autocast_device = device.type

    for batch_images, batch_masks in loader:
        batch_images = batch_images.to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type=autocast_device,
            enabled=autocast_device == "cuda",
        ):
            logits = model(batch_images)
        batch_preds = logits.argmax(dim=1).cpu().numpy()

        for index in range(batch_images.size(0)):
            if len(images) >= num_samples:
                break
            images.append(denormalize_image(batch_images[index], mean, std))
            masks.append(batch_masks[index].cpu().numpy())
            preds.append(batch_preds[index])
        if len(images) >= num_samples:
            break

    return save_prediction_grid(
        images,
        masks,
        preds,
        cfg.data.palette,
        output_path,
        class_names=cfg.data.classes,
        ignore_index=cfg.data.ignore_index,
    )


def evaluate_run(
    run_dir: Path,
    split: Split = "val",
    *,
    save_viz: bool = False,
    viz_samples: int = DEFAULT_VIZ_SAMPLES,
) -> dict[str, Any]:
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

    if save_viz:
        save_png_predictions(
            model,
            loader,
            cfg,
            device,
            run_dir / "predictions.png",
            mean=datamodule.mean,
            std=datamodule.std,
            num_samples=viz_samples,
        )

    return output


def _load_class_weights(run_dir: Path) -> torch.Tensor | None:
    path = run_dir / "class_weights.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    return torch.tensor(raw["weights"], dtype=torch.float32)


__all__ = [
    "DEFAULT_VIZ_SAMPLES",
    "evaluate_loader",
    "evaluate_run",
    "metrics_from_confusion",
    "save_png_predictions",
]
