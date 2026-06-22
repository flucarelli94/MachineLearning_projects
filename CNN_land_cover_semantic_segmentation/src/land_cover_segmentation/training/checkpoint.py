"""Checkpoint and run-artifact I/O for training and evaluation."""

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from land_cover_segmentation import __version__
from land_cover_segmentation.config import Config
from land_cover_segmentation.utils.runs import (
    default_checkpoint_path,
    load_run_config,
)

class CheckpointIO:
    """Read and write PyTorch checkpoints plus JSON metadata sidecars.

    Parameters
    ----------
    cfg : Config
        Resolved project configuration (stored in metadata on save).
    datamodule : LoveDADataModule
        Supplies normalization `mean` / `std` for inference reload.
    run_dir : pathlib.Path
        Run output directory; `meta.json` is updated on best saves.
    """

    def __init__(
        self,
        cfg: Config,
        datamodule: "LoveDADataModule",
        run_dir: Path,
    ) -> None:
        self.cfg = cfg
        self.datamodule = datamodule
        self.run_dir = Path(run_dir)

    @staticmethod
    def load_run_config(run_dir: Path) -> Config:
        """Load the resolved config written by a training run.

        Parameters
        ----------
        run_dir : pathlib.Path
            Run output directory containing `config.yaml`.

        Returns
        -------
        Config
            Configuration restored from the run.

        Raises
        ------
        FileNotFoundError
            If `config.yaml` is missing under `run_dir`.
        """
        return load_run_config(run_dir)

    @staticmethod
    def default_checkpoint_path(run_dir: Path) -> Path:
        """Return the preferred checkpoint path for a run (`best.pth`)."""
        return default_checkpoint_path(run_dir)

    @staticmethod
    def load(
        model: nn.Module,
        checkpoint_path: Path,
        *,
        device: torch.device | None = None,
    ) -> dict[str, Any]:
        """Restore model weights from a `.pth` checkpoint.

        Parameters
        ----------
        model : nn.Module
            Model whose weights are loaded in place.
        checkpoint_path : pathlib.Path
            Path to a checkpoint saved by `CheckpointIO.save`.
        device : torch.device or None, optional
            Map location for `torch.load`. Defaults to CPU.

        Returns
        -------
        dict
            Full checkpoint payload (`model_state_dict`, `mean`, `std`, etc.).

        Raises
        ------
        FileNotFoundError
            If `checkpoint_path` does not exist.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

        map_location = device or torch.device("cpu")
        payload = torch.load(
            checkpoint_path, map_location=map_location, weights_only=False
        )
        model.load_state_dict(payload["model_state_dict"])
        return payload

    @staticmethod
    def _metrics_to_json(metrics: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.ndim == 0:
                    item = value.item()
                    out[key] = (
                        None if isinstance(item, float) and math.isnan(item) else item
                    )
                else:
                    out[key] = [
                        None if isinstance(v, float) and math.isnan(v) else v
                        for v in value.tolist()
                    ]
            elif isinstance(value, float) and math.isnan(value):
                out[key] = None
            else:
                out[key] = value
        return out

    def save(
        self,
        path: Path,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_metrics: dict[str, Any],
        best_val_miou: float,
        is_best: bool = False,
    ) -> None:
        """Persist model weights and optional run metadata.

        Parameters
        ----------
        path : pathlib.Path
            Checkpoint path (e.g. `last.pth` or `best.pth`).
        model : nn.Module
            Model whose `state_dict` is saved.
        optimizer : torch.optim.Optimizer
            Optimizer whose `state_dict` is saved.
        epoch : int
            Zero-based epoch index at save time.
        val_metrics : dict
            Validation metrics from the epoch (used in metadata).
        best_val_miou : float
            Best validation mIoU seen so far.
        is_best : bool, optional
            When `True`, also write/update `run_dir/meta.json`.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_miou": best_val_miou,
            "mean": self.datamodule.mean,
            "std": self.datamodule.std,
            "package_version": __version__,
        }
        torch.save(payload, path)

        meta = {
            "checkpoint": path.name,
            "epoch": epoch,
            "best_val_miou": best_val_miou,
            "val_metrics": self._metrics_to_json(val_metrics),
            "config": self.cfg.to_dict(),
            "timestamp": datetime.now(UTC).isoformat(),
            "package_version": __version__,
        }
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
        if is_best:
            (self.run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

__all__ = ["CheckpointIO"]
