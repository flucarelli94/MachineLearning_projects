"""Training callbacks: early stopping, JSONL logging, and checkpoint I/O."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn

from land_cover_segmentation import __version__
from land_cover_segmentation.config import Config
from land_cover_segmentation.dataset.loveda import LoveDADataModule


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Parameters
    ----------
    patience : int
        Number of epochs to wait after the last improvement before
        signalling stop.
    mode : {"max", "min"}, optional
        Whether a higher or lower metric value is better.
    metric : str, optional
        Metric name stored for logging only (default `val_miou`).
    """

    def __init__(
        self,
        patience: int,
        mode: Literal["max", "min"] = "max",
        metric: str = "val_miou",
    ) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.patience = patience
        self.mode = mode
        self.metric = metric
        self._best: float | None = None
        self._epochs_without_improvement = 0

    @property
    def should_stop(self) -> bool:
        """Return `True` when patience has been exceeded."""
        return self._epochs_without_improvement >= self.patience

    def _is_improvement(self, value: float) -> bool:
        if self._best is None:
            return True
        if self.mode == "max":
            return value > self._best
        return value < self._best

    def step(self, value: float) -> bool:
        """Record a new metric value and return whether to stop.

        Parameters
        ----------
        value : float
            Metric for the epoch just completed.

        Returns
        -------
        bool
            `True` if training should stop.
        """
        improved = self._is_improvement(value)
        if improved:
            self._best = value
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement += 1
        return self.should_stop


class JSONLLogger:
    """Append one JSON object per line to a run log file.

    Parameters
    ----------
    path : pathlib.Path
        Destination file (e.g. `run.jsonl` under the run directory).
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _sanitize_for_json(self, value: Any) -> Any:
        if isinstance(value, float) and math.isnan(value):
            return None
        if isinstance(value, Mapping):
            return {key: self._sanitize_for_json(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._sanitize_for_json(item) for item in value]
        return value

    def log(self, record: Mapping[str, Any]) -> None:
        """Append a serializable record as a single JSON line."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self._sanitize_for_json(record)) + "\n")


class CheckpointWriter:
    """Write PyTorch checkpoints and a JSON metadata sidecar.

    Parameters
    ----------
    cfg : Config
        Resolved project configuration (stored in metadata).
    datamodule : LoveDADataModule
        Supplies normalization `mean` / `std` for inference reload.
    run_dir : pathlib.Path
        Run output directory; `meta.json` is updated on best saves.
    """

    def __init__(
        self,
        cfg: Config,
        datamodule: LoveDADataModule,
        run_dir: Path,
    ) -> None:
        self.cfg = cfg
        self.datamodule = datamodule
        self.run_dir = Path(run_dir)

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


__all__ = ["CheckpointWriter", "EarlyStopping", "JSONLLogger"]
