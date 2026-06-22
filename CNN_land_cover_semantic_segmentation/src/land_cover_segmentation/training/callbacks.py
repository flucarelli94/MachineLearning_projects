"""Training callbacks: early stopping and JSONL logging."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal


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


__all__ = ["EarlyStopping", "JSONLLogger"]
