"""Torch-free helpers for reading training run artifacts on disk."""

from __future__ import annotations

from pathlib import Path

from land_cover_segmentation.config import Config, load


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
    path = Path(run_dir) / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing run config: {path}")
    return load(path)


def default_checkpoint_path(run_dir: Path) -> Path:
    """Return the preferred checkpoint path for a run (`best.pth`)."""
    return Path(run_dir) / "best.pth"


__all__ = ["default_checkpoint_path", "load_run_config"]
