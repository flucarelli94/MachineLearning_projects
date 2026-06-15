"""Generic utilities shared across the package.

Filesystem helpers (used by the downloader and checkpoint writer), color
helpers used by GeoTIFF export, image statistics helpers used by the
data module at `setup()` time, deterministic seeding for training runs,
and shared logging configuration for CLI and library entry points.
"""

import logging
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch


def human_bytes(n: float) -> str:
    """Format a byte count as a human-readable string (e.g. `"3.7 GiB"`)."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def dir_size(path: str | Path) -> int:
    """Total size in bytes of all files under `path` (recursive). Missing dirs return 0."""
    p = Path(path)
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    """Convert a `"#RRGGBB"` string to an `(R, G, B)` tuple of ints in `0..255`.

    Parameters
    ----------
    h : str
        Hex color string. A leading `"#"` is optional; the remaining
        characters must be exactly 6 hex digits.

    Returns
    -------
    tuple[int, int, int]
        Red, green, blue components in `0..255`.

    Raises
    ------
    ValueError
        If `h` does not contain exactly 6 hex digits (after stripping a
        leading `"#"`).
    """
    h = h.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Expected 6-digit hex color, got {h!r}")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def configure_logging(name: str, *, level: int = logging.INFO) -> logging.Logger:
    """Configure package logging and return a named logger.

    The root handler is configured on the first call; later calls reuse
    that setup and return ``logging.getLogger(name)``.

    Parameters
    ----------
    name : str
        Logger name, typically ``__name__`` of the calling module.
    level : int, optional
        Root log level (default ``logging.INFO``).

    Returns
    -------
    logging.Logger
        Logger for ``name``.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    return logging.getLogger(name)


def compute_channel_stats(
    images: Sequence[np.ndarray],
) -> tuple[list[float], list[float]]:
    """Per-channel `(mean, std)` over the given images.

    Values are rescaled to `[0, 1]` by dividing by 255 (the function
    assumes `uint8` input).

    Parameters
    ----------
    images : Sequence[np.ndarray]
        Indexable, sized collection where each item is an `(H, W, C)`
        `uint8` array.

    Returns
    -------
    mean, std : tuple[list[float], list[float]]
        Per-channel mean and standard deviation in `[0, 1]`,
        each of length `C`.
    """
    if len(images) == 0:
        raise ValueError("images is empty")

    imgs = np.stack(images)
    if imgs.dtype != np.uint8:
        raise ValueError(f"images has dtype {imgs.dtype}; expected uint8")
    if imgs.ndim != 4:
        raise ValueError(
            f"images has shape {imgs.shape}; expected 4 dimensions (N, H, W, C)"
        )

    # Reduce over (N, H, W) to leave per-channel (C,) results.
    x = imgs.astype(np.float64) / 255.0
    mean = np.mean(x, axis=(0, 1, 2))
    std = np.std(x, axis=(0, 1, 2))
    return mean.tolist(), std.tolist()


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible training.

    Parameters
    ----------
    seed : int
        Global seed written to `random`, `numpy`, and `torch`.
    deterministic : bool, optional
        When `True`, prefer deterministic cuDNN kernels over faster
        non-deterministic ones.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device: str) -> torch.device:
    """Map a config device string to a `torch.device`.

    Parameters
    ----------
    device : str
        One of ``"auto"``, ``"cpu"``, or ``"cuda"``. ``"auto"`` selects
        CUDA when available, otherwise CPU.

    Returns
    -------
    torch.device
        Resolved compute device.
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


__all__ = [
    "compute_channel_stats",
    "configure_logging",
    "dir_size",
    "hex_to_rgb",
    "human_bytes",
    "resolve_device",
    "seed_everything",
]
