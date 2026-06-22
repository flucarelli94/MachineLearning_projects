from __future__ import annotations

import random

import numpy as np
import torch


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


__all__ = ["resolve_device", "seed_everything"]
