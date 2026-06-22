from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from tqdm import tqdm


def compute_channel_stats(
    images: Sequence[np.ndarray],
    *,
    max_samples: int | None = None,
    seed: int | None = None,
) -> tuple[list[float], list[float]]:
    """Per-channel `(mean, std)` over the given images.

    Values are rescaled to `[0, 1]` by dividing by 255 (the function
    assumes `uint8` input). Images are accumulated one at a time so the
    full collection is never stacked in memory.

    Parameters
    ----------
    images : Sequence[np.ndarray]
        Indexable, sized collection where each item is an `(H, W, C)`
        `uint8` array.
    max_samples : int or None, optional
        When set and ``len(images)`` exceeds this value, a uniform random
        sample of this many indices is used. Requires ``seed``.
    seed : int or None, optional
        RNG seed for ``max_samples`` subsampling.

    Returns
    -------
    mean, std : tuple[list[float], list[float]]
        Per-channel mean and standard deviation in `[0, 1]`,
        each of length `C`.
    """
    n = len(images)
    if n == 0:
        raise ValueError("images is empty")

    if max_samples is not None and n > max_samples:
        if seed is None:
            raise ValueError("seed is required when max_samples limits sampling")
        rng = np.random.default_rng(seed)
        indices = rng.choice(n, size=max_samples, replace=False)
    else:
        indices = np.arange(n)

    n_pixels = 0
    sum_vals: np.ndarray | None = None
    sum_sq: np.ndarray | None = None

    for idx in tqdm(indices, desc="Computing channel stats"):
        img = images[int(idx)]
        if img.dtype != np.uint8:
            raise ValueError(f"images has dtype {img.dtype}; expected uint8")
        if img.ndim != 3:
            raise ValueError(
                f"images has shape {img.shape}; expected 3 dimensions (H, W, C)"
            )

        x = img.astype(np.float64) / 255.0
        if sum_vals is None:
            channels = x.shape[2]
            sum_vals = np.zeros(channels, dtype=np.float64)
            sum_sq = np.zeros(channels, dtype=np.float64)

        n_pixels += x.shape[0] * x.shape[1]
        sum_vals += x.sum(axis=(0, 1))
        sum_sq += (x * x).sum(axis=(0, 1))

    assert sum_vals is not None and sum_sq is not None
    mean = sum_vals / n_pixels
    variance = np.maximum(sum_sq / n_pixels - mean * mean, 0.0)
    std = np.sqrt(variance)
    return mean.tolist(), std.tolist()


__all__ = ["compute_channel_stats"]
