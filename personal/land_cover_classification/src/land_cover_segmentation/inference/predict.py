"""Sliding-window inference with optional Gaussian tile blending."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax


def _gaussian_kernel(size: int, sigma_ratio: float = 0.25) -> np.ndarray:
    """Build a normalized 2D Gaussian weight map for tile blending.

    Parameters
    ----------
    size : int
        Side length of the square kernel in pixels.
    sigma_ratio : float, optional
        Gaussian sigma as a fraction of `size`.

    Returns
    -------
    numpy.ndarray
        `(size, size)` weights in `[0, 1]` with peak 1.
    """
    sigma = size * sigma_ratio
    axis = np.linspace(-(size // 2), size // 2, size)
    g1 = np.exp(-0.5 * (axis / sigma) ** 2)
    kernel = np.outer(g1, g1)
    return (kernel / kernel.max()).astype(np.float64)


def tile_scene(
    image_chw: np.ndarray,
    tile: int = 512,
    overlap: int = 128,
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Split a scene into overlapping `(C, tile, tile)` crops.

    Parameters
    ----------
    image_chw : numpy.ndarray
        Input scene of shape `(C, H, W)`.
    tile : int, optional
        Crop side length in pixels.
    overlap : int, optional
        Overlap between adjacent crops along rows and columns.

    Returns
    -------
    tiles : list[numpy.ndarray]
        Copied crop arrays, each `(C, tile_h, tile_w)` where the spatial
        size is at most `tile` (smaller at image borders or when the scene
        is smaller than `tile`).
    positions : list[tuple[int, int]]
        Top-left `(row, col)` of each crop in the full scene.
    """
    _, height, width = image_chw.shape
    if height <= tile and width <= tile:
        return [image_chw.copy()], [(0, 0)]

    stride = tile - overlap
    tiles: list[np.ndarray] = []
    positions: list[tuple[int, int]] = []
    row = 0
    while row + tile <= height + overlap:
        row0 = min(row, max(0, height - tile))
        col = 0
        while col + tile <= width + overlap:
            col0 = min(col, max(0, width - tile))
            tiles.append(image_chw[:, row0 : row0 + tile, col0 : col0 + tile].copy())
            positions.append((row0, col0))
            if col0 + tile >= width:
                break
            col += stride
        if row0 + tile >= height:
            break
        row += stride
    return tiles, positions


def reconstruct(
    prob_tiles: Sequence[np.ndarray],
    positions: Sequence[tuple[int, int]],
    height: int,
    width: int,
    num_classes: int,
    tile: int = 512,
    blend: bool = True,
) -> np.ndarray:
    """Merge per-tile class probabilities into a full-scene map.

    Parameters
    ----------
    prob_tiles : Sequence[numpy.ndarray]
        Softmax probabilities per tile, each `(num_classes, tile_h, tile_w)`.
    positions : Sequence[tuple[int, int]]
        Top-left coordinates matching `tile_scene`.
    height : int
        Full scene height `H`.
    width : int
        Full scene width `W`.
    num_classes : int
        Number of class channels in each probability tile.
    tile : int, optional
        Nominal tile size used to build the Gaussian kernel.
    blend : bool, optional
        When `True`, weight overlaps with a Gaussian kernel; otherwise
        use uniform weights.

    Returns
    -------
    numpy.ndarray
        Blended probability map of shape `(num_classes, H, W)`, `float32`.
    """
    acc = np.zeros((num_classes, height, width), dtype=np.float64)
    weight_map = np.zeros((height, width), dtype=np.float64)
    kernel = (
        _gaussian_kernel(tile) if blend else np.ones((tile, tile), dtype=np.float64)
    )

    for probs, (row0, col0) in zip(prob_tiles, positions, strict=True):
        tile_h, tile_w = probs.shape[-2], probs.shape[-1]
        weights = kernel[:tile_h, :tile_w]
        acc[:, row0 : row0 + tile_h, col0 : col0 + tile_w] += probs * weights
        weight_map[row0 : row0 + tile_h, col0 : col0 + tile_w] += weights

    return (acc / np.maximum(weight_map[np.newaxis], 1e-8)).astype(np.float32)


@torch.no_grad()
def predict_scene(
    model: nn.Module,
    image_chw: np.ndarray,
    num_classes: int,
    tile: int = 512,
    overlap: int = 128,
    batch_size: int = 8,
    device: str | torch.device = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Run tiled inference and return probabilities plus a class map.

    Parameters
    ----------
    model : nn.Module
        Segmentation network producing `(B, C, H, W)` logits.
    image_chw : numpy.ndarray
        Normalized scene of shape `(C, H, W)` (`float` or `float32`).
    num_classes : int
        Number of output classes (logit channels).
    tile : int, optional
        Sliding-window crop size.
    overlap : int, optional
        Overlap between adjacent crops.
    batch_size : int, optional
        Number of tiles per forward pass.
    device : str or torch.device, optional
        Compute device for inference.

    Returns
    -------
    prob_map : numpy.ndarray
        Blended softmax probabilities `(num_classes, H, W)`, `float32`.
    class_map : numpy.ndarray
        Argmax labels `(H, W)`, `uint8`.

    Notes
    -----
    `image_chw` must already be normalized with the same `(mean, std)` used
    during training (stored in run checkpoints).
    """
    torch_device = torch.device(device)
    model.eval()
    model.to(torch_device)

    tiles, positions = tile_scene(image_chw, tile=tile, overlap=overlap)
    prob_tiles: list[np.ndarray] = []
    for start in range(0, len(tiles), batch_size):
        batch = torch.from_numpy(np.stack(tiles[start : start + batch_size])).float()
        batch = batch.to(torch_device)
        probs = softmax(model(batch), dim=1).cpu().numpy()
        prob_tiles.extend(probs)

    scene_height, scene_width = image_chw.shape[-2], image_chw.shape[-1]
    prob_map = reconstruct(
        prob_tiles,
        positions,
        scene_height,
        scene_width,
        num_classes,
        tile=tile,
    )
    class_map = prob_map.argmax(axis=0).astype(np.uint8)
    return prob_map, class_map


__all__ = ["predict_scene", "reconstruct", "tile_scene"]
