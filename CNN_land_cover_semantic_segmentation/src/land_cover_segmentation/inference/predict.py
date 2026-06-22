"""Sliding-window inference with optional Gaussian tile blending."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax

from land_cover_segmentation.inference.tiling import (
    load_normalized_scene,
    reconstruct,
    tile_scene,
)
from land_cover_segmentation.inference.write import write_georaster, write_image
from land_cover_segmentation.models.factory import build_model
from land_cover_segmentation.training.checkpoint import CheckpointIO
from land_cover_segmentation.utils.model import resolve_device


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
        The model to run inference on.
    image_chw : np.ndarray
        The image to run inference on.
    num_classes : int
        The number of classes in the model.
    tile : int, optional
        The size of the tile to use for inference.
    overlap : int, optional
        The overlap between tiles for inference.
    batch_size : int, optional
        The batch size to use for inference.
    device : str or torch.device, optional
        The device to use for inference.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The probabilities and class map.
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


def predict_run(
    run_dir: Path,
    input_path: Path,
    output_path: Path,
) -> Path:
    """Run tiled inference for a trained model and write a class-map GeoTIFF or PNG.

    Parameters
    ----------
    run_dir : pathlib.Path
        The training run directory.
    input_path : pathlib.Path
        The input image path.
    output_path : pathlib.Path
        The output path.

    Returns
    -------
    pathlib.Path
        The output path.
    """
    run_dir = Path(run_dir)
    input_path = Path(input_path)
    output_path = Path(output_path)

    cfg = CheckpointIO.load_run_config(run_dir)
    device = resolve_device(cfg.run.device)
    model = build_model(cfg)
    payload = CheckpointIO.load(
        model,
        CheckpointIO.default_checkpoint_path(run_dir),
        device=device,
    )
    model.to(device)

    image_chw, georef_path, source_hwc = load_normalized_scene(
        input_path,
        payload["mean"],
        payload["std"],
        in_channels=cfg.model.in_channels,
    )
    overlap = max(1, cfg.data.image_size // 4)
    _, class_map = predict_scene(
        model,
        image_chw,
        num_classes=cfg.data.num_classes,
        tile=cfg.data.image_size,
        overlap=overlap,
        batch_size=cfg.data.batch_size,
        device=device,
    )
    if output_path.suffix.lower() == ".png":
        write_image(
            class_map,
            output_path,
            cfg.data.palette,
            ignore_index=cfg.data.ignore_index,
            reference_rgb=source_hwc,
            class_names=cfg.data.classes,
        )
    elif output_path.suffix.lower() == ".tif":
        write_georaster(
            class_map,
            georef_path,
            output_path,
            cfg.data.palette,
        )
    else:
        raise ValueError(f"Unsupported output format: {output_path.suffix}")
    return output_path


__all__ = [
    "load_normalized_scene",
    "predict_run",
    "predict_scene",
    "reconstruct",
    "tile_scene",
]
