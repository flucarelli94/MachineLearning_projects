"""Tiled prediction with an ONNX Runtime inference session."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from land_cover_segmentation.inference.predict import (
    load_normalized_scene,
    reconstruct,
    tile_scene,
)
from land_cover_segmentation.inference.write import write_georaster, write_image
from land_cover_segmentation.training.checkpoint import CheckpointIO


def load_onnx_session(onnx_path: Path) -> ort.InferenceSession:
    """Load an ONNX model for CPU inference."""
    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"Missing ONNX model: {onnx_path}")
    return ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )


def _softmax_logits(logits: np.ndarray, axis: int = 1) -> np.ndarray:
    """Apply numerically stable softmax along ``axis``."""
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=axis, keepdims=True)).astype(np.float32)


def _tiled_prob_maps(
    image_chw: np.ndarray,
    num_classes: int,
    tile: int,
    overlap: int,
    batch_size: int,
    run_batch: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Tile a scene, run batched forward passes, blend softmax probabilities."""
    tiles, positions = tile_scene(image_chw, tile=tile, overlap=overlap)
    prob_tiles: list[np.ndarray] = []
    for start in range(0, len(tiles), batch_size):
        batch = np.stack(tiles[start : start + batch_size]).astype(np.float32, copy=False)
        logits = run_batch(batch)
        prob_tiles.extend(_softmax_logits(logits, axis=1))

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


def predict_scene_onnx(
    session: ort.InferenceSession,
    image_chw: np.ndarray,
    num_classes: int,
    tile: int = 512,
    overlap: int = 128,
    batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Run tiled inference via ONNX Runtime and return probabilities plus labels."""
    input_name = session.get_inputs()[0].name

    def run_batch(batch: np.ndarray) -> np.ndarray:
        return session.run(None, {input_name: batch.astype(np.float32)})[0]

    return _tiled_prob_maps(
        image_chw,
        num_classes,
        tile,
        overlap,
        batch_size,
        run_batch,
    )


def _load_norm_stats(run_dir: Path, onnx_path: Path) -> tuple[list[float], list[float]]:
    """Load normalization stats from a checkpoint or ONNX sidecar."""
    checkpoint_path = CheckpointIO.default_checkpoint_path(run_dir)
    if checkpoint_path.exists():
        payload = torch.load(
            checkpoint_path, map_location=torch.device("cpu"), weights_only=False
        )
        return payload["mean"], payload["std"]

    sidecar_path = Path(onnx_path).with_suffix(".meta.json")
    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text())
        return sidecar["mean"], sidecar["std"]

    raise FileNotFoundError(
        f"Missing normalization stats; expected {checkpoint_path.name} or "
        f"{sidecar_path.name}."
    )


def predict_run(
    run_dir: Path,
    onnx_path: Path,
    input_path: Path,
    output_path: Path,
) -> Path:
    """Run tiled ONNX inference and write a class-map GeoTIFF or PNG.

    Parameters
    ----------
    run_dir : pathlib.Path
        Training run directory containing `config.yaml`.
    onnx_path : pathlib.Path
        Exported `.onnx` model path.
    input_path : pathlib.Path
        Input image path (3-band `uint8` raster).
    output_path : pathlib.Path
        Destination GeoTIFF or PNG path.

    Returns
    -------
    pathlib.Path
        The `output_path` written on disk.
    """
    run_dir = Path(run_dir)
    onnx_path = Path(onnx_path)
    input_path = Path(input_path)
    output_path = Path(output_path)

    cfg = CheckpointIO.load_run_config(run_dir)
    mean, std = _load_norm_stats(run_dir, onnx_path)
    session = load_onnx_session(onnx_path)

    image_chw, georef_path, source_hwc = load_normalized_scene(
        input_path,
        mean,
        std,
        in_channels=cfg.model.in_channels,
    )
    overlap = max(1, cfg.data.image_size // 4)
    _, class_map = predict_scene_onnx(
        session,
        image_chw,
        num_classes=cfg.data.num_classes,
        tile=cfg.data.image_size,
        overlap=overlap,
        batch_size=cfg.data.batch_size,
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
    else:
        write_georaster(
            class_map,
            georef_path,
            output_path,
            cfg.data.palette,
        )
    return output_path


__all__ = [
    "load_onnx_session",
    "predict_run",
    "predict_scene_onnx",
]
