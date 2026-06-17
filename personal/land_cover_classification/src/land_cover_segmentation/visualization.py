"""Visualization helpers for segmentation masks and prediction grids."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from PIL import Image, ImageDraw, ImageFont

from land_cover_segmentation.utils import hex_to_rgb


def palette_to_rgb(palette: Sequence[str]) -> np.ndarray:
    """Convert a hex palette to an `(N, 3)` uint8 RGB array.

    Parameters
    ----------
    palette : Sequence[str]
        Hex colors (`"#RRGGBB"`), one per class.

    Returns
    -------
    np.ndarray
        Shape `(len(palette), 3)`, dtype `uint8`.
    """
    return np.array([hex_to_rgb(color) for color in palette], dtype=np.uint8)


def palette_to_cmap(palette: Sequence[str]) -> ListedColormap:
    """Build a matplotlib colormap from a hex class palette.

    Parameters
    ----------
    palette : Sequence[str]
        Hex colors (`"#RRGGBB"`), one per class.

    Returns
    -------
    ListedColormap
        Colormap with one entry per class, scaled to `[0, 1]`.
    """
    rgb = palette_to_rgb(palette).astype(np.float64) / 255.0
    return ListedColormap(rgb)


def colorize_mask(
    mask: np.ndarray,
    palette: Sequence[str],
    *,
    ignore_index: int | None = None,
    ignore_color: tuple[int, int, int] = (128, 128, 128),
) -> np.ndarray:
    """Render a class-index mask as an `(H, W, 3)` RGB image.

    Parameters
    ----------
    mask : np.ndarray
        Integer array of shape `(H, W)` with values in `0..num_classes-1`
        and optionally `ignore_index`.
    palette : Sequence[str]
        Hex colors (`"#RRGGBB"`), one per class.
    ignore_index : int or None, optional
        Label value to paint with `ignore_color` instead of a class color.
    ignore_color : tuple[int, int, int], optional
        RGB color for `ignore_index` pixels (default mid-gray).

    Returns
    -------
    np.ndarray
        RGB image, dtype `uint8`, shape `(H, W, 3)`.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")

    colors = palette_to_rgb(palette)
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(colors):
        rgb[mask == class_id] = color
    if ignore_index is not None:
        rgb[mask == ignore_index] = ignore_color

    if ignore_index is None:
        invalid = (mask < 0) | (mask >= len(colors))
    else:
        invalid = (mask < 0) | ((mask >= len(colors)) & (mask != ignore_index))
    if np.any(invalid):
        bad = int(mask[invalid].max())
        raise ValueError(
            f"mask contains label {bad} outside palette range 0..{len(colors) - 1}"
        )
    return rgb


def denormalize_image(
    image_chw: np.ndarray | torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    """Undo channel normalization and return an `(H, W, 3)` uint8 RGB image.

    Parameters
    ----------
    image_chw : np.ndarray or torch.Tensor
        Normalized image, shape `(3, H, W)`.
    mean, std : Sequence[float]
        Per-channel normalization statistics in `[0, 1]` scale.

    Returns
    -------
    np.ndarray
        Denormalized RGB image, dtype `uint8`.
    """
    if isinstance(image_chw, torch.Tensor):
        image_chw = image_chw.detach().cpu().numpy()
    if image_chw.ndim != 3 or image_chw.shape[0] != 3:
        raise ValueError(f"image_chw must have shape (3, H, W), got {image_chw.shape}")

    mean_arr = np.asarray(mean, dtype=np.float64).reshape(3, 1, 1)
    std_arr = np.asarray(std, dtype=np.float64).reshape(3, 1, 1)
    hwc = np.transpose(image_chw * std_arr + mean_arr, (1, 2, 0))
    return np.clip(hwc * 255.0, 0, 255).astype(np.uint8)


def content_bbox(image: np.ndarray) -> tuple[int, int, int, int]:
    """Return the tight bounding box of non-black pixels in an image.

    Parameters
    ----------
    image : np.ndarray
        `(H, W)` or `(H, W, C)` array. A pixel is treated as empty when all
        channels are zero (typical LoveDA tile padding).

    Returns
    -------
    tuple[int, int, int, int]
        `(row_start, row_end, col_start, col_end)` with end indices exclusive.
        When no non-black pixels exist, returns the full image extent.
    """
    if image.ndim == 2:
        valid = image > 0
    elif image.ndim == 3:
        valid = np.any(image != 0, axis=-1)
    else:
        raise ValueError(f"image must be 2D or 3D, got shape {image.shape}")

    return content_bbox_from_valid(valid)


def content_bbox_from_valid(valid: np.ndarray) -> tuple[int, int, int, int]:
    """Return the tight bounding box of ``True`` pixels in a boolean mask."""
    if valid.ndim != 2:
        raise ValueError(f"valid must be 2D, got shape {valid.shape}")

    rows = np.flatnonzero(valid.any(axis=1))
    cols = np.flatnonzero(valid.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        height, width = valid.shape
        return 0, height, 0, width
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def intersect_bboxes(
    *bboxes: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Return the intersection of one or more `(row_start, row_end, col_start, col_end)` boxes."""
    if not bboxes:
        raise ValueError("intersect_bboxes requires at least one bbox")

    row_start = max(bbox[0] for bbox in bboxes)
    row_end = min(bbox[1] for bbox in bboxes)
    col_start = max(bbox[2] for bbox in bboxes)
    col_end = min(bbox[3] for bbox in bboxes)
    if row_start >= row_end or col_start >= col_end:
        return bboxes[0]
    return row_start, row_end, col_start, col_end


def crop_to_content(
    array: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Crop ``array`` to ``bbox`` from :func:`content_bbox`."""
    row_start, row_end, col_start, col_end = bbox
    if array.ndim == 2:
        return array[row_start:row_end, col_start:col_end]
    return array[row_start:row_end, col_start:col_end, ...]


def attach_outside_legend(
    rgb: np.ndarray,
    palette: Sequence[str],
    class_names: Sequence[str],
    *,
    class_ids: Sequence[int] | None = None,
    swatch_size: int = 20,
    font_size: int = 24,
    padding: int = 6,
    bg_color: tuple[int, int, int] = (248, 248, 248),
) -> np.ndarray:
    """Append a compact class legend below ``rgb`` without covering the map."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must have shape (H, W, 3), got {rgb.shape}")

    if class_ids is None:
        ids = list(range(len(palette)))
    else:
        ids = sorted({int(class_id) for class_id in class_ids if 0 <= class_id < len(palette)})
    if not ids:
        return rgb

    image = Image.fromarray(rgb, mode="RGB")
    width = rgb.shape[1]
    font = ImageFont.load_default(size=font_size)
    probe = ImageDraw.Draw(image)

    entries: list[tuple[tuple[int, int, int], str, int, int]] = []
    for class_id in ids:
        name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        color = hex_to_rgb(palette[class_id])
        text_bbox = probe.textbbox((0, 0), name, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        entries.append((color, name, text_width, text_height))

    gap = 4
    row_width = sum(
        padding + swatch_size + gap + text_width + padding
        for _, _, text_width, _ in entries
    )
    row_content_height = max(swatch_size, max(text_height for *_, text_height in entries))
    strip_height = row_content_height + 2 * padding
    strip = Image.new("RGB", (width, strip_height), bg_color)
    draw = ImageDraw.Draw(strip)

    x = max(padding, (width - row_width) // 2)
    for color, name, text_width, text_height in entries:
        swatch_y = padding + (row_content_height - swatch_size) // 2
        text_y = padding + (row_content_height - text_height) // 2
        draw.rectangle(
            [x, swatch_y, x + swatch_size - 1, swatch_y + swatch_size - 1],
            fill=color,
            outline=(190, 190, 190),
        )
        draw.text((x + swatch_size + gap, text_y), name, fill=(64, 64, 64), font=font)
        x += padding + swatch_size + gap + text_width + padding

    combined = Image.new("RGB", (width, rgb.shape[0] + strip_height), bg_color)
    combined.paste(image, (0, 0))
    combined.paste(strip, (0, rgb.shape[0]))
    return np.array(combined)


def save_prediction_grid(
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    preds: Sequence[np.ndarray],
    palette: Sequence[str],
    path: str | Path,
    *,
    class_names: Sequence[str] | None = None,
    ignore_index: int | None = None,
    titles: tuple[str, str, str] = ("Input", "Ground truth", "Prediction"),
    dpi: int = 120,
) -> Path:
    """Save a qualitative grid: input | ground truth | prediction per row.

    Parameters
    ----------
    images : Sequence[np.ndarray]
        Denormalized RGB inputs, each `(H, W, 3)` uint8.
    masks, preds : Sequence[np.ndarray]
        Class-index maps, each `(H, W)` int.
    palette : Sequence[str]
        Hex colors per class.
    path : str or pathlib.Path
        Output PNG path.
    class_names : Sequence[str] or None, optional
        Class names for the legend (defaults to `Class 0`, `Class 1`, ...).
    ignore_index : int or None, optional
        Mask label painted as mid-gray in GT/pred panels.
    titles : tuple[str, str, str], optional
        Column headers.
    dpi : int, optional
        Figure resolution passed to `savefig`.

    Returns
    -------
    pathlib.Path
        Path to the written PNG.
    """
    if not (len(images) == len(masks) == len(preds)):
        raise ValueError("images, masks, and preds must have the same length")
    if len(images) == 0:
        raise ValueError("images is empty")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    num_rows = len(images)
    fig, axes = plt.subplots(
        num_rows,
        3,
        figsize=(9, 3 * num_rows),
        squeeze=False,
        layout="constrained",
    )

    for row, (image, mask, pred) in enumerate(zip(images, masks, preds, strict=True)):
        bbox = content_bbox(image)
        image = crop_to_content(image, bbox)
        mask = crop_to_content(mask, bbox)
        pred = crop_to_content(pred, bbox)
        axes[row, 0].imshow(image)
        axes[row, 1].imshow(colorize_mask(mask, palette, ignore_index=ignore_index))
        axes[row, 2].imshow(colorize_mask(pred, palette, ignore_index=ignore_index))
        for col in range(3):
            axes[row, col].axis("off")

    for col, title in enumerate(titles):
        axes[0, col].set_title(title)

    legend_names = (
        list(class_names)
        if class_names is not None
        else [f"Class {index}" for index in range(len(palette))]
    )
    legend_patches = [
        plt.matplotlib.patches.Patch(color=np.array(hex_to_rgb(color)) / 255.0, label=name)
        for color, name in zip(palette, legend_names, strict=True)
    ]
    fig.legend(
        handles=legend_patches,
        loc="outside lower center",
        ncol=min(len(legend_patches), 4),
        frameon=False,
    )

    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


__all__ = [
    "attach_outside_legend",
    "colorize_mask",
    "content_bbox",
    "content_bbox_from_valid",
    "crop_to_content",
    "denormalize_image",
    "intersect_bboxes",
    "palette_to_cmap",
    "palette_to_rgb",
    "save_prediction_grid",
]
