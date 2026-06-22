"""Write semantic segmentation predictions to georeferenced GeoTIFF files or RGB PNGs."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import Affine

from land_cover_segmentation.utils import hex_to_rgb
from land_cover_segmentation.utils.visualization import (
    attach_outside_legend,
    colorize_mask,
    content_bbox,
    content_bbox_from_valid,
    crop_to_content,
    intersect_bboxes,
)


def write_georaster(
    class_map: np.ndarray,
    src_path: str | Path | None,
    out_path: str | Path,
    palette: list[str],
) -> None:
    """Write a single-band class map as a compressed GeoTIFF with colormap.

    Parameters
    ----------
    class_map : numpy.ndarray
        Predicted labels of shape `(H, W)`, integer dtype.
    src_path : str, pathlib.Path, or None
        Reference GeoTIFF to copy georeferencing from. When `None`, writes
        an ungeoreferenced image with an identity transform.
    out_path : str or pathlib.Path
        Destination GeoTIFF path.
    palette : list[str]
        Hex colors (`"#RRGGBB"`) parallel to class IDs.
    """
    if src_path is not None:
        with rasterio.open(src_path) as src:
            meta = src.meta.copy()
    else:
        meta = {
            "driver": "GTiff",
            "transform": Affine.identity(),
            "crs": None,
        }

    meta.update(
        dtype="uint8",
        count=1,
        compress="lzw",
        width=class_map.shape[1],
        height=class_map.shape[0],
        nodata=255,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(class_map.astype(np.uint8), 1)
        dst.write_colormap(
            1,
            {index: hex_to_rgb(color) + (255,) for index, color in enumerate(palette)},
        )


def write_image(
    class_map: np.ndarray,
    out_path: str | Path,
    palette: list[str],
    *,
    ignore_index: int | None = None,
    reference_rgb: np.ndarray | None = None,
    class_names: Sequence[str] | None = None,
) -> None:
    """Write a class map as an RGB PNG with the LoveDA palette applied.

    Parameters
    ----------
    class_map : numpy.ndarray
        Predicted labels of shape `(H, W)`, integer dtype.
    out_path : str or pathlib.Path
        Destination PNG path.
    palette : list[str]
        Hex colors (`"#RRGGBB"`) parallel to class IDs.
    ignore_index : int or None, optional
        Label painted as mid-gray instead of a class color.
    reference_rgb : numpy.ndarray or None, optional
        `(H, W, 3)` source image used to crop away black LoveDA padding before
        saving. When omitted, only predicted-background margins are trimmed.
    class_names : Sequence[str] or None, optional
        Human-readable class labels for the legend. Defaults to ``Class 0``,
        ``Class 1``, ...
    """
    rgb = colorize_mask(class_map, palette, ignore_index=ignore_index)
    bboxes = [content_bbox_from_valid(class_map != 0)]
    if reference_rgb is not None:
        bboxes.append(content_bbox(reference_rgb))
    bbox = intersect_bboxes(*bboxes)
    cropped_map = crop_to_content(class_map, bbox)
    rgb = crop_to_content(rgb, bbox)
    names = (
        list(class_names)
        if class_names is not None
        else [f"Class {index}" for index in range(len(palette))]
    )
    present_ids = [
        int(class_id)
        for class_id in np.unique(cropped_map)
        if 0 <= int(class_id) < len(palette) and int(class_id) != ignore_index
    ]
    rgb = attach_outside_legend(rgb, palette, names, class_ids=present_ids)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(out_path)


__all__ = ["write_georaster", "write_image"]
