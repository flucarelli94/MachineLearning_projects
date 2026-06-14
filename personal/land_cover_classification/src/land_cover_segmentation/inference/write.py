"""Write semantic segmentation predictions to georeferenced GeoTIFF files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

from land_cover_segmentation.utils import hex_to_rgb


def write_prediction(
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
            {
                index: hex_to_rgb(color) + (255,)
                for index, color in enumerate(palette)
            },
        )


__all__ = ["write_prediction"]
