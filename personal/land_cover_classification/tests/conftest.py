from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


@pytest.fixture
def synthetic_geotiff(tmp_path: Path) -> Path:
    path = tmp_path / "scene.tif"
    transform = from_origin(10.0, 20.0, 0.5, 0.5)
    data = np.zeros((3, 16, 16), dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=16,
        width=16,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)
    return path
