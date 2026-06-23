import os

import pytest

from land_cover_segmentation.dataset.download import download_loveda

def test_download_loveda_invalid_split(tmp_path):
    with pytest.raises(ValueError):
        download_loveda(root=tmp_path, splits=["invalid"], scenes=["urban", "rural"])

def test_download_loveda_invalid_scene(tmp_path):
    with pytest.raises(ValueError):
        download_loveda(root=tmp_path, splits=["test"], scenes=["invalid"])
