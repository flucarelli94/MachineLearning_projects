import numpy as np
from matplotlib.colors import ListedColormap

from land_cover_segmentation.config import Config
from land_cover_segmentation.utils import hex_to_rgb
from land_cover_segmentation.visualization import (
    colorize_mask,
    denormalize_image,
    palette_to_cmap,
    palette_to_rgb,
    save_prediction_grid,
)


def test_palette_to_rgb_and_cmap():
    palette = Config().data.palette
    rgb = palette_to_rgb(palette)
    assert rgb.shape == (len(palette), 3)
    assert rgb.dtype == np.uint8
    assert rgb[1].tolist() == list(hex_to_rgb("#E30B0B"))

    cmap = palette_to_cmap(palette)
    assert isinstance(cmap, ListedColormap)
    assert len(cmap.colors) == len(palette)


def test_colorize_mask_maps_classes_and_ignore_index():
    palette = Config().data.palette
    mask = np.array([[0, 1], [2, 255]], dtype=np.int64)
    rgb = colorize_mask(mask, palette, ignore_index=255)
    assert rgb.shape == (2, 2, 3)
    assert rgb[0, 1].tolist() == list(hex_to_rgb("#E30B0B"))
    assert rgb[1, 1].tolist() == [128, 128, 128]


def test_denormalize_image_round_trip():
    mean = [0.5, 0.5, 0.5]
    std = [0.25, 0.25, 0.25]
    image = np.full((3, 4, 4), 0.0, dtype=np.float32)
    out = denormalize_image(image, mean, std)
    assert out.shape == (4, 4, 3)
    assert out.dtype == np.uint8
    assert np.all(out == 127)


def test_save_prediction_grid_writes_png(tmp_path):
    palette = Config().data.palette
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :4] = 255
    mask = np.zeros((8, 8), dtype=np.int64)
    mask[:, 4:] = 1
    pred = mask.copy()

    out_path = save_prediction_grid(
        [image],
        [mask],
        [pred],
        palette,
        tmp_path / "grid.png",
        class_names=Config().data.classes,
        ignore_index=255,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0
