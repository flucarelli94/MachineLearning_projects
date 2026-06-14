"""Tests for tiled inference and GeoTIFF export."""

import numpy as np
import rasterio

from land_cover_segmentation.config import Config, DataConfig, ModelConfig
from land_cover_segmentation.inference.predict import (
    predict_scene,
    reconstruct,
    tile_scene,
)
from land_cover_segmentation.inference.write import write_prediction
from land_cover_segmentation.models.factory import build_model


def test_tile_scene_covers_small_image():
    image = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    tiles, positions = tile_scene(image, tile=4, overlap=1)
    assert len(tiles) == 1
    assert positions == [(0, 0)]
    assert tiles[0].shape == (3, 2, 2)


def test_tile_scene_generates_overlapping_crops():
    image = np.zeros((3, 100, 100), dtype=np.float32)
    tiles, positions = tile_scene(image, tile=64, overlap=16)
    assert len(tiles) > 1
    assert all(tile.shape == (3, 64, 64) for tile in tiles)
    assert positions[0] == (0, 0)
    assert positions[-1][0] == 36
    assert positions[-1][1] == 36


def test_reconstruct_blends_constant_probabilities():
    num_classes = 3
    tile = 4
    probs = np.zeros((num_classes, tile, tile), dtype=np.float32)
    probs[1] = 1.0
    prob_map = reconstruct(
        [probs, probs],
        [(0, 0), (0, 2)],
        height=4,
        width=6,
        num_classes=num_classes,
        tile=tile,
    )
    assert prob_map.shape == (num_classes, 4, 6)
    assert np.allclose(prob_map[1], 1.0)
    assert np.allclose(prob_map[0], 0.0)
    assert np.allclose(prob_map[2], 0.0)


def test_predict_scene_returns_prob_and_class_maps():
    cfg = Config(
        data=DataConfig(classes=["a", "b", "c"]),
        model=ModelConfig(source="custom", in_channels=3),
    )
    model = build_model(cfg)
    image = np.random.randn(3, 80, 80).astype(np.float32)
    prob_map, class_map = predict_scene(
        model,
        image,
        num_classes=cfg.data.num_classes,
        tile=32,
        overlap=8,
        batch_size=2,
        device="cpu",
    )
    assert prob_map.shape == (3, 80, 80)
    assert prob_map.dtype == np.float32
    assert class_map.shape == (80, 80)
    assert class_map.dtype == np.uint8
    assert np.all((class_map >= 0) & (class_map < cfg.data.num_classes))


def test_write_prediction_round_trip(tmp_path, synthetic_geotiff):
    class_map = np.full((16, 16), 1, dtype=np.uint8)
    out_path = tmp_path / "pred.tif"
    palette = Config().data.palette

    write_prediction(class_map, synthetic_geotiff, out_path, palette)

    with rasterio.open(synthetic_geotiff) as src:
        src_meta = src.meta
    with rasterio.open(out_path) as dst:
        assert dst.count == 1
        assert dst.dtypes[0] == "uint8"
        assert dst.crs == src_meta["crs"]
        assert dst.transform == src_meta["transform"]
        assert dst.nodata == 255
        assert dst.read(1)[0, 0] == 1
        assert dst.colormap(1)[1] == (227, 11, 11, 255)
