import numpy as np
import pytest

from land_cover_segmentation.utils import (
    compute_channel_stats,
    dir_size,
    hex_to_rgb,
    human_bytes,
)


def test_human_bytes():
    assert human_bytes(1024) == "1.0 KiB"
    assert human_bytes(1024 * 1024) == "1.0 MiB"
    assert human_bytes(1024 * 1024 * 1024) == "1.0 GiB"
    assert human_bytes(1024 * 1024 * 1024 * 1024) == "1.0 TiB"


def test_dir_size():
    size = dir_size(".")
    assert isinstance(size, int)
    assert size > 0


def test_hex_to_rgb():
    assert hex_to_rgb("#000000") == (0, 0, 0)
    assert hex_to_rgb("#FFFFFF") == (255, 255, 255)
    assert hex_to_rgb("#FF0000") == (255, 0, 0)
    assert hex_to_rgb("#00FF00") == (0, 255, 0)
    assert hex_to_rgb("#0000FF") == (0, 0, 255)


def test_compute_channel_stats_constant_image():
    img = np.full((4, 4, 3), 128, dtype=np.uint8)
    mean, std = compute_channel_stats([img, img, img])
    assert np.allclose(mean, [128 / 255] * 3)
    assert np.allclose(std, [0.0] * 3, atol=1e-6)


def test_compute_channel_stats_two_extremes():
    img0 = np.full((2, 2, 3), 0, dtype=np.uint8)
    img1 = np.full((2, 2, 3), 255, dtype=np.uint8)
    mean, std = compute_channel_stats([img0, img1])
    assert np.allclose(mean, [0.5] * 3)  # 0-255 -> 0-1
    assert np.allclose(std, [0.5] * 3, atol=1e-12)  # 0-255 -> 0-1


def test_compute_channel_stats_per_channel_independent():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[..., 0] = 51
    img[..., 1] = 128
    img[..., 2] = 255
    mean, std = compute_channel_stats([img])
    assert np.allclose(mean, [51 / 255, 128 / 255, 1.0])
    assert np.allclose(std, [0.0, 0.0, 0.0], atol=1e-6)


def test_compute_channel_stats_validation():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="empty"):
        compute_channel_stats([])
    bad_dtype = np.zeros((4, 4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="dtype"):
        compute_channel_stats([bad_dtype])
    bad_shape = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape"):
        compute_channel_stats([bad_shape])
