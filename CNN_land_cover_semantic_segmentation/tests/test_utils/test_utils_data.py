import numpy as np
import pytest

from land_cover_segmentation.utils.data import compute_channel_stats


def test_compute_channel_stats_constant_image():
    constant = 128
    img = np.full((4, 4, 3), constant, dtype=np.uint8)
    mean, std = compute_channel_stats([img, img, img])
    assert np.allclose(mean, [constant / 255] * 3)
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


def test_compute_channel_stats_returns_1d_per_channel():
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[:, 1, 0] = 255  # channel 0: left col 0, right col 255
    mean, std = compute_channel_stats([img])
    assert np.array(mean).shape == (3,)
    assert np.array(std).shape == (3,)
    assert np.allclose(mean, [0.5, 0.0, 0.0])
    assert np.allclose(std, [0.5, 0.0, 0.0])


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
    many = [np.full((2, 2, 3), 128, dtype=np.uint8) for _ in range(20)]
    with pytest.raises(ValueError, match="seed"):
        compute_channel_stats(many, max_samples=5)


def test_compute_channel_stats_max_samples_matches_full_when_large():
    img0 = np.full((2, 2, 3), 0, dtype=np.uint8)
    img1 = np.full((2, 2, 3), 255, dtype=np.uint8)
    images = [img0, img1]
    full_mean, full_std = compute_channel_stats(images)
    sampled_mean, sampled_std = compute_channel_stats(images, max_samples=10, seed=0)
    assert np.allclose(full_mean, sampled_mean)
    assert np.allclose(full_std, sampled_std)


def test_compute_channel_stats_max_samples_is_deterministic():
    images = [np.full((2, 2, 3), value, dtype=np.uint8) for value in range(10)]
    first = compute_channel_stats(images, max_samples=4, seed=42)
    second = compute_channel_stats(images, max_samples=4, seed=42)
    assert first == second
