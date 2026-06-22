"""Tests for the LoveDA datamodule adapter and lifecycle wiring.

The torchgeo dataset itself is not exercised here — we use a small
in-memory fake split that matches torchgeo's LoveDA item shape
(``{"image": (C,H,W) uint8 tensor, "mask": (H,W) int tensor}``).
"""

import pytest
import torch

from land_cover_segmentation.config import DataConfig
from land_cover_segmentation.dataset.loveda import (
    LoveDADataModule,
    _LoveDAAdapter,
    _subset_indices,
)
from land_cover_segmentation.dataset.augmentation import (
    build_train_augmentation,
    build_val_augmentation,
)

class TestLoveDAAdapter:
    val_transform = build_val_augmentation(image_size=64, mean=[0.5] * 3, std=[0.5] * 3)
    image = torch.zeros((3, 64, 64), dtype=torch.uint8)
    mask = staticmethod(
        lambda mask_value: torch.full((64, 64), mask_value, dtype=torch.long)
    )

    def test_adapter_remaps_nodata_to_ignore_index(self, fake_split):
        ds = fake_split([{"image": self.image, "mask": self.mask(7)}])
        adapter = _LoveDAAdapter(
            ds, self.val_transform, nodata_label=7, ignore_index=255
        )
        _, mask = adapter[0]
        assert (
            mask == 255
        ).all(), "every nodata pixel should be remapped to ignore_index"
        assert (mask != 7).all(), "no original nodata label should survive"

    def test_adapter_preserves_non_nodata_labels(self, fake_split):
        ds = fake_split([{"image": self.image, "mask": self.mask(3)}])
        adapter = _LoveDAAdapter(
            ds, self.val_transform, nodata_label=7, ignore_index=255
        )
        _, mask = adapter[0]
        assert (mask == 3).all()

    def test_adapter_output_shapes_and_dtypes(self, fake_split):
        ds = fake_split([{"image": self.image, "mask": self.mask(0)}])
        adapter = _LoveDAAdapter(
            ds, self.val_transform, nodata_label=7, ignore_index=255
        )
        img, mask = adapter[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 64, 64)
        assert img.dtype == torch.float32
        assert isinstance(mask, torch.Tensor)
        assert mask.shape == (64, 64)
        assert mask.dtype == torch.int64

    def test_adapter_length_matches_underlying_split(self, fake_split):
        ds = fake_split([{"image": self.image, "mask": self.mask(0)} for _ in range(5)])
        adapter = _LoveDAAdapter(
            ds, self.val_transform, nodata_label=7, ignore_index=255
        )
        assert len(adapter) == 5

def test_datamodule_properties_require_setup():
    dm = LoveDADataModule(DataConfig())
    with pytest.raises(RuntimeError, match="setup"):
        _ = dm.mean
    with pytest.raises(RuntimeError, match="setup"):
        _ = dm.std

def test_datamodule_dataloaders_require_setup():
    dm = LoveDADataModule(DataConfig())
    with pytest.raises(RuntimeError, match="setup"):
        dm.train_dataloader()
    with pytest.raises(RuntimeError, match="setup"):
        dm.val_dataloader()

def test_train_dataloader_batch_shapes_and_dtypes(datamodule_with_fake_adapters):
    dm = datamodule_with_fake_adapters(image_size=64, n_items=8, batch_size=2)
    img, mask = next(iter(dm.train_dataloader()))
    assert img.shape == (2, 3, 64, 64) and img.dtype == torch.float32
    assert mask.shape == (2, 64, 64) and mask.dtype == torch.int64

def test_val_dataloader_batch_shapes_and_dtypes(datamodule_with_fake_adapters):
    dm = datamodule_with_fake_adapters(image_size=64, n_items=8, batch_size=2)
    img, mask = next(iter(dm.val_dataloader()))
    assert img.shape == (2, 3, 64, 64) and img.dtype == torch.float32
    assert mask.shape == (2, 64, 64) and mask.dtype == torch.int64

def test_train_dataloader_is_deterministic_with_same_seed(
    datamodule_with_fake_adapters,
):
    dm1 = datamodule_with_fake_adapters(image_size=64, n_items=8, batch_size=2)
    dm2 = datamodule_with_fake_adapters(image_size=64, n_items=8, batch_size=2)
    img1, mask1 = next(iter(dm1.train_dataloader()))
    img2, mask2 = next(iter(dm2.train_dataloader()))
    assert torch.equal(img1, img2)
    assert torch.equal(mask1, mask2)

def test_adapter_set_seed_changes_augmentation_output(fake_split):

    ds = fake_split(
        [
            {
                "image": torch.zeros((3, 64, 64), dtype=torch.uint8),
                "mask": torch.full((64, 64), 0, dtype=torch.long),
            }
        ]
    )
    transform = build_train_augmentation(
        image_size=32, ignore_index=255, mean=[0.5] * 3, std=[0.5] * 3, seed=0
    )
    adapter = _LoveDAAdapter(ds, transform, nodata_label=7, ignore_index=255)
    adapter.set_seed(1)
    img_a, _ = adapter[0]
    adapter.set_seed(1)
    img_a_again, _ = adapter[0]
    # same seed must reproduce the same output
    assert torch.equal(img_a, img_a_again), "same seed must reproduce the same output"

    # different seed must change the output
    adapter.set_seed(999)
    img_b, _ = adapter[0]
    assert not torch.equal(img_a, img_b), "different seed must change the output"

def test_subset_indices_uses_half_when_fraction_is_one_half():
    indices = _subset_indices(10, fraction=0.5, seed=214)
    assert len(indices) == 5
    assert indices == sorted(indices)
    assert all(0 <= idx < 10 for idx in indices)

def test_subset_indices_is_deterministic_for_same_seed():
    first = _subset_indices(20, fraction=0.5, seed=7)
    second = _subset_indices(20, fraction=0.5, seed=7)
    assert first == second

def test_subset_indices_returns_full_range_when_fraction_is_one():
    indices = _subset_indices(6, fraction=1.0, seed=0)
    assert indices == [0, 1, 2, 3, 4, 5]

def test_subset_wraps_dataset_when_fraction_below_one(fake_split):
    ds = fake_split([{"image": torch.zeros(1), "mask": torch.zeros(1)} for _ in range(10)])
    indices = _subset_indices(len(ds), fraction=0.5, seed=214)
    subset = torch.utils.data.Subset(ds, indices)
    assert len(subset) == 5
