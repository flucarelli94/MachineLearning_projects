"""Tests for the LoveDA datamodule adapter and lifecycle wiring.

The torchgeo dataset itself is not exercised here — we use a small
in-memory fake split that matches torchgeo's LoveDA item shape
(``{"image": (C,H,W) uint8 tensor, "mask": (H,W) int tensor}``).
"""

from __future__ import annotations

import pytest
import torch

from land_cover_segmentation.config import DataConfig
from land_cover_segmentation.dataset.loveda import (
    LoveDADataModule,
    _LoveDAAdapter,
)
from land_cover_segmentation.dataset.transforms import build_val_transform


class _FakeSplit:
    """Minimal indexable list-of-items, mimicking a torchgeo split."""

    def __init__(self, items: list[dict]) -> None:
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        return self._items[idx]


class TestLoveDAAdapter:
    val_transform = build_val_transform(image_size=64, mean=[0.5] * 3, std=[0.5] * 3)
    image = torch.zeros((3, 64, 64), dtype=torch.uint8)
    mask = staticmethod(
        lambda mask_value: torch.full((64, 64), mask_value, dtype=torch.long)
    )

    def test_adapter_remaps_nodata_to_ignore_index(self):
        ds = _FakeSplit([{"image": self.image, "mask": self.mask(7)}])
        adapter = _LoveDAAdapter(
            ds, self.val_transform, nodata_label=7, ignore_index=255
        )
        _, mask = adapter[0]
        assert (
            mask == 255
        ).all(), "every nodata pixel should be remapped to ignore_index"
        assert (mask != 7).all(), "no original nodata label should survive"

    def test_adapter_preserves_non_nodata_labels(self):
        ds = _FakeSplit([{"image": self.image, "mask": self.mask(3)}])
        adapter = _LoveDAAdapter(
            ds, self.val_transform, nodata_label=7, ignore_index=255
        )
        _, mask = adapter[0]
        assert (mask == 3).all()

    def test_adapter_output_shapes_and_dtypes(self):
        ds = _FakeSplit([{"image": self.image, "mask": self.mask(0)}])
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

    def test_adapter_length_matches_underlying_split(self):
        ds = _FakeSplit([{"image": self.image, "mask": self.mask(0)} for _ in range(5)])
        adapter = _LoveDAAdapter(
            ds, self.val_transform, nodata_label=7, ignore_index=255
        )
        assert len(adapter) == 5

    @staticmethod
    def test_datamodule_properties_require_setup():
        dm = LoveDADataModule(DataConfig())
        with pytest.raises(RuntimeError, match="setup"):
            _ = dm.mean
        with pytest.raises(RuntimeError, match="setup"):
            _ = dm.std
