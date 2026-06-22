import pytest
import torch

from land_cover_segmentation.config import DataConfig
from land_cover_segmentation.dataset.loveda import (
    LoveDADataModule,
    _LoveDAAdapter,
)
from land_cover_segmentation.dataset.augmentation import (
    build_train_augmentation,
    build_val_augmentation,
)


@pytest.fixture()
def fake_split():
    def _fake_split(items: list[dict]) -> "_FakeSplit":
        class _FakeSplit:
            """Minimal indexable list-of-items, mimicking a torchgeo split."""

            def __init__(self, items: list[dict]) -> None:
                self._items = items

            def __len__(self) -> int:
                return len(self._items)

            def __getitem__(self, idx: int) -> dict:
                return self._items[idx]

        return _FakeSplit(items)

    return _fake_split


@pytest.fixture
def datamodule_with_fake_adapters(request):
    """Build a datamodule whose train/val adapters wrap an in-memory fake
    split, bypassing torchgeo entirely. Used to avoid downloading ~20 GB.
    """
    fake_split = request.getfixturevalue("fake_split")

    def _datamodule_with_fake_adapters(
        image_size: int = 64, n_items: int = 8, batch_size: int = 2
    ) -> LoveDADataModule:
        cfg = DataConfig(image_size=image_size, batch_size=batch_size, num_workers=0)
        dm = LoveDADataModule(cfg)
        train_items = [
            {
                "image": torch.zeros((3, image_size, image_size), dtype=torch.uint8),
                "mask": torch.full((image_size, image_size), 0, dtype=torch.long),
            }
            for _ in range(n_items)
        ]
        val_items = [
            {
                "image": torch.zeros((3, image_size, image_size), dtype=torch.uint8),
                "mask": torch.full((image_size, image_size), 0, dtype=torch.long),
            }
            for _ in range(n_items)
        ]
        train_t = build_train_augmentation(
            image_size=image_size,
            ignore_index=cfg.ignore_index,
            mean=[0.5] * 3,
            std=[0.5] * 3,
            seed=cfg.seed,
        )
        val_t = build_val_augmentation(
            image_size=image_size, mean=[0.5] * 3, std=[0.5] * 3
        )
        dm.train_ds = _LoveDAAdapter(
            fake_split(train_items),
            train_t,
            nodata_label=cfg.nodata_label,
            ignore_index=cfg.ignore_index,
        )
        dm.val_ds = _LoveDAAdapter(
            fake_split(val_items),
            val_t,
            nodata_label=cfg.nodata_label,
            ignore_index=cfg.ignore_index,
        )
        dm._mean = [0.5] * 3
        dm._std = [0.5] * 3
        dm._is_setup = True
        return dm

    return _datamodule_with_fake_adapters
