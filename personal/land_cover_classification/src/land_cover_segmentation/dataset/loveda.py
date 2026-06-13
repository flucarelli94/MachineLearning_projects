"""LoveDA datamodule.

Lifecycle (mirrors Lightning's contract without inheriting from it, so
the datamodule stays dependency-light and easy to unit-test):

* :meth:`LoveDADataModule.prepare_data` — single-process; ensures the
  LoveDA archives are present on disk (idempotent torchgeo download).
* :meth:`LoveDADataModule.setup` — per-process; instantiates per-split
  torchgeo datasets, computes channel normalization statistics on a
  seeded subset of the train split, builds train/val transforms with
  those stats, and wraps each split in a ``_LoveDAAdapter`` so it
  yields ``(image, mask)`` tuples ready for a DataLoader.

The ``train/val/test_dataloader()`` methods are added in a subsequent
step.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import albumentations
import numpy as np
import torch
import torch.utils.data
from torchgeo.datasets import LoveDA

from land_cover_segmentation.config import DataConfig
from land_cover_segmentation.dataset.download import download_loveda
from land_cover_segmentation.dataset.transforms import (
    build_train_transform,
    build_val_transform,
)
from land_cover_segmentation.utils import compute_channel_stats


class _ImageView(Sequence):
    """Lazy sequence view of a torchgeo LoveDA dataset's images as HWC
    ``uint8`` numpy arrays, for consumption by :func:`compute_channel_stats`.

    Each ``__getitem__`` call pulls a single sample from the underlying
    dataset (i.e. one PNG decode), so passing this view to the stats
    helper never materializes all images.
    """

    def __init__(self, ds: LoveDA) -> None:
        self._ds = ds

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> np.ndarray:
        # torchgeo's LoveDA returns image as (C, H, W) uint8; we need HWC.
        img = self._ds[idx]["image"].numpy()
        return np.transpose(img, (1, 2, 0))


class _LoveDAAdapter(torch.utils.data.Dataset):
    """Wrap a torchgeo LoveDA split with an Albumentations pipeline.

    Converts torchgeo's dict items (image as ``(C, H, W)`` ``uint8``
    tensor, mask as ``(H, W)`` integer tensor) into the trainer-facing
    ``(image, mask)`` tuple contract:

    * ``image``: ``(3, H, W)`` ``float32`` torch tensor, normalized.
    * ``mask``:  ``(H, W)`` ``int64`` torch tensor with values in
      ``0..num_classes-1`` or ``ignore_index``.

    Single source of truth for the ``nodata_label → ignore_index`` remap:
    the remap happens **before** the augmentation pipeline so any pixels
    later filled by affine / coarse-dropout (which also write
    ``ignore_index``) stay consistent with naturally-occurring nodata.
    """

    def __init__(
        self,
        ds: Any,
        transform: albumentations.Compose,
        *,
        nodata_label: int,
        ignore_index: int,
    ) -> None:
        self._ds = ds
        self._transform = transform
        self._nodata_label = nodata_label
        self._ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self._ds[idx]
        image = np.transpose(item["image"].numpy(), (1, 2, 0))  # HWC uint8
        mask = item["mask"].numpy()
        if self._nodata_label != self._ignore_index:
            mask = np.where(mask == self._nodata_label, self._ignore_index, mask)

        out = self._transform(image=image, mask=mask)
        return out["image"], out["mask"].to(torch.int64)


class LoveDADataModule:
    """LoveDA datamodule: download → adapt → transform → loader.

    Parameters
    ----------
    cfg : DataConfig
        Resolved data-layer configuration. The datamodule reads every
        knob from here; it does not accept per-call overrides.

    Notes
    -----
    Call :meth:`prepare_data` once (typically from the trainer's main
    process) to ensure the dataset is on disk, then :meth:`setup` to
    instantiate per-split datasets and compute normalization stats.
    DataLoader-building methods are added in a later step.
    """

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self._train_ds_raw: LoveDA | None = None
        self._val_ds_raw: LoveDA | None = None
        self._test_ds_raw: LoveDA | None = None
        self.train_ds: _LoveDAAdapter | None = None
        self.val_ds: _LoveDAAdapter | None = None
        self._mean: list[float] | None = None
        self._std: list[float] | None = None
        self._is_setup = False

    def prepare_data(self) -> None:
        """Idempotently download the LoveDA archives for every split.

        Files already present on disk are skipped, so this is safe to
        re-run. Checksums are verified against torchgeo's manifest.
        """
        download_loveda(
            root=self.cfg.root,
            splits=("train", "val", "test"),
            scenes=tuple(self.cfg.scene),
            checksum=True,
        )

    def setup(self, stage: str | None = None) -> None:
        """Instantiate per-split datasets and compute normalization stats.

        Idempotent: a second call is a no-op.

        Parameters
        ----------
        stage : str, optional
            Mirrors Lightning's ``stage`` argument (``"fit"`` /
            ``"validate"`` / ``"test"``). Currently unused — all three
            splits are loaded regardless. Kept for forward compatibility.

        Raises
        ------
        torchgeo.datasets.errors.DatasetNotFoundError
            If the dataset is not on disk. Call :meth:`prepare_data`
            first to download it (we intentionally pass ``download=False``
            here so ``setup()`` never silently kicks off a 4 GB transfer).
        """
        if self._is_setup:
            return

        common_args = dict(
            root=self.cfg.root, scene=list(self.cfg.scene), download=False
        )
        self._train_ds_raw = LoveDA(split="train", **common_args)
        self._val_ds_raw = LoveDA(split="val", **common_args)
        self._test_ds_raw = LoveDA(split="test", **common_args)

        self._mean, self._std = compute_channel_stats(_ImageView(self._train_ds_raw))

        train_transform = build_train_transform(
            image_size=self.cfg.image_size,
            ignore_index=self.cfg.ignore_index,
            mean=self._mean,
            std=self._std,
            seed=self.cfg.seed,
        )
        val_transform = build_val_transform(
            image_size=self.cfg.image_size,
            mean=self._mean,
            std=self._std,
        )
        self.train_ds = _LoveDAAdapter(
            self._train_ds_raw,
            train_transform,
            nodata_label=self.cfg.nodata_label,
            ignore_index=self.cfg.ignore_index,
        )
        self.val_ds = _LoveDAAdapter(
            self._val_ds_raw,
            val_transform,
            nodata_label=self.cfg.nodata_label,
            ignore_index=self.cfg.ignore_index,
        )

        self._is_setup = True

    @property
    def mean(self) -> list[float]:
        """Per-channel mean computed at :meth:`setup` time, in ``[0, 1]``."""
        self._require_setup()
        assert self._mean is not None
        return self._mean

    @property
    def std(self) -> list[float]:
        """Per-channel standard deviation computed at :meth:`setup` time."""
        self._require_setup()
        assert self._std is not None
        return self._std

    def _require_setup(self) -> None:
        if not self._is_setup:
            raise RuntimeError(
                "LoveDADataModule.setup() must be called before accessing this property"
            )


__all__ = ["LoveDADataModule"]
