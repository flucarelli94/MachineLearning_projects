"""LoveDA datamodule.

Lifecycle:

* `LoveDADataModule.prepare_data` — single-process; ensures the
  LoveDA archives are present on disk (idempotent torchgeo download;
  ~20 GB for all splits).
* `LoveDADataModule.setup` — per-process; instantiates per-split
  torchgeo datasets, computes channel normalization statistics on the
  train split, builds train/val transforms with those stats, and wraps
  each split in a `_LoveDAAdapter` so it yields `(image, mask)`
  tuples ready for a DataLoader.
"""

from collections.abc import Sequence
from typing import Any

import albumentations
import numpy as np
import torch
import torch.utils.data
from torchgeo.datasets import LoveDA

from land_cover_segmentation.config import DataConfig
from land_cover_segmentation.dataset.download import download_loveda
from land_cover_segmentation.dataset.augmentation import (
    build_train_augmentation,
    build_val_augmentation,
)
from land_cover_segmentation.utils import configure_logging
from land_cover_segmentation.utils.data import compute_channel_stats

logger = configure_logging(__name__)

STATS_MAX_SAMPLES = 1000


def _loveda_image_hwc_uint8(image: torch.Tensor) -> np.ndarray:
    """Convert a torchgeo LoveDA image tensor to HWC uint8 numpy."""
    arr = np.transpose(image.detach().cpu().numpy(), (1, 2, 0))
    if arr.dtype == np.uint8:
        return arr
    return np.clip(arr, 0, 255).astype(np.uint8)


def _subset_indices(n: int, fraction: float, seed: int) -> list[int]:
    """Return sorted indices for a deterministic random subset of size ``k``.

    Parameters
    ----------
    n : int
        Size of the full index range ``0..n-1``.
    fraction : float
        Fraction in `(0, 1]` of indices to keep.
    seed : int
        RNG seed for reproducible selection.

    Returns
    -------
    list[int]
        Sorted subset indices, length ``max(1, int(n * fraction))`` capped at ``n``.
    """
    k = max(1, int(n * fraction))
    if k >= n:
        return list(range(n))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(n, size=k, replace=False).tolist())


class _ImageView(Sequence):
    """Lazy sequence view of a torchgeo LoveDA dataset's images as HWC
    `uint8` numpy arrays, for consumption by `compute_channel_stats`.

    Each `__getitem__` call pulls a single sample from the underlying
    dataset (i.e. one PNG decode), so passing this view to the stats
    helper never materializes all images.
    """

    def __init__(self, ds: LoveDA) -> None:
        self._ds = ds

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> np.ndarray:
        return _loveda_image_hwc_uint8(self._ds[idx]["image"])


class _LoveDAAdapter(torch.utils.data.Dataset):
    """Wrap a torchgeo LoveDA split with an Albumentations pipeline.

    Converts torchgeo's dict items (image as `(C, H, W)` `uint8`
    tensor, mask as `(H, W)` integer tensor) into the trainer-facing
    `(image, mask)` tuple contract:

    * `image`: `(3, H, W)` `float32` torch tensor, normalized.
    * `mask`:  `(H, W)` `int64` torch tensor with values in
      `0..num_classes-1` or `ignore_index`.

    Single source of truth for the `nodata_label → ignore_index` remap:
    the remap happens **before** the augmentation pipeline so any pixels
    later filled by affine / coarse-dropout (which also write
    `ignore_index`) stay consistent with naturally-occurring nodata.
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
        image = _loveda_image_hwc_uint8(item["image"])
        mask = item["mask"].numpy()
        if self._nodata_label != self._ignore_index:
            mask = np.where(mask == self._nodata_label, self._ignore_index, mask)

        out = self._transform(image=image, mask=mask)
        return out["image"], out["mask"].to(torch.int64)

    def set_seed(self, seed: int) -> None:
        """Reseed the Albumentations Compose. Called per-worker.

        Delegates to `set_random_seed()`, which
        rebuilds an internal `numpy.random.Generator` — Albumentations
        does not use the legacy global `np.random` module.
        """
        self._transform.set_random_seed(seed)


def _worker_init_fn(worker_id: int) -> None:
    """Per-worker reseed for Albumentations augmentations.

    PyTorch's DataLoader already seeds `torch` per worker via `info.seed`. Albumentations keeps its
    own internal `numpy.random.Generator` (created by
    `albumentations.Compose.set_random_seed`).
    Without that, every worker would emit identical augmentation sequences.
    """
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    ds = info.dataset
    if hasattr(ds, "set_seed"):
        ds.set_seed(info.seed)


class LoveDADataModule:
    """LoveDA datamodule: download → adapt → transform → loader.

    Parameters
    ----------
    cfg : DataConfig
        Resolved data-layer configuration. The datamodule reads every
        knob from here; it does not accept per-call overrides.

    Notes
    -----
    Call `prepare_data()` once (typically from the trainer's main
    process) to ensure the dataset is on disk, then `setup()` to
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
        self.test_ds: _LoveDAAdapter | None = None
        self._mean: list[float] | None = None
        self._std: list[float] | None = None
        self._is_setup = False

    def prepare_data(self) -> None:
        """Idempotently download the LoveDA archives for every split.

        Files already present on disk are skipped, so this is safe to
        re-run. Checksums are verified against torchgeo's manifest.
        """
        logger.info("Downloading LoveDA dataset")
        download_loveda(
            root=self.cfg.root,
            splits=("train", "val", "test"),
            scenes=tuple(self.cfg.scene),
            checksum=True,
        )

    def setup(self) -> None:
        """Instantiate per-split datasets and compute normalization stats.

        Idempotent: a second call is a no-op.

        Raises
        ------
        torchgeo.datasets.errors.DatasetNotFoundError
            If the dataset is not on disk. Call `prepare_data()`
            first to download it (we intentionally pass `download=False`
            here so `setup()` never silently kicks off a ~20 GB transfer).
        """
        if self._is_setup:
            return

        common_args = dict(
            root=self.cfg.root, scene=list(self.cfg.scene), download=False
        )
        train_full = LoveDA(split="train", **common_args)
        val_full = LoveDA(split="val", **common_args)
        test_full = LoveDA(split="test", **common_args)

        if self.cfg.fraction < 1.0:
            logger.info(
                "Using fraction=%.2f: train %d/%d, val %d/%d, test %d/%d",
                self.cfg.fraction,
                max(1, int(len(train_full) * self.cfg.fraction)),
                len(train_full),
                max(1, int(len(val_full) * self.cfg.fraction)),
                len(val_full),
                max(1, int(len(test_full) * self.cfg.fraction)),
                len(test_full),
            )

        train_indices = _subset_indices(
            len(train_full), self.cfg.fraction, self.cfg.seed
        )
        val_indices = _subset_indices(len(val_full), self.cfg.fraction, self.cfg.seed)
        test_indices = _subset_indices(len(test_full), self.cfg.fraction, self.cfg.seed)
        self._train_ds_raw = torch.utils.data.Subset(train_full, train_indices)
        self._val_ds_raw = torch.utils.data.Subset(val_full, val_indices)
        self._test_ds_raw = torch.utils.data.Subset(test_full, test_indices)

        logger.info("Computing channel statistics")
        max_samples = min(STATS_MAX_SAMPLES, len(self._train_ds_raw))
        self._mean, self._std = compute_channel_stats(
            _ImageView(self._train_ds_raw),
            max_samples=max_samples,
            seed=self.cfg.seed,
        )

        logger.info("Building train augmentation")
        train_transform = build_train_augmentation(
            image_size=self.cfg.image_size,
            ignore_index=self.cfg.ignore_index,
            mean=self._mean,
            std=self._std,
            seed=self.cfg.seed,
        )
        logger.info("Building val augmentation")
        val_transform = build_val_augmentation(
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
        self.test_ds = _LoveDAAdapter(
            self._test_ds_raw,
            val_transform,
            nodata_label=self.cfg.nodata_label,
            ignore_index=self.cfg.ignore_index,
        )

        self._is_setup = True

    @property
    def mean(self) -> list[float]:
        """Per-channel mean computed at `setup()` time, in `[0, 1]`."""
        self._require_setup()
        assert self._mean is not None
        return self._mean

    @property
    def std(self) -> list[float]:
        """Per-channel standard deviation computed at `setup()` time."""
        self._require_setup()
        assert self._std is not None
        return self._std

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Shuffled, augmented DataLoader over the train split.

        Determinism: shuffling is driven by a torch `Generator` seeded
        with `cfg.seed`; per-worker augmentation seeding is handled by
        `_worker_init_fn()`. Together these make every batch
        reproducible for a given `cfg.seed`.
        """
        self._require_setup()
        assert self.train_ds is not None
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
            generator=torch.Generator().manual_seed(self.cfg.seed),
            worker_init_fn=_worker_init_fn,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Sequential, non-augmented DataLoader over the val split.

        `drop_last=False` so every val sample contributes to the
        streaming confusion matrix. The val transform is deterministic
        (center-crop + normalize) so `worker_init_fn` is not strictly
        required, but we still pass it for consistency with the train
        loader.
        """
        self._require_setup()
        assert self.val_ds is not None
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
            worker_init_fn=_worker_init_fn,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Sequential DataLoader over the test split (val-style transforms)."""
        self._require_setup()
        assert self.test_ds is not None
        return torch.utils.data.DataLoader(
            self.test_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
            worker_init_fn=_worker_init_fn,
        )

    def _require_setup(self) -> None:
        if not self._is_setup:
            raise RuntimeError(
                "LoveDADataModule.setup() must be called before accessing this property"
            )


__all__ = ["LoveDADataModule", "STATS_MAX_SAMPLES", "_subset_indices"]
