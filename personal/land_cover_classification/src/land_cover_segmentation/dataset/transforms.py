"""Albumentations pipelines for LoveDA.

Two factory functions:

* `build_train_transform` — geometric + photometric augmentation.
* `build_val_transform`   — deterministic crop + normalize.

Contract (the datamodule honors it on both ends):

* **Input**  – `image` is HWC `uint8` ndarray, `mask` is HW `int`
  ndarray with values in `0..num_classes-1` or `ignore_index`. The
  torchgeo LoveDA dataset's `nodata_label` must already be remapped to
  `ignore_index`.
* **Output** – `image` is `(3, H, W) float32` torch tensor (normalized),
  `mask` is `(H, W)` torch tensor (caller casts to `int64` for loss).

Augmentations that drop pixels (affine fill, coarse dropout) write
`ignore_index` into the mask so the loss and metrics skip them.

`ignore_index` is caller-supplied (it lives in `DataConfig.ignore_index`).
"""

from collections.abc import Sequence

import albumentations


def build_train_transform(
    image_size: int,
    *,
    ignore_index: int,
    mean: Sequence[float],
    std: Sequence[float],
    seed: int,
) -> albumentations.Compose:
    """Training pipeline: random crop + flips/rotations + mild affine + photometric + normalize.

    Pixels introduced by affine fill or coarse dropout get the value
    `ignore_index` in the mask so they are skipped by loss and metrics.

    Parameters
    ----------
    image_size : int
        Side length (pixels) of the square crop fed to the model.
    ignore_index : int, keyword-only
        Mask value written wherever augmentation drops pixels. Required
        — pass `config.data.ignore_index` so transforms, loss, and
        metrics stay in lockstep.
    mean, std : Sequence[float], keyword-only
        Per-channel normalization statistics in `[0, 1]`, mean and
        standard deviation respectively.
    seed : int, keyword-only
        Seeds the Compose's internal RNG so the augmentation *sequence*
        is reproducible across runs while still varying within a run.
        Multi-worker DataLoaders must additionally offset this per
        worker (handled in :class:`LoveDADataModule` via
        `worker_init_fn`) to avoid all workers producing identical
        augmentations.

    Returns
    -------
    albumentations.Compose
        A callable taking `image=HWC uint8` and `mask=HW int` and
        returning `image=(3, H, W) float32` and `mask=(H, W)` tensors.
    """
    return albumentations.Compose(
        [
            albumentations.RandomCrop(height=image_size, width=image_size),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.RandomRotate90(p=0.5),
            # Equivalent to plan's ShiftScaleRotate(shift=0.05, scale=0.1, rot=15);
            # ShiftScaleRotate is deprecated in albumentations 2.x in favor of Affine.
            albumentations.Affine(
                translate_percent=(-0.05, 0.05),
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                fill=0,
                fill_mask=ignore_index,
                p=0.5,
            ),
            albumentations.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05,
                p=0.5,
            ),
            # std_range is in normalized [0,1] image units in albumentations 2.x
            # (~ plan's var_limit=(5,20) on uint8: std_pixel ~ 2-4.5 -> 0.01-0.02).
            albumentations.GaussNoise(std_range=(0.01, 0.05), p=0.2),
            albumentations.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(8, 32),
                hole_width_range=(8, 32),
                fill=0,
                fill_mask=ignore_index,
                p=0.3,
            ),
            albumentations.Normalize(
                mean=tuple(mean), std=tuple(std), max_pixel_value=255.0
            ),
            albumentations.pytorch.ToTensorV2(),
        ],
        seed=seed,
    )


def build_val_transform(
    image_size: int,
    *,
    mean: Sequence[float],
    std: Sequence[float],
) -> albumentations.Compose:
    """Validation pipeline: deterministic center crop + normalize.

    No randomness — same input always yields the same output, which keeps
    validation metrics comparable across runs and epochs.

    Parameters
    ----------
    image_size : int
        Side length (pixels) of the square center crop.
    mean, std : Sequence[float], keyword-only
        Per-channel normalization statistics in `[0, 1]`,
        mean and standard deviation, respectively.

    Returns
    -------
    albumentations.Compose
        A callable with the same input/output contract as `build_train_transform`.
    """
    return albumentations.Compose(
        [
            albumentations.CenterCrop(height=image_size, width=image_size),
            albumentations.Normalize(
                mean=tuple(mean), std=tuple(std), max_pixel_value=255.0
            ),
            albumentations.pytorch.ToTensorV2(),
        ]
    )


__all__ = ["build_train_transform", "build_val_transform"]
