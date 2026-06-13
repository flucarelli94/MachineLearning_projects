import albumentations
import numpy as np

from land_cover_segmentation.dataset.transforms import (
    build_train_transform,
    build_val_transform,
)


def test_build_train_transform_interface():
    transform = build_train_transform(
        image_size=512,
        ignore_index=255,
        mean=[0.4030, 0.4116, 0.3784],
        std=[0.2107, 0.2052, 0.2070],
        seed=214,
    )
    assert transform is not None
    assert isinstance(transform, albumentations.Compose)


def test_build_val_transform_interface():
    transform = build_val_transform(
        image_size=512,
        mean=[0.4030, 0.4116, 0.3784],
        std=[0.2107, 0.2052, 0.2070],
    )
    assert transform is not None
    assert isinstance(transform, albumentations.Compose)
