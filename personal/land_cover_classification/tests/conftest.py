from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from land_cover_segmentation.config import (
    Config,
    DataConfig,
    ModelConfig,
    RunConfig,
    dump,
)
from land_cover_segmentation.engine.checkpoint import CheckpointIO
from land_cover_segmentation.models.factory import build_model


@dataclass
class StubDataModule:
    """Minimal stand-in for LoveDADataModule when only mean/std are needed."""

    mean: list[float]
    std: list[float]


@pytest.fixture
def stub_data_module_cls():
    return StubDataModule


@pytest.fixture
def synthetic_geotiff(tmp_path):
    path = tmp_path / "scene.tif"
    transform = from_origin(10.0, 20.0, 0.5, 0.5)
    data = np.zeros((3, 16, 16), dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=16,
        width=16,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)
    return path


@pytest.fixture
def trained_run_dir(stub_data_module_cls, tmp_path):
    cfg = Config(
        data=DataConfig(classes=["a", "b", "c"], image_size=16, batch_size=2),
        model=ModelConfig(source="custom", in_channels=3),
        run=RunConfig(device="cpu"),
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    dump(cfg, run_dir / "config.yaml")

    model = build_model(cfg)
    datamodule = stub_data_module_cls(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    CheckpointIO(cfg, datamodule, run_dir).save(
        run_dir / "best.pth",
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        epoch=0,
        val_metrics={"loss": 0.0, "miou": 0.0, "per_class_iou": [0.0, 0.0, 0.0]},
        best_val_miou=0.0,
        is_best=True,
    )
    return run_dir
