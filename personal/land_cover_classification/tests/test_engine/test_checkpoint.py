"""Tests for checkpoint I/O."""

import json
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from land_cover_segmentation.config import Config
from land_cover_segmentation.engine.checkpoint import CheckpointIO


def test_checkpoint_io_load_run_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="config.yaml"):
        CheckpointIO.load_run_config(tmp_path)


def test_checkpoint_io_writes_pth_and_meta(tmp_path):
    model = nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    datamodule = MagicMock()
    datamodule.mean = [0.1, 0.2, 0.3]
    datamodule.std = [0.9, 0.8, 0.7]

    checkpoint_io = CheckpointIO(Config(), datamodule, tmp_path)
    ckpt_path = tmp_path / "best.pth"
    val_metrics = {
        "loss": 0.5,
        "miou": 0.42,
        "per_class_iou": [0.4, None],
    }
    checkpoint_io.save(
        ckpt_path,
        model=model,
        optimizer=optimizer,
        epoch=3,
        val_metrics=val_metrics,
        best_val_miou=0.42,
        is_best=True,
    )

    assert ckpt_path.exists()
    payload = torch.load(ckpt_path, weights_only=False)
    assert payload["epoch"] == 3
    assert payload["mean"] == datamodule.mean

    sidecar = json.loads(ckpt_path.with_suffix(".meta.json").read_text())
    assert sidecar["checkpoint"] == "best.pth"
    assert sidecar["config"]["data"]["root"] == Config().data.root
    assert sidecar["val_metrics"]["miou"] == pytest.approx(0.42)
    assert sidecar["val_metrics"]["per_class_iou"] == [0.4, None]

    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["epoch"] == 3
