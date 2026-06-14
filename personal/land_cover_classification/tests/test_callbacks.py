"""Tests for training callbacks."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from land_cover_segmentation.config import Config
from land_cover_segmentation.engine.callbacks import (
    CheckpointWriter,
    EarlyStopping,
    JSONLLogger,
)


def test_early_stopping_stops_after_patience():
    stopper = EarlyStopping(patience=2)
    assert stopper.step(0.5) is False
    assert stopper.step(0.4) is False
    assert stopper.step(0.3) is True


def test_early_stopping_resets_on_improvement():
    stopper = EarlyStopping(patience=2)
    assert stopper.step(0.5) is False
    assert stopper.step(0.4) is False
    assert stopper.step(0.6) is False
    assert stopper.step(0.55) is False
    assert stopper.step(0.54) is True


def test_jsonl_logger_appends_records(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    logger = JSONLLogger(path)
    logger.log({"epoch": 1, "val_miou": 0.42})
    logger.log({"epoch": 2, "val_miou": 0.44})
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["epoch"] == 1
    assert json.loads(lines[1])["val_miou"] == pytest.approx(0.44)


def test_jsonl_logger_serializes_nan(tmp_path: Path):
    logger = JSONLLogger(tmp_path / "log.jsonl")
    logger.log({"score": float("nan")})
    record = json.loads((tmp_path / "log.jsonl").read_text())
    assert record["score"] is None


def test_checkpoint_writer_writes_pth_and_meta(tmp_path: Path):
    model = nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    datamodule = MagicMock()
    datamodule.mean = [0.1, 0.2, 0.3]
    datamodule.std = [0.9, 0.8, 0.7]

    writer = CheckpointWriter(Config(), datamodule, tmp_path)
    ckpt_path = tmp_path / "best.pth"
    val_metrics = {
        "loss": 0.5,
        "miou": 0.42,
        "per_class_iou": [0.4, None],
    }
    writer.save(
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
