import json
from pathlib import Path

import pytest

from land_cover_segmentation.training.callbacks import EarlyStopping, JSONLLogger


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
    assert json.loads(lines[0]) == {"epoch": 1, "val_miou": 0.42}
    assert json.loads(lines[1]) == {"epoch": 2, "val_miou": 0.44}


def test_jsonl_logger_serializes_nan(tmp_path: Path):
    logger = JSONLLogger(tmp_path / "log.jsonl")
    logger.log({"score": float("nan")})
    record = json.loads((tmp_path / "log.jsonl").read_text())
    assert record["score"] is None
