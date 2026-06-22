import pytest

from land_cover_segmentation.config import Config, dump
from land_cover_segmentation.utils.runs import default_checkpoint_path, load_run_config


def test_load_run_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="config.yaml"):
        load_run_config(tmp_path)


def test_load_run_config_round_trip(trained_run_dir):
    cfg = load_run_config(trained_run_dir)
    assert cfg.data.image_size == 16
    assert cfg.model.source == "custom"


def test_default_checkpoint_path(tmp_path):
    dump(Config(), tmp_path / "config.yaml")
    assert default_checkpoint_path(tmp_path) == tmp_path / "best.pth"
