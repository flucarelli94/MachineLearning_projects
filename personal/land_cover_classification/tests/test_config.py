"""Tests for config dataclasses and strict YAML load/dump."""

from dataclasses import asdict
from pathlib import Path

import pytest

from land_cover_segmentation.config import (
    Config,
    DataConfig,
    ModelConfig,
    RunConfig,
    dump,
    load,
)


def test_run_config_keys():
    run_config = RunConfig()
    assert set(asdict(run_config).keys()) == {
        "output_name",
        "seed",
        "deterministic",
        "device",
    }


def test_model_config_keys():
    model_config = ModelConfig()
    assert set(asdict(model_config).keys()) == {
        "encoder",
        "encoder_weights",
        "in_channels",
    }


def test_config_keys():
    config = Config()
    assert set(asdict(config).keys()) == {"data", "model", "run"}


def test_data_config_keys():
    data_config = DataConfig()
    assert set(asdict(data_config).keys()) == {
        "root",
        "scene",
        "image_size",
        "batch_size",
        "num_workers",
        "ignore_index",
        "nodata_label",
        "seed",
        "classes",
        "palette",
    }


class TestLoad:
    @staticmethod
    def test_applies_run_overrides(tmp_path: Path):
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text(
            "run:\n  output_name: smoke\n  seed: 0\n  deterministic: false\n"
            "  device: cpu\n"
        )
        cfg = load(yaml_path)
        assert cfg.run.output_name == "smoke"
        assert cfg.run.seed == 0
        assert cfg.run.deterministic is False
        assert cfg.run.device == "cpu"

    @staticmethod
    def test_rejects_unknown_run_key(tmp_path: Path):
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text("run:\n  devce: cpu\n")
        with pytest.raises(ValueError, match="run.devce"):
            load(yaml_path)

    @staticmethod
    def test_applies_overrides(tmp_path: Path):
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text(
            "model:\n  encoder: resnet34\n  encoder_weights: null\n  in_channels: 3\n"
        )
        cfg = load(yaml_path)
        assert cfg.model.encoder == "resnet34"
        assert cfg.model.encoder_weights is None
        assert cfg.model.in_channels == 3

    @staticmethod
    def test_rejects_unknown_model_key(tmp_path: Path):
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text("model:\n  encodr: resnet34\n")
        with pytest.raises(ValueError, match="model.encodr"):
            load(yaml_path)

    @staticmethod
    def test_rejects_unknown_top_level_key(tmp_path: Path):
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text("modl:\n  encoder: resnet34\n")
        with pytest.raises(ValueError, match="modl"):
            load(yaml_path)


def test_dump_round_trip_preserves_model(tmp_path: Path):
    config = Config(model=ModelConfig(encoder="mobilenet_v2", encoder_weights=None))
    out = tmp_path / "out.yaml"
    dump(config, out)
    loaded = load(out)
    assert loaded.model.encoder == "mobilenet_v2"
    assert loaded.model.encoder_weights is None
    assert loaded.data.root == config.data.root
