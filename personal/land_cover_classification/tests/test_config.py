"""Tests for config dataclasses and strict YAML load/dump."""

from dataclasses import asdict
from pathlib import Path

import pytest

from land_cover_segmentation.config import Config, DataConfig, ModelConfig, dump, load


class TestModelConfig:
    @staticmethod
    def test_keys():
        m = ModelConfig()
        assert set(asdict(m).keys()) == {"encoder", "encoder_weights", "in_channels"}


class TestConfig:
    @staticmethod
    def test_keys():
        cfg = Config()
        assert set(asdict(cfg).keys()) == {"data", "model"}


class TestLoad:
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


class TestDump:
    @staticmethod
    def test_round_trip_preserves_model(tmp_path: Path):
        cfg = Config(model=ModelConfig(encoder="mobilenet_v2", encoder_weights=None))
        out = tmp_path / "out.yaml"
        dump(cfg, out)
        loaded = load(out)
        assert loaded.model.encoder == "mobilenet_v2"
        assert loaded.model.encoder_weights is None
        assert loaded.data.root == cfg.data.root
