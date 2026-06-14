"""Tests for config dataclasses and strict YAML load/dump."""

from dataclasses import asdict
import os
from pathlib import Path

import pytest

from land_cover_segmentation.config import (
    Config,
    DataConfig,
    LossConfig,
    ModelConfig,
    OptimConfig,
    RunConfig,
    TrainConfig,
    dump,
    load,
)

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


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
        "source",
        "encoder",
        "encoder_weights",
        "in_channels",
    }


def test_config_keys():
    config = Config()
    assert set(asdict(config).keys()) == {
        "data",
        "model",
        "run",
        "optim",
        "loss",
        "train",
    }


def test_optim_config_keys():
    optim = OptimConfig()
    assert set(asdict(optim).keys()) == {
        "name",
        "lr",
        "weight_decay",
        "encoder_lr_scale",
        "encoder_lr_warmup_epochs",
    }


def test_loss_config_keys():
    loss = LossConfig()
    assert set(asdict(loss).keys()) == {"name", "use_class_weights"}


def test_train_config_keys():
    train = TrainConfig()
    assert set(asdict(train).keys()) == {
        "epochs",
        "patience",
        "grad_clip",
        "artifacts_root",
    }


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
    def test_applies_train_optim_loss_overrides(tmp_path: Path):
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text(
            "optim:\n  lr: 0.001\n  encoder_lr_warmup_epochs: 2\n"
            "loss:\n  use_class_weights: false\n"
            "train:\n  epochs: 5\n  artifacts_root: /tmp/runs\n"
        )
        cfg = load(yaml_path)
        assert cfg.optim.lr == 0.001
        assert cfg.optim.encoder_lr_warmup_epochs == 2
        assert cfg.optim.name == "adamw"
        assert cfg.loss.use_class_weights is False
        assert cfg.train.epochs == 5
        assert cfg.train.artifacts_root == "/tmp/runs"

    @staticmethod
    def test_rejects_unknown_train_key(tmp_path: Path):
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text("train:\n  epocs: 5\n")
        with pytest.raises(ValueError, match="train.epocs"):
            load(yaml_path)

    @staticmethod
    def test_applies_model_source_custom(tmp_path: Path):
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text("model:\n  source: custom\n")
        cfg = load(yaml_path)
        assert cfg.model.source == "custom"
        assert cfg.model.encoder == "efficientnet-b0"

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


@pytest.mark.parametrize("profile", os.listdir(CONFIGS_DIR))
def test_config_profile_loads(profile):
    cfg = load(CONFIGS_DIR / profile)
    assert isinstance(cfg.data.image_size, int)
    assert cfg.data.image_size > 0
    assert isinstance(cfg.data.batch_size, int)
    assert cfg.data.batch_size > 0
    assert isinstance(cfg.data.num_workers, int)
    assert cfg.data.num_workers >= 0
    assert isinstance(cfg.model.encoder, str)
    assert cfg.run.device in ["cpu", "cuda", "auto"]
    assert isinstance(cfg.run.output_name, str)
    assert isinstance(cfg.train.epochs, int)
    assert cfg.train.epochs > 0
    assert isinstance(cfg.train.patience, int)
    assert cfg.train.patience > 0
    assert cfg.train.patience < cfg.train.epochs
