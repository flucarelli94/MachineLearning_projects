"""Tests for the `cls` CLI."""

from click.testing import CliRunner

from land_cover_segmentation.cli import cls


def test_cls_help():
    result = CliRunner().invoke(cls, ["--help"])
    assert result.exit_code == 0
    assert "model" in result.output
    assert "data" in result.output


def test_model_group_help():
    result = CliRunner().invoke(cls, ["model", "--help"])
    assert result.exit_code == 0
    assert "evaluate" in result.output
    assert "predict" in result.output
    assert "train" in result.output


class TestTrainCli:
    @staticmethod
    def test_train_cli_help():
        result = CliRunner().invoke(cls, ["model", "train", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--run-name" in result.output

    @staticmethod
    def test_train_cli_requires_config():
        result = CliRunner().invoke(cls, ["model", "train"])
        assert result.exit_code != 0


class TestEvaluateCli:
    @staticmethod
    def test_evaluate_cli_help():
        result = CliRunner().invoke(cls, ["model", "evaluate", "--help"])
        assert result.exit_code == 0
        assert "--run" in result.output
        assert "--split" in result.output

    @staticmethod
    def test_evaluate_cli_requires_run():
        result = CliRunner().invoke(cls, ["model", "evaluate"])
        assert result.exit_code != 0


class TestPredictCli:
    @staticmethod
    def test_predict_cli_help():
        result = CliRunner().invoke(cls, ["model", "predict", "--help"])
        assert result.exit_code == 0
        assert "--run" in result.output
        assert "--input" in result.output
        assert "--output" in result.output

    @staticmethod
    def test_predict_cli_requires_run():
        result = CliRunner().invoke(cls, ["model", "predict"])
        assert result.exit_code != 0


class TestDataCli:
    @staticmethod
    def test_data_group_help():
        result = CliRunner().invoke(cls, ["data", "--help"])
        assert result.exit_code == 0
        assert "download" in result.output

    @staticmethod
    def test_download_cli_help():
        result = CliRunner().invoke(cls, ["data", "download", "--help"])
        assert result.exit_code == 0
        assert "--root" in result.output
        assert "--splits" in result.output
        assert "--scenes" in result.output
