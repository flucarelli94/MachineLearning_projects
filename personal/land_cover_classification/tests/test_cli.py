"""Tests for the `lcs` CLI."""

from click.testing import CliRunner

from land_cover_segmentation.cli import cli


def test_cli_help():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "evaluate" in result.output
    assert "predict" in result.output
    assert "train" in result.output


class TestTrainCli:
    @staticmethod
    def test_train_cli_help():
        result = CliRunner().invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--run-name" in result.output

    @staticmethod
    def test_train_cli_requires_config():
        result = CliRunner().invoke(cli, ["train"])
        assert result.exit_code != 0


class TestEvaluateCli:
    @staticmethod
    def test_evaluate_cli_help():
        result = CliRunner().invoke(cli, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "--run" in result.output
        assert "--split" in result.output

    @staticmethod
    def test_evaluate_cli_requires_run():
        result = CliRunner().invoke(cli, ["evaluate"])
        assert result.exit_code != 0


class TestPredictCli:
    @staticmethod
    def test_predict_cli_help():
        result = CliRunner().invoke(cli, ["predict", "--help"])
        assert result.exit_code == 0
        assert "--run" in result.output
        assert "--input" in result.output
        assert "--output" in result.output

    @staticmethod
    def test_predict_cli_requires_run():
        result = CliRunner().invoke(cli, ["predict"])
        assert result.exit_code != 0
