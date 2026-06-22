"""Tests for the `lcs` CLI."""

from click.testing import CliRunner

from land_cover_segmentation.cli import lcs

def test_lcs_help():
    result = CliRunner().invoke(lcs, ["--help"])
    assert result.exit_code == 0
    assert "model" in result.output
    assert "onnx" in result.output
    assert "data" in result.output

def test_model_group_help():
    result = CliRunner().invoke(lcs, ["model", "--help"])
    assert result.exit_code == 0
    assert "evaluate" in result.output
    assert "predict" in result.output
    assert "train" in result.output
    assert "export" not in result.output
    assert "predict-onnx" not in result.output

def test_onnx_group_help():
    result = CliRunner().invoke(lcs, ["onnx", "--help"])
    assert result.exit_code == 0
    assert "export" in result.output
    assert "predict-onnx" in result.output

class TestTrainCli:
    @staticmethod
    def test_train_cli_help():
        result = CliRunner().invoke(lcs, ["model", "train", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--run-name" in result.output

    @staticmethod
    def test_train_cli_requires_config():
        result = CliRunner().invoke(lcs, ["model", "train"])
        assert result.exit_code != 0

class TestEvaluateCli:
    @staticmethod
    def test_evaluate_cli_help():
        result = CliRunner().invoke(lcs, ["model", "evaluate", "--help"])
        assert result.exit_code == 0
        assert "--run" in result.output
        assert "--split" in result.output
        assert "--save-viz" in result.output

    @staticmethod
    def test_evaluate_cli_requires_run():
        result = CliRunner().invoke(lcs, ["model", "evaluate"])
        assert result.exit_code != 0

class TestPredictCli:
    @staticmethod
    def test_predict_cli_help():
        result = CliRunner().invoke(lcs, ["model", "predict", "--help"])
        assert result.exit_code == 0
        assert "--run" in result.output
        assert "--input" in result.output
        assert "--output" in result.output

    @staticmethod
    def test_predict_cli_requires_run():
        result = CliRunner().invoke(lcs, ["model", "predict"])
        assert result.exit_code != 0

class TestPredictOnnxCli:
    @staticmethod
    def test_predict_onnx_cli_help():
        result = CliRunner().invoke(lcs, ["onnx", "predict-onnx", "--help"])
        assert result.exit_code == 0
        assert "--run" in result.output
        assert "--onnx" in result.output
        assert "--input" in result.output
        assert "--output" in result.output

    @staticmethod
    def test_predict_onnx_cli_requires_onnx(trained_run_dir, tmp_path, synthetic_geotiff):
        result = CliRunner().invoke(
            lcs,
            [
                "onnx",
                "predict-onnx",
                "--run",
                str(trained_run_dir),
                "--input",
                str(synthetic_geotiff),
                "--output",
                str(tmp_path / "pred.tif"),
            ],
        )
        assert result.exit_code != 0

class TestExportCli:
    @staticmethod
    def test_export_cli_help():
        result = CliRunner().invoke(lcs, ["onnx", "export", "--help"])
        assert result.exit_code == 0
        assert "--run" in result.output
        assert "--output" in result.output
        assert "--checkpoint" in result.output
        assert "--opset" in result.output

    @staticmethod
    def test_export_cli_requires_run():
        result = CliRunner().invoke(lcs, ["onnx", "export"])
        assert result.exit_code != 0

    @staticmethod
    def test_export_cli_requires_output():
        result = CliRunner().invoke(lcs, ["onnx", "export", "--run", "."])
        assert result.exit_code != 0

    @staticmethod
    def test_export_cli_writes_onnx(trained_run_dir, tmp_path):
        output_path = tmp_path / "deploy.onnx"
        result = CliRunner().invoke(
            lcs,
            [
                "onnx",
                "export",
                "--run",
                str(trained_run_dir),
                "--output",
                str(output_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert output_path.exists()
        assert output_path.with_suffix(".meta.json").exists()
        assert "ONNX written to" in result.output

class TestDataCli:
    @staticmethod
    def test_data_group_help():
        result = CliRunner().invoke(lcs, ["data", "--help"])
        assert result.exit_code == 0
        assert "download" in result.output

    @staticmethod
    def test_download_cli_help():
        result = CliRunner().invoke(lcs, ["data", "download", "--help"])
        assert result.exit_code == 0
        assert "--root" in result.output
        assert "--splits" in result.output
        assert "--scenes" in result.output
