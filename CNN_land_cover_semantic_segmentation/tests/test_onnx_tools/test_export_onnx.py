"""Tests for ONNX export."""

import json

import onnx
import pytest

from land_cover_segmentation.onnx_tools.export_onnx import export_run_to_onnx


def test_export_run_to_onnx_writes_valid_graph(trained_run_dir, tmp_path):
    output_path = tmp_path / "model.onnx"
    written = export_run_to_onnx(trained_run_dir, output_path=output_path)

    assert written == output_path
    assert output_path.exists()

    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)

    sidecar = json.loads(output_path.with_suffix(".meta.json").read_text())
    assert sidecar["artifact"] == "model.onnx"
    assert sidecar["checkpoint"] == "best.pth"
    assert sidecar["model_source"] == "custom"
    assert sidecar["mean"] == [0.0, 0.0, 0.0]
    assert sidecar["std"] == [1.0, 1.0, 1.0]


def test_export_run_to_onnx_missing_checkpoint(trained_run_dir, tmp_path):
    missing = trained_run_dir / "missing.pth"
    with pytest.raises(FileNotFoundError, match="Missing checkpoint"):
        export_run_to_onnx(
            trained_run_dir,
            output_path=tmp_path / "model.onnx",
            checkpoint_path=missing,
        )
