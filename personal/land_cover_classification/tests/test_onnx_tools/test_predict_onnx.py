"""Tests for ONNX Runtime prediction."""

import numpy as np
import pytest
import rasterio
import torch

from land_cover_segmentation.inference.predict import predict_scene
from land_cover_segmentation.models.factory import build_model
from land_cover_segmentation.onnx_tools.export_onnx import export_run_to_onnx
from land_cover_segmentation.onnx_tools.predict import (
    load_onnx_session,
    predict_run,
    predict_scene_onnx,
)
from land_cover_segmentation.training.checkpoint import CheckpointIO


def test_predict_scene_onnx_matches_pytorch(trained_run_dir, tmp_path):
    onnx_path = export_run_to_onnx(trained_run_dir, output_path=tmp_path / "model.onnx")
    session = load_onnx_session(onnx_path)

    loaded_cfg = CheckpointIO.load_run_config(trained_run_dir)
    model = build_model(loaded_cfg)
    CheckpointIO.load(
        model,
        trained_run_dir / "best.pth",
        device=torch.device("cpu"),
    )

    image = np.random.randn(3, 80, 80).astype(np.float32)
    num_classes = loaded_cfg.data.num_classes
    kwargs = dict(num_classes=num_classes, tile=32, overlap=8, batch_size=2)

    pt_prob, pt_class = predict_scene(model, image, device="cpu", **kwargs)
    onnx_prob, onnx_class = predict_scene_onnx(session, image, **kwargs)

    assert pt_prob.shape == onnx_prob.shape == (num_classes, 80, 80)
    assert pt_class.shape == onnx_class.shape == (80, 80)
    assert np.mean(pt_class == onnx_class) >= 0.95


def test_onnx_and_pytorch_predictions_are_similar(trained_run_dir, tmp_path):
    """ONNX Runtime and PyTorch should produce close probability maps and labels."""
    onnx_path = export_run_to_onnx(trained_run_dir, output_path=tmp_path / "model.onnx")
    session = load_onnx_session(onnx_path)

    cfg = CheckpointIO.load_run_config(trained_run_dir)
    model = build_model(cfg)
    CheckpointIO.load(
        model,
        trained_run_dir / "best.pth",
        device=torch.device("cpu"),
    )

    rng = np.random.default_rng(0)
    scene_kwargs = dict(
        num_classes=cfg.data.num_classes,
        tile=32,
        overlap=8,
        batch_size=2,
    )

    for height, width in ((80, 80), (83, 77), (65, 65)):
        image = rng.standard_normal((3, height, width), dtype=np.float32)
        pt_prob, pt_class = predict_scene(model, image, device="cpu", **scene_kwargs)
        onnx_prob, onnx_class = predict_scene_onnx(session, image, **scene_kwargs)

        assert pt_prob.shape == onnx_prob.shape == (cfg.data.num_classes, height, width)
        assert pt_class.shape == onnx_class.shape == (height, width)

        prob_diff = np.abs(pt_prob - onnx_prob)
        assert prob_diff.mean() < 1e-4
        assert prob_diff.max() < 1e-2
        np.testing.assert_allclose(pt_prob, onnx_prob, rtol=1e-3, atol=1e-3)

        label_agreement = float(np.mean(pt_class == onnx_class))
        assert label_agreement >= 0.99


def test_predict_run_onnx_writes_geotiff(trained_run_dir, tmp_path, synthetic_geotiff):
    onnx_path = tmp_path / "model.onnx"
    export_run_to_onnx(trained_run_dir, output_path=onnx_path)
    out_path = tmp_path / "pred_onnx.tif"
    predict_run(trained_run_dir, onnx_path, synthetic_geotiff, out_path)

    assert out_path.exists()
    with rasterio.open(out_path) as dst:
        assert dst.count == 1
        assert dst.crs is not None


def test_predict_run_onnx_missing_model_raises(trained_run_dir, synthetic_geotiff, tmp_path):
    out_path = tmp_path / "pred.tif"
    missing = tmp_path / "missing.onnx"
    with pytest.raises(FileNotFoundError, match="Missing ONNX model"):
        predict_run(trained_run_dir, missing, synthetic_geotiff, out_path)
