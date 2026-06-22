"""Export trained segmentation runs to ONNX."""

import json
from datetime import UTC, datetime
from pathlib import Path

import onnx
from onnx import helper
import torch

from land_cover_segmentation import __version__
from land_cover_segmentation.models.factory import build_model
from land_cover_segmentation.training.checkpoint import CheckpointIO

def export_run_to_onnx(
    run_dir: Path,
    *,
    output_path: Path,
    checkpoint_path: Path | None = None,
    opset_version: int = 17,
) -> Path:
    """Export a trained run checkpoint to ONNX with a JSON metadata sidecar.

    Parameters
    ----------
    run_dir : pathlib.Path
        Run output directory containing `config.yaml`.
    checkpoint_path : pathlib.Path or None, optional
        Checkpoint to load. Defaults to `run_dir / "best.pth"`.
    output_path : pathlib.Path
        Destination ``.onnx`` file to write (required).
    opset_version : int, optional
        ONNX opset version passed to `torch.onnx.export`.

    Returns
    -------
    pathlib.Path
        Path to the written ONNX artifact.

    Raises
    ------
    FileNotFoundError
        If the run config or checkpoint is missing.
    RuntimeError
        If `torch.onnx.export` fails for the resolved model.
    """
    run_dir = Path(run_dir)
    cfg = CheckpointIO.load_run_config(run_dir)
    checkpoint_path = Path(
        checkpoint_path or CheckpointIO.default_checkpoint_path(run_dir)
    )
    output_path = Path(output_path)

    model = build_model(cfg)
    device = torch.device("cpu")
    payload = CheckpointIO.load(model, checkpoint_path, device=device)
    model.eval()

    mean = payload["mean"]
    std = payload["std"]
    num_classes = cfg.data.num_classes
    in_channels = cfg.model.in_channels
    image_size = cfg.data.image_size

    dummy = torch.zeros(1, in_channels, image_size, image_size, device=device)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    onnx_metadata = {
        "mean": json.dumps(mean),
        "std": json.dumps(std),
        "num_classes": str(num_classes),
        "in_channels": str(in_channels),
        "package_version": __version__,
    }

    try:
        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch", 2: "height", 3: "width"},
                "logits": {0: "batch", 2: "height", 3: "width"},
            },
            opset_version=opset_version,
            dynamo=False,
        )
        onnx_model = onnx.load(str(output_path))
        helper.set_model_props(onnx_model, onnx_metadata)
        onnx.save(onnx_model, str(output_path))
    except Exception as exc:
        raise RuntimeError(
            f"ONNX export failed for model.source={cfg.model.source!r}, "
            f"model.encoder={cfg.model.encoder!r}"
        ) from exc

    sidecar = {
        "artifact": output_path.name,
        "checkpoint": checkpoint_path.name,
        "opset": opset_version,
        "mean": mean,
        "std": std,
        "model_source": cfg.model.source,
        "model_encoder": cfg.model.encoder,
        "timestamp": datetime.now(UTC).isoformat(),
        "package_version": __version__,
    }
    output_path.with_suffix(".meta.json").write_text(json.dumps(sidecar, indent=2))

    return output_path

__all__ = ["export_run_to_onnx"]
