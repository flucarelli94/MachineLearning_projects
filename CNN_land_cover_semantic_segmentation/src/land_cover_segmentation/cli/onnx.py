"""Export and run ONNX models."""

from pathlib import Path

import click

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def onnx() -> None:
    """Export and run ONNX models."""

def _resolve_checkpoint_path(run_dir: Path, checkpoint: str) -> Path:
    """Resolve a checkpoint CLI value to an absolute path."""
    path = Path(checkpoint)
    if path.is_file():
        return path
    return run_dir / path.name

@onnx.command("predict-onnx")
@click.option(
    "--run",
    "run_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Training run directory (contains config.yaml).",
)
@click.option(
    "--onnx",
    "onnx_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to exported .onnx model.",
)
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Input image (3-band uint8 GeoTIFF or PNG).",
)
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output path: GeoTIFF (.tif) with colormap, or RGB PNG (.png).",
)
def predict_onnx(
    run_dir: Path,
    onnx_path: Path,
    input_path: Path,
    output_path: Path,
) -> None:
    """Predict a land-cover map using an ONNX Runtime session."""
    from land_cover_segmentation.onnx_tools.predict import predict_run as predict_onnx_run

    out = predict_onnx_run(run_dir, onnx_path, input_path, output_path)
    click.echo(f"Prediction written to {out}")

@onnx.command("export")
@click.option(
    "--run",
    "run_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Training run directory (contains config.yaml and a checkpoint).",
)
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Destination .onnx path.",
)
@click.option(
    "--checkpoint",
    default="best.pth",
    show_default=True,
    help="Checkpoint filename under --run or path to a .pth file.",
)
@click.option(
    "--opset",
    default=17,
    show_default=True,
    type=int,
    help="ONNX opset version for torch.onnx.export.",
)
def export(
    run_dir: Path,
    output_path: Path,
    checkpoint: str,
    opset: int,
) -> None:
    """Export a trained checkpoint to ONNX (+ JSON metadata sidecar)."""
    from land_cover_segmentation.onnx_tools.export_onnx import export_run_to_onnx

    checkpoint_path = _resolve_checkpoint_path(run_dir, checkpoint)
    written = export_run_to_onnx(
        run_dir,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        opset_version=opset,
    )
    sidecar = written.with_suffix(".meta.json")
    click.echo(f"ONNX written to {written}")
    click.echo(f"Metadata written to {sidecar}")

__all__ = ["export", "onnx", "predict_onnx"]
