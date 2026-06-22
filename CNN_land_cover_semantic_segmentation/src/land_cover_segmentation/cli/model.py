"""Train, evaluate, and predict with PyTorch segmentation models."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import click

from land_cover_segmentation.config import load


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def model() -> None:
    """Train, evaluate, and run inference with segmentation models."""


@model.command("train")
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="YAML config profile to load.",
)
@click.option(
    "--run-name",
    default=None,
    help="Override `run.output_name` for this run's artifact directory.",
)
def train(config: Path, run_name: str | None) -> None:
    """Train a segmentation model and write run artifacts."""
    from land_cover_segmentation.dataset.loveda import LoveDADataModule
    from land_cover_segmentation.models.factory import build_model
    from land_cover_segmentation.training.trainer import Trainer

    cfg = load(config)
    if run_name is not None:
        cfg = replace(cfg, run=replace(cfg.run, output_name=run_name))

    built_model = build_model(cfg)
    datamodule = LoveDADataModule(cfg.data)
    result = Trainer(built_model, cfg, datamodule).fit()

    click.echo(f"Run directory: {result['run_dir']}")
    click.echo(f"Best val mIoU: {result['best_val_miou']:.4f}")
    click.echo(f"Epochs run: {result['epochs_run']}")
    if result["stopped_early"]:
        click.echo("Stopped early (patience exceeded).")


@model.command("evaluate")
@click.option(
    "--run",
    "run_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Training run directory (contains config.yaml and best.pth).",
)
@click.option(
    "--split",
    type=click.Choice(["val", "test"], case_sensitive=False),
    default="val",
    show_default=True,
    help="Dataset split to evaluate.",
)
@click.option(
    "--save-viz",
    is_flag=True,
    default=False,
    help="Write predictions.png qualitative grid.",
)
def evaluate(run_dir: Path, split: str, save_viz: bool) -> None:
    """Evaluate a trained run and write metrics.json."""
    from land_cover_segmentation.training.evaluator import evaluate_run

    metrics = evaluate_run(run_dir, split=split, save_viz=save_viz)

    click.echo(f"Split: {metrics['split']}")
    click.echo(f"mIoU: {metrics['miou']:.4f}")
    click.echo(f"Pixel accuracy: {metrics['pixel_acc']:.4f}")
    click.echo(f"Loss: {metrics['loss']:.4f}")
    click.echo(f"Metrics written to {run_dir / 'metrics.json'}")
    if save_viz:
        click.echo(f"Qualitative grid written to {run_dir / 'predictions.png'}")


@model.command("predict")
@click.option(
    "--run",
    "run_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Training run directory (contains config.yaml and best.pth).",
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
    help="Output path: GeoTIFF (.tif) with colormap, or RGB PNG (.png) with palette colors.",
)
def predict(
    run_dir: Path,
    input_path: Path,
    output_path: Path,
) -> None:
    """Predict a land-cover map for an input image (PyTorch checkpoint)."""
    from land_cover_segmentation.inference.predict import predict_run

    out = predict_run(run_dir, input_path, output_path)
    click.echo(f"Prediction written to {out}")


__all__ = ["evaluate", "model", "predict", "train"]
