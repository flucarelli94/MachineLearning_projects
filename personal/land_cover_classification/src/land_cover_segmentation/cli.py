"""Unified command-line interface for training, evaluation, and prediction."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import click

from land_cover_segmentation.config import load
from land_cover_segmentation.dataset.loveda import LoveDADataModule
from land_cover_segmentation.engine.evaluator import Split, evaluate_run
from land_cover_segmentation.engine.trainer import Trainer
from land_cover_segmentation.inference.predict import predict_run
from land_cover_segmentation.models.factory import build_model


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """Land cover segmentation tools."""


@cli.command("train")
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
    cfg = load(config)
    if run_name is not None:
        cfg = replace(cfg, run=replace(cfg.run, output_name=run_name))

    model = build_model(cfg)
    datamodule = LoveDADataModule(cfg.data)
    result = Trainer(model, cfg, datamodule).fit()

    click.echo(f"Run directory: {result['run_dir']}")
    click.echo(f"Best val mIoU: {result['best_val_miou']:.4f}")
    click.echo(f"Epochs run: {result['epochs_run']}")
    if result["stopped_early"]:
        click.echo("Stopped early (patience exceeded).")


@cli.command("evaluate")
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
def evaluate(run_dir: Path, split: Split) -> None:
    """Evaluate a trained run and write metrics.json."""
    metrics = evaluate_run(run_dir, split=split)

    click.echo(f"Split: {metrics['split']}")
    click.echo(f"mIoU: {metrics['miou']:.4f}")
    click.echo(f"Pixel accuracy: {metrics['pixel_acc']:.4f}")
    click.echo(f"Loss: {metrics['loss']:.4f}")
    click.echo(f"Metrics written to {run_dir / 'metrics.json'}")


@cli.command("predict")
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
    help="Output GeoTIFF path for the class map.",
)
def predict(run_dir: Path, input_path: Path, output_path: Path) -> None:
    """Predict a land-cover map for an input image."""
    out = predict_run(run_dir, input_path, output_path)
    click.echo(f"Prediction written to {out}")


__all__ = ["cli", "evaluate", "predict", "train"]
