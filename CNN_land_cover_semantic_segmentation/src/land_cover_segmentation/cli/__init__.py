"""Unified command-line interface for land cover segmentation."""

from __future__ import annotations

import logging

import click

from land_cover_segmentation import utils
from land_cover_segmentation.cli.data import data, download
from land_cover_segmentation.cli.model import evaluate, model, predict, train
from land_cover_segmentation.cli.onnx import export, onnx, predict_onnx


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def lcs() -> None:
    """Land cover segmentation tools."""
    utils.configure_logging(__name__, level=logging.INFO)


lcs.add_command(model)
lcs.add_command(onnx)
lcs.add_command(data)


__all__ = [
    "data",
    "download",
    "evaluate",
    "export",
    "lcs",
    "model",
    "onnx",
    "predict",
    "predict_onnx",
    "train",
]
