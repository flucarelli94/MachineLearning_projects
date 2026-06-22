"""Unified command-line interface for land cover segmentation."""

import logging

import click

from land_cover_segmentation import utils
from land_cover_segmentation.cli.data import data
from land_cover_segmentation.cli.model import model
from land_cover_segmentation.cli.onnx import onnx

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def lcs() -> None:
    """Land cover segmentation tools."""
    utils.configure_logging(__name__, level=logging.INFO)

lcs.add_command(model)
lcs.add_command(onnx)
lcs.add_command(data)

__all__ = ["lcs"]
