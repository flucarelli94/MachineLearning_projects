"""Dataset download and preparation."""

from pathlib import Path

import click

from land_cover_segmentation.utils.general import human_bytes, dir_size
from land_cover_segmentation.config import VALID_SCENES, VALID_SPLITS


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def data() -> None:
    """Dataset download and preparation."""


@data.command("download")
@click.option(
    "--root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("./data/loveda"),
    show_default=True,
    help="Destination directory.",
)
@click.option(
    "--splits",
    type=click.Choice(VALID_SPLITS),
    multiple=True,
    default=VALID_SPLITS,
    show_default=True,
    help="Splits to fetch (repeatable).",
)
@click.option(
    "--scenes",
    type=click.Choice(VALID_SCENES),
    multiple=True,
    default=VALID_SCENES,
    show_default=True,
    help="Scenes to verify and count (repeatable). Does not limit download size.",
)
@click.option(
    "--checksum/--no-checksum",
    default=True,
    show_default=True,
    help="Verify file checksums against torchgeo's manifest.",
)
def download(
    root: Path,
    splits: tuple[str, ...],
    scenes: tuple[str, ...],
    checksum: bool,
) -> None:
    """Download LoveDA splits and report per-split sample counts."""
    from land_cover_segmentation.dataset.download import download_loveda

    click.echo(f"Root: {root.resolve()}")
    click.echo(
        f"Splits: {list(splits)}    Scenes: {list(scenes)}    Checksum: {checksum}"
    )

    counts = download_loveda(root=root, splits=splits, scenes=scenes, checksum=checksum)

    for split, n in counts.items():
        click.echo(f"  {split}: {n} samples")
    click.echo(f"\nDone. Total on disk under {root}: {human_bytes(dir_size(root))}")


__all__ = ["data", "download"]
