"""Bootstrap the LoveDA dataset.

CLI entry point for downloading the LoveDA dataset. The installed console
script (see `[project.scripts]` in `pyproject.toml`) is
`land-cover-seg-download`; `python -m` also works since this module
defines an `if __name__ == "__main__"` block.

Roughly 4 GB total: (train + val + test) × (urban + rural). Subsequent runs
skip files that already exist on disk.

TorchGeo ships one archive per split; each zip contains both urban and rural.
The `--scenes` / `scenes` argument filters verification and sample counts,
not download size.

Examples
--------
Default — fetch every split and scene::

    uv run land-cover-seg-download

Equivalent `python -m` form (handy when the project isn't installed)::

    uv run python -m land_cover_segmentation.dataset.download

Custom root, skip checksum verification::

    uv run land-cover-seg-download --root ./data/loveda --no-checksum

Subset of splits (the option is repeatable)::

    uv run land-cover-seg-download --splits train --splits val
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import click
from torchgeo.datasets import LoveDA

from land_cover_segmentation import utils

Split = Literal["train", "val", "test"]
Scene = Literal["urban", "rural"]

VALID_SPLITS: tuple[Split, ...] = ("train", "val", "test")
VALID_SCENES: tuple[Scene, ...] = ("urban", "rural")


def download_loveda(
    root: Path | str = Path("./data/loveda"),
    splits: Sequence[Split] = VALID_SPLITS,
    scenes: Sequence[Scene] = VALID_SCENES,
    checksum: bool = True,
) -> dict[str, int]:
    """Download (or verify) LoveDA for the requested splits.

    Delegates to `~torchgeo.datasets.LoveDA` with `download=True`.
    Files already present under `root` are skipped, so this function is
    idempotent and safe to re-run.

    Parameters
    ----------
    root : pathlib.Path or str, optional
        Destination directory. Created if missing.
    splits : Sequence[str], optional
        Subset of `("train", "val", "test")` whose archives to download.
        Each split zip contains both urban and rural scene folders.
    scenes : Sequence[str], optional
        Subset of `("urban", "rural")` forwarded to LoveDA. Controls which
        scene directories must exist, which samples are indexed, and the
        returned counts. Does **not** limit download size — TorchGeo always
        fetches and extracts the full split archive.
    checksum : bool, optional
        If `True`, verify downloaded archives against torchgeo's manifest.

    Returns
    -------
    dict[str, int]
        Mapping `{split: num_samples}` for the requested splits, counting
        only samples from the selected `scenes`.

    Raises
    ------
    ValueError
        If `splits` or `scenes` contains a value not in the allowed set.

    Notes
    -----
    LoveDA exposes per-scene loading but not per-scene download URLs. Use
    `scenes` to match how you plan to load the dataset later (e.g. urban
    only), not to reduce bandwidth or disk usage on a fresh download.
    """
    bad_splits = [s for s in splits if s not in VALID_SPLITS]
    if bad_splits:
        raise ValueError(f"unknown splits {bad_splits}; allowed: {VALID_SPLITS}")
    bad_scenes = [s for s in scenes if s not in VALID_SCENES]
    if bad_scenes:
        raise ValueError(f"unknown scenes {bad_scenes}; allowed: {VALID_SCENES}")

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for split in splits:
        ds = LoveDA(
            root=str(root),
            split=split,
            scene=list(scenes),
            download=True,
            checksum=checksum,
        )
        counts[split] = len(ds)
    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
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
def main(
    root: Path,
    splits: tuple[str, ...],
    scenes: tuple[str, ...],
    checksum: bool,
) -> None:
    """Download LoveDA splits and report per-split sample counts."""
    click.echo(f"Root: {root.resolve()}")
    click.echo(
        f"Splits: {list(splits)}    Scenes: {list(scenes)}    Checksum: {checksum}"
    )

    counts = download_loveda(root=root, splits=splits, scenes=scenes, checksum=checksum)

    for split, n in counts.items():
        click.echo(f"  {split}: {n} samples")
    click.echo(
        f"\nDone. Total on disk under {root}: {utils.human_bytes(utils.dir_size(root))}"
    )


if __name__ == "__main__":
    main()


__all__ = ["download_loveda", "VALID_SPLITS", "VALID_SCENES"]
