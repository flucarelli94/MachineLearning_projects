"""Bootstrap the LoveDA dataset.

Download via the CLI: ``uv run lcs data download``.

This module exposes ``download_loveda()`` for programmatic use. The Click
command lives in ``land_cover_segmentation.cli``.

Roughly **20 GB** on disk for a full download (train + val + test, urban +
rural). Subsequent runs skip files that already exist on disk.

TorchGeo ships one archive per split; each zip contains both urban and rural.
The ``scenes`` argument filters verification and sample counts, not download
size.

Examples
--------
Default — fetch every split and scene::

    uv run lcs data download

Custom root, skip checksum verification::

    uv run lcs data download --root ./data/loveda --no-checksum

Subset of splits (the option is repeatable)::

    uv run lcs data download --splits train --splits val
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from torchgeo.datasets import LoveDA

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

    Delegates to `torchgeo.datasets.LoveDA` with `download=True`.
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
    A full download of all splits uses roughly 20 GB on disk. LoveDA exposes
    per-scene loading but not per-scene download URLs. Use `scenes` to match
    how you plan to load the dataset later (e.g. urban only), not to reduce
    bandwidth or disk usage on a fresh download.
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


__all__ = ["download_loveda", "VALID_SPLITS", "VALID_SCENES"]
