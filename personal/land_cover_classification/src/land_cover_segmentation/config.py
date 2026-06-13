"""Typed configuration for the data layer.

Scope (intentionally minimal): only fields needed to build datasets and
dataloaders. Model / optimizer / training configuration will be added in a
later phase as we build those layers — keeping the schema lean now means
unused config can't drift out of sync with code.

This module is the **single home** for every LoveDA-specific value. Most
live as :class:`DataConfig` fields and can be overridden via YAML.
Per-channel normalization statistics (mean / std) are *not* declared
here — the data module computes them at runtime on a sampled subset of
the training set and passes them directly into the transforms, so the
config never carries stale placeholder numbers.

Design notes
------------
* No third-party config framework (Hydra/OmegaConf) — pure stdlib + PyYAML.
* Strict: unknown keys at any nesting level raise `ValueError` so typos
  like `dta: {root: ...}` fail fast.
* No type coercion — we trust PyYAML's native parsing.
"""

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

# Default LoveDA constants.
_LOVEDA_CLASSES: list[str] = [
    "background",
    "building",
    "road",
    "water",
    "barren",
    "forest",
    "agriculture",
]
_LOVEDA_PALETTE: list[str] = [
    "#000000",  # background
    "#E30B0B",  # building
    "#A9A9A9",  # road
    "#1E56C8",  # water
    "#A0754A",  # barren
    "#1A7A1A",  # forest
    "#F5E642",  # agriculture
]


@dataclass
class DataConfig:
    """Configuration for the data layer (dataset + dataloader).

    Only LoveDA is supported, so the dataset identifier is implicit and not
    a field. `ignore_index` is the single source of truth for the "skip
    this pixel" value used by transforms, loss, and metrics — other modules
    must read it from here rather than redefining it.

    Attributes
    ----------
    root : str
        Filesystem path where the dataset lives (or will be downloaded to).
    scene : list[str]
        LoveDA scenes to include; subset of `["urban", "rural"]`.
    image_size : int
        Side length (pixels) of the square crops fed to the model.
    batch_size : int
        Mini-batch size used by both train and val dataloaders.
    num_workers : int
        Number of worker processes per dataloader.
    ignore_index : int
        Mask value the loss and metrics must skip. LoveDA's "no-data" label
        is remapped to this value upstream, and augmentations that drop
        pixels (affine fill, coarse dropout) also write this value.
    nodata_label : int
        Label value torchgeo's LoveDA emits for "no-data" pixels (scene
        edges with no annotation). The data module remaps this to
        `ignore_index` before training.
    seed : int
        RNG seed for shuffling and augmentation. Controls reproducibility.
    classes : list[str]
        Foreground class names, ordered by integer label. `num_classes`
        is derived from this list.
    palette : list[str]
        Hex colors (`"#RRGGBB"`) per class, parallel to `classes`.

    Notes
    -----
    Per-channel normalization statistics are *not* fields here — they are
    computed at runtime by the data module from a sampled subset of the
    training set, and passed directly to the albumentations pipelines.
    """

    root: str = "./data/loveda"
    scene: list[str] = field(default_factory=lambda: ["urban", "rural"])
    image_size: int = 512
    batch_size: int = 8
    num_workers: int = 4
    ignore_index: int = 255
    nodata_label: int = 7
    seed: int = 214
    classes: list[str] = field(default_factory=lambda: list(_LOVEDA_CLASSES))
    palette: list[str] = field(default_factory=lambda: list(_LOVEDA_PALETTE))

    @property
    def num_classes(self) -> int:
        """Number of foreground classes."""
        return len(self.classes)


@dataclass
class Config:
    """Top-level project configuration.

    Currently only carries the data layer. Model / optimizer / training
    sections will be added as those layers come online.

    Attributes
    ----------
    data : DataConfig
        Data layer configuration.
    """

    data: DataConfig = field(default_factory=DataConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain nested `dict` view (suitable for YAML/JSON dump)."""
        return asdict(self)


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------


def _merge_into_dataclass(
    dc_instance: Any, overrides: Mapping[str, Any], path: str
) -> Any:
    """Recursively apply `overrides` onto a dataclass instance.

    Returns a *new* dataclass instance (defaults stay untouched). Unknown keys
    raise :class:`ValueError` carrying the full dotted path for the typo.
    """
    if not is_dataclass(dc_instance):
        raise TypeError(f"{path or '<root>'} is not a dataclass")

    field_map = {f.name: f for f in fields(dc_instance)}
    unknown = set(overrides) - set(field_map)
    if unknown:
        prefix = f"{path}." if path else ""
        keys = ", ".join(sorted(prefix + k for k in unknown))
        allowed = ", ".join(sorted(prefix + k for k in field_map))
        raise ValueError(f"Unknown config key(s): {keys}. Allowed: {allowed}")

    kwargs: dict[str, Any] = {}
    for name, _ in field_map.items():
        current = getattr(dc_instance, name)
        if name not in overrides:
            kwargs[name] = current
            continue
        new_value = overrides[name]
        sub_path = f"{path}.{name}" if path else name
        if is_dataclass(current):
            if not isinstance(new_value, Mapping):
                raise ValueError(
                    f"Expected a mapping for nested config '{sub_path}', got {type(new_value).__name__}"
                )
            kwargs[name] = _merge_into_dataclass(current, new_value, sub_path)
        else:
            kwargs[name] = new_value
    return type(dc_instance)(**kwargs)


def load(path: str | Path) -> Config:
    """Load a YAML config file and deep-merge it onto :class:`Config` defaults.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a YAML file. An empty file is treated as `{}` (all defaults).

    Returns
    -------
    Config
        A fully-resolved configuration with user overrides applied.

    Raises
    ------
    ValueError
        If the top-level YAML node is not a mapping, or if any key (at any
        nesting level) is not declared on the matching dataclass. The error
        message includes the dotted path of the offending key(s).
    """
    p = Path(path)
    with p.open("r") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"Top-level YAML in {p} must be a mapping, got {type(raw).__name__}"
        )
    return _merge_into_dataclass(Config(), raw, "")


def dump(cfg: Config, path: str | Path) -> None:
    """Write a fully-resolved :class:`Config` to YAML.

    Parent directories of `path` are created if missing. Field order from
    the dataclasses is preserved (`sort_keys=False`) so the dump stays
    diff-friendly across runs.

    Parameters
    ----------
    cfg : Config
        Configuration to serialize.
    path : str or pathlib.Path
        Destination file path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))


__all__ = ["Config", "DataConfig", "load", "dump"]
