"""Typed configuration for the land-cover segmentation pipeline.

This module is the **single home** for data ingestion parameters (LoveDA-specific, values (via
`DataConfig`), model architecture design (via `ModelConfig`), and run-level
settings (via `RunConfig`).
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
class ModelConfig:
    """Configuration for the segmentation model.

    The factory builds a U-Net via segmentation-models-pytorch.

    Attributes
    ----------
    encoder : str
        Backbone name understood by segmentation-models-pytorch
        (e.g. `"efficientnet-b0"`, `"resnet34"`).
    encoder_weights : str or None
        Pretrained weights for the encoder (`"imagenet"`) or `None` to
        train from scratch.
    in_channels : int
        Number of input image channels. LoveDA RGB uses `3`.
    """

    encoder: str = "efficientnet-b0"
    encoder_weights: str | None = "imagenet"
    in_channels: int = 3


@dataclass
class RunConfig:
    """Configuration for a single training or evaluation run.

    Attributes
    ----------
    output_name : str
        Subdirectory name under ``TrainConfig.ckpt_root`` where the run's
        checkpoints, logs, and metadata are written.
    seed : int
        Global RNG seed for training.
        Separate from ``DataConfig.seed``, which controls dataloader shuffle and augmentations.
    deterministic : bool
        If `True`, prefer reproducible CUDA/cuDNN algorithms over speed.
    device : str
        Compute device: ``"auto"`` (CUDA if available, else CPU), ``"cpu"``,
        or ``"cuda"``.
    """

    output_name: str = "default"
    seed: int = 142
    deterministic: bool = False
    device: str = "auto"


@dataclass
class Config:
    """Top-level project configuration.

    Attributes
    ----------
    data : DataConfig
        Data layer configuration.
    model : ModelConfig
        Model architecture configuration.
    run : RunConfig
        Run-level settings (output directory name, global seed, device).
    """

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    run: RunConfig = field(default_factory=RunConfig)

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


__all__ = ["Config", "DataConfig", "ModelConfig", "RunConfig", "load", "dump"]
