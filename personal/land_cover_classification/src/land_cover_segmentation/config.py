"""Typed configuration for the land-cover segmentation pipeline.

This module is the **single home** for data settings (LoveDA-specific, via `DataConfig`),
model design (via `ModelConfig`), run settings (via `RunConfig`), optimizer (via `OptimConfig`),
loss (via `LossConfig`), and training loop configuration (via `TrainConfig`).
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
        RNG seed for shuffling, augmentation, and dataset subsampling.
        Controls reproducibility.
    fraction : float
        Fraction of each split to use, in `(0, 1]`. `1.0` uses the full
        split; `0.5` uses a deterministic random half of train, val, and test.
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
    fraction: float = 1.0
    classes: list[str] = field(default_factory=lambda: list(_LOVEDA_CLASSES))
    palette: list[str] = field(default_factory=lambda: list(_LOVEDA_PALETTE))

    def __post_init__(self) -> None:
        if not 0 < self.fraction <= 1.0:
            raise ValueError(
                f"Unsupported data.fraction {self.fraction!r}; expected a value in (0, 1]."
            )

    @property
    def num_classes(self) -> int:
        """Number of foreground classes."""
        return len(self.classes)


@dataclass
class ModelConfig:
    """Configuration for the segmentation model.

    `source` selects which builder `land_cover_segmentation.models.factory.build_model`
    uses:

    * `"smp"` (default) — U-Net via segmentation-models-pytorch; `encoder`, `encoder_weights`,
        and `in_channels` apply.
    * `"custom"` — calls `build_model` in `land_cover_segmentation.models.custom_model`
        (fixed module path and function name; not configurable).

    Attributes
    ----------
    source : str
        `"smp"` (segmentation_models.pytorch) or `"custom"`.
    encoder : str
        Backbone name for the built-in smp U-Net (e.g. `"efficientnet-b0"`, `"resnet34"`).
        Ignored when `source="custom"`.
    encoder_weights : str or None
        Pretrained encoder weights (`"imagenet"`) or `None`. Ignored when `source="custom"`.
    in_channels : int
        Input image channels. LoveDA RGB uses `3`.
    """

    source: str = "smp"
    encoder: str = "efficientnet-b0"
    encoder_weights: str | None = "imagenet"
    in_channels: int = 3

    def __post_init__(self) -> None:
        if self.source not in ("smp", "custom"):
            raise ValueError(
                f"Unsupported model.source {self.source!r}; expected 'smp' or 'custom'."
            )


@dataclass
class RunConfig:
    """Configuration for a single training or evaluation run.

    Attributes
    ----------
    output_name : str
        Subdirectory name under `TrainConfig.artifacts_root` where the run's
        checkpoints, logs, and metadata are written.
    seed : int
        Global RNG seed for training.
        Separate from `DataConfig.seed`, which controls dataloader shuffle and augmentations.
    deterministic : bool
        If `True`, prefer reproducible CUDA/cuDNN algorithms over speed.
    device : str
        Compute device: `"auto"` (CUDA if available, else CPU), `"cpu"`,
        or `"cuda"`.
    """

    output_name: str = "default"
    seed: int = 142
    deterministic: bool = False
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError(
                f"Unsupported run.device {self.device!r}; expected 'auto', 'cpu', or 'cuda'."
            )


@dataclass
class OptimConfig:
    """Configuration for the optimizer and learning-rate settings.

    Attributes
    ----------
    name : str
        Optimizer identifier. Only `"adamw"` is supported.
    lr : float
        Peak learning rate for decoder / non-encoder parameters.
    weight_decay : float
        AdamW weight decay coefficient.
    encoder_lr_scale : float
        Encoder learning rate as a fraction of `lr` after warmup
        (e.g. `0.1` → encoder LR is `0.1 * lr`).
    encoder_lr_warmup_epochs : int
        Number of epochs to linearly ramp the encoder LR from `0` to
        `encoder_lr_scale * lr`.
    """

    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 1e-4
    encoder_lr_scale: float = 0.1
    encoder_lr_warmup_epochs: int = 5

    def __post_init__(self) -> None:
        if self.name != "adamw":
            raise ValueError(f"Unsupported optim.name {self.name!r}; expected 'adamw'.")


@dataclass
class LossConfig:
    """Configuration for the training loss.

    Attributes
    ----------
    name : str
        Loss identifier. Only `"dice_ce"` (0.5 CE + 0.5 Dice) is supported.
    use_class_weights : bool
        If `True`, derive per-class CE weights from train-set pixel counts.
    """

    name: str = "dice_ce"
    use_class_weights: bool = True

    def __post_init__(self) -> None:
        if self.name != "dice_ce":
            raise ValueError(
                f"Unsupported loss.name {self.name!r}; expected 'dice_ce'."
            )


@dataclass
class TrainConfig:
    """Configuration for the training loop.

    Attributes
    ----------
    epochs : int
        Maximum number of training epochs.
    patience : int
        Early-stopping patience on validation mean IoU.
    grad_clip : float
        Max gradient norm for global clipping; `0` disables clipping.
    artifacts_root : str
        Root directory for run outputs (checkpoints, logs, metadata). Each run
        writes under `{artifacts_root}/{RunConfig.output_name}/`.
    """

    epochs: int = 5
    patience: int = 6
    grad_clip: float = 1.0
    artifacts_root: str = "artifacts/runs"

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError(
                f"train.patience must be >= 1, got {self.patience}."
            )


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
    optim : OptimConfig
        Optimizer and learning-rate schedule.
    loss : LossConfig
        Training loss.
    train : TrainConfig
        Training loop (epochs, early stopping, artifact paths).
    """

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    run: RunConfig = field(default_factory=RunConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

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
    raise `ValueError` carrying the full dotted path for the typo.
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
    """Load a YAML config file and deep-merge it onto `Config` defaults.

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
    """Write a fully-resolved `Config` to YAML.

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


__all__ = [
    "Config",
    "DataConfig",
    "LossConfig",
    "ModelConfig",
    "OptimConfig",
    "RunConfig",
    "TrainConfig",
    "load",
    "dump",
]
