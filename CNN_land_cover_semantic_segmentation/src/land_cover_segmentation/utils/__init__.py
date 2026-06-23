"""Generic utilities shared across the package (torch-free re-exports).

Logging setup and color helpers are re-exported here for convenience.
Dataset statistics live in `utils.data`; PyTorch helpers in `utils.model`.
"""

from land_cover_segmentation.utils.general import configure_logging, hex_to_rgb

__all__ = [
    "configure_logging",
    "hex_to_rgb",
]
