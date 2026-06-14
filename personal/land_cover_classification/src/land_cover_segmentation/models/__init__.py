"""Model factory and optional user-defined architecture hook.

Use `factory.build_model` from training and inference code. When `model.source` is `"custom"`,
that function delegates to `custom_model.build_model`.
"""

from land_cover_segmentation.models.factory import build_model

__all__ = ["build_model"]
