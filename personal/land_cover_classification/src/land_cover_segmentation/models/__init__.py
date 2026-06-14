"""Model factory and optional user-defined architecture hook.

Built-in models are constructed by `models.factory` (Phase C). For a custom
architecture, implement `custom_model.build_model` and set
`model.source: custom` in the YAML config.
"""
