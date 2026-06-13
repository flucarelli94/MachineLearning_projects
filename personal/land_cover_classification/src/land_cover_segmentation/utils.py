"""Generic utilities shared across the package.

Filesystem helpers (used by the downloader and, later, the checkpoint
writer) plus small color helpers used to render label palettes. Kept
dependency-free so it stays cheap to import from anywhere.
"""

from pathlib import Path


def human_bytes(n: float) -> str:
    """Format a byte count as a human-readable string (e.g. ``"3.7 GiB"``)."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def dir_size(path: str | Path) -> int:
    """Total size in bytes of all files under ``path`` (recursive). Missing dirs return 0."""
    p = Path(path)
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    """Convert a ``"#RRGGBB"`` string to an ``(R, G, B)`` tuple of ints in ``0..255``.

    Parameters
    ----------
    h : str
        Hex color string. A leading ``"#"`` is optional; the remaining
        characters must be exactly 6 hex digits.

    Returns
    -------
    tuple[int, int, int]
        Red, green, blue components in ``0..255``.

    Raises
    ------
    ValueError
        If ``h`` does not contain exactly 6 hex digits (after stripping a
        leading ``"#"``).
    """
    h = h.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Expected 6-digit hex color, got {h!r}")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


__all__ = ["human_bytes", "dir_size", "hex_to_rgb"]
