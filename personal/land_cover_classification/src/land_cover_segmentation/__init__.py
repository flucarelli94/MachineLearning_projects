"""Land cover semantic segmentation prototype (LoveDA)."""

from pathlib import Path


def _package_version() -> str:
    version_file = Path(__file__).resolve().parents[2] / "VERSION"
    return version_file.read_text(encoding="utf-8").strip()


__version__ = _package_version()
