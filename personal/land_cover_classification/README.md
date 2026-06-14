# Land Cover Segmentation

Semantic segmentation prototype for [LoveDA](https://github.com/Junjue-Wang/LoveDA) land cover (RGB, 7 classes). Built with PyTorch, [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch), and [TorchGeo](https://github.com/microsoft/torchgeo).

## Install

Requires Python 3.10+.

### For users (recommended)

Install a fixed copy of the package into your environment (not editable — suitable for
training, inference, and downloading data):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install .
```

Run this from the project root after cloning or unpacking the source. This registers the
`cls` console script on your `PATH` inside the virtual environment.

To reinstall after pulling updates: `pip install .` again (or `pip install --upgrade .`).

**PyTorch note:** `pip install .` pulls `torch` and `torchvision` from PyPI. For a specific
CUDA build, install them first from the
[PyTorch install selector](https://pytorch.org/get-started/locally/), then run
`pip install .`.

### For developers

Editable install — source changes take effect without reinstalling:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .
```

Dev tools (pytest, ruff, Jupyter):

```bash
pip install pytest ruff ipykernel jupyterlab
```

### With uv (optional)

If you use [uv](https://docs.astral.sh/uv/) for development:

```bash
uv sync
```

Then prefix commands below with `uv run` (e.g. `uv run cls data download`).

## Download the dataset

LoveDA is fetched through TorchGeo. A full download (train, val, and test splits) uses roughly **20 GB** on disk. Downloads are **idempotent**: files already present under the destination directory are skipped, so re-running the command is safe.

### Default download

Stores data under `./data/loveda` (matching the default in `config.py`):

```bash
cls data download
```

With uv: `uv run cls data download`

On success, the CLI prints per-split sample counts and the total size on disk.

### Custom destination

```bash
cls data download --root /path/to/loveda
```

If you change the root, set the same path in your config YAML under `data.root`.

### Subset of splits

Useful while iterating — each split is a separate archive:

```bash
cls data download --splits train --splits val
```

Allowed splits: `train`, `val`, `test`.

### Scene filter

LoveDA contains **urban** and **rural** scenes. The `--scenes` flag controls which scene
directories are verified and counted; it does **not** reduce download size (TorchGeo always
fetches the full split archive):

```bash
cls data download --scenes urban
```

### Skip checksum verification

```bash
cls data download --no-checksum
```

Run `cls data download --help` for all options.

### Notes

- Downloaded data lives under `data/` and is listed in `.gitignore` — it is not committed to
the repository.

