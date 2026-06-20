# Land Cover Segmentation

Semantic segmentation prototype for [LoveDA](https://github.com/Junjue-Wang/LoveDA) land cover (RGB, 7 classes). Built with PyTorch, [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch), and [TorchGeo](https://github.com/microsoft/torchgeo).

## Install

Requires Python 3.10+.

Dependencies are split into optional extras (pip) and dependency groups (uv), aligned with
CLI commands. A bare `pip install .` only installs the CLI shell (`click`, `numpy`, `pyyaml`);
install the extras you need, or `all` for the full stack.

| Extra / group | CLI commands | Pulls in |
| --- | --- | --- |
| `data` | `lcs data download` | TorchGeo |
| `training` | `lcs model train` | `data` + PyTorch, smp, albumentations |
| `evaluation` | `lcs model evaluate` | `training` + matplotlib, pillow |
| `inference` | `lcs model predict` | PyTorch, smp, rasterio, matplotlib |
| `onnx-inference` | `lcs onnx export`, `lcs onnx predict-onnx` | `inference` + onnx, onnxruntime |
| `all` | all of the above | every extra |

### For users (recommended)

Install a fixed copy of the package (not editable):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install ".[all]"
```

Run this from the project root after cloning or unpacking the source. This registers the
`lcs` console script on your `PATH` inside the virtual environment.

Selective installs:

```bash
pip install ".[data]"              # download only
pip install ".[training]"          # train (+ data)
pip install ".[onnx-inference]"    # export + ONNX predict
```

To reinstall after pulling updates: `pip install ".[all]"` again (or add `--upgrade`).

**PyTorch note:** the extras pull `torch` and `torchvision` from PyPI. For a specific CUDA
build, install them first from the
[PyTorch install selector](https://pytorch.org/get-started/locally/), then install the
package extras.

### For developers

Editable install — source changes take effect without reinstalling:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[all]"
```

Dev tools (pytest, ruff, Jupyter) are included when using `uv sync` below, or install
manually:

```bash
pip install pytest ruff ipykernel jupyterlab
```

### With uv (optional)

If you use [uv](https://docs.astral.sh/uv/) for development:

```bash
uv sync   # dev group: all CLI groups + pytest, ruff, Jupyter
```

Selective groups:

```bash
uv sync --group training
uv sync --group onnx-inference
```

Then prefix commands below with `uv run` (e.g. `uv run lcs data download`).

## Download the dataset

LoveDA is fetched through TorchGeo. A full download (train, val, and test splits) uses roughly **20 GB** on disk. Downloads are **idempotent**: files already present under the destination directory are skipped, so re-running the command is safe.

### Default download

Stores data under `./data/loveda` (matching the default in `config.py`):

```bash
lcs data download
```

With uv: `uv run lcs data download`

On success, the CLI prints per-split sample counts and the total size on disk.

### Custom destination

```bash
lcs data download --root /path/to/loveda
```

If you change the root, set the same path in your config YAML under `data.root`.

### Subset of splits

Useful while iterating — each split is a separate archive:

```bash
lcs data download --splits train --splits val
```

Allowed splits: `train`, `val`, `test`.

### Scene filter

LoveDA contains **urban** and **rural** scenes. The `--scenes` flag controls which scene
directories are verified and counted; it does **not** reduce download size (TorchGeo always
fetches the full split archive):

```bash
lcs data download --scenes urban
```

### Skip checksum verification

```bash
lcs data download --no-checksum
```

Run `lcs data download --help` for all options.

### Notes

- Downloaded data lives under `data/` and is listed in `.gitignore` — it is not committed to
the repository.

## Train

Training is driven by YAML profiles under `configs/`. Each run writes artifacts to
`artifacts/runs/<run-name>/` (`best.pth`, `last.pth`, `config.yaml`, `run.jsonl`, `meta.json`,
`class_weights.json`).

```bash
lcs model train --config configs/base.yaml
```

Override the run directory name (defaults to `run.output_name` in the YAML):

```bash
lcs model train --config configs/fast.yaml --run-name smoke
```

With uv:

```bash
uv run lcs model train --config configs/base.yaml --run-name my-run
```

### Config profiles

| Profile | Purpose |
| ------- | ------- |
| `configs/base.yaml` | Default GPU run (EfficientNet-B0 U-Net, 30 epochs, full data) |
| `configs/full.yaml` | Longer GPU run (50 epochs) |
| `configs/fast.yaml` | CPU smoke test (MobileNetV2, 5 epochs, 50% data subset, 256 px crops) |
| `configs/custom.yaml` | Custom U-Net via `model.source: custom` and `model.unet_features` |

Normalization statistics are computed from the training split at setup time and stored in the
checkpoint payload (`best.pth`), not in the YAML.

## Evaluate

Load a trained run and score it on the validation or test split:

```bash
lcs model evaluate --run ./artifacts/runs/smoke --split val
```

Optional qualitative grid (`predictions.png` next to `metrics.json`):

```bash
lcs model evaluate --run ./artifacts/runs/smoke --split val --save-viz
```

Output: `metrics.json` with mIoU, per-class IoU, pixel accuracy, and loss.

## Export ONNX

After training, export the best checkpoint to a portable ONNX graph (manual step — not run
automatically during training):

```bash
lcs onnx export --run ./artifacts/runs/smoke \
  --output ./artifacts/runs/smoke/model.onnx
```

Writes the ONNX file and a sibling `model.meta.json` sidecar. Override the checkpoint source:

```bash
lcs onnx export --run ./artifacts/runs/smoke \
  --output ./artifacts/runs/smoke/deploy.onnx \
  --checkpoint last.pth
```

`--output` is **required** — there is no default ONNX path.

### ONNX I/O contract

The exported graph is **only the segmentation network**:

| | |
| --- | --- |
| **Input** | `float32` NCHW, already normalized with the run's `mean` / `std` (same as PyTorch predict) |
| **Output** | Multiclass logits `(batch, num_classes, height, width)` — no softmax inside the graph |

Tiling, softmax, Gaussian blending, and GeoTIFF/PNG writers stay in Python. Normalization
stats are stored in `best.pth`, copied into the ONNX `.meta.json` sidecar on export, and
read by `predict-onnx` from the checkpoint when present (otherwise from the sidecar).

## Predict

Run tiled inference on a single RGB image (PNG or 3-band GeoTIFF) using the PyTorch
checkpoint (`best.pth`):

```bash
lcs model predict --run ./artifacts/runs/smoke \
  --input ./data/loveda/Train/Rural/images_png/2.png \
  --output ./artifacts/runs/smoke/pred.png

lcs model predict --run ./artifacts/runs/smoke \
  --input ./path/to/scene.tif \
  --output ./artifacts/runs/smoke/pred.tif
```

For ONNX Runtime inference, use the separate command (requires an exported model):

```bash
lcs onnx predict-onnx --run ./artifacts/runs/smoke \
  --onnx ./artifacts/runs/smoke/model.onnx \
  --input ./path/to/scene.tif \
  --output ./artifacts/runs/smoke/pred_onnx.tif
```

- **`.png`** — palette-colored RGB map with a compact legend below the image; margins where
  the model predicts background are cropped for display.
- **`.tif`** — single-band GeoTIFF with georeferencing copied from the input and an embedded
  colormap (opens correctly in QGIS).

## Expected mIoU

Rough targets on LoveDA **val** with the default U-Net + EfficientNet-B0 profile
(`configs/base.yaml`, full data, GPU):

- **~0.50** val mIoU after a full training run (30 epochs) — a strong baseline, not SOTA.

The **`fast.yaml`** smoke profile is for pipeline verification, not accuracy: on CPU with half
the data and 256 px crops, expect **~0.20–0.30** val mIoU after 5 epochs (the smoke run in
`artifacts/runs/smoke/` reached **0.25**).

## Known limitations

- **RGB-only training** — 3-channel LoveDA imagery; multispectral Sentinel-2 training is not
  implemented.
- **GeoTIFF output** — georeferencing is preserved only when the input has a valid CRS and
  transform. PNG inputs produce ungeoreferenced PNG outputs.
- **LoveDA tile geometry** — some tiles have irregular valid regions; PNG predictions crop
  predicted-background margins for visualization, which can differ from the full tile extent.
- **CPU training time** — `fast.yaml` completes in tens of minutes on CPU; full runs expect a
  GPU (T4-class or better).
- **Dataset size** — a full LoveDA download is ~20 GB on disk.

## Extension hooks

- **Swap the smp encoder** — edit `model.encoder` and `model.encoder_weights` in
  `configs/*.yaml` (requires `model.source: smp`).
- **Custom U-Net** — set `model.source: custom` and tune `model.unet_features` (per-level
  channel widths; one entry per encoder/decoder level) in YAML — see `configs/custom.yaml`.
  Edit `build_model(cfg)` in `src/land_cover_segmentation/models/custom_model.py` only when
  you need structural changes beyond width and depth.
- **Data subset / RAM** — tune `data.fraction` (0–1] in YAML to use a deterministic subset of
  each split; useful for smoke runs (`fast.yaml` uses `0.5`).
- **ONNX deployment** — `lcs onnx export --run … --output …` then
  `lcs onnx predict-onnx --run … --onnx …`.
- **Predict input** — `lcs model predict` expects a 3-band `uint8` RGB raster (GeoTIFF or PNG).

Run `lcs model train --help`, `lcs model evaluate --help`, `lcs model predict --help`,
`lcs onnx export --help`, and `lcs onnx predict-onnx --help` for all CLI options.
