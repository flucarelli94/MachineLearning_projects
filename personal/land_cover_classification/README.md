# Land Cover Segmentation

Semantic segmentation prototype for [LoveDA](https://github.com/Junjue-Wang/LoveDA) land cover (RGB, 7 classes). Built with PyTorch, [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch), and [TorchGeo](https://github.com/microsoft/torchgeo).

## Install

Requires Python 3.10+.

Dependencies are split into optional **extras** (pip and uv), aligned with CLI commands. A bare
`pip install .` only installs the CLI shell (`click`, `numpy`, `pyyaml`); add the extras you
need for each workflow.

| Extra | CLI commands | Pulls in |
| --- | --- | --- |
| `data` | `lcs data download` | TorchGeo |
| `training` | `lcs model train` | `data` + PyTorch, smp, albumentations |
| `evaluation` | `lcs model evaluate` (also covers train) | `training` + matplotlib, pillow |
| `inference` | `lcs model predict` | PyTorch, smp, rasterio, matplotlib |
| `onnx-export` | `lcs onnx export` | `evaluation` + onnx |
| `onnx-inference` | `lcs onnx predict-onnx` | onnxruntime, rasterio, pillow, matplotlib (no PyTorch) |

### For users (recommended)

Install a fixed copy of the package (not editable):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install ".[evaluation]"    # train + evaluate; add others as needed
```

Run this from the project root after cloning or unpacking the source. This registers the
`lcs` console script on your `PATH` inside the virtual environment.

Selective installs:

```bash
pip install ".[data]"              # download only
pip install ".[training]"          # train (+ data)
pip install ".[inference]"         # PyTorch predict
pip install ".[onnx-export]"       # ONNX export (needs PyTorch)
pip install ".[onnx-inference]"    # ONNX predict only (no PyTorch)
```

Combine extras when needed, e.g. `pip install ".[evaluation,onnx-export,onnx-inference]"`.

To reinstall after pulling updates, re-run the same `pip install` command (add `--upgrade`
if you want).

**PyTorch note:** extras pull `torch` and `torchvision` from PyPI (CPU wheels via uv’s
PyTorch index in this repo). For a specific CUDA build, install them first from the
[PyTorch install selector](https://pytorch.org/get-started/locally/), then install the
package extras.

### For developers

Editable install — source changes take effect without reinstalling:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[evaluation,inference,onnx-export,onnx-inference]"
```

Dev tools (pytest, ruff, Jupyter) with uv:

```bash
uv sync --all-extras --group dev
```

Or install dev tools manually with pip:

```bash
pip install pytest ruff ipykernel jupyterlab
```

### With uv (optional)

If you use [uv](https://docs.astral.sh/uv/) for development:

```bash
uv sync --all-extras --group dev   # every extra + pytest, ruff, Jupyter
```

Selective extras:

```bash
uv sync --extra training
uv sync --extra evaluation
uv sync --extra onnx-export
uv sync --extra onnx-inference
```

Plain `uv sync` only installs core deps plus the `dev` group — not enough to run training
or inference. Always pass `--extra …` or `--all-extras`.

Then prefix commands below with `uv run` (e.g. `uv run lcs data download`).

### Verify installs

Full dev/test environment (all extras + pytest):

```bash
uv sync --all-extras --group dev
uv run pytest
```

Slim ONNX predict-only environment (no PyTorch):

```bash
UV_PROJECT_ENVIRONMENT=.venv-onnx-inference uv sync --extra onnx-inference --no-default-groups
UV_PROJECT_ENVIRONMENT=.venv-onnx-inference uv run lcs onnx predict-onnx --help
```

Use a separate `UV_PROJECT_ENVIRONMENT` so the slim sync does not overwrite your main dev
`.venv`. Export still requires the `onnx-export` extra (or `lcs-export-onnx` Docker image).
Predict-onnx requires the `{model.onnx}` sibling `model.meta.json` sidecar from export.

## Docker

Four images under [`docker/`](docker/), built with [uv](https://docs.astral.sh/uv/) and aligned
with pip extras. **Data and run artifacts are never copied into images** — bind-mount them at
runtime.

| Image | Dockerfile | uv extra | Use |
| --- | --- | --- | --- |
| `lcs-training` | `docker/Dockerfile.training` | `evaluation` | train, evaluate, download |
| `lcs-export-onnx` | `docker/Dockerfile.export-onnx` | `onnx-export` | ONNX export |
| `lcs-inference-pytorch` | `docker/Dockerfile.inference-pytorch` | `inference` | PyTorch predict |
| `lcs-inference-onnx` | `docker/Dockerfile.inference-onnx` | `onnx-inference` | ONNX predict (no PyTorch) |

### Mount points (inside the container)

| Path | Purpose |
| --- | --- |
| `/data/loveda` | LoveDA dataset (`data.root` in [`configs/docker/base.yaml`](configs/docker/base.yaml)) |
| `/artifacts` | Run outputs under `/artifacts/runs/<name>/` |
| `/input` | Inference input rasters (any host file mounted here) |
| `/output` | Inference outputs |

### Docker config profiles

Configs under `configs/docker/` are baked into images at build time (not bind-mounted).
They only override paths to match the mount points above; other fields fall back to
[`config.py`](src/land_cover_segmentation/config.py) defaults unless set in the YAML.

| Profile | Purpose |
| ------- | ------- |
| `configs/docker/base.yaml` | Path overlay only (`data.root`, `train.artifacts_root`) |
| `configs/docker/fast.yaml` | CPU smoke in containers (MobileNetV2, 1 epoch, 20% data, 256 px crops) |

Pass `--config` explicitly on every `docker run` / `docker compose run` — Compose does not
pick a default. CPU smoke example below uses `fast.yaml`; for a longer GPU run inside
CUDA image, use `base.yaml` (inherits code defaults: EfficientNet-B0, 5 epochs, full data).

### Build

From the project root (requires `uv.lock` present — run `uv lock` locally if missing):

```bash
# Training — CPU smoke
docker build -f docker/Dockerfile.training \
  --build-arg BUILD_TARGET=cpu \
  -t lcs-training:cpu .

# Training — CUDA (default BUILD_TARGET=cuda)
docker build -f docker/Dockerfile.training \
  --build-arg BUILD_TARGET=cuda \
  -t lcs-training:cuda .

docker build -f docker/Dockerfile.inference-pytorch -t lcs-inference-pytorch:latest .
docker build -f docker/Dockerfile.export-onnx -t lcs-export-onnx:latest .
docker build -f docker/Dockerfile.inference-onnx -t lcs-inference-onnx:latest .
```

Or build all services via Compose:

```bash
docker compose -f docker/docker-compose.yml build
```

### Run examples

Use `--user "$(id -u):$(id -g)"` so files written to bind mounts (e.g. `/artifacts`,
`/output`) are owned by your host user, not root.

Train (CPU smoke image, mounted data + artifacts):

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD/data/loveda:/data/loveda" \
  -v "$PWD/artifacts:/artifacts" \
  lcs-training:cpu \
  model train --config configs/docker/fast.yaml --run-name exp1
```

CUDA training: use `lcs-training:cuda`, add `--gpus all`, and a GPU host with
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
Example: same mounts as above with `--config configs/docker/base.yaml`.

PyTorch predict:

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD/artifacts/runs/exp1:/run:ro" \
  -v "$PWD/scene.tif:/input/scene.tif:ro" \
  -v "$PWD/output:/output" \
  lcs-inference-pytorch:latest \
  model predict --run /run --input /input/scene.tif --output /output/pred.tif
```

ONNX export (writes `model.onnx` and `model.meta.json` sidecar):

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD/artifacts:/artifacts" \
  lcs-export-onnx:latest \
  onnx export --run /artifacts/runs/exp1 --output /artifacts/runs/exp1/model.onnx
```

ONNX predict (requires export sidecar; no PyTorch in image):

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD/artifacts/runs/exp1:/run:ro" \
  -v "$PWD/scene.tif:/input/scene.tif:ro" \
  -v "$PWD/output:/output" \
  lcs-inference-onnx:latest \
  onnx predict-onnx --run /run --onnx /run/model.onnx \
    --input /input/scene.tif --output /output/pred.tif
```

Compose wrapper (same mounts preconfigured; set `UID`/`GID` for host ownership):

Train:

```bash
UID=$(id -u) GID=$(id -g) docker compose -f docker/docker-compose.yml run --rm training \
  model train --config configs/docker/fast.yaml --run-name exp1
```

PyTorch predict (add input/output bind mounts; run dir is under the preconfigured `/artifacts` mount):

```bash
UID=$(id -u) GID=$(id -g) docker compose -f docker/docker-compose.yml run --rm \
  -v "$PWD/scene.tif:/input/scene.tif:ro" \
  -v "$PWD/output:/output" \
  inference-pytorch \
  model predict --run /artifacts/runs/exp1 --input /input/scene.tif --output /output/pred.tif
```

ONNX export:

```bash
UID=$(id -u) GID=$(id -g) docker compose -f docker/docker-compose.yml run --rm export-onnx \
  onnx export --run /artifacts/runs/exp1 --output /artifacts/runs/exp1/model.onnx --opset 17
```

ONNX predict:

```bash
UID=$(id -u) GID=$(id -g) docker compose -f docker/docker-compose.yml run --rm \
  -v "$PWD/scene.tif:/input/scene.tif:ro" \
  -v "$PWD/output:/output" \
  inference-onnx \
  onnx predict-onnx --run /artifacts/runs/exp1 --onnx /artifacts/runs/exp1/model.onnx \
    --input /input/scene.tif --output /output/pred.tif
```

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
| `configs/docker/base.yaml` | Docker path overlay (`/data/loveda`, `/artifacts/runs`) |
| `configs/docker/fast.yaml` | Docker CPU smoke (1 epoch, 20% data; see Docker section) |

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

Override the ONNX opset (default **17**):

```bash
lcs onnx export --run ./artifacts/runs/smoke \
  --output ./artifacts/runs/smoke/model.onnx \
  --opset 17
```

`--output` is **required** — there is no default ONNX path.

### ONNX I/O contract

The exported graph is **only the segmentation network**:

| | |
| --- | --- |
| **Input** | `float32` NCHW, already normalized with the run's `mean` / `std` (same as PyTorch predict) |
| **Output** | Multiclass logits `(batch, num_classes, height, width)` — no softmax inside the graph |

Tiling, softmax, Gaussian blending, and GeoTIFF/PNG writers stay in Python. Normalization
stats are copied from the checkpoint into the ONNX `.meta.json` sidecar on export; the slim
`onnx-inference` install reads them from that sidecar only (no `best.pth` required).

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

For ONNX Runtime inference, use the separate command (requires an exported model and
`model.meta.json` sidecar from export):

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
- **ONNX deployment** — export with `onnx-export` extra (`lcs onnx export --run … --output …`),
  then predict with `onnx-inference` extra (`lcs onnx predict-onnx --run … --onnx …`).
- **Predict input** — `lcs model predict` expects a 3-band `uint8` RGB raster (GeoTIFF or PNG).

Run `lcs model train --help`, `lcs model evaluate --help`, `lcs model predict --help`,
`lcs onnx export --help`, and `lcs onnx predict-onnx --help` for all CLI options.
