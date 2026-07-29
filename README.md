# generatedata

A Python library for generating synthetic datasets in a standardized format for testing and benchmarking machine learning algorithms in dynamical systems settings.

## Overview

This library provides a collection of synthetic datasets that represent start-target pairs for dynamical systems research. Each dataset consists of "start" points (often noisy or perturbed data) and "target" points (clean data lying on the manifold of interest). This format is particularly useful for:

- Testing trajectory-based machine learning algorithms
- Benchmarking denoising methods
- Evaluating manifold learning techniques
- Dynamical systems modeling and analysis
- Time-series / sequence modeling (via the `load_data_as_sequence` API)

## Available Datasets

### 2D Geometric Datasets

- **`regression_line`**: Noisy points projected onto a line (1000 points, 2 dims)
- **`pca_line`**: Points scattered around a line in 2D space (1000 points, 2 dims)
- **`circle`**: Points near a unit circle with radial noise (1000 points, 2 dims)
- **`regression_circle`**: Points on a circle with added y-axis noise (1000 points, 2 dims)

### Higher-Dimensional Datasets

- **`manifold`**: 3D Swiss roll manifold data (1000 points, 3 dims)
- **`MNIST1D`**: 1D MNIST-like data (1000 points, 40 features + 10 one-hot labels)
- **`MNIST1Dcustom_*`**: Custom MNIST1D variants with configurable scale, translation, correlated noise, IID noise, and shear transforms (1000 points, 40 features + 10 labels)
- **`MNIST`**: Standard MNIST digits (1000 points, 784 features + 10 one-hot labels)
- **`MNIST_custom_*`**: Custom MNIST variants resized to 50×50 with configurable rotation, translation, scaling, and random erasing transforms (1000 points, 2500 features + 10 labels)

### Real-World Datasets

- **`EMlocalization`**: Electromagnetic localization data (3260 points, 160 features + 1 label)
- **`LunarLander`**: Lunar Lander game state data (4069 points, 404 features + 4 actions)
- **`MassSpec`**: Mass spectrometry data (572 points, 921 features + 512 labels)

### Long Range Arena (LRA) Benchmark Datasets

Native implementations of the [Long Range Arena](https://github.com/google-research/long-range-arena) benchmark tasks for evaluating sequence models on long-context problems. These are part of the core dataset set, so a plain `generate_all(...)` produces them — no extra flag is needed.

- **`lra_listops`**: Hierarchical expression evaluation — nested MIN/MAX/MEDIAN/SUM_MOD operators over single-digit integers (10,000 points, 2048 token sequence + 10 classes)
- **`lra_text`**: IMDB byte-level sentiment classification — movie reviews encoded as raw byte sequences (10,000 points, 4096 byte sequence + 2 classes)
- **`lra_image`**: CIFAR-10 sequential classification — grayscale images flattened in raster-scan order (10,000 points, 1024 pixel sequence + 10 classes)
- **`lra_pathfinder`**: Synthetic visual path connectivity — determine whether two dots in a 32×32 image are connected by a curve (10,000 points, 1024 pixel sequence + 2 classes)
- **`lra_pathx`**: Extended Pathfinder at 128×128 resolution — same task with much longer sequences (2,000 points, 16384 pixel sequence + 2 classes)

### Weight-Space Estimation (whest) Datasets

Datasets for the ARC White-Box Estimation Challenge 2026: given **only the weights** of a deep ReLU MLP, predict its mean layer activations under a standard normal input. Writing one network's forward map as `z_l = relu(z_{l-1} W_l)`, the target is

    F(W) = E_{x ~ N(0, I)} [ z_depth ]   in R^width

a *deterministic* function of `W` alone — the input distribution is integrated out. One row is one random network: the features are its weight tensor, the labels are the mean activations.

Ten datasets span width 8–256 and depth 8–32, four of them carrying the **officially published competition networks** with ground truth at 1e9 samples per network. Only `whest_w8_d8` is built by default; the rest are opt-in (`--whest` / `--whest-xl`).

| dataset | width | depth | networks | size | labels from |
| --- | --- | --- | --- | --- | --- |
| `whest_w256_d32` | 256 | **32** | 11 | 92.7 MB | **official**, 1e9 samples |
| `whest_w128_d32` | 128 | 32 | 44 | 93.0 MB | ours, 2²⁴ samples |
| `whest_w64_d32` | 64 | 32 | 178 | 94.9 MB | ours, 2²⁴ samples |
| `whest_w32_d32` | 32 | 32 | 701 | 94.9 MB | ours, 2²⁴ samples |
| `whest_w16_d32` | 16 | 32 | 2,700 | 94.3 MB | ours, 2²⁴ samples |
| `whest_w256_d8` | 256 | 8 | 45 | 94.8 MB | **official**, 1e9 samples |
| `whest_w16_d8` | 16 | 8 | 10,000 | 88.3 MB | ours, 2²⁴ samples |
| `whest_w8_d8` | 8 | 8 | 10,000 | 23.7 MB | ours, 2¹⁸ samples |
| `whest_w256_d32_xl` | 256 | 32 | 56 | 471.7 MB | **official**, 1e9 samples |
| `whest_w128_d32_xl` | 128 | 32 | 224 | 473.7 MB | ours, 2²⁴ samples |

Unlike every other family here they are stored as `.npy` arrays rather than a flat parquet pair, and are loadable **only** through `load_data_as_sequence` — one weight matrix per timestep, in layer order.

> 📖 **See [`README_whest.md`](./README_whest.md)** for the full reference: the competition conventions and why the forward map is `z @ W` and not `z @ W.T`, how each rung's size and network count were chosen, per-rung label noise and baseline errors, the deadness structure that shapes the ladder, usage and training examples, attribution for the official data, and the metadata schema.

When the full parameter sweep is enabled (`generate_all(..., all=True)`), the library generates hundreds of MNIST custom and MNIST1D custom variants across grids of transform parameters — including EMNIST, KMNIST, and FashionMNIST families.

## Installation

### Using uv (Recommended)

```bash
git clone <repository-url>
cd generatedata
uv sync
```

### Using uv with development dependencies

```bash
git clone <repository-url>
cd generatedata
uv sync --group dev
```

## Usage

### Loading Data

```python
from generatedata import load_data

# List available datasets
datasets = load_data.data_names()
print(datasets)

# Load a specific dataset
data = load_data.load_data('MNIST')
start_points = data['start']  # Noisy/perturbed data
target_points = data['target']  # Clean data on manifold
info = data['info']  # Dataset metadata
```

### Loading as Features / Labels

For supervised learning tasks, datasets with an `x_y_index` split can be loaded directly as `(X, Y)` pairs:

```python
from generatedata.load_data import load_data_as_xy, load_data_as_xy_onehot

# Continuous labels
X, Y = load_data_as_xy('EMlocalization')

# One-hot encoded labels
X, Y = load_data_as_xy_onehot('MNIST')
```

### Loading as Sequences (Time-Series)

Any dataset with `x_y_index` metadata can be reshaped into a time-series at load time — no special datasets need to be generated:

```python
from generatedata.load_data import load_data_as_sequence

# Reshape MNIST into a sequence: one pixel per timestep → seq_len=784
X_seq, labels = load_data_as_sequence('MNIST', step_size=1)
# X_seq shape: (1000, 784, 11)  — 784 timesteps, 1 pixel + 10 label dims per step
# labels shape: (1000, 10)

# One row per timestep → seq_len=28
X_seq, labels = load_data_as_sequence('MNIST', step_size=28)
# X_seq shape: (1000, 28, 38)  — 28 timesteps, 28 pixels + 10 label dims per step

# Pixels only (no label broadcast)
X_seq, labels = load_data_as_sequence('MNIST', step_size=28, label_every_step=False)
# X_seq shape: (1000, 28, 28)  — just pixels
```

Key points:

- `step_size` controls how many feature values form one timestep
- `seq_len` is computed as `x_y_index // step_size` (must divide evenly)
- `label_every_step=True` broadcasts labels to every timestep and concatenates them
- `label_every_step=False` returns pixels only; labels are returned separately
- `label_every_step=None` (the default) resolves per dataset: `True` for everything
  above, `False` for the whest family, where broadcasting the labels into the
  features would leak the target (passing `True` there raises)
- `part` selects which block the features and labels come from: `"target"` (the
  default), `"start"`, or `"all_layers"` for the whest family's per-layer stack

### Loading LRA Datasets

LRA datasets follow the same API as all other datasets. They are especially well suited for the sequence loading API. Note that they are not yet part of the published remote snapshot, so these examples pass `local=True` (see [Local vs Remote Data](#local-vs-remote-data)):

```python
from generatedata.load_data import load_data_as_sequence, load_data_as_xy_onehot

# ListOps: one token per timestep
X_seq, labels = load_data_as_sequence('lra_listops', step_size=1, local=True)
# X_seq shape: (10000, 2048, 11)  — 1 token + 10 label dims per step

# Tokens only, without the broadcast labels
X_seq, labels = load_data_as_sequence('lra_listops', step_size=1, local=True,
                                      label_every_step=False)
# X_seq shape: (10000, 2048, 1)

# Pathfinder: one row of pixels per timestep
X_seq, labels = load_data_as_sequence('lra_pathfinder', step_size=32, local=True,
                                      label_every_step=False)
# X_seq shape: (10000, 32, 32)

# LRA datasets record a `default_step_size`, so step_size may be omitted
X_seq, labels = load_data_as_sequence('lra_image', local=True)
# X_seq shape: (10000, 1024, 11)  — default_step_size=1

# Or load as flat features / one-hot labels.  This emits a UserWarning for
# sequence datasets, since it returns padded fixed-size rows.
X, Y = load_data_as_xy_onehot('lra_image', local=True)
# X shape: (10000, 1024), Y shape: (10000, 10)
```

### Loading whest Datasets

The whest datasets are reachable **only** through `load_data_as_sequence` — they have
no flat DataFrame form, and `load_data` / `load_data_as_xy` raise a `ValueError`
saying so. They are not in the published remote snapshot yet, so pass `local=True`:

```python
from generatedata.load_data import load_data_as_sequence

# One weight matrix per timestep, in layer order.  The recorded default_step_size
# is width**2, so step_size may be omitted.
W_seq, F = load_data_as_sequence('whest_w256_d32', local=True)
# W_seq: (11, 32, 65536) float32 memmap    F: (11, 256) final-layer means
W = W_seq.reshape(-1, 32, 256, 256)          # (nets, depth, width, width)

# `part` selects the label block: the cheap UT baseline, or the per-layer stack
_, ut_fixed = load_data_as_sequence('whest_w256_d32', local=True, part='start')
residual = F - ut_fixed                       # R(W) = F - UT_fixed(W)
```

Local arrays are **memory-mapped**, so that costs no RAM until you index into
`W_seq`; call `np.asarray(W_seq)` if you want it resident. The layer-ordered view is
the natural input for a recurrent model of the layer dynamics: the forward pass is a
product of random matrices acting on a distribution, and `F` is a functional of its
terminal distribution.

See [`README_whest.md`](./README_whest.md) for a training-loop example, how to score
an estimator the way the contest does, and the conventions that bite.

### Using with PyTorch

```python
from generatedata import load_data
from generatedata.StartTargetData import StartTargetData
from generatedata.df_to_tensor import df_to_tensor
import torch

# Load data
data = load_data.load_data('circle')
z_start = df_to_tensor(data['start'])
z_target = df_to_tensor(data['target'])

# Create PyTorch dataset
dataset = StartTargetData(z_start, z_target)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Use in training loop
for start_batch, target_batch in dataloader:
    # Your training code here
    pass
```

### Local vs Remote Data

The library supports both local and remote data loading:

```python
# Load from local files (requires local data generation)
data = load_data.load_data('MNIST', local=True)

# Load from remote URL (default)
data = load_data.load_data('MNIST', local=False)
```

The remote snapshot at `config.DATA_URL` predates the LRA generators and holds
the full `all=True` sweep (334 datasets: the core sets plus the EMNIST,
FashionMNIST, MNIST custom, and MNIST1D custom families). It does not yet
contain the `lra_*` datasets, the `whest_*` datasets, or the KMNIST family, so
those must be generated locally and loaded with `local=True`. Requesting an
unavailable name raises `ValueError` listing what is available.

## Custom Dataset Transforms

The MNIST custom generator supports these torchvision v2 transforms:

| Parameter | Description | Default |
| --- | --- | --- |
| `degrees` | Rotation range `(min, max)` | `(0, 0)` |
| `translate` | Translation range `(min, max)` as fraction of image size | `(0, 0)` |
| `scale` | Scaling range `(min, max)` | `(1, 1)` |
| `random_erasing_prob` | Probability of random erasing augmentation | `0.0` |

Images are resized to 50×50, grayscaled, and normalized before transform application. The MNIST1D custom generator offers `scale_coeff`, `max_translation`, `corr_noise_scale`, `iid_noise_scale`, and `shear_scale` parameters.

## Data Format

All datasets follow a consistent format:

- **Start points**: Initial/noisy data points
- **Target points**: Clean data points on the target manifold
- **Info**: Metadata including:
  - `num_points`: Number of data points
  - `size`: Total dimensionality
  - `x_y_index`: Split index for features/labels (if applicable)
  - `x_size`: Number of input features
  - `y_size`: Number of output labels
  - `onehot_y`: Whether labels are one-hot encoded
  - `is_sequence`: Present on sequence-native datasets whose flat rows are padded
    (the LRA family), so that loading them as flat X/Y warns
  - `default_step_size`: Lets `load_data_as_sequence` be called without an
    explicit `step_size` (1 for the LRA family, `width**2` for the whest family)
  - `storage`: Absent for the flat start/target parquet datasets; `"npy"` for the
    whest family, which stores arrays instead and is loadable only as a sequence
  - `label_every_step_allowed`: False where broadcasting the labels into the
    feature axis would leak the target (the whest family)

## Repository Structure

```
generatedata/
├── generatedata/           # Main library code
│   ├── load_data.py       # Data loading (flat, X/Y, sequence)
│   ├── save_data.py       # Data saving utilities
│   ├── data_generators.py # Core dataset generators + transforms
│   ├── lra_generators.py  # Long Range Arena (LRA) benchmark generators
│   ├── whest_generators.py # Weight-space estimation (whest) generators
│   ├── hf_data.py         # Pinned HuggingFace Hub downloads + dataset wrapper
│   ├── StartTargetData.py # PyTorch dataset class
│   ├── df_to_tensor.py    # DataFrame to tensor conversion
│   ├── plot_2D_start_end.py # Plotting helper for 2D start/target pairs
│   └── config.py          # Configuration (data URL, etc.)
├── scripts/               # Data generation scripts
├── notebooks/             # Example notebooks
├── tests/                 # Test suite
└── data/                  # Generated datasets
    ├── processed/         # Processed parquet files (and .npy arrays for whest)
    ├── raw/              # Raw data files (EM, LunarLander, MassSpec sources)
    └── external/         # Cached upstream downloads (MNIST, CIFAR-10, IMDB, KMNIST)
```

## Development

### Running Tests

```bash
uv run pytest
```

### Generating Data Locally

The main entry point for generating all datasets is:

```bash
# Generate the core datasets (including the LRA benchmark datasets)
uv run python scripts/generatedata_local.py

# Also generate the full parameter sweeps (hundreds of custom variants)
uv run python scripts/generatedata_local.py --all
```

This script will generate datasets and place them in the `data/processed/` directory. Datasets whose data files already exist are skipped, so re-running it only fills in what is missing.

#### Generating the whest ladder

The whest datasets beyond `whest_w8_d8` are opt-in, and are *not* part of `--all`:

```bash
# The eight-dataset grid: ~680 MB, ~1 GB of official data downloaded, ~40 min on a GPU
uv run python scripts/generatedata_local.py --whest

# Also the two 500 MB datasets (implies --whest)
uv run python scripts/generatedata_local.py --whest-xl
```

They are gated because, unlike the rest of the library, they want a GPU worth using
and fetch about a gigabyte from the HuggingFace Hub. Both commands print a preflight
summary of every dataset, its size and the total before writing anything, and skip
datasets that already exist with matching parameters. From Python:

```python
from generatedata.whest_generators import generate_whest_all
generate_whest_all('data/processed', include_xl=True)
```

The full ladder is 1,622 MB. See [`README_whest.md`](./README_whest.md) for the
per-rung sizes and for generating geometries that are not on the ladder.

#### Upstream Data Sources

Datasets that are not synthetic are downloaded and cached under `data/external/`. Several upstream projects publish only from a single academic web server with no CDN, which has proven unreliable, so those are fetched from CDN-backed mirrors instead:

| Dataset | Source |
| --- | --- |
| MNIST | `ossci-datasets` S3 mirror (via torchvision) |
| MNIST1D | GitHub, `greydanus/mnist1d` |
| CIFAR-10 (for `lra_image`) | HuggingFace `uoft-cs/cifar10`, revision-pinned |
| IMDB (for `lra_text`) | HuggingFace `stanfordnlp/imdb`, revision-pinned |
| Competition MLPs (`--whest` only) | HuggingFace `aicrowd/arc-whestbench-public-2026`, revision-pinned, CC-BY-4.0 |
| KMNIST (`all=True` only) | HuggingFace `tanganke/kmnist`, revision-pinned |
| FashionMNIST (`all=True` only) | Zalando S3 website endpoint (via torchvision) |
| EMNIST (`all=True` only) | NIST `biometrics.nist.gov` — still a single host with no CDN, and a 536 MB zip |

Each HuggingFace source is pinned to a commit hash in the code, so the downloaded bytes are reproducible. Downloads are cached and written atomically, so an interrupted download does not leave a truncated file behind. The `MNIST1Dcustom_*` variants need no download — they are synthesized locally by the `mnist1d` package.

#### Advanced: Generate Individual Datasets

The core dataset generation functions are in `generatedata/data_generators.py`. Each function generates a specific dataset and can be called directly for custom workflows. Example (from Python):

```python
from generatedata.data_generators import generate_circle
from pathlib import Path
generate_circle(Path('data/processed/'), num_points=2000)
```

See the source for available generators: `generate_regression_line`, `generate_pca_line`, `generate_circle`, `generate_regression_circle`, `generate_manifold`, `generate_mnist1d`, `generate_mnist1d_custom`, `generate_mnist`, `generate_mnist_custom`, `generate_emlocalization`, `generate_lunarlander`, `generate_massspec`, and `generate_all`. LRA generators are in `generatedata/lra_generators.py`: `generate_lra_listops`, `generate_lra_text`, `generate_lra_image`, `generate_lra_pathfinder`, and `generate_lra_pathx`. The whest generators are in `generatedata/whest_generators.py` — `generate_whest` for a bake of your own and `generate_whest_official` to carve rows out of the published competition data:

```python
from generatedata.whest_generators import generate_whest, generate_whest_official
from pathlib import Path
generate_whest(Path('data/processed/'), width=16, depth=16, num_points=2000)
generate_whest_official(Path('data/processed/'), key='phase1', num_points=11)
```

That module also exposes the pieces on their own — `he_weights` (draw competition-faithful networks), `mc_layer_means` (the Monte-Carlo ground truth for every layer, returning the means and their second moments so the labels' own standard error is available), `ut_fixed_final_mean` (the cheap deterministic estimate), and `whest_dataset_bytes` (the exact size of a geometry, which is what fixes each rung's `num_points`).

#### Copying Data to HTTP-Served Directory

To make generated data available via HTTP (e.g., for remote loading), use:

```bash
cd scripts
./copy_data_to_http.sh
```

The script takes no arguments and is specific to the author's WPI hosting setup. It mounts the HTTP directory with `rcp drive mount -d html`, copies everything in `data/processed/` into a new timestamped directory under `~/mnt/html/public_html/data/generatedata/`, and then **rewrites `generatedata/config.py`** so that `DATA_URL` points at the new snapshot. It uses paths relative to `scripts/`, so run it from that directory.

### Example Notebooks

- `notebooks/1-rcp-visualize-data.ipynb`: Visualization examples and data exploration patterns.
- `notebooks/2-rcp-scikit-learn.ipynb`: Integration with scikit-learn RandomForest models for regression and classification tasks.
- `notebooks/3-rcp-load_data.ipynb`: Demonstrates the `load_data` API and dataset metadata.
- `notebooks/4-rcp-timeseries-datasets.ipynb`: Interactive sequence builder — step-by-step pixel reveal, heatmap visualisation, and a complete LSTM classifier training example using `load_data_as_sequence`.
- `notebooks/5-rcp-whest-visualization.ipynb`: The whest datasets — verifies the `z @ W` forward convention against the stored labels, then visualises the heavy tail of `F`, the ReLU cone collapsing with depth, width scaling, and the structure of the cheap baseline's residual. Adapts to whichever whest datasets are present locally.

## License

BSD 3-Clause License

## Author

Randy Paffenroth (rcpaffenroth@wpi.edu)
