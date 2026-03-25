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

Native implementations of the [Long Range Arena](https://github.com/google-research/long-range-arena) benchmark tasks for evaluating sequence models on long-context problems. Generated via `generate_all(..., lra=True)` or the `--lra` CLI flag.

- **`lra_listops`**: Hierarchical expression evaluation — nested MIN/MAX/MEDIAN/SUM_MOD operators over single-digit integers (10,000 points, 2048 token sequence + 10 classes)
- **`lra_text`**: IMDB byte-level sentiment classification — movie reviews encoded as raw byte sequences (10,000 points, 4096 byte sequence + 2 classes)
- **`lra_image`**: CIFAR-10 sequential classification — grayscale images flattened in raster-scan order (10,000 points, 1024 pixel sequence + 10 classes)
- **`lra_pathfinder`**: Synthetic visual path connectivity — determine whether two dots in a 32×32 image are connected by a curve (10,000 points, 1024 pixel sequence + 2 classes)
- **`lra_pathx`**: Extended Pathfinder at 128×128 resolution — same task with much longer sequences (2,000 points, 16384 pixel sequence + 2 classes)

When the full parameter sweep is enabled (`generate_all(..., all=True)`), the library generates hundreds of MNIST custom and MNIST1D custom variants across grids of transform parameters — including EMNIST and FashionMNIST families.

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
uv sync --extra dev
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
- `label_every_step=True` (default) broadcasts labels to every timestep and concatenates them
- `label_every_step=False` returns pixels only; labels are returned separately

### Loading LRA Datasets

LRA datasets follow the same API as all other datasets. They are especially well suited for the sequence loading API:

```python
from generatedata.load_data import load_data_as_sequence, load_data_as_xy_onehot

# ListOps: one token per timestep
X_seq, labels = load_data_as_sequence('lra_listops', step_size=1)
# X_seq shape: (10000, 2048, 1)

# Pathfinder: one row of pixels per timestep
X_seq, labels = load_data_as_sequence('lra_pathfinder', step_size=32)
# X_seq shape: (10000, 32, 32)

# Or load as flat features / one-hot labels
X, Y = load_data_as_xy_onehot('lra_image')
# X shape: (10000, 1024), Y shape: (10000, 10)
```

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

## Repository Structure

```
generatedata/
├── generatedata/           # Main library code
│   ├── load_data.py       # Data loading (flat, X/Y, sequence)
│   ├── save_data.py       # Data saving utilities
│   ├── data_generators.py # Core dataset generators + transforms
│   ├── lra_generators.py  # Long Range Arena (LRA) benchmark generators
│   ├── StartTargetData.py # PyTorch dataset class
│   ├── df_to_tensor.py    # DataFrame to tensor conversion
│   └── config.py          # Configuration (data URL, etc.)
├── scripts/               # Data generation scripts
├── notebooks/             # Example notebooks
├── tests/                 # Test suite
└── data/                  # Generated datasets
    ├── processed/         # Processed parquet files
    ├── raw/              # Raw data files
    └── external/         # External datasets (e.g., MNIST)
```

## Development

### Running Tests

```bash
uv run pytest
```

### Generating Data Locally

The main entry point for generating all datasets is:

```bash
# Generate core datasets only
uv run python scripts/generatedata_local.py

# Generate core datasets + full parameter sweeps
uv run python scripts/generatedata_local.py --all

# Generate Long Range Arena (LRA) benchmark datasets
uv run python scripts/generatedata_local.py --lra

# Generate everything
uv run python scripts/generatedata_local.py --all --lra
```

This script will generate datasets and place them in the `data/processed/` directory.

#### Advanced: Generate Individual Datasets

The core dataset generation functions are in `generatedata/data_generators.py`. Each function generates a specific dataset and can be called directly for custom workflows. Example (from Python):

```python
from generatedata.data_generators import generate_circle
from pathlib import Path
generate_circle(Path('data/processed/'), num_points=2000)
```

See the source for available generators: `generate_regression_line`, `generate_pca_line`, `generate_circle`, `generate_regression_circle`, `generate_manifold`, `generate_mnist1d`, `generate_mnist1d_custom`, `generate_mnist`, `generate_mnist_custom`, `generate_emlocalization`, `generate_lunarlander`, `generate_massspec`, and `generate_all`. LRA generators are in `generatedata/lra_generators.py`: `generate_lra_listops`, `generate_lra_text`, `generate_lra_image`, `generate_lra_pathfinder`, and `generate_lra_pathx`.

#### Copying Data to HTTP-Served Directory

To make generated data available via HTTP (e.g., for remote loading), use:

```bash
./scripts/copy_data_to_http.sh /path/to/http/dir
```

This will copy all processed data to the specified directory. Ensure you have write permissions and that your web server is configured to serve from this location.

### Example Notebooks

- `notebooks/1-rcp-visualize-data.ipynb`: Visualization examples and data exploration patterns.
- `notebooks/2-rcp-scikit-learn.ipynb`: Integration with scikit-learn RandomForest models for regression and classification tasks.
- `notebooks/3-rcp-load_data.ipynb`: Demonstrates the `load_data` API and dataset metadata.
- `notebooks/4-rcp-timeseries-datasets.ipynb`: Interactive sequence builder — step-by-step pixel reveal, heatmap visualisation, and a complete LSTM classifier training example using `load_data_as_sequence`.

## License

BSD 3-Clause License

## Author

Randy Paffenroth (rcpaffenroth@wpi.edu)
