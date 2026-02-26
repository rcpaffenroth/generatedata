# Task 2 of 3 — Add sequence generator functions for MNIST datasets

## Series context

This is the second of three tasks adding time-series support to `generatedata`.
**Task 1** extended `save_data()` with `seq_len` and `step_size` metadata parameters.
This task adds generator functions that produce sequence-flavoured variants of the
existing MNIST datasets.  **Task 3** (not yet written) will add a load function that
consumes the metadata written here.

## Background

`generatedata/data_generators.py` already contains two MNIST-family save helpers:

```python
def mnist1d_save_data(data_dir, name, num_points, mnist1d_dataset,
                      vector_dim=40, additional_info=None) -> None: ...

def mnist_save_data(data_dir, name, num_points, mnist_dataset,
                    vector_dim=28*28, additional_info=None) -> None: ...
```

Both build a flat tensor `[pixel_features | one_hot_label]`, store it via `save_data()`,
and record `x_y_index` so the loader knows where pixels end and labels begin.

The corresponding public generators are `generate_mnist1d`, `generate_mnist1d_custom`,
`generate_mnist`, and `generate_mnist_custom`.

### Current flat format (must remain unchanged)

For standard MNIST the stored parquet has shape `(num_points, 794)`:

```
columns: x0 … x783  x784 … x793
content: pixel[0]…pixel[783]  label[0]…label[9]
```

The sequence generator **must store data in exactly the same flat format** so the
existing `load_data` / `load_data_as_xy` functions continue to work unchanged.  The only
difference is that `save_data()` is called with the additional `seq_len` and `step_size`
arguments introduced in Task 1.

## Changes required

### 1. Add `pixels_per_step` parameter to both save-helpers

Extend the signatures so callers can pass sequence metadata through:

```python
def mnist1d_save_data(
    data_dir: Path,
    name: str,
    num_points: int,
    mnist1d_dataset: dict,
    vector_dim: int = 40,
    additional_info: dict | None = None,
    pixels_per_step: int | None = None,      # NEW
) -> None:
```

```python
def mnist_save_data(
    data_dir: Path,
    name: str,
    num_points: int,
    mnist_dataset,                            # torchvision Dataset, leave untyped
    vector_dim: int = 28 * 28,
    additional_info: dict | None = None,
    pixels_per_step: int | None = None,      # NEW
) -> None:
```

Inside each helper, forward the new argument to `save_data()`:

```python
# Compute seq_len only when pixels_per_step is given
seq_len = vector_dim // pixels_per_step if pixels_per_step is not None else None

save_data(data_dir, name, start_data, target_data,
          x_y_index=vector_dim, onehot_y=True,
          additional_info=additional_info,
          seq_len=seq_len, step_size=pixels_per_step)   # NEW keyword args
```

When `pixels_per_step` is `None` the call falls back to the existing behaviour (no
sequence metadata written).

### 2. Add two new public generator functions

Add these functions **after** the existing `generate_mnist` / `generate_mnist1d` blocks:

```python
def generate_mnist1d_sequence(
    data_dir: Path,
    num_points: int = 1000,
    pixels_per_step: int = 1,
) -> None:
    """Generate an MNIST1D dataset laid out as a time series.

    Downloads the standard MNIST1D dataset and stores it with sequence metadata
    so that load_data_as_sequence() can reshape it to
    (num_points, seq_len, pixels_per_step).

    Args:
        data_dir: Path to save the data.
        num_points: Number of samples.
        pixels_per_step: Features consumed per timestep (default 1).
    """
```

```python
def generate_mnist_sequence(
    data_dir: Path,
    num_points: int = 1000,
    pixels_per_step: int = 1,
    dataset_name: str = "MNIST",
) -> None:
    """Generate a torchvision MNIST-family dataset laid out as a time series.

    Stores data with sequence metadata so that load_data_as_sequence() can
    reshape it to (num_points, seq_len, pixels_per_step).

    Args:
        data_dir: Path to save the data.
        num_points: Number of samples.
        pixels_per_step: Features consumed per timestep (default 1).
        dataset_name: One of 'MNIST', 'EMNIST', 'KMNIST', 'FashionMNIST'.
    """
```

**Naming convention for generated datasets**

Each generator must derive a unique dataset name that encodes the variant:

- `generate_mnist1d_sequence(pixels_per_step=1)` → name `"MNIST1D_seq1"`
- `generate_mnist1d_sequence(pixels_per_step=5)` → name `"MNIST1D_seq5"`
- `generate_mnist_sequence(pixels_per_step=1, dataset_name="MNIST")` → `"MNIST_seq1"`
- `generate_mnist_sequence(pixels_per_step=28, dataset_name="FashionMNIST")` → `"FashionMNIST_seq28"`

Pattern: `f"{dataset_name}_seq{pixels_per_step}"`

### 3. Add default sequence variants to `generate_all()`

Inside `generate_all()`, after the existing `generate_mnist(data_dir)` call, add:

```python
generate_mnist1d_sequence(data_dir)           # pixels_per_step=1 default
generate_mnist_sequence(data_dir)             # MNIST, pixels_per_step=1 default
```

Do **not** add every possible `pixels_per_step` combination to `generate_all()` — only
the default (1 pixel per step).  Custom variants can be called directly by users.

## File to modify

`generatedata/data_generators.py` only.

## What NOT to change

- `save_data.py` (already updated in Task 1).
- `load_data.py` (updated in Task 3).
- The existing `generate_mnist`, `generate_mnist1d`, and their `_custom` variants.
- The flat parquet layout — sequence data is stored identically to non-sequence data.

## Verification

After these changes:

```python
generate_mnist_sequence(data_dir, num_points=100, pixels_per_step=1)
```

must create `MNIST_seq1_start.parquet` and `MNIST_seq1_target.parquet` with shape
`(100, 794)`, and `info.json` must contain:

```json
"MNIST_seq1": {
    "num_points": 100,
    "size": 794,
    "x_y_index": 784,
    "x_size": 784,
    "y_size": 10,
    "onehot_y": 1,
    "seq_len": 784,
    "step_size": 1
}
```

And the existing call `generate_mnist(data_dir, num_points=100)` must produce output
identical to before this change (no `seq_len` / `step_size` keys in `info.json`).
