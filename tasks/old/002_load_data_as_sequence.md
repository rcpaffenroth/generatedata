# Task 2 of 4 — Implement simplified `load_data_as_sequence()`

## Series context

This is the core task of the `feature/lra_branch_v2` redesign.  The key change from v1
is that `step_size` is now a **runtime parameter** passed by the caller, and `seq_len` is
**computed** from the data dimensions — neither value is stored in `info.json`.  This
means any dataset with `x_y_index` defined can be loaded as a sequence with any valid
`step_size`, without needing to generate special `_seq` variant datasets.

## Background

In v1 (`feature/lra_branch`), `load_data_as_sequence()` required `seq_len` and
`step_size` to be present in the dataset's `info.json`.  This forced users to generate
separate datasets for each `step_size` (e.g. `MNIST_seq1`, `MNIST_seq28`).

The v2 approach eliminates this complexity:

- **Any** existing dataset with `x_y_index` can be treated as a sequence.
- The caller provides `step_size` (how many feature columns per timestep).
- `seq_len` is derived: `seq_len = x_y_index // step_size`.
- A `ValueError` is raised if `x_y_index` is not evenly divisible by `step_size`.

## Change required

Add `load_data_as_sequence()` to `generatedata/load_data.py`.

### Function signature

```python
def load_data_as_sequence(
    name: str,
    step_size: int,
    local: bool = False,
    data_dir: Path | str | None = None,
    label_every_step: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
```

Note that `import numpy as np` will need to be added to the imports at the top of the
file.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Dataset name (e.g. `"MNIST"`, `"MNIST1D"`) |
| `step_size` | `int` | Number of feature values per timestep |
| `local` | `bool` | If `True`, load from local processed data directory |
| `data_dir` | `Path \| str \| None` | Override the default data directory |
| `label_every_step` | `bool` | If `True`, broadcast labels across all timesteps and concatenate with pixel data; if `False`, return pixel data only |

### Logic

1. Call `load_data(name, local=local, data_dir=data_dir)` to get the raw data dict.
2. Extract `info = data["info"]`.
3. Validate that `x_y_index` is present in `info`. If not, raise `ValueError`:
   ```
   ValueError: Dataset '{name}' has no x_y_index metadata. Cannot reshape as sequence.
   ```
4. Extract `x_y_index = info["x_y_index"]`.
5. Validate that `x_y_index` is evenly divisible by `step_size`. If not, raise
   `ValueError`:
   ```
   ValueError: x_y_index ({x_y_index}) is not evenly divisible by step_size ({step_size}).
   ```
6. Compute `seq_len = x_y_index // step_size`.
7. Split the target DataFrame:
   - `pixels = target_df.iloc[:, :x_y_index].to_numpy()`  → shape `(num_points, x_y_index)`
   - `labels = target_df.iloc[:, x_y_index:].to_numpy()`  → shape `(num_points, label_dim)`
8. Reshape pixels: `X_seq = pixels.reshape(num_points, seq_len, step_size)`.
9. If `label_every_step is True`:
   - Broadcast labels to shape `(num_points, seq_len, label_dim)`.
   - Concatenate with `X_seq` along axis 2 → shape `(num_points, seq_len, step_size + label_dim)`.
10. Return `(X_seq, labels)`.

### Return value

```
tuple[np.ndarray, np.ndarray]:
    X_seq: shape (num_points, seq_len, step_size + label_dim) if label_every_step=True,
           shape (num_points, seq_len, step_size) if label_every_step=False
    labels: shape (num_points, label_dim)
```

### Docstring

```python
"""Load any dataset with x_y_index and reshape it into a sequence.

The sequence length is computed as x_y_index // step_size.  This allows
any flat dataset to be treated as a time-series without storing sequence
metadata in info.json.

Args:
    name: Dataset name.
    step_size: Number of feature values per timestep.
    local: If True, load from local processed data directory.
    data_dir: Override the default data directory.
    label_every_step: If True, broadcast labels across all timesteps
        and concatenate with pixel sequence; if False, return pixels only.

Returns:
    (X_seq, labels) where X_seq has shape
    (num_points, seq_len, step_size [+ label_dim]) and
    labels has shape (num_points, label_dim).

Raises:
    ValueError: If x_y_index is missing or not divisible by step_size.
"""
```

## File to modify

`generatedata/load_data.py` — add `import numpy as np` to the imports and add the new
function at the end of the file.

## What NOT to change

- The existing functions (`data_names`, `get_random_data_name`, `load_data`,
  `load_data_as_xy`, `load_data_as_xy_onehot`).
- Any other file in the repository.

## Verification

After this change the following should work with any dataset that has `x_y_index`:

```python
from generatedata.load_data import load_data_as_sequence

# Load MNIST as a pixel-by-pixel sequence (784 timesteps × 1 pixel each)
X_seq, labels = load_data_as_sequence("MNIST", step_size=1, local=True)
assert X_seq.shape == (1000, 784, 11)   # 1 pixel + 10 one-hot labels
assert labels.shape == (1000, 10)

# Load MNIST as a row-by-row sequence (28 timesteps × 28 pixels each)
X_seq, labels = load_data_as_sequence("MNIST", step_size=28, local=True)
assert X_seq.shape == (1000, 28, 38)    # 28 pixels + 10 one-hot labels
assert labels.shape == (1000, 10)

# Without label broadcasting
X_seq, labels = load_data_as_sequence("MNIST", step_size=1, local=True,
                                       label_every_step=False)
assert X_seq.shape == (1000, 784, 1)    # just pixels
assert labels.shape == (1000, 10)

# Invalid step_size raises ValueError
load_data_as_sequence("MNIST", step_size=3, local=True)
# ValueError: x_y_index (784) is not evenly divisible by step_size (3).

# Dataset without x_y_index raises ValueError
load_data_as_sequence("circle", step_size=1, local=True)
# ValueError: Dataset 'circle' has no x_y_index metadata. Cannot reshape as sequence.
```
