# Task 1 of 3 — Extend `save_data()` with sequence metadata

## Series context

This is the first of three tasks that together add time-series / sequence support to the
`generatedata` library.  This task is **purely about the storage layer**: no generator
functions and no load functions are changed here.  The later tasks depend on this one.

## Background

`generatedata/save_data.py` contains the single function `save_data()` which writes two
parquet files and maintains a shared `info.json` metadata file.  The current signature is:

```python
def save_data(
    data_dir,
    name: str,
    start_data: dict,
    target_data: dict,
    x_y_index: int | None = None,
    onehot_y: bool = False,
    additional_info: dict | None = None,
) -> None:
```

When `x_y_index` is provided it records the split between the feature columns (`x_size`)
and the label columns (`y_size`) inside `info.json`.  A typical MNIST entry looks like:

```json
"MNIST": {
    "num_points": 1000,
    "size": 794,
    "x_y_index": 784,
    "x_size": 784,
    "y_size": 10,
    "onehot_y": 1
}
```

The **flat storage format must not change** — data continues to be stored as a 2-D
parquet with shape `(num_points, total_features)`.  The new metadata fields are purely
descriptive; they tell downstream loaders how to interpret / reshape the flat array.

## Change required

Add two optional keyword parameters to `save_data()`:

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `seq_len` | `int \| None` | `None` | Number of timesteps in the sequence |
| `step_size` | `int \| None` | `None` | Number of feature values per timestep (not counting the label) |

When both `seq_len` and `step_size` are provided:

1. Validate that `seq_len * step_size == x_y_index` (i.e. the pixel columns are exactly
   covered by the sequence).  Raise a `ValueError` with a clear message if not:
   ```
   ValueError: seq_len (784) * step_size (1) = 784 must equal x_y_index (785).
   ```
2. Write both values into `info.json` under the dataset name:
   ```json
   "seq_len": 784,
   "step_size": 1
   ```

When either parameter is `None` (the default) write nothing extra — full backward
compatibility is preserved.

## Updated signature

```python
def save_data(
    data_dir: Path | str,
    name: str,
    start_data: dict,
    target_data: dict,
    x_y_index: int | None = None,
    onehot_y: bool = False,
    additional_info: dict | None = None,
    seq_len: int | None = None,
    step_size: int | None = None,
) -> None:
```

## File to modify

`generatedata/save_data.py` — add two parameters and approximately 6–8 lines of logic
inside the existing `if x_y_index is not None:` block.

## What NOT to change

- The parquet storage format and file-naming convention.
- The existing `data_info` keys (`num_points`, `size`, `x_y_index`, `x_size`, `y_size`,
  `onehot_y`).
- Any other file in the repository.

## Verification

After the change the following call must write `"seq_len": 784, "step_size": 1` into
`info.json` and must not affect any other existing key:

```python
save_data(data_dir, "MNIST_seq1", start_data, target_data,
          x_y_index=784, onehot_y=True, seq_len=784, step_size=1)
```

And this call must behave identically to the current implementation (no new keys written):

```python
save_data(data_dir, "MNIST", start_data, target_data,
          x_y_index=784, onehot_y=True)
```
