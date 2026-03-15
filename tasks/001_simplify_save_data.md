# Task 1 of 4 — Simplify `save_data()` by removing sequence metadata

## Series context

This is the first of four tasks that together implement the simplified time-series /
sequence approach described in `RCP-better-way`.  The key insight is that `seq_len` can
be computed at load time from the data size and `step_size`, so there is no need to store
sequence metadata in `info.json`.

## Background

The current `save_data()` in `generatedata/save_data.py` has `seq_len` and `step_size`
parameters carried over from the `feature/lra_branch`.  These are no longer needed
because the new `load_data_as_sequence()` (Task 2) will compute `seq_len` on the fly as
`x_y_index // step_size`, where `step_size` is a runtime parameter passed by the caller.

## Change required

Remove the `seq_len` and `step_size` parameters from `save_data()` and all related
validation / writing logic.

### Updated signature

```python
def save_data(
    data_dir: Path | str,
    name: str,
    start_data: dict,
    target_data: dict,
    x_y_index: int | None = None,
    onehot_y: bool = False,
    additional_info: dict | None = None,
) -> None:
```

### Code to remove

Inside the `if x_y_index is not None:` block, delete the entire sub-block that handles
`seq_len` and `step_size`:

```python
        if seq_len is not None and step_size is not None:
            if seq_len * step_size != x_y_index:
                raise ValueError(
                    f"seq_len ({seq_len}) * step_size ({step_size}) = {seq_len * step_size} "
                    f"must equal x_y_index ({x_y_index})."
                )
            data_info['seq_len'] = seq_len
            data_info['step_size'] = step_size
```

## File to modify

`generatedata/save_data.py`

## What NOT to change

- The parquet storage format and file-naming convention.
- The existing `data_info` keys (`num_points`, `size`, `x_y_index`, `x_size`, `y_size`,
  `onehot_y`).
- The `additional_info` parameter and its handling.
- Any other file in the repository.

## Verification

After this change:

1. `save_data()` should no longer accept `seq_len` or `step_size` keyword arguments.
2. The `info.json` files should never contain `seq_len` or `step_size` keys.
3. All existing non-sequence tests (`test_save_data.py`, `test_load_data.py`, etc.)
   should continue to pass.
4. Callers in `data_generators.py` that currently pass `seq_len`/`step_size` (if any)
   must be updated — but in v2 there are none, since the sequence generator functions
   were already removed.
