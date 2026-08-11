# Task 3 of 4 — Focused tests for `load_data_as_sequence()`

## Series context

This task replaces the v1 test suite (`test_timeseries.py`) with focused tests that
validate the simplified v2 `load_data_as_sequence()`.  The v1 tests were spread across
three concerns (save metadata, generators, loader) — many of which no longer apply.
The v2 tests concentrate on the core value: reshaping any flat dataset into a sequence
with a caller-supplied `step_size`.

## Background

In v2:

- `save_data()` has no `seq_len` / `step_size` parameters, so no save-metadata tests are
  needed beyond what already exists in `test_save_data.py`.
- There are no `generate_mnist_sequence()` / `generate_mnist1d_sequence()` functions, so
  no generator tests are needed.
- The only new functionality is `load_data_as_sequence(name, step_size, ...)`, so all
  tests focus on that.

## Change required

Create `tests/test_timeseries.py` with the following test class and tests.

### Helper

Reuse a minimal helper to write a synthetic dataset using `save_data()` and
`compile_info_json()`:

```python
def _make_dataset(data_dir: Path, name: str, num_points: int,
                  num_features: int, label_dim: int,
                  x_y_index: int | None = None) -> None:
    """Write a synthetic flat dataset using save_data()."""
    rng = np.random.default_rng(42)
    total_cols = num_features + label_dim
    cols = {f"x{i}": rng.random(num_points) for i in range(total_cols)}
    save_data(data_dir, name, cols, cols,
              x_y_index=x_y_index, onehot_y=True if x_y_index else False)
    compile_info_json(data_dir)
```

### Test class: `TestLoadDataAsSequence`

Use `unittest.mock.patch` to mock `data_names` so it returns the synthetic dataset name,
same pattern as v1.

#### Tests to implement

1. **`test_shape_step_size_1`**
   - Create dataset with `num_features=784`, `label_dim=10`, `x_y_index=784`.
   - Call `load_data_as_sequence(name, step_size=1, label_every_step=True)`.
   - Assert `X_seq.shape == (num_points, 784, 11)` (1 pixel + 10 labels).
   - Assert `labels.shape == (num_points, 10)`.

2. **`test_shape_step_size_28`**
   - Same dataset as above.
   - Call with `step_size=28`.
   - Assert `X_seq.shape == (num_points, 28, 38)` (28 pixels + 10 labels).
   - Assert `labels.shape == (num_points, 10)`.

3. **`test_shape_step_size_784`**
   - Same dataset as above.
   - Call with `step_size=784`.
   - Assert `X_seq.shape == (num_points, 1, 794)` (784 pixels + 10 labels).
   - Assert `labels.shape == (num_points, 10)`.

4. **`test_shape_label_every_step_false`**
   - Same dataset, `step_size=1`, `label_every_step=False`.
   - Assert `X_seq.shape == (num_points, 784, 1)` (pixels only).
   - Assert `labels.shape == (num_points, 10)`.

5. **`test_returns_numpy_arrays`**
   - Verify both return values are `np.ndarray`.

6. **`test_labels_consistent_across_modes`**
   - The `labels` array should be identical regardless of `label_every_step`.

7. **`test_labels_broadcast_into_x_seq`**
   - When `label_every_step=True`, verify that `X[:, t, step_size:]` equals `labels`
     for every timestep `t`.

8. **`test_different_dataset_same_step_size`**
   - Create a second dataset with `num_features=40`, `label_dim=10`, `x_y_index=40`
     (mimicking MNIST1D).
   - Call with `step_size=1`.
   - Assert `X_seq.shape == (num_points, 40, 11)`.

9. **`test_invalid_step_size_not_divisible`**
   - Use the 784-feature dataset, call with `step_size=3`.
   - Assert `ValueError` is raised with message about divisibility.

10. **`test_dataset_without_x_y_index`**
    - Create a dataset **without** `x_y_index` (e.g. simple geometric data).
    - Assert `ValueError` is raised with message about missing `x_y_index`.

11. **`test_seq_len_computed_correctly`**
    - For multiple `(x_y_index, step_size)` pairs, verify that the returned shape's
      second dimension equals `x_y_index // step_size`.
    - Pairs to test: `(784, 1)`, `(784, 28)`, `(784, 784)`, `(40, 1)`, `(40, 10)`.

## File to create

`tests/test_timeseries.py`

## What NOT to change

- Existing test files (`test_save_data.py`, `test_load_data.py`, etc.).
- Any source files (those are covered by Tasks 1 and 2).

## Verification

Run:

```bash
cd generatedata_lra_v2
poetry run pytest tests/test_timeseries.py -v
```

All 11 tests should pass. The tests should not require network access or real dataset
downloads — they use only synthetic data created via `save_data()`.
