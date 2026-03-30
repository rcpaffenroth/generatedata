# Add random seeds to geometric dataset generators

## Problem

The geometric dataset generators in `generatedata/data_generators.py` use `np.random.uniform()` and `np.random.normal()` (the legacy global random state) without setting a seed:

- `generate_regression_line()` (line 94)
- `generate_pca_line()` (line 109)
- `generate_circle()` (line 127)
- `generate_regression_circle()` (line 144)
- `generate_manifold()` (line 176) via `swiss_roll()`

This means every time you regenerate the data, you get different results. This is inconsistent with the LRA generators which all accept a `seed` parameter and use `np.random.default_rng(seed)`.

For a research project, reproducibility is important -- if a student regenerates data, their previous results become non-reproducible.

## Suggested Fix

Add an optional `seed` parameter to each geometric generator and use `np.random.default_rng(seed)` instead of the global random state. For example:

```python
def generate_regression_line(data_dir: Path, num_points: int = 1000, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    x_on = rng.uniform(0, 1, num_points)
    y_on = 0.73 * x_on
    x_off = x_on
    y_off = y_on + rng.normal(0, 0.1, num_points)
    ...
```

Apply the same pattern to all five geometric generators. The `swiss_roll()` helper should also accept an `rng` parameter.

Also update `generate_mnist1d()` (line 222) -- it downloads a fixed pickle so the data is deterministic, but the function interface should be consistent.

## Files to Modify

- `generatedata/data_generators.py`

## Testing

- `uv run pytest tests/` (full suite)
- Verify that calling a generator twice with the same seed produces identical output
