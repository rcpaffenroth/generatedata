# Consolidate load_data_as_xy and load_data_as_xy_onehot

## Problem

In `generatedata/load_data.py`, `load_data_as_xy()` (line 94) and `load_data_as_xy_onehot()` (line 123) are nearly identical functions. The only difference is that the `_onehot` variant adds an extra check that `onehot_y` is present and equals 1 in the metadata. Both functions:

1. Call `load_data()`
2. Warn about sequence datasets
3. Check for `x_y_index`, `x_size`, `y_size`
4. Return the same slice of `data["target"]`

This duplication means any future change (e.g., a bug fix) must be made in two places.

## Suggested Fix

Remove `load_data_as_xy_onehot()` and add an optional `require_onehot` parameter to `load_data_as_xy()`:

```python
def load_data_as_xy(
    name: str,
    local: bool = False,
    data_dir: Path | str | None = None,
    require_onehot: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
```

When `require_onehot=True`, perform the `onehot_y` check. This preserves the functionality while eliminating ~40 lines of duplicated code.

Alternatively, keep `load_data_as_xy_onehot()` as a thin wrapper that calls `load_data_as_xy(..., require_onehot=True)` for backwards compatibility.

## Files to Modify

- `generatedata/load_data.py`

## Files to Update (callers)

- `tests/test_load_data.py` (update test to use new signature)
- Any notebooks that call `load_data_as_xy_onehot` (check `notebooks/`)

## Testing

- `uv run pytest tests/test_load_data.py`
- Verify notebooks still work: `uv run pytest --nbmake notebooks/`
