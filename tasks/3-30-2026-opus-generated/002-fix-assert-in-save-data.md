# Replace assert with proper error raising in save_data

## Problem

In `generatedata/save_data.py` line 30, data validation uses `assert`:

```python
assert start_df.shape[1]==target_df.shape[1], 'shape mismatch'
```

Python's `assert` statements are **completely removed** when Python is run with the `-O` (optimize) flag. This means this validation silently disappears in optimized mode, which could lead to silently saving corrupted data with mismatched shapes.

## Suggested Fix

Replace the `assert` with an explicit `ValueError`:

```python
if start_df.shape[1] != target_df.shape[1]:
    raise ValueError(
        f"Shape mismatch: start_data has {start_df.shape[1]} columns "
        f"but target_data has {target_df.shape[1]} columns."
    )
```

This is a one-line change that ensures validation always runs regardless of Python optimization flags. The improved error message also tells the user exactly what went wrong.

## Files to Modify

- `generatedata/save_data.py`

## Testing

- `uv run pytest tests/test_save_data.py`
- Optionally add a test that verifies a `ValueError` is raised on mismatched shapes
